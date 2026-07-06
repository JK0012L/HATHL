import random
import numpy as np
import sys
from scipy.optimize import linprog
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.base_brain import ContinuousSequencingBrain, ContinuousSchedulingNetwork


class sequencing_brain(ContinuousSequencingBrain):
    """HATHL-based scheduling brain, inherits shared base class, retains multi-step returns, TD(lambda) and other features"""
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initialization first
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # HATHL specific parameters        
        self.trajectory_buffer_size = 64
        self.rep_memo_size = 320        
        self.sequencing_action_NN_training_interval = 50
        self.n_step = 5
        self.lambda_param = 0.7
        self.baseline_ema_alpha = 0.99
        self.running_baseline = 0.0
        self.mixed_training_ratio = 0.7
        self.min_trajectory_length = 10
        self.good_experience_threshold = 0.7
        self.job_count = 0
        self.worker_count = 0
        # Network initialization (reuses base class network structure)
        if self.train:           
            self.network = ContinuousSchedulingNetwork(self.input_size)  # Use shared network 
            self.env.process(self.training_process_parameter_sharing())
        else:          
            self.network = ContinuousSchedulingNetwork(self.input_size)
            self.network.load_state_dict(torch.load(self.address))
            self.network.eval()
            self.multi_obj_manager.get_per_baselines(self.address)
            
    # ========== Training process (HATHL specific) ==========
    def training_process_parameter_sharing(self):
        """Training process (multi-machine shared training)"""
        yield self.env.timeout(self.warm_up + 1)
        
        # After warm-up phase, first use warm-up experience for supervised learning
        if len(self.rep_memo) > 1:
            for i in range(10):
                self.train_from_replay_memory()
        self.samples = 0
        # Start mixed training
        while not (self.job_creator.in_system_job_no == 0 and self.job_creator.index_jobs >= self.job_numbers) : 
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > 0.5 * self.trajectory_buffer_size:
                self.samples -= 0.5 * self.trajectory_buffer_size           
                if random.random() < self.mixed_training_ratio:
                    self.train_policy_gradient_with_baseline()
                else:
                    if len(self.rep_memo) >= self.rep_memo_size:
                        self.train_from_replay_memory()
            yield self.env.timeout(100)
        
        # Save model
        address = self.address.format(sys.path[0])
        torch.save(self.network.state_dict(), address)
        pref_address = address.replace('.pt', '_preferences.pkl')
        self.multi_obj_manager.save_training_results(pref_address)
        print(f"Model and preference vector saved: {address}, {pref_address}")

    # ========== Multi-step Return Calculation Functions ==========
    def compute_n_step_returns(self, trajectory, start_idx, n_step):
        """Compute n-step cumulative return"""
        if start_idx + n_step > len(trajectory):
            return None
        
        discounted_return = 0
        for i in range(n_step):
            discounted_return += (self.discount_factor ** i) * trajectory[start_idx + i][2]
        
        # If there are subsequent states, add state value estimation
        if start_idx + n_step < len(trajectory):
            next_state = torch.tensor(trajectory[start_idx + n_step][3], dtype=torch.float32).unsqueeze(0)
            next_preference = torch.tensor(trajectory[start_idx + n_step][4], dtype=torch.float32).unsqueeze(0)
            next_alpha = self.network.forward(next_state, next_preference)
            next_job_weights = trajectory[start_idx + n_step][6]
            _, next_value = self.select_by_chebyshev(next_job_weights, next_alpha)
            discounted_return += (self.discount_factor ** n_step) * next_value
        
        return discounted_return
    

    def compute_lambda_returns(self, trajectory, lambda_param=0.7):
        """Compute TD(lambda) return"""
        T = len(trajectory)
        if T == 0:
            return []
        
        # Compute all n-step returns
        n_step_returns = {}
        max_n = min(self.n_step, T)
        
        for n in range(1, max_n + 1):
            returns = []
            for t in range(T - n + 1):
                n_step_return = self.compute_n_step_returns(trajectory, t, n)
                if n_step_return is not None:
                    returns.append(n_step_return)
            if returns:
                n_step_returns[n] = torch.tensor(returns, dtype=torch.float32)
        
        # Compute lambda return
        lambda_returns = []
        for t in range(T):
            lambda_return = 0
            total_weight = 0
            for n in range(1, max_n + 1):
                if t + n <= T and n in n_step_returns and len(n_step_returns[n]) > t:
                    if t + n < T:
                        weight = (1 - lambda_param) * (lambda_param ** (n - 1))
                    else:
                        weight = lambda_param ** (n - 1)
                    lambda_return += weight * n_step_returns[n][t]
                    total_weight += weight
            
            if not torch.isnan(lambda_return / total_weight):
                lambda_returns.append(lambda_return / total_weight)
            else:
                # Fallback to 1-step return
                if t < T:
                    one_step = np.nan_to_num(self.compute_n_step_returns(trajectory, t, 1), nan=0.0, posinf=0.0, neginf=0.0)
                    step_val = torch.tensor(one_step if one_step is not None else 0, dtype=torch.float32)
                    lambda_returns.append(step_val)
        
        return lambda_returns


  # ========== Policy Gradient Training Method ==========
    def train_policy_gradient_with_baseline(self):
        """Policy gradient training method with baseline (using multi-step returns)"""
        # Check trajectory buffer
        if len(self.trajectory_buffer) < self.n_step + 1:
            if len(self.rep_memo) > self.batch_size:
                return self.train_from_replay_memory()
            else:
                return
        
        # Compute trajectory returns
        returns = self.compute_lambda_returns(self.trajectory_buffer, self.lambda_param)
        if not returns:
            if len(self.rep_memo) > self.batch_size:
                return self.train_from_replay_memory()
            else:
                return
        
        returns = torch.stack(returns)
        
        # Extract trajectory data
        states = torch.stack([torch.tensor(step[0], dtype=torch.float32)
                              for step in self.trajectory_buffer[:len(returns)]])
        actions = torch.stack([torch.tensor(step[1], dtype=torch.float32)
                               for step in self.trajectory_buffer[:len(returns)]])
        preferences = torch.stack([torch.tensor(step[4], dtype=torch.float32)
                                   for step in self.trajectory_buffer[:len(returns)]])
        
        # Normalize returns
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update baseline
        self.running_baseline = (self.baseline_ema_alpha * self.running_baseline +
                                 (1 - self.baseline_ema_alpha) * returns.mean().item())
        
        # Compute advantage function
        advantages = returns_normalized - self.running_baseline
        
        # Policy gradient loss
        predicted_alphas = self.network.forward(states, preferences)
        mse_losses = F.mse_loss(predicted_alphas, actions, reduction='none').mean(dim=1)
        log_probs = -mse_losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Supervised learning loss (learn good demonstrations from experience replay pool)
        supervised_loss = torch.tensor(0.0, device=states.device)
        if len(self.rep_memo) >= self.rep_memo_size and random.random() < 0.3:
            supervised_loss = self.compute_supervised_loss()
        
        # Total loss
        total_loss = policy_loss + 0.1 * supervised_loss
        
        # Backpropagation
        self.network.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.network.optimizer.step()
        
        # Store good trajectories to experience replay pool
        self.store_good_experiences_to_replay(states, actions, returns)
        
        # Record and cleanup
        self.record_training_metrics(total_loss, policy_loss, supervised_loss, returns)
        
        
        return total_loss.item()

    # ========== Functions for Training from Experience Replay Pool ==========
    def train_from_replay_memory(self):
        """Supervised learning from experience replay pool (behavioral cloning)"""        
        sample_num = min(len(self.rep_memo),self.batch_size)
        # Random sample a batch of experience
        minibatch = random.sample(self.rep_memo, sample_num)
        
        # Extract data
        states = torch.stack([data[0] for data in minibatch], dim=0)
        actions = torch.stack([torch.tensor(data[1], dtype=torch.float32) for data in minibatch], dim=0)
        preferences = torch.stack([data[4] for data in minibatch], dim=0)
        
        # Network forward propagation
        predicted_actions = self.network.forward(states, preferences)
        
        # Supervised learning loss
        loss = F.mse_loss(predicted_actions, actions)
        
        # Backpropagation
        self.network.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.network.optimizer.step()
        
        if self.training_step_count % 10 == 0 :
            print(f'\033[32m[HATHL Supervised Learning]\033[0m Count:{self.training_step_count},'
                f'Arrived jobs: {self.job_creator.index_jobs},'
                f'WIP: {self.job_creator.in_system_job_no}, '
                f' Loss:{loss.item():.5f} '
                )
        
        self.training_step_count += 1
        return loss.item()

    # ========== Helper Functions ==========
    def compute_supervised_loss(self):
        """Compute supervised learning loss from experience replay pool"""
        if len(self.rep_memo) < 32:
            return torch.tensor(0.0)
        
        batch_size = min(32, len(self.rep_memo))
        minibatch = random.sample(self.rep_memo, batch_size)
        
        states = torch.stack([data[0] for data in minibatch], dim=0)
        actions = torch.stack([torch.tensor(data[1], dtype=torch.float32) for data in minibatch], dim=0)
        preferences = torch.stack([data[4] for data in minibatch], dim=0)
        
        predicted = self.network.forward(states, preferences)
        loss = F.mse_loss(predicted, actions)
        return loss
    
    def store_good_experiences_to_replay(self, states, actions, returns):
        """Store good experiences into replay pool"""
        mean_return = returns.mean().item()
        threshold = mean_return * (1 - self.good_experience_threshold)
        good_indices = (returns < threshold).nonzero(as_tuple=True)[0]
        
        for idx in good_indices:
            if idx < len(self.trajectory_buffer):
                step = self.trajectory_buffer[idx]      
                if not (self.ablation == "Ablation2"):          
                    selected_idx, _ = self.select_by_chebyshev(step[5], torch.tensor(step[1],dtype=torch.float32))
                    optimal_weights, new_score = self.compute_optimal_weights(step[5], selected_idx, step[1] )
                    if new_score < step[2]:
                        step[1] = optimal_weights
                experience = (
                    torch.tensor(step[0], dtype=torch.float32),
                    step[1],#step[1],
                    step[2],
                    torch.tensor(step[3], dtype=torch.float32),
                    torch.tensor(step[4], dtype=torch.float32),
                    step[5],
                    step[6]
                )

                self.rep_memo.append(experience)
        
        # Limit replay pool size
        if len(self.rep_memo) > self.rep_memo_size:
            self.rep_memo = self.rep_memo[-self.rep_memo_size:]
       
    
    def record_training_metrics(self, total_loss, policy_loss, supervised_loss, returns):
        """Record training metrics"""
        
        avg_return = returns.mean().item()
         
        # Learning rate scheduling
        self.network.lr_scheduler.step(total_loss)
        if self.training_step_count % 10 == 0 :
            print(f'[HATHL Reinforcement Learning] Count:{self.training_step_count},'
                f'Arrived jobs: {self.job_creator.index_jobs}, '
                f'WIP: {self.job_creator.in_system_job_no}, ' 
                f'Avg return: {avg_return:.5f}, '
                f'Total loss: {total_loss.item():.5f}, '
                f'Policy loss: {policy_loss.item():.5f}, '
                f'Supervised loss: {supervised_loss.item():.5f}')
        
        self.training_step_count += 1
    


    # ========== Override base class training process method ==========
    def training_process(self):
        """Override base class training process, use HATHL specific process"""
        return self.training_process_parameter_sharing()
