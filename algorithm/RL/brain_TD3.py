# brain_TD3.py - TD3 Network Inheriting ContinuousSchedulingNetwork
import random 
import numpy as np 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.base_brain import ContinuousSequencingBrain, ContinuousSchedulingNetwork


class sequencing_brain(ContinuousSequencingBrain):
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initialization
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # TD3 specific parameters
        self.tau = 0.005  # Target network soft update parameter
        self.policy_noise = 0.2  # TD3 target policy noise
        self.noise_clip = 0.5  # TD3 noise clipping range
        self.policy_delay = 2  # TD3 policy delayed update frequency
        self.total_train_steps = 0  # Total training step counter
        
        # Initialize TD3 network
        if self.train == True: 
            self.network = TD3Network(self.input_size)
            self.env.process(self.training_process_td3())  # TD3 training process
        else:
            self.network = TD3Network(self.input_size)  
            self.network.load_state_dict(torch.load(self.address))            
            self.network.eval()  
            self.multi_obj_manager.get_per_baselines(self.address)
    
    def training_process_td3(self):  
        """TD3 training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting TD3 training...")
        train_steps = 0
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodically perform training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > self.trajectory_buffer_size:
                self.samples -= 0.5 * self.trajectory_buffer_size 
                actor_loss, critic_loss = self.train_td3()
                if actor_loss is not None: 
                    if self.total_train_steps % 10 == 0:
                        print(f'TD3 training Step:{self.total_train_steps}, '
                              f'Arrived jobs: {self.job_creator.index_jobs}, '
                              f'WIP: {self.job_creator.in_system_job_no}, '                                
                              f'Actor loss:{actor_loss:.3f}, '
                              f'Critic loss:{critic_loss:.3f},')
                    
            self.total_train_steps += 1
            yield self.env.timeout(100)  # Check every 100 time units
         
        # Save model
        if self.address:
            address = self.address.format(sys.path[0])
            torch.save(self.network.state_dict(), address)
            pref_address = address.replace('.pt', '_preferences.pkl')
            self.multi_obj_manager.save_training_results(pref_address)
            print(f"TD3 model and preference vector saved: {address}, {pref_address}")
   
    def train_td3(self):
        """TD3 training function"""
        if len(self.trajectory_buffer) < self.batch_size:
            return None, None
        
        # Sample from experience replay buffer
        batch = self.sample(self.batch_size)
        if batch is None:
            return None, None
        
        # Convert data to tensors
        states = torch.FloatTensor(batch[0]).to(self.network.device)
        actions = torch.FloatTensor(batch[1]).to(self.network.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.network.device)
        next_states = torch.FloatTensor(batch[3]).to(self.network.device)
        preferences = torch.FloatTensor(batch[4]).to(self.network.device)
        
        # TD3 training step
        actor_loss, critic_loss = self.network.update_parameters(
            states, actions, rewards, next_states, preferences,
            self.total_train_steps
        )
        
        return actor_loss, critic_loss
    
    def sample(self, batch_size):
        """Sample batch data, return list instead of dictionary"""
        if len(self.trajectory_buffer) < batch_size:
            return None
        
        batch = random.sample(self.trajectory_buffer, batch_size)
        
        # Initialize lists
        states_list, actions_list, rewards_list = [], [], []
        next_states_list, preferences_list = [], []
        
        for state, action, reward, next_state, preference, w1, w2 in batch:
            states_list.append(state.detach().cpu().numpy())
            actions_list.append(action.detach().cpu().numpy())
            rewards_list.append(reward)
            next_states_list.append(next_state.detach().cpu().numpy())
            preferences_list.append(preference.detach().cpu().numpy())
        
        # Return list instead of dictionary
        return [
            np.array(states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.float32),
            np.array(rewards_list, dtype=np.float32).reshape(-1, 1),
            np.array(next_states_list, dtype=np.float32),
            np.array(preferences_list, dtype=np.float32)
        ]

# ========== TD3 Network Architecture - Inherits ContinuousSchedulingNetwork ==========
class TD3Network(ContinuousSchedulingNetwork):
    """
    TD3 Network - Inherits ContinuousSchedulingNetwork
    Adds dual Q networks and target networks on top of base class
    """
    def __init__(self, input_size=15, preference_size=3):
        super(TD3Network, self).__init__(input_size, preference_size)
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== Get parent class state encoder output dimension ==========
        # Parent class ContinuousSchedulingNetwork's state_extractor outputs 10 dimensions
        state_dim = 10
        
        # ========== 1. Actor Network (deterministic policy) ==========
        # Input: state encoding(10) + preference features(3) = 13
        actor_input_size = state_dim + self.preference_size  # 10 + 3 = 13
        
        self.actor = nn.Sequential(
            nn.Linear(actor_input_size, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, self.preference_size)
        )
        
        # ========== 2. Critic Network (dual Q network) ==========
        # Input: state encoding(10) + preference features(3) + action(3) = 16
        critic_input_size = state_dim + self.preference_size + self.preference_size  # 10 + 3 + 3 = 16
        
        # Critic 1
        self.critic1 = nn.Sequential(
            nn.Linear(critic_input_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )
        
        # Critic 2
        self.critic2 = nn.Sequential(
            nn.Linear(critic_input_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )
        
        # ========== 3. Target Networks ==========
        # Target Actor
        self.actor_target = nn.Sequential(
            nn.Linear(actor_input_size, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, self.preference_size)
        )
        
        # Target Critic 1
        self.critic1_target = nn.Sequential(
            nn.Linear(critic_input_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )
        
        # Target Critic 2
        self.critic2_target = nn.Sequential(
            nn.Linear(critic_input_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # ========== 4. Optimizer ==========
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=0.00005)
        self.critic_optimizer = optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr, weight_decay=0.00005
        )
        
        # ========== TD3 Hyperparameters ==========
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.gamma = 0.99
        
        # Reinitialize weights of newly added parts
        self._init_td3_weights()
        self.to(self.device)
    
    def _init_td3_weights(self):
        """Initialize weights of newly added TD3 parts"""
        for module in [self.actor, self.critic1, self.critic2,
                       self.actor_target, self.critic1_target, self.critic2_target]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0)
    
    def _extract_features(self, state_features, preference_vector):
        """
        Extract state features and preference features
        Consistent with parent class method signature, used by forward
        """
        # Use parent class group normalization
        ratio_feat, count_feat, time_feat, diff_feat = self._split_features(state_features)
        
        ratio_norm = self.norm_ratio(ratio_feat)
        count_norm = self.norm_count(count_feat)
        time_norm = self.norm_time(time_feat)
        diff_norm = self.norm_diff(diff_feat)
        
        normalized_features = torch.cat([ratio_norm, count_norm, time_norm, diff_norm], dim=-1)
        state_encoded = self.state_extractor(normalized_features)
        
        enhanced_pref = self.pref_enhancer(preference_vector)
        
        return state_encoded, enhanced_pref
    
    def _safe_softmax(self, x, dim=-1, temperature=1.0):
        """Numerically stable Softmax implementation"""
        if torch.isnan(x).any():
            return torch.ones_like(x) / x.shape[dim]
        
        x = x / temperature
        max_vals = torch.max(x, dim=dim, keepdim=True)[0]
        x_stable = x - max_vals
        exp_x = torch.exp(x_stable)
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True) + 1e-8
        output = exp_x / sum_exp
        
        if torch.isnan(output).any():
            return torch.ones_like(x) / x.shape[dim]
        return output
    
    def _actor_forward(self, fused_state_pref, enhanced_pref):
        """
        Actor forward propagation - outputs deterministic action
        """
        actor_input = torch.cat([fused_state_pref, enhanced_pref], dim=-1)  # [batch, 13]
        raw_action = self.actor(actor_input)
        action = self._safe_softmax(raw_action, dim=-1, temperature=2.0)
        return action
    
    def forward(self, state_features, preference_vector, mode="policy"):
        """
        Forward propagation - TD3 specific
        """
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        action = self._actor_forward(fused_state_pref, enhanced_pref)
        
        if mode == "policy":
            return action
        
        # Compute Q value
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        q1 = self.critic1(critic_input)
        q2 = self.critic2(critic_input)
        value = torch.min(q1, q2)
        
        if mode == "value":
            return value
        elif mode == "both":
            return action, value
    
    def compute_q_values(self, fused_state_pref, enhanced_pref, action):
        """Compute Q values"""
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        q1 = self.critic1(critic_input)
        q2 = self.critic2(critic_input)
        return q1, q2
    
    def compute_target_q_values(self, fused_state_pref, enhanced_pref, action):
        """Compute target Q values"""
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        q1 = self.critic1_target(critic_input)
        q2 = self.critic2_target(critic_input)
        return q1, q2
    
    def update_parameters(self, states, actions, rewards, next_states, preferences, total_steps):
        """TD3 parameter update"""
        # Convert to correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        # ========== 1. Input data validation ==========
        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"⚠️ {name} contains NaN or Inf, skipping this batch")
                return False
            return True
        
        if not (check_tensor(states, "states") and check_tensor(actions, "actions") and
                check_tensor(rewards, "rewards") and check_tensor(next_states, "next_states") and
                check_tensor(preferences, "preferences")):
            return 0.0, 0.0
        
        # ========== 2. Update Critic network ==========
        # Extract current state features
        fused_state_pref, enhanced_pref = self._extract_features(states, preferences)
        
        # Compute current Q value
        current_q1, current_q2 = self.compute_q_values(fused_state_pref, enhanced_pref, actions)
        
        # Extract next state features
        next_fused_state_pref, next_enhanced_pref = self._extract_features(next_states, preferences)
        
        with torch.no_grad():
            # Target policy action
            next_actor_input = torch.cat([next_fused_state_pref, next_enhanced_pref], dim=-1)
            raw_next_action = self.actor_target(next_actor_input)
            next_action = self._safe_softmax(raw_next_action, dim=-1, temperature=2.0)
            
            # Add policy noise
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_action + noise
            next_action = torch.clamp(next_action, 0, 1)
            next_action = next_action / (next_action.sum(dim=1, keepdim=True) + 1e-8)
            
            # Compute target Q value
            target_q1, target_q2 = self.compute_target_q_values(next_fused_state_pref, next_enhanced_pref, next_action)
            target_q = torch.min(target_q1, target_q2)
            next_q_value = rewards + self.gamma * target_q
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, next_q_value)
        critic2_loss = F.mse_loss(current_q2, next_q_value)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # ========== 3. Delayed update Actor network ==========
        actor_loss = None
        if total_steps % self.policy_delay == 0:
            # Re-extract current state features
            fused_state_pref_actor, enhanced_pref_actor = self._extract_features(states, preferences)
            
            # Compute Actor loss
            actor_input = torch.cat([fused_state_pref_actor, enhanced_pref_actor], dim=-1)
            raw_new_actions = self.actor(actor_input)
            new_actions = self._safe_softmax(raw_new_actions, dim=-1, temperature=2.0)
            
            # Recompute Q value (for Actor loss)
            q1_new, _ = self.compute_q_values(fused_state_pref_actor, enhanced_pref_actor, new_actions)
            actor_loss = -q1_new.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # ========== 4. Soft update target networks ==========
            self.soft_update(self.critic1, self.critic1_target, self.tau)
            self.soft_update(self.critic2, self.critic2_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
        
        return actor_loss.item() if actor_loss is not None else 0.0, critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_action(self, state_features, preference_vector, exploration=True):
        """Get action (for inference and exploration)"""
        self.eval()
        with torch.no_grad():
            action = self.forward(state_features, preference_vector, mode="policy")
            
            if exploration and self.training:
                noise = torch.normal(0, 0.1, size=action.shape).to(self.device)
                action = action + noise
                action = torch.clamp(action, 0, 1)
                action = action / (action.sum(dim=1, keepdim=True) + 1e-8)
        self.train()
        return action
