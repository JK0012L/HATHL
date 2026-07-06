# brain_A2C.py
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
        
        # A2C specific parameters
        self.gae_lambda = 0.95  # GAE parameter
        self.entropy_coeff = 0.01  # Entropy regularization coefficient
        self.value_loss_coeff = 0.5  # Value loss coefficient
        self.max_grad_norm = 0.5  # Gradient clipping
        self.training_epochs = 5  # A2C training epochs
        self.total_train_steps = 0  # Total training step counter
        
        # Initialize A2C network
        if self.train == True: 
            self.network = A2CNetwork(self.input_size)
            self.env.process(self.training_process_a2c())  # A2C training process
        else:
            self.network = A2CNetwork(self.input_size)  
            self.network.load_state_dict(torch.load(self.address))            
            self.network.eval()  
            self.multi_obj_manager.get_per_baselines(self.address)
    
    def training_process_a2c(self):  
        """A2C training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting A2C training...")
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodically perform training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > self.trajectory_buffer_size:
                self.samples -= 0.5 * self.trajectory_buffer_size 
                policy_loss, value_loss, entropy = self.train_a2c()
                if policy_loss is not None:
                    if self.total_train_steps % 10 == 0:
                        print(f'A2C training Step:{self.total_train_steps}, '
                              f'Arrived jobs: {self.job_creator.index_jobs}, '
                              f'WIP: {self.job_creator.in_system_job_no}, '
                              f'Policy loss:{policy_loss:.3f}, '
                              f'Value loss:{value_loss:.3f}, '
                              f'Entropy:{entropy:.3f}')
                    
            self.total_train_steps += 1
            yield self.env.timeout(100)  # Check every 100 time units
         
        # Save model
        if self.address:
             address = self.address.format(sys.path[0])
             torch.save(self.network.state_dict(), address)
             pref_address = address.replace('.pt', '_preferences.pkl')
             self.multi_obj_manager.save_training_results(pref_address)
             print(f"A2C model and preference vector saved: {address}, {pref_address}")
   
    def train_a2c(self):
        """A2C training function"""
        if len(self.trajectory_buffer) < self.batch_size:
            return None, None, None
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Perform multiple epochs of training
        for epoch in range(self.training_epochs):
            # Sample from experience replay buffer
            batch = self.sample(self.batch_size)
            if batch is None:
                return None, None, None
            
            # Convert data to tensors
            states = torch.FloatTensor(batch[0]).to(self.network.device)
            actions = torch.FloatTensor(batch[1]).to(self.network.device)
            rewards = torch.FloatTensor(batch[2]).to(self.network.device)
            next_states = torch.FloatTensor(batch[3]).to(self.network.device)
            preferences = torch.FloatTensor(batch[4]).to(self.network.device)
            
            # A2C training step
            policy_loss, value_loss, entropy = self.network.update_parameters(
                states, actions, rewards, next_states, preferences,
                self.discount_factor, self.gae_lambda, self.value_loss_coeff,
                self.entropy_coeff, self.max_grad_norm
            )
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
        
        # Return average losses
        return (total_policy_loss / self.training_epochs,
                total_value_loss / self.training_epochs,
                total_entropy / self.training_epochs)
    
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
        
        # Return list
        return [
            np.array(states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.float32),
            np.array(rewards_list, dtype=np.float32).reshape(-1, 1),
            np.array(next_states_list, dtype=np.float32),
            np.array(preferences_list, dtype=np.float32)
        ]

# ========== A2C Network Architecture - Inherits ContinuousSchedulingNetwork ==========
class A2CNetwork(ContinuousSchedulingNetwork):
    """
    A2C Network - Inherits ContinuousSchedulingNetwork
    Adds value network and GAE advantage calculation on top of base class
    """
    def __init__(self, input_size=15, preference_size=3):
        super(A2CNetwork, self).__init__(input_size, preference_size)
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== Get parent class state encoder output dimension ==========
        # Parent class ContinuousSchedulingNetwork's state_extractor outputs 10 dimensions
        state_dim = 10
        
        # ========== 1. Actor Network (policy network) ==========
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
            nn.Linear(8, self.preference_size),
            nn.Softmax(dim=-1)
        )
        
        # ========== 2. Critic Network (value network) ==========
        # Input: state encoding(10) + preference features(3) = 13
        critic_input_size = state_dim + self.preference_size  # 10 + 3 = 13
        
        self.critic = nn.Sequential(
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
        
        # ========== 3. Optimizer ==========
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=0.00005)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=0.00005)
        
        # Reinitialize weights of newly added parts
        self._init_a2c_weights()
        self.to(self.device)
    
    def _init_a2c_weights(self):
        """Initialize weights of newly added A2C parts"""
        for module in [self.actor, self.critic]:
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
    
    def _actor_forward(self, fused_state_pref, enhanced_pref):
        """
        Actor forward propagation - outputs action probability distribution
        """
        actor_input = torch.cat([fused_state_pref, enhanced_pref], dim=-1)  # [batch, 13]
        action_probs = self.actor(actor_input)
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return action_probs
    
    def _critic_forward(self, fused_state_pref, enhanced_pref):
        """
        Critic forward propagation - outputs state value
        """
        critic_input = torch.cat([fused_state_pref, enhanced_pref], dim=-1)  # [batch, 13]
        value = self.critic(critic_input)
        return value
    
    def forward(self, state_features, preference_vector, mode="policy"):
        """
        Forward propagation - A2C specific
        """
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        action_probs = self._actor_forward(fused_state_pref, enhanced_pref)
        
        if mode == "policy":
            return action_probs
        
        value = self._critic_forward(fused_state_pref, enhanced_pref)
        
        if mode == "value":
            return value
        elif mode == "both":
            return action_probs, value
    
    def get_value(self, state_features, preference_vector):
        """Get state value"""
        self.eval()
        with torch.no_grad():
            fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
            value = self._critic_forward(fused_state_pref, enhanced_pref)
        self.train()
        return value
    
    def compute_gae_advantages(self, rewards, values, next_values, done_mask, gamma, gae_lambda):
        """
        Compute GAE advantage function
        """
        td_errors = rewards + gamma * next_values * (1 - done_mask) - values
        
        advantages = torch.zeros_like(td_errors)
        running_advantage = 0
        for i in reversed(range(len(td_errors))):
            running_advantage = td_errors[i] + gamma * gae_lambda * (1 - done_mask[i]) * running_advantage
            advantages[i] = running_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update_parameters(self, states, actions, rewards, next_states, preferences,
                          gamma, gae_lambda, value_loss_coeff, entropy_coeff, max_grad_norm):
        """
        A2C parameter update
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"⚠️ {name} contains NaN or Inf, skipping this batch")
                return False
            return True
        
        if not (check_tensor(states, "states") and check_tensor(actions, "actions") and
                check_tensor(rewards, "rewards") and check_tensor(next_states, "next_states") and
                check_tensor(preferences, "preferences")):
            return 0.0, 0.0, 0.0
        
        # Extract features
        fused_state_pref_actor, enhanced_pref_actor = self._extract_features(states, preferences)
        fused_state_pref_critic, enhanced_pref_critic = self._extract_features(states, preferences)
        next_fused_state_pref, next_enhanced_pref = self._extract_features(next_states, preferences)
        
        # Actor output
        action_probs = self._actor_forward(fused_state_pref_actor, enhanced_pref_actor)
        
        # Critic output
        current_values = self._critic_forward(fused_state_pref_critic, enhanced_pref_critic)
        
        with torch.no_grad():
            next_values = self._critic_forward(next_fused_state_pref, next_enhanced_pref)
        
        # Compute GAE advantages
        done_mask = torch.zeros_like(rewards)
        advantages, returns = self.compute_gae_advantages(
            rewards, current_values, next_values, done_mask, gamma, gae_lambda
        )
        
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Actor Loss
        action_indices = torch.argmax(actions, dim=1, keepdim=True)
        action_dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = action_dist.log_prob(action_indices.squeeze(-1))
        
        policy_loss = -(log_probs * advantages.squeeze(-1)).mean()
        entropy = action_dist.entropy().mean()
        actor_loss = policy_loss - entropy_coeff * entropy
        
        # Critic Loss
        value_loss = F.mse_loss(current_values, returns)
        critic_loss = value_loss_coeff * value_loss
        
        # Backpropagation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def get_action(self, state_features, preference_vector, deterministic=False):
        """
        Get action (for inference and exploration)
        """
        self.eval()
        with torch.no_grad():
            action_probs = self.forward(state_features, preference_vector, mode="policy")
            
            if deterministic:
                action = torch.zeros_like(action_probs)
                action.scatter_(1, torch.argmax(action_probs, dim=1, keepdim=True), 1.0)
            else:
                action_dist = torch.distributions.Categorical(probs=action_probs)
                action_idx = action_dist.sample()
                action = torch.zeros_like(action_probs)
                action.scatter_(1, action_idx.unsqueeze(1), 1.0)
        self.train()
        return action      