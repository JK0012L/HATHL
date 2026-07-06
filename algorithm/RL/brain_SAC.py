# brain_SAC.py - SAC Network Inheriting ContinuousSchedulingNetwork
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
        
        # ========== SAC specific parameters ==========
        self.tau = 0.005  # Target network soft update parameter
        self.target_entropy_coeff = 0.98  # Target entropy coefficient
        
        # ========== Network initialization ==========
        if self.train == True: 
            self.network = SACNetwork(self.input_size)
            self.algorithm_name = "SAC"
            self.training_epochs = 5  # SAC training epochs
            self.env.process(self.training_process_sac())
            
        if self.train == False:         
            self.network = SACNetwork(self.input_size)
            self.network.load_state_dict(torch.load(self.address))            
            self.network.eval()  
            self.multi_obj_manager.get_per_baselines(self.address)
            
    # ========== SAC Training Process ==========
    def training_process_sac(self):  
        """SAC training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting SAC training...")
        train_steps = 0
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodically perform training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > self.trajectory_buffer_size:
                self.samples -= 0.5 * self.trajectory_buffer_size 
                actor_loss, critic_loss, alpha_loss = self.train_sac()                
                # Record losses
                if actor_loss is not None:
                    self.rule_loss_record.append(actor_loss)  
                    self.value_loss_record.append(critic_loss)  
                    self.loss_record.append(alpha_loss) 
                    
                    # Print training info every 100 steps
                    if train_steps % 10 == 0:
                        current_alpha = self.network.log_alpha.exp().item()
                        print(f'[SAC Training] Step:{train_steps}, '
                              f'Arrived jobs: {self.job_creator.index_jobs}, '
                              f'WIP: {self.job_creator.in_system_job_no}, '                              
                              f'Actor loss:{actor_loss:.4f}, '
                              f'Critic loss:{critic_loss:.4f}, '
                              f'Alpha loss:{alpha_loss:.4f} ')
            
            train_steps += 1
            yield self.env.timeout(100) 
         
        # Save model
        address = self.address.format(sys.path[0])
        torch.save(self.network.state_dict(), address)
        pref_address = address.replace('.pt', '_preferences.pkl')
        self.multi_obj_manager.save_training_results(pref_address)
        print(f"SAC model saved: {address}")
   
    def train_sac(self):
        """SAC training function"""
        if len(self.trajectory_buffer) < self.batch_size:
            return None, None, None
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_alpha_loss = 0
        
        # Perform multiple epochs of training
        for epoch in range(self.training_epochs):
            # Sample from experience replay buffer
            batch = self.sample_sac_batch(self.batch_size)
            
            if batch is None:
                continue
            
            # Convert data to tensors
            states = torch.FloatTensor(batch['states']).to(self.network.device)
            actions = torch.FloatTensor(batch['actions']).to(self.network.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.network.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.network.device)
            preferences = torch.FloatTensor(batch['preferences']).to(self.network.device)
            
            # SAC training step
            actor_loss, critic_loss, alpha_loss = self.network.update_parameters(
                states, actions, rewards, next_states, preferences
            )
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_alpha_loss += alpha_loss
        
        # Calculate average losses
        avg_actor_loss = total_actor_loss / self.training_epochs
        avg_critic_loss = total_critic_loss / self.training_epochs
        avg_alpha_loss = total_alpha_loss / self.training_epochs
        
        return avg_actor_loss, avg_critic_loss, avg_alpha_loss
    
    def sample_sac_batch(self, batch_size):
        """Sample batch data for SAC training"""
        if len(self.trajectory_buffer) < batch_size:
            return None
        
        batch = random.sample(self.trajectory_buffer, batch_size)
        
        # Initialize lists
        states_list, actions_list, rewards_list = [], [], []
        next_states_list, preferences_list = [], []
        
        for experience in batch:
            # Assume experience storage format: (state, action, reward, next_state, preference)
            if len(experience) >= 5:
                state, action, reward, next_state, preference = experience[:5]
                
                # Convert to numpy arrays
                if torch.is_tensor(state):
                    states_list.append(state.detach().cpu().numpy())
                else:
                    states_list.append(state)
                    
                if torch.is_tensor(action):
                    actions_list.append(action.detach().cpu().numpy())
                else:
                    actions_list.append(action)
                    
                rewards_list.append(reward)
                
                if torch.is_tensor(next_state):
                    next_states_list.append(next_state.detach().cpu().numpy())
                else:
                    next_states_list.append(next_state)
                    
                if torch.is_tensor(preference):
                    preferences_list.append(preference.detach().cpu().numpy())
                else:
                    preferences_list.append(preference)
        
        # Convert to numpy arrays
        return {
            'states': np.array(states_list, dtype=np.float32),
            'actions': np.array(actions_list, dtype=np.float32),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'next_states': np.array(next_states_list, dtype=np.float32),
            'preferences': np.array(preferences_list, dtype=np.float32)
        }

# ========== SAC Network Architecture - Inherits ContinuousSchedulingNetwork ==========
class SACNetwork(ContinuousSchedulingNetwork):
    """
    SAC Network - Inherits ContinuousSchedulingNetwork
    Adds dual Q networks and temperature parameter on top of base class
    """
    def __init__(self, input_size=15, preference_size=3):
        super(SACNetwork, self).__init__(input_size, preference_size)
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        state_dim = 10
        
        # ========== 1. Actor Network (policy network) ==========
        # Input: state encoding(10) + preference features(3) = 13
        actor_input_size = state_dim + self.preference_size  # 10 + 3 = 13
        
        # Output mean and standard deviation (for Gaussian policy)
        self.actor_mean = nn.Sequential(
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
        
        self.actor_log_std = nn.Sequential(
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
        
        # ========== 3. Target Critic Networks ==========
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
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # ========== 4. Temperature parameter (auto-adjusted) ==========
        self.target_entropy = -torch.prod(torch.Tensor([self.preference_size]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        
        # ========== 5. Optimizer ==========
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr
        )
        
        self.actor_optimizer = optim.Adam(
            list(self.actor_mean.parameters()) + list(self.actor_log_std.parameters()),
            lr=self.lr
        )
        
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        
        # Reinitialize weights of newly added parts
        self._init_sac_weights()
        self.to(self.device)
    
    def _init_sac_weights(self):
        """Initialize weights of newly added SAC parts"""
        for module in [self.actor_mean, self.actor_log_std, 
                       self.critic1, self.critic2,
                       self.critic1_target, self.critic2_target]:
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
        Actor forward propagation - outputs mean and standard deviation of Gaussian distribution
        """
        actor_input = torch.cat([fused_state_pref, enhanced_pref], dim=-1)  # [batch, 13]
        
        mean = self.actor_mean(actor_input)
        log_std = self.actor_log_std(actor_input)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std
    
    def sample_action(self, state_features, preference_vector, deterministic=False):
        """
        Sample action (for training and exploration)
        """
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        mean, std = self._actor_forward(fused_state_pref, enhanced_pref)
        
        if deterministic:
            action = torch.softmax(mean, dim=-1)
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.softmax(x_t, dim=-1)
        
        return action
    
    def evaluate(self, state_features, preference_vector):
        """
        Evaluate action (for computing log_prob and entropy)
        """
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        mean, std = self._actor_forward(fused_state_pref, enhanced_pref)
        
        normal = torch.distributions.Normal(mean, std)
        
        x_t = normal.rsample()
        action = torch.softmax(x_t, dim=-1)
        
        # Compute log probability
        logits = x_t
        log_prob = -F.cross_entropy(logits, logits.argmax(dim=-1), reduction='none')
        
        # Compute entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob.unsqueeze(-1), entropy
    
    def compute_q_values(self, state_features, preference_vector, action):
        """Compute Q values"""
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)  # [batch, 16]
        
        q1 = self.critic1(critic_input)
        q2 = self.critic2(critic_input)
        
        return q1, q2
    
    def compute_target_q_values(self, state_features, preference_vector, action):
        """Compute target Q values"""
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        
        q1 = self.critic1_target(critic_input)
        q2 = self.critic2_target(critic_input)
        
        return q1, q2
    
    def update_parameters(self, states, actions, rewards, next_states, preferences):
        """SAC parameter update"""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.evaluate(next_states, preferences)
            
            target_q1, target_q2 = self.compute_target_q_values(next_states, preferences, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            gamma = 0.99
            next_q_value = rewards + gamma * target_q
        
        # Update Critic network
        current_q1, current_q2 = self.compute_q_values(states, preferences, actions)
        critic1_loss = F.mse_loss(current_q1, next_q_value)
        critic2_loss = F.mse_loss(current_q2, next_q_value)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update Actor network
        new_action, log_prob, _ = self.evaluate(states, preferences)
        q1_new, q2_new = self.compute_q_values(states, preferences, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_mean.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.actor_log_std.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        tau = 0.005
        self.soft_update(self.critic1, self.critic1_target, tau)
        self.soft_update(self.critic2, self.critic2_target, tau)
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def forward(self, state_features, preference_vector):
        """Forward propagation (for compatibility)"""
        action = self.sample_action(state_features, preference_vector, deterministic=True)
        return action
