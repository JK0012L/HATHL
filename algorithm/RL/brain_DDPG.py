# brain_DDPG.py
import random 
import numpy as np 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.base_brain import ContinuousSequencingBrain, ContinuousSchedulingNetwork


class sequencing_brain(ContinuousSequencingBrain):
    """DDPG scheduling brain (inherits from base class)"""
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initialization
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # DDPG specific parameters
        self.tau = 0.005  # Target network soft update parameter
        self.policy_noise = 0.2  # Target policy noise (for exploration)
        self.noise_clip = 0.5  # Noise clipping range
        self.total_train_steps = 0  # Total training step counter
        
        # Initialize DDPG network
        if self.train == True: 
            self.network = DDPGNetwork(self.input_size)
            self.env.process(self.training_process_ddpg())  # DDPG training process
        else:
            self.network = DDPGNetwork(self.input_size)  
            self.network.load_state_dict(torch.load(self.address))            
            self.network.eval()  
            self.multi_obj_manager.get_per_baselines(self.address)
    
    def training_process_ddpg(self):  
        """DDPG training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting DDPG training...")
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodically perform training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > self.trajectory_buffer_size:
                self.samples -= 0.5 * self.trajectory_buffer_size 
                actor_loss, critic_loss = self.train_ddpg()
                if actor_loss is not None:
                    if self.total_train_steps % 10 == 0:
                        print(f'DDPG training Step:{self.total_train_steps}, '
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
            print(f"DDPG model and preference vector saved: {address}, {pref_address}")
   
    def train_ddpg(self):
        """DDPG training function"""
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
        
        # DDPG training step
        actor_loss, critic_loss = self.network.update_parameters(
            states, actions, rewards, next_states, preferences,
            self.total_train_steps, self.discount_factor, self.tau,
            self.policy_noise, self.noise_clip
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

# ========== DDPG Network Architecture - Inherits ContinuousSchedulingNetwork ==========
class DDPGNetwork(ContinuousSchedulingNetwork):
    """
    DDPG Network - Inherits ContinuousSchedulingNetwork
    Adds target networks and deterministic policy on top of base class
    """
    def __init__(self, input_size=15, preference_size=3):
        super(DDPGNetwork, self).__init__(input_size, preference_size)
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
       
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
            nn.Linear(8, self.preference_size)  # Output 3-dim action (preference weights)
        )
        
        # Output Softmax layer (converts output to probability distribution)
        self.actor_output_softmax = nn.Softmax(dim=-1)
        
        # ========== 2. Critic Network (single Q network) ==========
        # Input: state encoding(10) + preference features(3) + actions(3) = 16
        critic_input_size = state_dim + self.preference_size + self.preference_size  # 10 + 3 + 3 = 16
        
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
        self.actor_target_output_softmax = nn.Softmax(dim=-1)
        
        # Target Critic
        self.critic_target = nn.Sequential(
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
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # ========== 4. Optimizer ==========
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=0.00005)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=0.00005)
        
        # ========== DDPG Hyperparameters ==========
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.gamma = 0.99
        
        # Reinitialize weights of newly added parts
        self._init_ddpg_weights()
        self.to(self.device)
    
    def _init_ddpg_weights(self):
        """Initialize weights of newly added DDPG parts"""
        for module in [self.actor, self.critic, self.actor_target, self.critic_target]:
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
        action = self._safe_softmax(raw_action, dim=-1, temperature=1.0)
        return action
    
    def forward(self, state_features, preference_vector, mode="policy"):
        """
        Forward propagation - DDPG specific
        
        Args:
            state_features: [batch_size, input_size] state features
            preference_vector: [batch_size, preference_size] preference vector
            mode: "policy" returns action, "value" returns Q value, "both" returns both
        
        Returns:
            action: [batch_size, preference_size] action probability distribution
            value: [batch_size, 1] Q value (only when mode is "value" or "both")
        """
        fused_state_pref, enhanced_pref = self._extract_features(state_features, preference_vector)
        action = self._actor_forward(fused_state_pref, enhanced_pref)
        
        if mode == "policy":
            return action
        
        # Compute Q value
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        value = self.critic(critic_input)
        
        if mode == "value":
            return value
        elif mode == "both":
            return action, value
    
    def compute_q_value(self, fused_state_pref, enhanced_pref, action):
        """Compute Q value"""
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        q = self.critic(critic_input)
        return q
    
    def compute_target_q_value(self, fused_state_pref, enhanced_pref, action):
        """Compute target Q value"""
        critic_input = torch.cat([fused_state_pref, enhanced_pref, action], dim=-1)
        q = self.critic_target(critic_input)
        return q
    
    def compute_target_action(self, fused_state_pref, enhanced_pref, add_noise=False):
        """Compute target action"""
        actor_input = torch.cat([fused_state_pref, enhanced_pref], dim=-1)
        raw_action = self.actor_target(actor_input)
        action = self._safe_softmax(raw_action, dim=-1, temperature=1.0)
        
        if add_noise:
            # Add exploration noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
            action = torch.clamp(action, 0, 1)
            action = action / (action.sum(dim=1, keepdim=True) + 1e-8)
        
        return action
    
    def update_parameters(self, states, actions, rewards, next_states, preferences,
                          total_steps, gamma, tau, policy_noise, noise_clip):
        """
        DDPG parameter update
        
        Args:
            states: [batch_size, input_size] states
            actions: [batch_size, preference_size] actions
            rewards: [batch_size, 1] rewards
            next_states: [batch_size, input_size] next states
            preferences: [batch_size, preference_size] preference vector
            total_steps: Total training steps
            gamma: Discount factor
            tau: Target network soft update parameter
            policy_noise: Target policy noise
            noise_clip: Noise clipping range
        
        Returns:
            actor_loss: Actor loss
            critic_loss: Critic loss
        """
        # Convert to correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        # Update hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        # ========== Input data validation ==========
        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"⚠️ {name} contains NaN or Inf, skipping this batch")
                return False
            return True
        
        if not (check_tensor(states, "states") and check_tensor(actions, "actions") and
                check_tensor(rewards, "rewards") and check_tensor(next_states, "next_states") and
                check_tensor(preferences, "preferences")):
            return 0.0, 0.0
        
        # ========== Extract features ==========
        # Current state features
        fused_state_pref, enhanced_pref = self._extract_features(states, preferences)
        
        # Next state features
        next_fused_state_pref, next_enhanced_pref = self._extract_features(next_states, preferences)
        
        # ========== Update Critic network ==========
        # Compute current Q value
        current_q = self.compute_q_value(fused_state_pref, enhanced_pref, actions)
        
        # Compute target Q value
        with torch.no_grad():
            # Target network computes next action (with noise)
            next_action = self.compute_target_action(next_fused_state_pref, next_enhanced_pref, add_noise=True)
            
            # Compute target Q value
            target_q = self.compute_target_q_value(next_fused_state_pref, next_enhanced_pref, next_action)
            
            # TD target
            next_q_value = rewards + self.gamma * target_q
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, next_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # ========== Update Actor network ==========
        # Re-extract current state features (for Actor update)
        fused_state_pref_actor, enhanced_pref_actor = self._extract_features(states, preferences)
        
        # Compute new action
        new_actions = self._actor_forward(fused_state_pref_actor, enhanced_pref_actor)
        
        # Compute Actor loss (maximize Q value)
        q_new = self.compute_q_value(fused_state_pref_actor, enhanced_pref_actor, new_actions)
        actor_loss = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # ========== Soft update target networks ==========
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_action(self, state_features, preference_vector, exploration=True):
        """
        Get action (for inference and exploration)
        
        Args:
            state_features: State features
            preference_vector: Preference vector
            exploration: Whether to add exploration noise
        
        Returns:
            action: Action (probability distribution)
        """
        self.eval()
        with torch.no_grad():
            action = self.forward(state_features, preference_vector, mode="policy")
            
            if exploration and self.training:
                # Add exploration noise
                noise = torch.randn_like(action) * self.policy_noise
                action = action + noise
                action = torch.clamp(action, 0, 1)
                action = action / (action.sum(dim=1, keepdim=True) + 1e-8)
        self.train()
        return action
