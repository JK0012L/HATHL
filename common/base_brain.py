# sequencing_brain.py - Scheduling Brain Class
import numpy as np
import sys
import random
import copy
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import sequencing
from common.cfunctions import (state_multi_channel, build_experience,
                               adapt_for_job_rule)
import common.multiobject as mutilobjectivemanager

class ContinuousSchedulingNetwork(nn.Module):
    """Scheduling network with dual-path + gated fusion"""
    def __init__(self, input_size=15, preference_size=3):
        super(ContinuousSchedulingNetwork, self).__init__()
        
        self.lr = 0.0002
        self.input_size = input_size
        self.preference_size = preference_size
        self.clip_norm = 5.0
        
        # ========== Feature group definitions (by value range) ==========
        # Group 1: Ratio features (5 dimensions, range 0-1)
        # Group 2: Count features (4 dimensions, range 0~large integer)
        # Group 3: Time features (4 dimensions, range 0~large real number)
        # Group 4: Difference features (2 dimensions, range -∞~+∞)
        
        self.ratio_size = 5
        self.count_size = 4
        self.time_size = 4
        self.diff_size = 2
        
        # ========== Group normalization layers ==========
        self.norm_ratio = nn.Sequential(nn.LayerNorm(self.ratio_size), nn.Flatten())
        self.norm_count = nn.Sequential(nn.LayerNorm(self.count_size), nn.Flatten())
        self.norm_time = nn.Sequential(nn.LayerNorm(self.time_size), nn.Flatten())
        self.norm_diff = nn.Sequential(nn.LayerNorm(self.diff_size), nn.Flatten())
        
        # ========== 1. State feature extraction ==========
        # Total dimension after concatenation: 5+4+4+2 = 15
        self.state_extractor = nn.Sequential(
            nn.Linear(15, 20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(20),
            nn.Dropout(0.1),
            nn.Linear(20, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1)
        )
        
        # ========== 2. Preference feature enhancement ==========
        self.pref_enhancer = nn.Sequential(
            nn.Linear(preference_size, 6),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(6),
            nn.Dropout(0.1),
            nn.Linear(6, preference_size),
            nn.Softmax(dim=-1)
        )
        
        # ========== 3. Feature fusion (with residual connection) ==========
        # State features 10 dims + Preference features 3 dims = 13 dims
        self.feature_fusion = nn.Sequential(
            nn.Linear(13, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1),
            nn.Linear(10, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 6),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_transform = nn.Linear(13, 6)
        
        # ========== 4. Dual output heads ==========
        # Path A: Fusion feature path
        self.alpha_head = nn.Sequential(
            nn.Linear(6 + preference_size, 8),  # 6 + 3 = 9
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(4),
            nn.Dropout(0.1),
            nn.Linear(4, preference_size),
            nn.Softmax(dim=-1)
        )
        
        # Path B: Direct preference path
        self.pref_direct_alpha = nn.Sequential(
            nn.Linear(preference_size, 6),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(6),
            nn.Dropout(0.1),
            nn.Linear(6, preference_size),
            nn.Softmax(dim=-1)
        )
        
        # ========== 5. Gated fusion ==========
        self.fusion_gate = nn.Sequential(
            nn.Linear(6, 3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
        # ========== 6. Optimizer ==========
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00005,
                                     betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                 factor=0.8, patience=1000,
                                                                 threshold=1e-4, min_lr=1e-6)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def _split_features(self, state_features):
        """
        Split 15-dimensional state features by group
        - Index 0-4: Ratio features (5 dims)
        - Index 5-8: Count features (4 dims)
        - Index 9-12: Time features (4 dims)
        - Index 13-14: Difference features (2 dims)
        """
        ratio_features = state_features[:, :5]
        count_features = state_features[:, 5:9]
        time_features = state_features[:, 9:13]
        diff_features = state_features[:, 13:15]
        return ratio_features, count_features, time_features, diff_features
    
    def forward(self, state_features, preference_vector):
        """
        Forward propagation
        Args:
            state_features: [batch_size, 15] state features
            preference_vector: [batch_size, 3] preference vector
        Returns:
            alpha_output: [batch_size, 3] scheduling weights
        """
        # ========== 1. Group normalization ==========
        ratio_feat, count_feat, time_feat, diff_feat = self._split_features(state_features)
        
        ratio_norm = self.norm_ratio(ratio_feat)
        count_norm = self.norm_count(count_feat)
        time_norm = self.norm_time(time_feat)
        diff_norm = self.norm_diff(diff_feat)
        
        # Concatenate normalized features
        normalized_features = torch.cat([
            ratio_norm, count_norm, time_norm, diff_norm
        ], dim=-1)  # [batch, 15]
        
        # ========== 2. State feature extraction ==========
        state_encoded = self.state_extractor(normalized_features)  # [batch, 10]
        
        # ========== 3. Preference feature enhancement ==========
        enhanced_pref = self.pref_enhancer(preference_vector)  # [batch, 3]
        
        # ========== 4. Feature fusion (with residual connection) ==========
        combined = torch.cat([state_encoded, enhanced_pref], dim=-1)  # [batch, 13]
        fusion_out = self.feature_fusion(combined)  # [batch, 6]
        residual = self.residual_transform(combined)  # [batch, 6]
        fused_features = fusion_out + residual  # [batch, 6]
        
        # ========== 5. Dual path output ==========
        # Path A: Fusion features + preference
        alpha_from_fusion = self.alpha_head(
            torch.cat([fused_features, enhanced_pref], dim=-1)  # [batch, 9]
        )  # [batch, 3]
        
        # Path B: Direct preference
        alpha_from_pref = self.pref_direct_alpha(enhanced_pref)  # [batch, 3]
        
        # ========== 6. Gated fusion ==========
        gate = self.fusion_gate(fused_features)  # [batch, 1]
        alpha_output = gate * alpha_from_fusion + (1 - gate) * alpha_from_pref  # [batch, 3]
        
        return alpha_output

class ContinuousSequencingBrain:
    """Base sequencing brain class"""
    def __init__(self, env, job_creator, WorkerManager, all_machines, job_numbers, *args, **kwargs):
        self.env = env
        self.job_creator = job_creator
        self.worker_manager = WorkerManager
        self.m_list = all_machines
        self.m_no = len(self.m_list)
        self.job_numbers = job_numbers
        self.warm_up = 0.1 * job_numbers * job_creator.avg_pt
        self.span = job_numbers * job_creator.avg_pt
        self.job_creator.build_sqc_experience_repository(self.m_list)
        
        # Experience pool separation: jobs and workers
        self.rep_memo = []
        self.trajectory_buffer = []
        self.balance_ratio = 0.5
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.loss_record = []
        self.rule_loss_record = []
        self.value_loss_record = []
        self.sample_reward_record = []
        self.train_reward_record = []
        self.current_preference_vector = kwargs.get('preference_vector', None)
        self.train = kwargs.get('train', False)
        self.address = kwargs.get('address', None)
        self.ablation = kwargs.get('ablation', None)
        self.gae_lambda = 0.95
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.5
        self.trajectory_buffer_size = 64
        self.batch_size = 64
        self.samples = 0 
        self.input_size = 15      # Original state dimension (for feature extraction)        
        self.training_epochs = None
        self.algorithm_name = None
        
        # Job rule list
        self.job_func_list = [sequencing.SPT, sequencing.LWKR, sequencing.WINQ,
                              sequencing.SRO, sequencing.NPT]        
        self.multi_obj_manager = mutilobjectivemanager.MultiObjectiveManager(num_objectives=3)
        self.training_step_count = 0
        for m in self.m_list:
            m.job_sequencing = self.action_default
        # Create two networks
        if self.train:
            self.env.process(self.warm_up_process())
        else:
            self.multi_obj_manager.get_per_baselines(self.address)
            for m in self.m_list:
                m.job_sequencing = self.action_rule
     
        
    def warm_up_process(self):
        """Warm-up process"""
        for m in self.m_list:
            m.job_sequencing = self.action_warm_up            
        
        for idx, func in enumerate(self.job_func_list):
            self.func_selection = idx
            print('Time {}: Machine job selection rule set to {}'.format(self.env.now, func))
            yield self.env.timeout(int(self.warm_up / (len(self.job_func_list) + 1)))
        
        for m in self.m_list:
            m.job_sequencing = self.action_default
           
        
        print("From time {} to time {}, each machine uses random rules to select jobs.".format(self.env.now, self.warm_up))
        yield self.env.timeout(self.warm_up - self.env.now - 1)
        
        # Deep copy warm-up phase experience to rep_memo
        self.rep_memo = copy.deepcopy(self.trajectory_buffer)
        
        print("Time {}: Each machine starts using deep reinforcement learning to select pending jobs.".format(self.env.now))
        for m in self.m_list:
            m.job_sequencing = self.action_rule

    def action_default(self, sqc_data):
        """Default action phase (random rule)"""
        m_idx = sqc_data.get('machine_idx', 0)
        job_position = 0  # Initialize default value
       
        current_queue = sqc_data.get('queue', 0)
        queue_size = len(current_queue)
      
        if queue_size > 1:
            importance = self.multi_obj_manager.calculate_job_importance(self.job_creator, current_queue)
            _data = adapt_for_job_rule(self.m_list[m_idx])
           
            s_t = state_multi_channel(self.m_list[m_idx], sqc_data)
            preference = self.multi_obj_manager.calculate_preference(self.m_list[m_idx].job_idx)
            a_rule = torch.tensor(preference, dtype=torch.float32)
            job_position = sequencing.RAND(_data)
            _, reward = self.select_by_chebyshev(importance, preference)
            
            build_experience(self, self.env.now, m_idx, s_t, a_rule, reward, preference, importance)
          
            job_position = random.randint(0, queue_size - 1)
        
        return job_position 

    def action_warm_up(self, sqc_data):
        """Warm-up action phase"""
        m_idx = sqc_data.get('machine_idx', 0)
        job_position = 0  # Initialize default value
        
        current_queue = sqc_data.get('queue', 0)
        queue_size = len(current_queue)
       
        if queue_size > 1:
            importance = self.multi_obj_manager.calculate_job_importance(self.job_creator, current_queue)
            _data = adapt_for_job_rule(self.m_list[m_idx])
            job_position = self.job_func_list[self.func_selection](_data)
            
            s_t = state_multi_channel(self.m_list[m_idx], sqc_data)
            preference = self.multi_obj_manager.calculate_preference(self.m_list[m_idx].job_idx)
            a_rule = torch.tensor(preference, dtype=torch.float32)
            _, reward = self.select_by_chebyshev(importance, preference)
            build_experience(self, self.env.now, m_idx, s_t, a_rule, reward, preference, importance)
        
        return job_position 

    def action_rule(self, sqc_data):
        """Deep reinforcement learning action phase"""
        m_idx = sqc_data.get('machine_idx', 0)
        job_position = 0
        
        current_queue = sqc_data.get('queue', 0)
        queue_size = len(current_queue)
               
        if queue_size > 1:
            s_t_full = state_multi_channel(self.m_list[m_idx], sqc_data)
            s_t_reshaped = s_t_full.reshape([1, self.input_size])
            
            if self.train:                
                if self.ablation == "Ablation3":
                    preference = torch.tensor(np.array(np.random.dirichlet([1, 1, 1])), dtype=torch.float32)
                else:
                    preference = self.multi_obj_manager.calculate_preference(self.m_list[m_idx].job_idx)
            else:
                preference = self.current_preference_vector            
                       
            if random.random() < self.epsilon:
                order_coeff = torch.tensor(np.random.dirichlet([1, 1, 1]), dtype=torch.float32)
            else:
                order_coeff = self.network.forward(s_t_reshaped, preference.reshape(1, 3)).squeeze(0)
            
            importance = self.multi_obj_manager.calculate_job_importance(self.job_creator, current_queue)
            
            job_position, reward = self.select_by_chebyshev(importance, order_coeff)
            
            if self.train:
                build_experience(self, self.env.now, m_idx, s_t_full, order_coeff.detach(), reward, preference, importance)
            self.samples += 1
        return job_position

    def select_by_chebyshev(self, candidates_estimates, preference):
        """
        Select the best candidate using the Chebyshev method
        
        Args:
            candidates_estimates: [n_candidates, 3] Three objective estimates for each candidate
            preference: [3] Preference weight vector (neural network output)
        
        Returns:
            selected_idx: Index of the selected candidate
            score: Minimum Chebyshev score
        """
        # Check if candidates is empty
        if candidates_estimates is None or len(candidates_estimates) == 0:
            return 0, 0.0
        
        # Convert to numpy array
        if torch.is_tensor(candidates_estimates):
            candidates = candidates_estimates.detach().cpu().numpy()
        else:
            candidates = np.array(candidates_estimates, dtype=np.float32)
        
        # Check again if empty after conversion
        if candidates.size == 0:
            return 0, 0.0
        
        if torch.is_tensor(preference):
            w = preference.detach().cpu().numpy().flatten()
        else:
            w = np.array(preference, dtype=np.float32).flatten()
        
        # Ensure weight sum equals 1
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)
        
        # Normalize objective values (eliminate dimensional effects)
        min_vals = candidates.min(axis=0, keepdims=True)
        max_vals = candidates.max(axis=0, keepdims=True)
        
        # Prevent division by zero
        ranges = max_vals - min_vals
        ranges[ranges < 1e-10] = 1.0
        
        # Normalize to [0, 1] range
        normalized = (candidates - min_vals) / ranges
        
        # Calculate weighted Chebyshev scores
        weighted = normalized * w[np.newaxis, :]
        if self.ablation == "Ablation1":
            # Randomly select a candidate
            n_candidates = len(candidates)
            selected_idx = random.randint(0, n_candidates - 1)
            # Score uses the Chebyshev score of this candidate (for convenient subsequent comparison)
            scores = np.max(weighted, axis=1)
            score = scores[selected_idx]
        else:
            # Original logic: select the candidate with the smallest Chebyshev score
            scores = np.max(weighted, axis=1)
            selected_idx = np.argmin(scores)
            score = np.min(scores)
        
        return int(selected_idx), float(score)

    def compute_optimal_weights(self, A, k, w, beta=0.5, lambda_reg=0.1, min_weight=0.05, max_weight=0.95):
        """Weight optimization using Chebyshev scoring"""
        n = A.shape[1]
        m = A.shape[0]
        
        if torch.is_tensor(w):
            w_np = w.detach().cpu().numpy()
        else:
            w_np = np.array(w, dtype=np.float64)
        
        min_vals = A.min(axis=0, keepdims=True)
        max_vals = A.max(axis=0, keepdims=True)
        ranges = max_vals - min_vals
        ranges[ranges < 1e-10] = 1.0
        A_normalized = (A - min_vals) / ranges
        
        def chebyshev_score(weights, preference):
            weighted = weights * preference
            return np.max(weighted)
        
        old_score = chebyshev_score(A_normalized[k], w_np)
        uniform = np.ones(n) / n
        
        def objective(p):
            score_k = chebyshev_score(A_normalized[k], p)
            reg = lambda_reg * np.sum((p - uniform)**2)
            return score_k + reg
        
        constraints = []
        constraints.append({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})
        
        for j in range(m):
            if j != k:
                def make_constraint(j):
                    def constraint_func(p):
                        score_k = chebyshev_score(A_normalized[k], p)
                        score_j = chebyshev_score(A_normalized[j], p)
                        return score_j - score_k
                    return {'type': 'ineq', 'fun': constraint_func}
                constraints.append(make_constraint(j))
        
        constraints.append({'type': 'ineq', 'fun': lambda p: old_score - chebyshev_score(A_normalized[k], p)})
        bounds = [(min_weight, max_weight) for _ in range(n)]
        p0 = np.clip(w_np, min_weight, max_weight)
        p0 = p0 / np.sum(p0)
        
        try:
            res = minimize(objective, p0, bounds=bounds, constraints=constraints,
                        method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False})
            
            if res.success:
                p_optimal = res.x
                p_optimal = np.clip(p_optimal, min_weight, max_weight)
                p_optimal = p_optimal / np.sum(p_optimal)
                new_score = chebyshev_score(A_normalized[k], p_optimal)
                
                constraints_ok = True
                tolerance = 1e-6
                for j in range(m):
                    if j != k:
                        score_j = chebyshev_score(A_normalized[j], p_optimal)
                        if score_j < new_score - tolerance:
                            constraints_ok = False
                            break
                
                if new_score > old_score + tolerance:
                    constraints_ok = False
                
                if constraints_ok:
                    return torch.tensor(p_optimal, dtype=torch.float32), new_score
                else:
                    return torch.tensor(p0, dtype=torch.float32), old_score
            else:
                return torch.tensor(p0, dtype=torch.float32), old_score
        except Exception as e:
            print(f"Optimization exception: {e}")
            return torch.tensor(p0, dtype=torch.float32), old_score
    
