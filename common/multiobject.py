# multiobject.py - Multi-Objective Management Module
import numpy as np
import os
import pickle
from scipy import stats
import torch
import itertools
import random

class MultiObjectiveManager:
    """
    Multi-objective manager
    Responsible for external archive maintenance, preference vector generation, and importance matrix calculation
    """
    def __init__(self, num_objectives=3, max_gen=500):
        """
        Initialize multi-objective manager
        
        Args:
            num_objectives: Number of objectives
            max_gen: Maximum window size for preference selection
        """
        self.num_objectives = num_objectives
        self.max_gen = max_gen
        
        # ========== External archive (stores non-dominated solutions based on estimated values) ==========
        self.external_archive = []  # Each element is [f1_est, f2_est, f3_est, f4_est]
        
        # ========== Preference vector collection ==========
        self.preference_samples = []  # Preference vector list
        self.samples_num = 10  # Generate 10 preference vectors by default
        self.global_preference_baseline = None
        # ========== Normalization bounds (dynamically updated based on current archive) ==========
        self.normalization_bounds = {
            'min_vals': None,
            'max_vals': None,
            'last_update': 0
        }
        
        self.update_counter = 0
    
    # ========== External Archive Maintenance ==========
    
    def _round_solution(self, solution, decimals=6):
        """
        Round each objective value of the solution to specified decimal places
        
        Args:
            solution: Solution vector
            decimals: Number of decimal places, default 2
        
        Returns:
            np.ndarray: Rounded solution
        """
        return np.round(np.array(solution), decimals)
    
    def is_dominated(self, new_solution, archive, decimals=6):
        """
        Check if a new solution is dominated by any solution in the archive (minimization problem)
        Consider numerical precision, keeping 2 decimal places for comparison
        
        Args:
            new_solution: New solution [f1, f2, f3, f4]
            archive: List of solutions in the archive
            decimals: Decimal places, default 2
        
        Returns:
            bool: True if new solution is dominated, False otherwise
        """
        if not archive:
            return False
        
        # Round to specified decimal places
        new_rounded = self._round_solution(new_solution, decimals)
        
        for solution in archive:
            sol_rounded = self._round_solution(solution, decimals)
            
            # Check if all objectives are not worse than the new solution (sol <= new)
            all_not_worse = all(s <= n for s, n in zip(sol_rounded, new_rounded))
            
            # Check if at least one objective is strictly better than the new solution (sol < new)
            strictly_better = any(s < n for s, n in zip(sol_rounded, new_rounded))
            
            # If all objectives are not worse and at least one is better, new solution is dominated
            if all_not_worse and strictly_better:
                return True
        
        return False
    
    def update_archive(self, estimated_solution):
        """
        Update external archive (based on estimated solution)
        
        Args:
            estimated_solution: Estimated solution vector [f1_est, f2_est, f3_est]
        
        Returns:
            bool: Whether the archive was updated
        """
        if estimated_solution is None or len(estimated_solution) != self.num_objectives:
            return False
        
        # Check for NaN
        if any(np.isnan(estimated_solution)):
            return False
        
        for sol in self.external_archive:
            if np.array_equal(np.round(sol, 4), np.round(estimated_solution,4)):
                # Equal solution exists, do not update archive (equality is not dominance)
                return False
            
        # Check if new solution is dominated by existing archive
        if self.is_dominated(estimated_solution, self.external_archive):
            return False
        
        # Remove solutions dominated by new solution
        self.external_archive = [
            sol for sol in self.external_archive 
            if not self.is_dominated(sol, [estimated_solution])
        ]
        
        # Add new solution
        self.external_archive.append(np.array(estimated_solution))
        
        if len(self.external_archive)>100:
            self.external_archive=self.external_archive[-100:]
               
        self.update_counter += 1

        if self.update_counter % 10 == 0 :
            print(f"  Archive size: {len(self.external_archive)}")

        # Update normalization bounds
        self._update_normalization_bounds()
        self._update_preference_baseline()

        return True
  
    def _update_preference_baseline(self):
        """Update baseline reference points (ideal point, nadir point, etc.)"""
        if len(self.external_archive) == 0:
            return
        # Extract all solution objective values
        objectives_array = self.external_archive
        if len(objectives_array) == 0:
            return
        ideal_point = np.min(objectives_array, axis=0)  # Ideal point: minimum value of each objective
        nadir_point = np.max(objectives_array, axis=0)  # Nadir point: maximum value of each objective
        mean_point = np.mean(objectives_array, axis=0)  # Mean point
        self.global_preference_baseline = {
            'ideal_point': ideal_point,
            'nadir_point': nadir_point,
            'mean_point': mean_point  
        }

    def _update_normalization_bounds(self):
        """Update normalization bounds based on current archive""" 
        archive_array = np.array(self.external_archive)
        min_vals = np.min(archive_array, axis=0)
        max_vals = np.max(archive_array, axis=0)
        
        # Ensure min < max to avoid division by zero
        for i in range(self.num_objectives):
            if max_vals[i] - min_vals[i] < 1e-6:
                max_vals[i] = min_vals[i] + 1.0
        
        self.normalization_bounds = {
            'min_vals': min_vals,
            'max_vals': max_vals,
            'last_update': self.update_counter
        }
    
    def _get_normalization_range(self, use_archive=True):
        """
        Get normalization range
        
        Args:
            use_archive: Whether to use archive bounds
        
        Returns:
            (min_vals, max_vals): Minimum and maximum values
        """
        min_vals = self.normalization_bounds['min_vals']
        max_vals = self.normalization_bounds['max_vals']
       
        return min_vals, max_vals
    
    def generate_preference_samples(self, num_samples=None):
        """
        Generate preference vector set based on current archive
        Called after each job completion and archive update
        
        Args:
            num_samples: Number of preference vectors to generate
        """
        if num_samples is not None:
            self.samples_num = num_samples
        
        if len(self.external_archive) < 3:
        # Insufficient archive, use default uniform distribution
            self.preference_samples = self._generate_uniform_samples(self.samples_num)
            return
        
        # 1. Normalize current archive
        normalized_archive = self._normalize_archive()
        
        # 2. Fit distributions for each objective
        distributions = self._fit_objective_distributions(normalized_archive)
        
        # 3. Generate preference vectors (50% distribution-based, 50% uniform)
        self.preference_samples = self._generate_mixed_samples(
            distributions, self.samples_num
        )
    
    def _normalize_archive(self):
        """Normalize solutions in archive to [0,1]"""
        if len(self.external_archive) < 2:
            return np.array(self.external_archive)
        
        archive_array = np.array(self.external_archive)
        min_vals, max_vals = self._get_normalization_range(use_archive=True)
        ranges = max_vals - min_vals
        ranges[ranges < 1e-10] = 1.0
        
        normalized = (archive_array - min_vals) / ranges
        return np.clip(normalized, 0, 1)
    
    def _fit_objective_distributions(self, normalized_archive):
        """
        Fit probability distributions for each objective
        
        Returns:
            dict: Distribution information for each dimension
        """
        distributions = {}
        n_dims = normalized_archive.shape[1]
        
        for dim in range(n_dims):
            dim_data = normalized_archive[:, dim]
            
            if len(dim_data) < 4:
                # Insufficient data, use uniform distribution
                distributions[dim] = {
                    'type': 'uniform',
                    'params': [0, 1],
                    'function': lambda x, a=0, b=1: stats.uniform.pdf(x, a, b-a)
                }
                continue
            
            # Try fitting multiple distributions
            best_dist = self._fit_best_distribution(dim_data)
            distributions[dim] = best_dist
        
        return distributions
    
    def _fit_best_distribution(self, data):
        """Fit the best probability distribution for the data"""
        distributions_to_try = [
            ('normal', stats.norm),
            ('beta', stats.beta),
            ('gamma', stats.gamma),
            ('lognorm', stats.lognorm),
            ('uniform', stats.uniform)
        ]
        
        best_dist_name = 'uniform'
        best_dist_params = [0, 1]
        best_aic = np.inf
        best_function = lambda x: stats.uniform.pdf(x, 0, 1)
        
        for dist_name, dist_class in distributions_to_try:
            try:
                if dist_name == 'uniform':
                    continue
                
                params = dist_class.fit(data)
                log_likelihood = np.sum(dist_class.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                
                if aic < best_aic and not np.isnan(aic):
                    best_aic = aic
                    best_dist_name = dist_name
                    best_dist_params = params
                    best_function = lambda x, params=params: dist_class.pdf(x, *params)
                    
            except Exception:
                continue
        
        return {
            'type': best_dist_name,
            'params': best_dist_params,
            'function': best_function
        }
    
    def _generate_uniform_samples(self, num_samples):
        """Generate uniformly distributed preference vector samples"""
        samples = []
        for _ in range(num_samples):
            sample = np.random.dirichlet([1, 1, 1])
            samples.append(sample)
        return samples

    def _generate_mixed_samples(self, distributions, num_samples):
        """
        Generate preference vectors using mixed method: 50% distribution-based, 50% uniform
        """
        n_dims = len(distributions)
        samples = []
        
        # 50% distribution-based samples
        num_dist = num_samples // 2
        for _ in range(num_dist):
            sample = np.zeros(n_dims)
            for dim in range(n_dims):
                dist_info = distributions[dim]
                
                # ========== Fix: correctly generate samples based on distribution type ==========
                if dist_info['type'] == 'uniform':
                    sample[dim] = np.random.uniform(0, 1)
                    
                elif dist_info['type'] == 'normal':
                    # Normal distribution: typically has 2 parameters (loc, scale)
                    params = dist_info['params']
                    if len(params) >= 2:
                        loc, scale = params[0], params[1]
                    else:
                        loc, scale = 0, 1
                    val = np.random.normal(loc, scale)
                    sample[dim] = np.clip(val, 0, 1)
                    
                elif dist_info['type'] == 'beta':
                    # ========== Fix: Beta distribution only takes the first two parameters ==========
                    params = dist_info['params']
                    # Beta distribution only needs two parameters a and b
                    if len(params) >= 2:
                        a, b = params[0], params[1]
                    else:
                        a, b = 2, 2  # Default symmetric Beta distribution
                    val = np.random.beta(a, b)
                    sample[dim] = np.clip(val, 0, 1)
                    
                elif dist_info['type'] == 'gamma':
                    # Gamma distribution
                    params = dist_info['params']
                    if len(params) >= 3:
                        shape, loc, scale = params[0], params[1], params[2]
                        val = np.random.gamma(shape, scale) + loc
                    else:
                        val = np.random.gamma(2, 1)
                    sample[dim] = np.clip(val, 0, 1)
                    
                elif dist_info['type'] == 'lognorm':
                    # Log-normal distribution
                    params = dist_info['params']
                    if len(params) >= 3:
                        s, loc, scale = params[0], params[1], params[2]
                        val = np.random.lognormal(mean=np.log(scale), sigma=s) + loc
                    else:
                        val = np.random.lognormal(0, 1)
                    sample[dim] = np.clip(val, 0, 1)
                    
                else:
                    sample[dim] = np.random.uniform(0, 1)
            
            if np.sum(sample) > 0:
                sample = sample / np.sum(sample)
            else:
                sample = np.ones(n_dims) / n_dims
            samples.append(sample)
        
        # 50% uniform samples
        num_uniform = num_samples - num_dist
        samples.extend(self._generate_uniform_samples(num_uniform))
        
        # Shuffle order
        random.shuffle(samples)
        
        return samples
  
    def calculate_preference(self, job_idx):
        """
        Select current preference vector based on job index
        Use modulo strategy to ensure consistent preference within the same job lifecycle
        
        Args:
            job_idx: Job index
        
        Returns:
            torch.Tensor: Preference vector [3]
        """
        if len(self.preference_samples) > 0:
            index = int(job_idx / self.max_gen) % len(self.preference_samples)
            selected_preference = self.preference_samples[index].copy()
            return torch.tensor(selected_preference, dtype=torch.float32)
        else:
            # Return uniform weights when no preference samples available
            return torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
    
    def calculate_job_importance(self, job_creator, job_queue):
        """
        Calculate job importance matrix J
        
        Args:
            job_creator: Job creator
            job_queue: List of job indices in current queue
        
        Returns:
            np.ndarray: [len(job_queue), 3] Normalized importance matrix
        """
        # ========== Handle empty queue ==========
        if job_queue is None or len(job_queue) == 0:
            return np.array([])
        
        # ========== Handle case where baseline does not exist ==========
        if (self.global_preference_baseline is None or 
            len(self.global_preference_baseline) == 0 or
            self.global_preference_baseline.get('ideal_point') is None):
            # Return random importance for warm-up phase
            return np.array([np.random.dirichlet([1, 1, 1]) for _ in range(len(job_queue))])
        
        baseline = self.global_preference_baseline
        ideal_point = np.array(baseline['ideal_point'])      
        nadir_point = np.array(baseline['nadir_point'])
        
        # Get estimated values for all jobs in current queue
        estimates = []
        for job_idx in job_queue:
            if job_idx in job_creator.objects:
                est = np.array([
                    job_creator.objects[job_idx][1],  # f1 estimated value
                    job_creator.objects[job_idx][2],  # f2 estimated value
                    job_creator.objects[job_idx][3]   # f3 estimated value
                ])
            else:
                est = np.array([1000, 0.5, 0.3])
            estimates.append(est)
        
        # Add reference points
        estimates.append(ideal_point)
        estimates.append(nadir_point)
        estimates_matrix = np.array(estimates)
        
        # ========== Get normalization range ==========
        min_vals = self.normalization_bounds.get('min_vals')
        max_vals = self.normalization_bounds.get('max_vals')
        
        # If bounds are None, use bounds from current estimated values
        if min_vals is None or max_vals is None:
            min_vals = np.min(estimates_matrix, axis=0)
            max_vals = np.max(estimates_matrix, axis=0)
            # Ensure bounds are valid
            for i in range(len(min_vals)):
                if max_vals[i] - min_vals[i] < 1e-10:
                    max_vals[i] = min_vals[i] + 1.0
        
        # Normalize
        ranges = max_vals - min_vals
        ranges[ranges < 1e-10] = 1.0
        
        normalized = (estimates_matrix - min_vals) / ranges
        importance = normalized[:-2, :]  # Remove added reference points
        
        return importance

    def save_training_results(self, file_path):
        """Save training results - simplified version (supports fixed baseline mode only)"""
        # Create save directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build save data
        save_data = {            
            # Core data: preference vectors and baselines           
            'global_preference_baseline': self.global_preference_baseline,
            # External archive (save first 100 solutions to avoid excessive file size)
            'external_archive': self.external_archive[:100] if len(self.external_archive) > 0 else [],
            'archive_size': len(self.external_archive),           
        }
       # Save to file
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Training results saved to: {file_path}")          
            print(f"Non-dominated solution count: {len(self.external_archive)}")
            return True
            
        except Exception as e:
            print(f"Save failed: {e}")
            return False
    
    def get_per_baselines(self, file_path):
   
        file_path = file_path.replace('.pt', '_preferences.pkl')
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        base_baseline = save_data.get('global_preference_baseline') 
        
        self.global_preference_baseline = base_baseline
   