# agent_worker.py - Worker resource manager (dynamic worker heterogeneity version)

import numpy as np
import math
from typing import List, Dict, Optional


class WorkerLoadTracker:
    """Worker load tracker - retains history records and average calculation only"""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.load_history = [[] for _ in range(num_workers)]  # Load history for each worker
        self.max_history = 100  # Keep at most 100 entries
    
    def record_load(self, worker_idx: int, load_value: float):
        """Record instantaneous worker load (called when loading/unloading operation completes)"""
        self.load_history[worker_idx].append(load_value)
        if len(self.load_history[worker_idx]) > self.max_history:
            self.load_history[worker_idx] = self.load_history[worker_idx][-self.max_history:]
    
    def get_average_loads(self) -> List[float]:
        """Get the average load of each worker"""
        avg_loads = []
        for worker_idx in range(self.num_workers):
            if self.load_history[worker_idx]:
                avg_loads.append(float(np.mean(self.load_history[worker_idx])))
            else:
                avg_loads.append(0.0)
        return avg_loads
    
    def get_load_imbalance(self) -> float:
        """
        Get the coefficient of variation of load across workers (Objective 3)
        Formula: CV = std(average_load) / mean(average_load)
        """
        avg_loads = self.get_average_loads()
        mean_load = np.mean(avg_loads)
        if mean_load < 1e-6:
            return 0.0
        std_load = np.std(avg_loads)
        return float(std_load / mean_load)


class WorkerManager:
    def __init__(self, env, num_workers, num_machines, job_creator, machine_positions=None):
        self.env = env
        self.num_workers = num_workers
        self.num_machines = num_machines
        self.job_creator = job_creator
        self.machine_positions = machine_positions or self._default_positions(num_machines)
        
        # Worker basic attributes
        self.worker_machine_matrix = self._generate_worker_machine_matrix()
        self.physical_capacity = np.zeros(num_workers)
        self.environmental_stress = np.zeros(num_workers)        
        
        # Fatigue parameters (fixed)
        self.physical_fatigue_rate = 0.05
        self.mental_fatigue_rate = 0.08
        self.physical_recovery_rate = 0.03
        self.mental_recovery_rate = 0.02
        self.walking_fatigue_coef = 0.8
        self.loading_physical_intensity = 1.2
        self.loading_mental_intensity = 0.7
        self.unloading_physical_intensity = 1.1
        self.unloading_mental_intensity = 0.6
        self.repair_physical_intensity = 0.5
        self.repair_mental_intensity = 1.5
        
        # Worker dynamic state
        self.worker_available_time = [0.0] * num_workers    # Earliest available time
        self.current_position = [-1] * num_workers
        
        # Statistics
        self.total_walking_distance = np.zeros(num_workers)  # Total walking distance per worker
        self.total_loading_time = np.zeros(num_workers)
        self.total_unloading_time = np.zeros(num_workers)
        self.total_repair_time = np.zeros(num_workers)
        self.total_rest_time = np.zeros(num_workers)
        
        # Worker-level estimates
        self.last_update_time = np.zeros(num_workers)
        self.walking_speed = 1.2
        self.load_tracker = WorkerLoadTracker(num_workers)
        
        # Physical and mental fatigue
        self.physical_fatigue = np.zeros(num_workers)
        self.mental_fatigue = np.zeros(num_workers)
        self.total_fatigue = np.zeros(num_workers)        
      
        # Machine list reference (for machine load collection)
        self.m_list = None
        
        # ========== Scenario freezer reference ==========
        self.scenario_freezer = None
        # =================================

    def set_machine_list(self, machine_list):
        """Set machine list reference"""
        self.m_list = machine_list

    def _default_positions(self, num_machines):
        positions = {}
        for i in range(num_machines):
            positions[i] = (i * 5, 0)
        return positions
    
    def _generate_worker_machine_matrix(self):
        matrix = np.zeros((self.num_workers, self.num_machines))
        for w_idx in range(self.num_workers):
            for m_idx in range(self.num_machines):
                efficiency = np.random.normal(1.0, 0.2)
                efficiency = np.clip(efficiency, 0.8, 1.2)
                matrix[w_idx, m_idx] = round(efficiency, 2)
        return matrix
    
    # ========== Core: Compute physical capacity index from continuous heterogeneity value ==========
    def _compute_physical_capacity(self, worker_heterogeneity):
      
        base_capacity = 0.6 - (worker_heterogeneity - 0.1) * 0.5
        base_capacity = np.clip(base_capacity, 0.2, 0.6)
        
        # Add random variation, volatility increases with heterogeneity
        volatility = worker_heterogeneity * 0.5
        variation = np.random.normal(0, volatility)
        
        capacity = base_capacity + variation
        return np.clip(capacity, 0.2, 0.8)
    
    def _compute_environmental_stress(self, worker_heterogeneity):
       
        base_stress = 0.25 + worker_heterogeneity * 0.3
        base_stress = np.clip(base_stress, 0.25, 0.45)
        
        # Add random variation
        volatility = worker_heterogeneity * 0.3
        variation = np.random.normal(0, volatility)
        
        stress = base_stress + variation
        return np.clip(stress, 0.2, 0.6)
    
    # ========== Core: Update worker heterogeneity (called when each job arrives) ==========
    def update_workers_for_new_job(self, worker_heterogeneity):
        
        for w_idx in range(self.num_workers):
            individual_h = worker_heterogeneity + np.random.normal(0, worker_heterogeneity * 0.2)
            individual_h = np.clip(individual_h, 0.05, 0.65)
            
            # Compute physical capacity and environmental stress based on individual heterogeneity
            self.physical_capacity[w_idx] = self._compute_physical_capacity(individual_h)
            self.environmental_stress[w_idx] = self._compute_environmental_stress(individual_h)
         
        
        # Regenerate efficiency matrix (based on new physical capacity)
        self._regenerate_efficiency_matrix()
    
    def _regenerate_efficiency_matrix(self):
        """Regenerate efficiency matrix based on current worker physical capacities"""
        for w_idx in range(self.num_workers):
            fitness = self.physical_capacity[w_idx]
            for m_idx in range(self.num_machines):
                # Efficiency positively correlates with physical fitness, with added randomness
                efficiency = 0.7 + 0.6 * fitness + np.random.normal(0, 0.05)
                efficiency = np.clip(efficiency, 0.6, 1.3)
                self.worker_machine_matrix[w_idx, m_idx] = efficiency
    
    def record_worker_initial_state(self):
        """Record initial states of all workers to scenario freezer"""
        if not self.scenario_freezer:
            return
        
        for w_idx in range(self.num_workers):
            efficiency_matrix = [self.worker_machine_matrix[w_idx, m_idx] 
                                for m_idx in range(self.num_machines)]
            
            self.scenario_freezer.record_worker_initial_state(
                worker_idx=w_idx,
                position=self.current_position[w_idx],
                efficiency_matrix=efficiency_matrix,
                physical_fatigue=self.physical_fatigue[w_idx],
                mental_fatigue=self.mental_fatigue[w_idx]
            )
    
    def record_worker_load(self, worker_idx: int):      
       
        self.load_tracker.record_load(worker_idx, self.total_fatigue[worker_idx])

    def get_worker_load_imbalance(self) -> float:
        """Get worker load balance (Objective 3)"""
        return self.load_tracker.get_load_imbalance()

    
    def get_available_workers_for_machine(self, machine_idx: int, operation_type: str = 'load'):
        
        available_workers = []
        current_time = self.env.now
        
        for w_idx in range(self.num_workers):            
            efficiency = self.worker_machine_matrix[w_idx, machine_idx]
            if efficiency <= 0:
                continue
            
            wait_time = self.worker_available_time[w_idx] - current_time
            fatigue = self.total_fatigue[w_idx]
            
            # Product score: wait_time × fatigue / efficiency
            product_score = ((wait_time + 0.01) * (fatigue + 0.01)) / (efficiency + 0.01)
            
            available_workers.append({
                'worker_idx': w_idx,
                'efficiency': efficiency,
                'available_time': self.worker_available_time[w_idx],
                'total_fatigue': self.total_fatigue[w_idx],
                'product_score': product_score,
                'wait_time': wait_time
            })
        
        # Sort by product score in ascending order (lower score = higher priority)
        available_workers.sort(key=lambda x: x['product_score'])
        
        return available_workers
    
    def calculate_walking_distance(self, worker_idx: int, from_machine: int, to_machine: int) -> float:
        if from_machine == -1:
            from_pos = (-5, 0)
        else:
            from_pos = self.machine_positions.get(from_machine, (0, 0))
        if to_machine == -1:
            to_pos = (-5, 0)
        else:
            to_pos = self.machine_positions.get(to_machine, (0, 0))
        distance = math.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
        self.total_walking_distance[worker_idx] += distance
        return distance
  
    def _update_fatigue(self, worker_idx: int, duration: float, machine_idx: int, 
                        fatigue_type: str, intensity: float) -> None:
        if duration <= 0:
            return
        if machine_idx >= 0:
            efficiency = self.worker_machine_matrix[worker_idx, machine_idx]
            perceived_duration = duration / efficiency
        else:
            perceived_duration = duration
        if fatigue_type == 'physical':
            current = self.physical_fatigue[worker_idx]
            factor = self.physical_capacity[worker_idx]
            rate = self.physical_fatigue_rate
        else:
            current = self.mental_fatigue[worker_idx]
            factor = self.environmental_stress[worker_idx]
            rate = self.mental_fatigue_rate
        delta = 1 - np.exp(-rate * perceived_duration * factor * intensity)
        new_value = 1 - (1 - current) * (1 - delta)
        if fatigue_type == 'physical':
            self.physical_fatigue[worker_idx] = new_value
        else:
            self.mental_fatigue[worker_idx] = new_value
        self._update_total_fatigue(worker_idx)
    
    def update_fatigue_from_walking(self, worker_idx: int, distance: float):
        walking_time = distance / self.walking_speed
        self._update_fatigue(worker_idx, walking_time, -1, 'physical', self.walking_fatigue_coef)
    
    def update_fatigue_from_loading(self, worker_idx: int, duration: float, machine_idx: int):
        self._update_fatigue(worker_idx, duration, machine_idx, 'physical', self.loading_physical_intensity)
        self.total_loading_time[worker_idx] += duration
    
    def update_fatigue_from_unloading(self, worker_idx: int, duration: float, machine_idx: int):
        self._update_fatigue(worker_idx, duration, machine_idx, 'physical', self.unloading_physical_intensity)
        self.total_unloading_time[worker_idx] += duration
    
    def update_fatigue_from_repair(self, worker_idx: int, duration: float, machine_idx: int):
        self._update_fatigue(worker_idx, duration, machine_idx, 'physical', self.repair_physical_intensity)
        self._update_fatigue(worker_idx, duration, machine_idx, 'mental', self.repair_mental_intensity)
        self.total_repair_time[worker_idx] += duration
    
    def update_mental_fatigue_from_handling(self, worker_idx: int, duration: float, 
                                            machine_idx: int, operation_type: str):
        intensity = self.loading_mental_intensity if operation_type == 'load' else self.unloading_mental_intensity
        self._update_fatigue(worker_idx, duration, machine_idx, 'mental', intensity)
    
    def _update_total_fatigue(self, worker_idx: int):
        self.total_fatigue[worker_idx] = self.physical_fatigue[worker_idx] + self.mental_fatigue[worker_idx]
    
    def update_all_workers_state(self, current_time):
        for w_idx in range(self.num_workers):
            if self.last_update_time[w_idx] < current_time:
                idle_duration = current_time - self.last_update_time[w_idx]
                if idle_duration > 0:
                    self.physical_fatigue[w_idx] *= np.exp(-self.physical_recovery_rate * idle_duration)
                    self.mental_fatigue[w_idx] *= np.exp(-self.mental_recovery_rate * idle_duration)
                    self._update_total_fatigue(w_idx)
                    self.last_update_time[w_idx] = current_time