# agent_job.py - Support dynamic worker heterogeneity

import numpy as np
import random
import matplotlib.pyplot as plt
from common.cfunctions import add_job, get_machine_load_imbalance

class creation:
    def __init__(self, env=None, job_numbers=10000, machine_list=None, pt_range=[5,15], 
                 loading_range=[1,3], unloading_range=[1,3], due_tightness=4, E_utliz=0.9, 
                 train=False, GA=None, **kwargs):
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            random.seed(kwargs['seed'])
        
        self.env = env
        self.train = train
        self.m_list = machine_list
        self.no_machines = len(self.m_list)
        
        self.production_record = {}
        self.objects = {}
            
        self.pt_range = pt_range
        self.loading_range = loading_range
        self.unloading_range = unloading_range
        self.avg_pt = (sum(self.pt_range) / len(self.pt_range)) - 0.5
        self.avg_loading = (loading_range[0] + loading_range[1]) / 2
        self.avg_unloading = (unloading_range[0] + unloading_range[1]) / 2
        
        self.span = job_numbers * (self.avg_pt + self.avg_loading + self.avg_unloading) * self.no_machines
        self.E_utliz = E_utliz
        
        self.sequence_seed = list(range(self.no_machines))
        self.in_system_job_no = 0
        self.index_jobs = 0
        
        self.sequence_list = []
        self.pt_list = []
        self.loading_time_list = []
        self.unloading_time_list = []
        self.arrival_list = []
        
        self.op_completion_times = {}
        self.op_wait_times = {}
        self.op_start_times = {}
        
        # Arrival interval control
        avg_op_time = self.avg_pt + self.avg_loading + self.avg_unloading
        self.beta = avg_op_time / self.E_utliz
        self.total_no = job_numbers
        
        # Dynamically adjust arrival fluctuation based on utilization
        k = 8 - 6 * np.clip((self.E_utliz - 0.7) / 0.3, 0, 1)
        scale = self.beta / k
        self.arrival_interval = [round(x) for x in np.random.gamma(shape=k, scale=scale, size=self.total_no)]
        
        # ========== Dynamic worker heterogeneity parameters ==========
        self.worker_heterogeneity_range = kwargs.get('worker_heterogeneity_range', [0.1, 0.6])
        # =====================================
        
        # ========== Scenario freezer reference ==========
        self.scenario_freezer = kwargs.get('scenario_freezer', None)
        # ===================================
        
        self.ptl_generation = self.ptl_generation_random
        self.loading_generation = self.loading_generation_random
        self.unloading_generation = self.unloading_generation_random
        
        self.sqc_brain = None
        self.initial_job_assignment()
        self.env.process(self.new_job_arrival())
    
    def initial_job_assignment(self):
        """Assign initial jobs to each machine (warm start)"""
        sqc_seed = list(range(self.no_machines))
        if self.index_jobs < self.total_no:
            for m_idx, m in enumerate(self.m_list):
                random.shuffle(sqc_seed)
                sqc = [m_idx] + [x for x in sqc_seed if x != m_idx]
                self.sequence_list.append(sqc)
                
                ptl = self.ptl_generation()
                loading_times = self.loading_generation()
                unloading_times = self.unloading_generation()
                
                self.pt_list.append(ptl)
                self.loading_time_list.append(loading_times)
                self.unloading_time_list.append(unloading_times)
                
                remaining_ptl = ptl[1:] if len(ptl) > 0 else []
                remaining_loading = loading_times[1:] if len(loading_times) > 0 else []
                remaining_unloading = unloading_times[1:] if len(unloading_times) > 0 else []
                
                total_op_time = sum(ptl) + sum(loading_times) + sum(unloading_times)               
               
                self.arrival_list.append(int(self.env.now))
                
                self.production_record[self.index_jobs] = [0, 0, 0, 0, 0]
                self.objects[self.index_jobs] = [0, 0, 0, 0]
                self.objects[self.index_jobs][1] = total_op_time
                self.objects[self.index_jobs][2] = 0.3
                self.objects[self.index_jobs][3] = 0.2
    
                self.in_system_job_no += 1
                
                self.op_completion_times[self.index_jobs] = []
                self.op_wait_times[self.index_jobs] = []
                self.op_start_times[self.index_jobs] = []
                
                m.queue.append(self.index_jobs)
                m.sequence_list.append(sqc[1:])
                m.remaining_pt_list.append(remaining_ptl)
                m.remaining_loading_times.append(remaining_loading)
                m.remaining_unloading_times.append(remaining_unloading)                
               
                add_job(m, pt=self.pt_list[self.index_jobs], 
                       loading=self.loading_time_list[self.index_jobs],
                       unloading=self.unloading_time_list[self.index_jobs])
                
                # ========== Record initial job parameters ==========
                if self.scenario_freezer:
                    self.scenario_freezer.start_job_recording(self.index_jobs)
                    for op_idx, machine in enumerate(sqc):
                        self.scenario_freezer.record_job_operation(
                            machine_idx=machine,
                            processing_time=ptl[op_idx],
                            loading_time=loading_times[op_idx],
                            unloading_time=unloading_times[op_idx]
                        )
                    self.scenario_freezer.finish_job_recording(self.arrival_list[self.index_jobs])
                # =====================================
                
                self.index_jobs += 1
    
    def new_job_arrival(self):
        """New job arrival process"""
        while self.index_jobs < self.total_no:
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)
            
            # ========== Core: Update worker heterogeneity when each job arrives ==========
            if self.m_list and hasattr(self.m_list[0], 'worker_manager'):
                worker_mgr = self.m_list[0].worker_manager
                if worker_mgr is not None:
                    # Randomly sample worker heterogeneity level for the current job
                    current_worker_h = np.random.uniform(
                        self.worker_heterogeneity_range[0],
                        self.worker_heterogeneity_range[1]
                    )
                    # Update heterogeneity attributes for all workers
                    worker_mgr.update_workers_for_new_job(current_worker_h)
            # ======================================================
            
            random.shuffle(self.sequence_seed)
            self.sequence_list.append(self.sequence_seed.copy())
            
            ptl = self.ptl_generation()
            loading_times = self.loading_generation()
            unloading_times = self.unloading_generation()
            
            self.pt_list.append(ptl)
            self.loading_time_list.append(loading_times)
            self.unloading_time_list.append(unloading_times)
            
            total_op_time = sum(ptl) + sum(loading_times) + sum(unloading_times)
           
            self.arrival_list.append(int(self.env.now))
            self.in_system_job_no += 1
            
            first_machine = self.sequence_seed[0]
            
            self.production_record[self.index_jobs] = [0, 0, 0, 0, 0]
            self.objects[self.index_jobs] = [0, total_op_time, 0.2, 0.2]
                       
            self.op_completion_times[self.index_jobs] = []
            self.op_wait_times[self.index_jobs] = []
            self.op_start_times[self.index_jobs] = []
            
            self.m_list[first_machine].queue.append(self.index_jobs)
            self.m_list[first_machine].sequence_list.append(self.sequence_list[self.index_jobs][1:])
            self.m_list[first_machine].remaining_pt_list.append(self.pt_list[self.index_jobs][1:])
            self.m_list[first_machine].remaining_loading_times.append(loading_times[1:])
            self.m_list[first_machine].remaining_unloading_times.append(unloading_times[1:])            
           
            add_job(self.m_list[first_machine], 
                   pt=self.pt_list[self.index_jobs],
                   loading=loading_times,
                   unloading=unloading_times)
            
            # ========== Record new job parameters ==========
            if self.scenario_freezer:
                self.scenario_freezer.start_job_recording(self.index_jobs)
                for op_idx, machine in enumerate(self.sequence_seed):
                    self.scenario_freezer.record_job_operation(
                        machine_idx=machine,
                        processing_time=ptl[op_idx],
                        loading_time=loading_times[op_idx],
                        unloading_time=unloading_times[op_idx]
                    )
                self.scenario_freezer.finish_job_recording(self.arrival_list[self.index_jobs])
            # ===================================
            
            try:
                if not self.m_list[first_machine].sufficient_stock.triggered:
                    self.m_list[first_machine].sufficient_stock.succeed()
            except:
                pass
            
            self.index_jobs += 1
    
    def ptl_generation_random(self):
        return [random.randint(self.pt_range[0], self.pt_range[1]-1) 
                for _ in range(self.no_machines)]
    
    def loading_generation_random(self):
        return [round(random.uniform(self.loading_range[0], self.loading_range[1]), 2)
                for _ in range(self.no_machines)]
    
    def unloading_generation_random(self):
        return [round(random.uniform(self.unloading_range[0], self.unloading_range[1]), 2)
                for _ in range(self.no_machines)]
    
    # ========== The following methods remain unchanged ==========
    
    def record_op_start(self, job_idx, op_idx, start_time):
        if job_idx not in self.op_start_times:
            self.op_start_times[job_idx] = []
        while len(self.op_start_times[job_idx]) <= op_idx:
            self.op_start_times[job_idx].append(0)
        self.op_start_times[job_idx][op_idx] = start_time
    
    def record_op_completion(self, job_idx, op_idx, completion_time, wait_time):
        if job_idx not in self.op_completion_times:
            self.op_completion_times[job_idx] = []
            self.op_wait_times[job_idx] = []
        while len(self.op_completion_times[job_idx]) <= op_idx:
            self.op_completion_times[job_idx].append(0)
            self.op_wait_times[job_idx].append(0)
        self.op_completion_times[job_idx][op_idx] = completion_time
        self.op_wait_times[job_idx][op_idx] = wait_time
        if job_idx in self.production_record:
            self.production_record[job_idx][4] = op_idx + 1
    
    def get_remaining_pt(self, job_idx):
        for m in self.m_list:
            if job_idx in m.queue:
                pos = m.queue.index(job_idx)
                if pos < len(m.remaining_pt_list):
                    return sum(m.remaining_pt_list[pos])
        return 0
    
    def get_remaining_loading(self, job_idx):
        for m in self.m_list:
            if job_idx in m.queue:
                pos = m.queue.index(job_idx)
                if hasattr(m, 'remaining_loading_times') and pos < len(m.remaining_loading_times):
                    return sum(m.remaining_loading_times[pos])
        return 0
    
    def get_remaining_unloading(self, job_idx):
        for m in self.m_list:
            if job_idx in m.queue:
                pos = m.queue.index(job_idx)
                if hasattr(m, 'remaining_unloading_times') and pos < len(m.remaining_unloading_times):
                    return sum(m.remaining_unloading_times[pos])
        return 0
    
    def get_remaining_total_time(self, job_idx):
        """Get the total remaining time of a job (including processing, loading, unloading)"""
        remaining_pt = self.get_remaining_pt(job_idx)
        remaining_loading = self.get_remaining_loading(job_idx)
        remaining_unloading = self.get_remaining_unloading(job_idx)
        return remaining_pt + remaining_loading + remaining_unloading

    def get_remaining_ops(self, job_idx):
        for m in self.m_list:
            if job_idx in m.queue:
                pos = m.queue.index(job_idx)
                if pos < len(m.remaining_pt_list):
                    return len(m.remaining_pt_list[pos])
        return 0
    
    def get_completed_pt(self, job_idx):
        if job_idx >= len(self.pt_list):
            return 0
        total_pt = sum(self.pt_list[job_idx])
        remaining = self.get_remaining_pt(job_idx)
        return total_pt - remaining
    
    def get_completed_loading(self, job_idx):
        if job_idx >= len(self.loading_time_list):
            return 0
        total_loading = sum(self.loading_time_list[job_idx])
        remaining = self.get_remaining_loading(job_idx)
        return total_loading - remaining
    
    def get_completed_unloading(self, job_idx):
        if job_idx >= len(self.unloading_time_list):
            return 0
        total_unloading = sum(self.unloading_time_list[job_idx])
        remaining = self.get_remaining_unloading(job_idx)
        return total_unloading - remaining
    
    def get_completed_total_time(self, job_idx):
        return (self.get_completed_pt(job_idx) + 
                self.get_completed_loading(job_idx) + 
                self.get_completed_unloading(job_idx))
    
    def get_completed_ops(self, job_idx):
        if job_idx in self.production_record:
            return self.production_record[job_idx][4]
        return 0
    
    def get_wait_history(self, job_idx):
        return self.op_wait_times.get(job_idx, [])
    
    def estimate_flow_time(self, job_idx, current_time):
        arrival_time = self.arrival_list[job_idx] if job_idx < len(self.arrival_list) else current_time
        elapsed = current_time - arrival_time
        completed_ops = self.get_completed_ops(job_idx)
        
        if completed_ops == 0:
            remaining_pt = self.get_remaining_pt(job_idx)
            remaining_loading = self.get_remaining_loading(job_idx)
            remaining_unloading = self.get_remaining_unloading(job_idx)
            total_remaining = remaining_pt + remaining_loading + remaining_unloading
            return total_remaining + elapsed
        
        completed_total = self.get_completed_total_time(job_idx)
        occurred_wait = max(0, elapsed - completed_total)
        avg_wait_per_op = occurred_wait / completed_ops
        remaining_ops = self.get_remaining_ops(job_idx)
        
        wait_history = self.get_wait_history(job_idx)
        if len(wait_history) >= 2:
            weights = np.exp(np.linspace(0, 1, len(wait_history)))
            weights = weights / np.sum(weights)
            weighted_avg_wait = np.sum(np.array(wait_history) * weights)
            avg_wait_per_op = 0.7 * weighted_avg_wait + 0.3 * avg_wait_per_op
        
        expected_wait = avg_wait_per_op * remaining_ops
        remaining_pt = self.get_remaining_pt(job_idx)
        remaining_loading = self.get_remaining_loading(job_idx)
        remaining_unloading = self.get_remaining_unloading(job_idx)
        total_remaining = remaining_pt + remaining_loading + remaining_unloading
        estimated_flow = elapsed + total_remaining + expected_wait
        return estimated_flow
    
    def get_estimated_solution(self, job_idx):
        """Get the estimated solution for a job (3-dimensional)"""
        if job_idx not in self.objects:
            return np.array([1000, 0.5, 0.5])
        return np.array([
            self.objects[job_idx][1],
            self.objects[job_idx][2],
            self.objects[job_idx][3]
        ])
    
    def update_job_estimates(self, job_idx, current_time, worker_manager, bit=0):
        """Update job estimates (3 objectives)"""
        if job_idx not in self.objects:
            return
        
        # Objective 1: Estimated flow time
        est_f1 = self.estimate_flow_time(job_idx, current_time)
        
        # Objective 2: Machine load balance (obtained from machine list)
        if hasattr(worker_manager, 'm_list') and worker_manager.m_list:
            est_f2 = get_machine_load_imbalance(worker_manager.m_list)
        else:
            est_f2 = 0.5
        
        # Objective 3: Worker load balance
        if worker_manager is not None:
            est_f3 = worker_manager.get_worker_load_imbalance()
        else:
            est_f3 = 0.5
        
        self.objects[job_idx][1] = est_f1
        self.objects[job_idx][2] = est_f2
        self.objects[job_idx][3] = est_f3
        
        if bit == 1:
            self.production_record[job_idx][1] = est_f1
            self.production_record[job_idx][2] = est_f2
            self.production_record[job_idx][3] = est_f3
    
    def dynamic_seed_change(self, interval):
        while self.in_system_job_no >= 1:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
    
    def build_sqc_experience_repository(self, m_list):
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}
            self.rep_memo[m.m_idx] = []