# agent_machine.py - Support dynamic machine heterogeneity

import sys
sys.path
import numpy as np
import torch
import random
from copy import deepcopy
from tabulate import tabulate
import common.sequencing as sequencing
from common.cfunctions import (before_operation, state_update_all, 
                               complete_experience, sequencing_data_generation, 
                               after_operation)

class machine:
    def __init__(self, env, index, *args, **kwargs):
        self.env = env
        self.m_idx = index
        self.queue = []
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        
        # ========== Loading/unloading time lists ==========
        self.loading_time_list = []
        self.unloading_time_list = []
        self.remaining_loading_times = []
        self.remaining_unloading_times = []
        self.current_loading = []
        self.current_unloading = []
        
        self.decision_point = 0
        self.available_time = 0
        self.average_workcontent = 0
        self.delay_records = []
        self.before_op_slack = []
        self.before_op_winq_loser = []        
        self.average_waiting = 0
        self.cumulative_run_time = 0
        self.global_exp_tard_rate = 0
        self.sufficient_stock = self.env.event()
        self.working_event = self.env.event()
        self.current_pt = []
        self.waiting_jobs = 0
        self.position = 0
        self.being_time = 0
        self.delay = 0
        self.job_idx = 0
        self.total_idle_time = 0
        self.idle_start_time = 0
        self.use_ratio = 1      
        self.ahead_delay_record = np.array([0], dtype=np.float32)
        self.ahead_delay_record_ga = np.array([0], dtype=np.float32)
        self.avg_tardiness = 0
        self.bit = 0
        
        # Worker-related
        self.current_worker = None
        self.worker_assigned_time = 0
        self.worker_release_time = 0
        self.last_walking_distance = 0
        self.last_worker_idx = -1
        
        self.before_op_time = 0        
        self.schedule_records = []
        self.current_record = None
        
        # ========== Machine failure parameters (support dynamic heterogeneity) ==========
        self.base_failure_rate_mean = kwargs.get('failure_rate_mean', 1000)
        self.base_repair_time_mean = kwargs.get('repair_time_mean', 60)
        
        # Heterogeneity ranges
        self.machine_heterogeneity_range = kwargs.get('machine_heterogeneity_range', [0.1, 0.5])
        self.repair_heterogeneity_range = kwargs.get('repair_heterogeneity_range', [0.2, 0.8])
        
        # Current heterogeneity level (resampled on each failure)
        self.current_machine_heterogeneity = np.random.uniform(
            self.machine_heterogeneity_range[0],
            self.machine_heterogeneity_range[1]
        )
        self.current_repair_heterogeneity = np.random.uniform(
            self.repair_heterogeneity_range[0],
            self.repair_heterogeneity_range[1]
        )
        
        # Current failure parameters (computed from current heterogeneity)
        self.failure_rate_mean = self.base_failure_rate_mean
        self.failure_rate_std = self.base_failure_rate_mean * self.current_machine_heterogeneity
        self.repair_time_mean = self.base_repair_time_mean
        self.repair_time_std = self.base_repair_time_mean * self.current_repair_heterogeneity
        
        self.is_broken = False
        self.breakdown_process = None
        self.repair_start_time = 0
        self.repair_end_time = 0
        self.total_breakdown_time = 0
        self.breakdown_count = 0
        self.last_repair_worker = None
        
        self.load_history = []           # Machine instantaneous load history
        self.max_history = 100           # Keep at most 100 entries
        
        # ========== Scenario freezer reference ==========
        self.scenario_freezer = None
        # =================================

        if not len(self.queue):
            self.sufficient_stock.succeed()
        self.working_event.succeed()
        
        if 'rule' in kwargs:
            order = "self.job_sequencing = sequencing." + kwargs['rule']
            try:
                exec(order)
            except:
                raise Exception
        else:
            self.job_sequencing = sequencing.RAND
    
    def get_current_instant_load(self):
        """Get the current instantaneous load of the machine"""
        queue_load = sum(self.current_pt) if self.current_pt else 0
        processing_remaining = getattr(self, 'current_processing_remaining', 0)
        return queue_load + processing_remaining
    
    def record_machine_load(self):
        """Record instantaneous loads of all machines (called when an operation completes)"""
        # Get instantaneous loads of all machines
        all_machine_loads = []
        for m in self.m_list:
            current_load = m.get_current_instant_load()
            all_machine_loads.append(current_load)
        
        # Record to each machine's load history
        for idx, m in enumerate(self.m_list):
            m.load_history.append(all_machine_loads[idx])
            if len(m.load_history) > m.max_history:
                m.load_history = m.load_history[-m.max_history:]
    
    def get_average_load(self):
        """Get the average load of the machine"""
        if not self.load_history:
            return 0.0
        return float(np.mean(self.load_history))

    def _update_failure_parameters(self):
        """
        After each failure, resample heterogeneity and update failure parameters
        """
        # Resample heterogeneity levels
        self.current_machine_heterogeneity = np.random.uniform(
            self.machine_heterogeneity_range[0],
            self.machine_heterogeneity_range[1]
        )
        self.current_repair_heterogeneity = np.random.uniform(
            self.repair_heterogeneity_range[0],
            self.repair_heterogeneity_range[1]
        )
        
        # Update failure parameters
        self.failure_rate_std = self.base_failure_rate_mean * self.current_machine_heterogeneity
        self.repair_time_std = self.base_repair_time_mean * self.current_repair_heterogeneity
    
    def initialization(self, machine_list, job_creator, worker_manager=None):
        self.m_list = machine_list
        self.m_no = len(self.m_list)
        self.no_ops = len(self.m_list)
        self.job_creator = job_creator
        self.worker_manager = worker_manager       
        self.cur_ops = 0
        state_update_all(self)
        self.env.process(self.production())
        
        if self.worker_manager is not None:
            self.breakdown_process = self.env.process(self.breakdown_generation())

    def breakdown_generation(self):
        while True:
            # Generate next failure interval using current failure parameters
            next_breakdown_interval = max(1, np.random.normal(
                self.failure_rate_mean, 
                self.failure_rate_std
            ))
            
            if self.job_creator.in_system_job_no > 0:
                yield self.env.timeout(next_breakdown_interval)
            else:
                break
            
            if not self.is_broken and len(self.queue) > 0 and self.working_event.triggered:
                self.is_broken = True
                self.breakdown_count += 1
                self.working_event = self.env.event()
                self.env.process(self.handle_breakdown())

    def execute_worker_operation(self, operation_type, duration, target_machine=None):
        if self.worker_manager is None:
            yield self.env.timeout(duration)
            return None, duration, 0, None
        
        decision_time = self.env.now
        self.worker_manager.update_all_workers_state(self.env.now)       
        available_workers = self.worker_manager.get_available_workers_for_machine(target_machine, operation_type)
        selected_worker = available_workers[0]

        worker_idx = selected_worker['worker_idx']
        efficiency = selected_worker['efficiency']
        worker_available = selected_worker['available_time']
        
        if operation_type == 'load':
            if self.current_record is not None:
                self.current_record = {
                    'job_idx': self.job_idx,
                    'op_idx': self.cur_ops,
                    'machine_idx': self.m_idx,
                    'loading_start': decision_time,
                    'loading_worker': worker_idx,
                    'loading_end': None,
                    'process_start': None,
                    'process_end': None,
                    'unloading_start': None,
                    'unloading_worker': None,
                    'unloading_end': None
                }
        elif operation_type == 'unload':
            if self.current_record is not None:
                self.current_record['unloading_start'] = decision_time
                self.current_record['unloading_worker'] = worker_idx
        
        from_pos = self.worker_manager.current_position[worker_idx]
        distance = self.worker_manager.calculate_walking_distance(worker_idx, from_pos, target_machine)
        walking_time = distance / self.worker_manager.walking_speed
        
        actual_start = max(decision_time, worker_available) + walking_time/60
        if actual_start > decision_time:
            yield self.env.timeout(actual_start - decision_time)
        
        self.worker_manager.update_fatigue_from_walking(worker_idx, distance)
        
        actual_duration = duration / efficiency
        
        if operation_type == 'load':
            self.worker_manager.update_fatigue_from_loading(worker_idx, actual_duration, target_machine)
            self.worker_manager.update_mental_fatigue_from_handling(worker_idx, actual_duration, target_machine, 'load')
        elif operation_type == 'unload':
            self.worker_manager.update_fatigue_from_unloading(worker_idx, actual_duration, target_machine)
            self.worker_manager.update_mental_fatigue_from_handling(worker_idx, actual_duration, target_machine, 'unload')
        elif operation_type == 'repair':
            self.worker_manager.update_fatigue_from_repair(worker_idx, actual_duration, target_machine)
        
        yield self.env.timeout(actual_duration)
        
        end_time = actual_start + actual_duration
        self.worker_manager.worker_available_time[worker_idx] = end_time
        self.worker_manager.current_position[worker_idx] = target_machine
        self.current_worker = worker_idx
        
        if operation_type == 'load':
            if self.current_record is not None:
                self.current_record['loading_end'] = end_time
        elif operation_type == 'unload':
            if self.current_record is not None:
                self.current_record['unloading_end'] = end_time
                self.schedule_records.append(self.current_record)
                self.current_record = None
                self.worker_manager.update_all_workers_state(self.env.now)   
        self.worker_manager.record_worker_load(worker_idx)
       
        return worker_idx, actual_duration, distance

    def execute_handling(self, handling_time, operation_type):
        worker_idx, actual_duration, distance = yield self.env.process(
            self.execute_worker_operation(operation_type, handling_time, self.m_idx)
        )
        return worker_idx, actual_duration, distance

    def handle_breakdown(self):
        breakdown_start = self.env.now
        self.repair_start_time = breakdown_start
        
        # Use current repair parameters
        repair_duration = max(1, np.random.normal(self.repair_time_mean, self.repair_time_std))
        
        worker_idx, actual_repair_duration, distance = yield self.env.process(
            self.execute_worker_operation('repair', repair_duration, self.m_idx)
        )
        
        self.last_repair_worker = worker_idx
        self.repair_end_time = self.env.now        
       
        self.is_broken = False
        self.working_event.succeed()
        breakdown_duration = self.repair_end_time - breakdown_start
        self.total_breakdown_time += breakdown_duration
        
        # ========== Key: After repair completes, update parameters for next failure ==========
        self._update_failure_parameters()
        
        # ========== Record breakdown event to scenario freezer ==========
        if self.scenario_freezer:
            self.scenario_freezer.record_breakdown(
                machine_idx=self.m_idx,
                start_time=breakdown_start,
                end_time=self.repair_end_time
            )
        # =================================================
        
        if len(self.queue) > 0:
            try:
                if not self.sufficient_stock.triggered:
                    self.sufficient_stock.succeed()
            except:
                pass

    def get_current_handling_time(self, position, operation_type):
        if position >= len(self.queue):
            return 0
        
        current_sequence = self.sequence_list[position] if position < len(self.sequence_list) else []
        completed_ops = self.no_ops - len(current_sequence) - 1
        
        if operation_type == 'load':
            if hasattr(self, 'loading_time_list') and position < len(self.loading_time_list):
                loading_times = self.loading_time_list[position]
                if completed_ops < len(loading_times):
                    return loading_times[completed_ops]
        else:
            if hasattr(self, 'unloading_time_list') and position < len(self.unloading_time_list):
                unloading_times = self.unloading_time_list[position]
                if completed_ops < len(unloading_times):
                    return unloading_times[completed_ops]
        
        return 0

    def production(self):
        if not len(self.queue):
            yield self.env.process(self.starvation())
        state_update_all(self)
        
        while True:
            if self.is_broken:
                yield self.working_event
                
            self.decision_point = self.env.now
           
            sqc_data = sequencing_data_generation(self)
            self.position = self.job_sequencing(sqc_data)
            self.job_idx = self.queue[self.position]
            
            pt = self.pt_list[self.position][self.m_idx] if self.position < len(self.pt_list) else 0
            loading_time = self.get_current_handling_time(self.position, 'load')
            unloading_time = self.get_current_handling_time(self.position, 'unload')
           
            current_sequence = self.sequence_list[self.position] if self.position < len(self.sequence_list) else []
            self.cur_ops = self.no_ops - len(current_sequence)
            
            self.before_op_time = self.env.now
          
            if loading_time > 0:
                self.current_record = {
                    'job_idx': self.job_idx,
                    'op_idx': self.cur_ops,
                    'machine_idx': self.m_idx,
                    'loading_start': None,
                    'loading_worker': None,
                    'loading_end': None,
                    'process_start': None,
                    'process_end': None,
                    'unloading_start': None,
                    'unloading_worker': None,
                    'unloading_end': None
                }
                yield self.env.process(self.execute_handling(loading_time, 'load'))
                
                if self.current_record is not None:
                    self.current_record['process_start'] = self.env.now
            
            before_operation(self)
            
            remaining_time = pt
            while remaining_time > 0:
                if not self.is_broken:
                    time_slice = min(remaining_time, 10)
                    yield self.env.timeout(time_slice)
                    remaining_time -= time_slice
                    self.cumulative_run_time += time_slice
                    
                    if self.is_broken:
                        yield self.working_event
                else:
                    yield self.working_event
            
            if self.current_record is not None:
                self.current_record['process_end'] = self.env.now
            
            if unloading_time > 0:
                yield self.env.process(self.execute_handling(unloading_time, 'unload'))
            
            after_operation(self, self.decision_point)
           
            if not len(self.queue):
                yield self.env.process(self.starvation())

    def starvation(self): 
        self.sufficient_stock = self.env.event()
        yield self.sufficient_stock
        if self.is_broken or not self.working_event.triggered:
            yield self.working_event
        state_update_all(self)
    
    def get_gantt_data(self):
        return self.schedule_records