# shop_floor.py - Shop Floor Simulation Class (supports dynamic heterogeneity)

import numpy as np
import sys
import os
import warnings
from agent import agent_job, agent_worker, agent_machine
from common.scenario_freeze import ScenarioFreezer

warnings.filterwarnings("ignore")


class ShopFloor:
    def __init__(self, env, job_numbers, parameters, brain_machine, train=False, **kwargs):
        self.train = train
        self.env = env
        
        self.m_no = parameters['machine_count']
        self.p_no = parameters['worker_count']
        self.use_ratio = parameters['utilization']
        
        self.worker_heterogeneity_range = parameters.get('worker_heterogeneity_range', [0.1, 0.6])
        self.worker_heterogeneity = np.random.uniform(*self.worker_heterogeneity_range)
        self.machine_heterogeneity_range = parameters.get('machine_heterogeneity_range', [0.1, 0.5])
        self.machine_heterogeneity = np.random.uniform(*self.machine_heterogeneity_range)
        
        self.repair_heterogeneity_range = parameters.get('repair_heterogeneity_range', [0.2, 0.8])
        self.repair_heterogeneity = np.random.uniform(*self.repair_heterogeneity_range)
        # ==================================================
        
        self.h_t = parameters['process_time_range'][1]
        self.l_t = parameters['process_time_range'][0]
        self.h_load = parameters['loading_range'][1]
        self.l_load = parameters['loading_range'][0]
        self.h_unload = parameters['unloading_range'][1]
        self.l_unload = parameters['unloading_range'][0]
        
        self.job_numbers = job_numbers
        self.m_list = []
        self.job_objectives_records = []
        
        self.preference_vector = kwargs.get('preference_vector', None)
        self.ablation = kwargs.get('ablation', None)
        self.bit = kwargs.get('bit', 0)
        
        # ========== Scenario Freezer ==========
        self.freeze_scenario = kwargs.get('freeze_scenario', False)
        self.scenario_freezer = None
        self.frozen_scenario = None
        
        if self.freeze_scenario:
            seed = kwargs.get('seed', np.random.randint(1, 10000))
            scenario_id = kwargs.get('scenario_id', 'unknown')
            self.scenario_freezer = ScenarioFreezer(
                scenario_id=scenario_id,
                seed=seed,
                num_machines=self.m_no,
                num_workers=self.p_no
            )
        # =================================
        
        # Create machines (pass heterogeneity ranges)
        for i in range(self.m_no):
            machine_kwargs = {
                'failure_rate_mean': parameters['failure_rate_mean'],
                'repair_time_mean': parameters['repair_time_mean'],
                'machine_heterogeneity_range': self.machine_heterogeneity_range,
                'repair_heterogeneity_range': self.repair_heterogeneity_range,
            }
            
            expr1 = f"self.m_{i} = agent_machine.machine(env, {i}, **machine_kwargs)"
            exec(expr1)
            expr2 = f"self.m_list.append(self.m_{i})"
            exec(expr2)
            
            # Pass scenario freezer reference
            if self.scenario_freezer:
                self.m_list[i].scenario_freezer = self.scenario_freezer
        
        # Create job creator (pass worker heterogeneity range and scenario freezer)
        job_creator_kwargs = {
            'worker_heterogeneity_range': self.worker_heterogeneity_range,
            'scenario_freezer': self.scenario_freezer
        }
        
        if 'seed' in kwargs:
            job_creator_kwargs['seed'] = kwargs['seed']
        
        self.job_creator = agent_job.creation(
            self.env, self.job_numbers, self.m_list,
            pt_range=[self.l_t, self.h_t],
            loading_range=[self.l_load, self.h_load],
            unloading_range=[self.l_unload, self.h_unload],
            E_utliz=self.use_ratio,
            train=train,
            **job_creator_kwargs
        )
        
        # Create worker manager (pass heterogeneity ranges)
        self.worker_manager = agent_worker.WorkerManager(env, self.p_no, self.m_no, self.job_creator)
        self.worker_manager.worker_heterogeneity_range = self.worker_heterogeneity_range
        self.worker_manager.machine_heterogeneity_range = self.machine_heterogeneity_range
        self.worker_manager.set_machine_list(self.m_list)
        
        # Pass scenario freezer reference
        if self.scenario_freezer:
            self.worker_manager.scenario_freezer = self.scenario_freezer
        
        # Initially generate worker attributes (using current job heterogeneity)
        initial_worker_h = np.random.uniform(*self.worker_heterogeneity_range)
        self.worker_manager.update_workers_for_new_job(initial_worker_h)
        
        # Record worker initial state
        if self.scenario_freezer:
            self.worker_manager.record_worker_initial_state()
        
        # Initialize machines
        for i, m in enumerate(self.m_list):
            m.initialization(self.m_list, self.job_creator, self.worker_manager)
            if self.bit == 1:
                m.bit = 1
        
        if 'sequencing_rule' in kwargs:
            self._setup_scheduling_rule(kwargs['sequencing_rule'])
        else:
            self._setup_neural_network(brain_machine, kwargs)
    
    def _setup_neural_network(self, brain_machine, kwargs=None):
        address = kwargs.get('address', None)
        
        if self.train:
            self.sqc_brain = brain_machine.sequencing_brain(
                self.env, self.job_creator, self.worker_manager, self.m_list, self.job_numbers,
                ma_no=self.m_no, tightness=3.0,
                address=address, ablation=self.ablation,
                preference_vector=self.preference_vector,
                worker_manager=self.worker_manager,
                train=True
            )
        else:
            self.sqc_brain = brain_machine.sequencing_brain(
                self.env, self.job_creator, self.worker_manager, self.m_list, self.job_numbers,
                ma_no=self.m_no, tightness=3.0,
                address=address, ablation=self.ablation,
                preference_vector=self.preference_vector,
                worker_manager=self.worker_manager,
                train=False
            )
        
        for m in self.m_list:
            m.sqc_brain = self.sqc_brain
        self.job_creator.sqc_brain = self.sqc_brain
    
    def _setup_scheduling_rule(self, rule_name):
        for m in self.m_list:
            order = f"m.job_sequencing = sequencing.{rule_name}"
            exec(order)
    
    def simulation(self, bit=0):
        self.env.run()
        
        # ========== Freeze scenario ==========
        if self.freeze_scenario and self.scenario_freezer:
            self.frozen_scenario = self.scenario_freezer.freeze()
            # print(f"Scenario frozen, contains {len(self.frozen_scenario.jobs)} jobs, "
            #       f"{len(self.frozen_scenario.machines)} machines, "
            #       f"{len(self.frozen_scenario.workers)} workers")
        # ==============================
        
        if bit == 0:
            self.collect_job_level_objectives()
        else:
            self.collect_total_objectives()
    
    def get_frozen_scenario(self):
        """Get the frozen scenario (for static algorithms)"""
        return self.frozen_scenario
    
    def collect_job_level_objectives(self):
        self.job_objectives_records = []
        for job_id, record in self.job_creator.production_record.items():
            objective1 = record[1]
            objective2 = record[2]
            objective3 = record[3]
            job_record = {
                'job_id': job_id,
                'arrival_order': job_id,
                'objectives': np.array([objective1, objective2, objective3])
            }
            self.job_objectives_records.append(job_record)
        self.job_objectives_records.sort(key=lambda x: x['arrival_order'])
    
    def collect_total_objectives(self):
        self.job_objectives_records = []
        objective1 = 0
        objective2 = 0
        objective3 = 0
        for job_id, record in self.job_creator.production_record.items():
            objective1 += record[1]
            objective2 += record[2]
            objective3 += record[3]
        job_record = {
            'objectives': np.array([objective1, objective2, objective3])
        }
        self.job_objectives_records.append(job_record)