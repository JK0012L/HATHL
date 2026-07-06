# main_training_rule.py
import simpy
import sys
sys.path
import random
import os
from openpyxl import load_workbook,Workbook
import numpy as np
import torch
from common.experiment_scene import get_all_scenarios
from common.shop_floor import ShopFloor
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
import importlib


out_analysis = 1     
run_num = 30
job_num = 50      
drl_preference_runs = 5
baseline = 'HATHL'
benchmark = ['HATHL','Ablation1','Ablation2','Ablation3']
# Ablation experiment settings:
# Experiment 1 (Ablation1): Do not use Chebyshev scoring, instead randomly select jobs and return random scores as reward values
# Experiment 2 (Ablation2): Do not use mathematical programming to optimize neural network output
# Experiment 3 (Ablation3): Use random sampling instead of Pareto front analysis to form neural network preference vector input

all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']  
    for cyc in range(run_num):
        print(f'*** Scenario: {scenario_id}, Experiment Round {cyc+1}/{run_num} ***')  
        mo_manager = MultiObjectiveManager()
        seed = np.random.randint(20000*(1+random.random()))        
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        for rule in benchmark:    
            if rule != baseline:
                address = f"{sys.path[0]}/sequencing_models/{baseline}_{rule}_{scenario_id}.pt"
            else:                   
                address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt" 
            brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}") 
            for pref_run  in range(drl_preference_runs):
                print(f'{rule} Test, Round {cyc}: {pref_run+1}/{drl_preference_runs}:')                       
                env = simpy.Environment()   
                perturbed_pref =  perturbed_preferences[pref_run]                  
                spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                    seed=seed, ablation=rule,preference_vector=perturbed_pref,address=address)
                spf.simulation()
                run_id = f'{scenario_id}_{cyc}_pref_{pref_run}'
                mo_manager.add_experiment_data(f'{rule}', run_id, spf.job_objectives_records)                
                
        print("\n=== Multi-Objective Performance Analysis ===")
        report = mo_manager.generate_comprehensive_report()    
        rule_metrics = report['rule_metrics']   
        for rule_name, metrics in rule_metrics.items():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume: {metrics['hypervolume']:.6f}")
            print(f"  Spread: {metrics['spread']:.6f}")
            print(f"  IGD: {metrics['igd']:.6f}")
        mo_manager.save_to_excel(report, scenario_id, cyc, benchmark, cpath='Ablation')

if out_analysis:    
    data_analysis_report(sub_path= 'Ablation')
       
            
        