import random
import simpy
import sys
sys.path
import numpy as np
import importlib
import warnings
from common.shop_floor import ShopFloor
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
from common.experiment_scene import get_all_scenarios
import torch
import os
warnings.filterwarnings("ignore")
out_analysis = 1
run_num = 30
job_num = 50
drl_preference_runs = 5
baseline = 'HATHL'
benchmark = ['SPT', 'LWKR', 'WINQ','SRO', 'NPT']
all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']     
    # if scenario_id  in ['EXP-1','EXP-2','EXP-3','EXP-4']:
    for cyc in range(run_num):
        print(f'*** Scenario: {scenario_id}, Experiment Round {cyc+1}/{run_num} ***')
        mo_manager = MultiObjectiveManager()        
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        seed = np.random.randint(20000*(1+random.random()))
        address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt"
        brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}") 
        for rule in benchmark:                            
            print(f'Testing rule: {rule}')
            env = simpy.Environment()
            spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                sequencing_rule=rule , seed=seed, address=address)
            spf.simulation()
            mo_manager.add_experiment_data('RULE', f'run_{cyc+1}', spf.job_objectives_records)      
        for pref_run  in range(drl_preference_runs):             
            print(f'DRL Preference test {pref_run+1}')            
            perturbed_pref =  perturbed_preferences[pref_run]
            env = simpy.Environment()
            spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine, 
                    seed=seed,preference_vector=perturbed_pref, address=address)
            spf.simulation()
            mo_manager.add_experiment_data("DRL", f'run_{cyc}', spf.job_objectives_records)

        # Multi-objective analysis
        print("\n=== Multi-Objective Performance Analysis ===")
        report = mo_manager.generate_comprehensive_report()    
        rule_metrics = report['rule_metrics']   
        for rule_name, metrics in rule_metrics.items():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume: {metrics['hypervolume']:.6f}")
            print(f"  Spread: {metrics['spread']:.6f}")
            print(f"  IGD: {metrics['igd']:.6f}")
        
        out_list = ['RULE',"DRL"] 
        mo_manager.save_to_excel(report, scenario_id, cyc, out_list, cpath='Rule')

if out_analysis:    
    data_analysis_report(sub_path= 'Rule')




