import simpy
import sys
sys.path #sometimes need this to refresh the path
import importlib
import numpy as np
import random
import torch
import warnings
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
from common.shop_floor import ShopFloor
from common.experiment_sensitivity import get_sensitivity_scenarios_by_type
from common.experiment_scene import get_all_scenarios
from common.sensitivity_analysis import run_heterogeneity_analysis

warnings.filterwarnings("ignore")

out_analysis = 1
run_num = 30
job_num = 50
drl_preference_runs = 5
baseline = 'HATHL'
benchmark = ['repair','machine','worker'] 
all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:  
    scenario_id=scenario['scenario_id'] 
    for rule in benchmark: 
        for cyc in range(run_num):  
            sen_scenarios = get_sensitivity_scenarios_by_type(rule)
            mo_manager = MultiObjectiveManager() 
            for sen_scenario in sen_scenarios:        
                sen_scenario_id=sen_scenario['scenario_id']
                scenario['parameters']['worker_heterogeneity_range']=sen_scenario['parameters']['worker_heterogeneity_range']
                scenario['parameters']['machine_heterogeneity_range']=sen_scenario['parameters']['machine_heterogeneity_range']
                scenario['parameters']['repair_heterogeneity_range']=sen_scenario['parameters']['repair_heterogeneity_range']
                print(f'*** Scenario {scenario_id}, Sensitivity parameter {rule}, ID {sen_scenario_id}, Experiment Round {cyc+1}/{run_num} ***')                    
                perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
                for pref_run  in range(drl_preference_runs):             
                    seed = np.random.randint(20000*(1+random.random())) 
                    brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}")              
                    address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt"
                    print(f'Round {pref_run+1}/{drl_preference_runs}:')                      
                    env = simpy.Environment()   
                    perturbed_pref =  perturbed_preferences[pref_run]                  
                    spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                        seed=seed, preference_vector=perturbed_pref,address=address)                      
                    spf.simulation()
                    run_id = f'{sen_scenario_id}_{cyc}_pref_{pref_run}'
                    mo_manager.add_experiment_data(f'{sen_scenario_id}', run_id, spf.job_objectives_records)  
                
                # Multi-objective analysis
            print("\n=== Multi-Objective Performance Analysis ===")
            report = mo_manager.generate_comprehensive_report()    
            rule_metrics = report['rule_metrics']   
            for rule_name, metrics in rule_metrics.items():
                print(f"\nRule {rule_name}:")
                print(f"  Hypervolume: {metrics['hypervolume']:.6f}")
                print(f"  Spread: {metrics['spread']:.6f}")
                print(f"  IGD: {metrics['igd']:.6f}")

            scenario_ids = [scenario['scenario_id'] for scenario in sen_scenarios]
            mo_manager.save_to_excel(report, scenario_id+'-'+rule, cyc, scenario_ids, 'Sensitivity')

if out_analysis:
    run_heterogeneity_analysis(output_dir= 'experimental_analysis/Sensitivity')





    
   
        