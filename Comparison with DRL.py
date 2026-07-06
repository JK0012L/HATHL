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
from common.experiment_scene import get_all_scenarios
from common.cfunctions import plot_all_machines_gantt,plot_combined_gantt

warnings.filterwarnings("ignore")

out_analysis = 1
show_gatt = 0
run_num = 30
job_num = 50
drl_preference_runs = 5

benchmark = ['SAC','TD3','A2C','DDPG','HATHL'] 
all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']    
    for cyc in range(run_num):  
        print(f'*** Scenario: {scenario_id}, Experiment Round {cyc+1}/{run_num} ***')
        mo_manager = MultiObjectiveManager()        
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        for pref_run  in range(drl_preference_runs):             
            seed = np.random.randint(20000*(1+random.random()))            
            for rule in benchmark:  
                brain_machine = importlib.import_module(f"algorithm.RL.brain_{rule}")              
                address = f"{sys.path[0]}/sequencing_models/{rule}_{scenario_id}.pt"
                print(f'{scenario_id} Scenario, {rule} Test, Round {cyc+1}: {pref_run+1}/{drl_preference_runs}:')                      
                env = simpy.Environment()   
                perturbed_pref =  perturbed_preferences[pref_run]                  
                spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                    seed=seed, preference_vector=perturbed_pref,address=address)                      
                spf.simulation()
                run_id = f'{scenario_id}_{cyc}_pref_{pref_run}'
                mo_manager.add_experiment_data(f'{rule}', run_id, spf.job_objectives_records)                
            

                if show_gatt:
                    """
                    Collect Gantt chart data for all machines and plot
                    """
                    all_records = []
                    for m in spf.m_list:
                        all_records.append(m.get_gantt_data())
                    
                    # Draw combined Gantt chart    
                    plot_combined_gantt(all_records, spf.m_no,save_path=f"{sys.path[0]}/{scenario_id}_gantt.png" )
                    
                    # Can also draw each machine individually    
                    plot_all_machines_gantt(all_records, spf.m_no, save_dir=f"{sys.path[0]}")   
                
            # Multi-objective analysis
        print("\n=== Multi-Objective Performance Analysis ===")
        report = mo_manager.generate_comprehensive_report()    
        rule_metrics = report['rule_metrics']   
        for rule_name, metrics in rule_metrics.items():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume: {metrics['hypervolume']:.6f}")
            print(f"  Spread: {metrics['spread']:.6f}")
            print(f"  IGD: {metrics['igd']:.6f}")
        
        mo_manager.save_to_excel(report, scenario_id, cyc, benchmark, 'Ref_Learning')

if out_analysis:    
    data_analysis_report(sub_path = 'Ref_Learning')




    
   
        