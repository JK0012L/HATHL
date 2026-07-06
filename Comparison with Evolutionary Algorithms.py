# main_training_SI.py
import simpy
import sys
sys.path
import random
import numpy as np
import torch
from common.experiment_scene import get_all_scenarios
from common.shop_floor import ShopFloor
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
from algorithm.SI import MOEA_DD,RVEA,NSGA2,SMS_EMOA
import importlib
import time 

def run_algorithm(algorithm_module, static_instance, run_id_base, mo_manager):
    """Run static multi-objective optimization algorithm"""
    start_time = time.time()
    
    # Create problem instance (pass frozen scenario)
    job = algorithm_module.JobShopProblem(static_instance=static_instance)
    
    # Create optimizer and run
    module_name = algorithm_module.__name__.split('.')[-1]
    optimizer_class = getattr(algorithm_module, f"{module_name}Scheduler")
    optimizer = optimizer_class(job, pop_size=50, max_gen=100)
    results = optimizer.run()
    best_solutions = results['pareto_front']
    
    # Process and store results
    algorithm_name = module_name
    for i, solution in enumerate(best_solutions):
        run_id = f"{run_id_base}_{i}"
        job_record = {'objectives': tuple(solution)}
        mo_manager.add_experiment_data(algorithm_name, run_id, [job_record])
    
    return (time.time() - start_time) / (job.num_jobs * job.num_machines)

# Training phase remains unchanged
out_analysis = 1
run_num = 20
job_num = 50
drl_preference_runs = 50
baseline = 'HATHL'
benchmark = ['SMS_EMOA','MOEA_DD','NSGA2','RVEA']

all_scenarios = get_all_scenarios() 
brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}") 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id'] 
    # if  scenario_id  in ['EXP-10']:       
    print(f'\n=== Starting Multi-Objective Comparison Experiment: Scenario={scenario_id} ===')  
    for cyc in range(run_num): 
        algorithm_times = {}
        mo_manager = MultiObjectiveManager()        
        print(f'*** Experiment Round {cyc+1}/{run_num} ***')                             
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        drl_start_time = time.time()
        for pref_run  in range(drl_preference_runs): 
            seed = np.random.randint(20000*(1+random.random())) 
            address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt" 
            print(f'DRL Preference test {pref_run+1}/{drl_preference_runs}:')
            perturbed_pref =  perturbed_preferences[pref_run]            
            env = simpy.Environment()
            spf = ShopFloor(env, job_num,scenario['parameters'],brain_machine, freeze_scenario=True,
                seed=seed,preference_vector=perturbed_pref, address=address,scenario_id=scenario_id)
            # spf.job_creator.arrival_interval = [0]*(job_num) # If static multi-objective, enable this statement
            spf.simulation(bit=1)    
            run_id = f'{cyc}_{pref_run}'
            mo_manager.add_experiment_data("DRL", run_id, spf.job_objectives_records) 
            frozen_scenario = spf.get_frozen_scenario()
            static_instance = frozen_scenario.to_dict()

        algorithm_times['DRL'] = (time.time() - drl_start_time)/(job_num*scenario['parameters']['machine_count'])

                
        for rule in benchmark:    
            print(f"\n=== Scenario {scenario_id}, {rule} Algorithm Multi-Objective Optimization ===")
            run_id_base = f'{scenario_id}_{cyc}'
            algorithm_times[f'{rule}'] = run_algorithm(globals()[rule], static_instance, run_id_base, mo_manager)            
        
        
        print("\n=== Multi-Objective Performance Analysis ===")
        report = mo_manager.generate_comprehensive_report()    
        rule_metrics = report['rule_metrics']   
        for rule_name, metrics in rule_metrics.items():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume: {metrics['hypervolume']:.6f}")
            print(f"  Spread: {metrics['spread']:.6f}")
            print(f"  IGD: {metrics['igd']:.6f}")

        out_list = ['DRL'] + benchmark
        mo_manager.save_to_excel(report, scenario_id, cyc, out_list,algorithm_times, cpath='DMO')
        

if out_analysis:    
    data_analysis_report(sub_path= 'DMO')


       