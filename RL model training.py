# main_training_RL.py
import simpy
import sys
sys.path
from common.experiment_scene import get_all_scenarios
from common.shop_floor import ShopFloor
import importlib

# benchmark = ['SAC','TD3','A2C','DDPG','HATHL'] 
#benchmark = ['HATHL'] 
benchmark = ['Ablation1','Ablation2','Ablation3']  # Ablation experiment
job_numbers = 10000
if __name__ == "__main__":   
    all_scenarios=get_all_scenarios() 
    for scenario in all_scenarios:             
        scenario_id=scenario['scenario_id']         
        for rule in benchmark:
            # RL model training  
            # brain_machine = importlib.import_module(f"algorithm.RL.brain_{rule}") 
            # address=f"{sys.path[0]}/sequencing_models/{rule}_{scenario_id}.pt"  
            # print(f'\n=== Starting Scenario={scenario_id}, Algorithm {rule} Model Training:===')  
            # Ablation experiment model training 
            brain_machine = importlib.import_module(f"algorithm.RL.brain_HATHL") 
            address=f"{sys.path[0]}/sequencing_models/HATHL_{rule}_{scenario_id}.pt"         
            print(f'\n=== Starting Scenario={scenario_id}, Algorithm HATHL_{rule} Model Training:===')       
            env = simpy.Environment()  
            spf = ShopFloor(env, job_numbers, scenario['parameters'],brain_machine,
                            train=True,address=address,ablation=rule)
            spf.simulation() 
                    
