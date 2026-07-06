# experiment_sensitivity.py - Dynamic Worker Training Scenario Definitions

import numpy as np

# ========== Fixed Parameters (consistent across all scenarios) ==========

FIXED_PARAMETERS = {
    'process_time_range': (10, 20),   # Processing time range
    'failure_rate_mean': 1000,       # Baseline mean failure interval
    'repair_time_mean': 5,          # Baseline mean repair time    
    'worker_heterogeneity_range': [0.1, 0.6],   # Worker heterogeneity random range
    'machine_heterogeneity_range': [0.1, 0.5],  # Machine heterogeneity random range
    'repair_heterogeneity_range':[0.2, 0.8],
    'loading_range': [0.1, 0.2],     # Loading time range
    'unloading_range': [0.05, 0.12],  # Unloading time range
}
# ========== L9 Orthogonal Experiment Scenarios (3 factors, 3 levels) ==========
# Factor A: Number of machines M (5, 10, 20)
# Factor B: Number of workers W (3,5,10 for M=5; 5,10,20 for M=10; 10,20,40 for M=20)
# Factor C: Machine utilization U (0.75, 0.85, 0.95)
SCENARIO_DEFINITIONS = [
    # ========== Original positive correlation combinations ==========
    # Combo 1: A1B1C1 (M=5, W=2, U=0.75) - Few workers, low load
    {'scenario_id': 'EXP-1', 'factors': {'machine_count': 5, 'worker_count': 2, 'utilization': 0.75}},
    # Combo 2: A1B2C2 (M=5, W=3, U=0.85) - Medium workers, medium load
    {'scenario_id': 'EXP-2', 'factors': {'machine_count': 5, 'worker_count': 3, 'utilization': 0.85}},
    # Combo 3: A1B3C3 (M=5, W=4, U=0.95) - Many workers, high load
    {'scenario_id': 'EXP-3', 'factors': {'machine_count': 5, 'worker_count': 4, 'utilization': 0.95}},
    # Combo 10: A1 negative correlation (M=5, W=2, U=0.95) - Few workers, high load (extreme challenge)
    {'scenario_id': 'EXP-4', 'factors': {'machine_count': 5, 'worker_count': 2, 'utilization': 0.95}},
    
    # Combo 4: A2B1C2 (M=10, W=3, U=0.85) - Medium workers, medium load
    {'scenario_id': 'EXP-5', 'factors': {'machine_count': 10, 'worker_count': 3, 'utilization': 0.85}},
    # Combo 5: A2B2C3 (M=10, W=5, U=0.95) - Many workers, high load
    {'scenario_id': 'EXP-6', 'factors': {'machine_count': 10, 'worker_count': 5, 'utilization': 0.95}},
    # Combo 6: A2B3C1 (M=10, W=7, U=0.75) - Many workers, low load
    {'scenario_id': 'EXP-7', 'factors': {'machine_count': 10, 'worker_count': 7, 'utilization': 0.75}},
    # Combo 11: A2 negative correlation (M=10, W=3, U=0.95) - Few workers, high load
    {'scenario_id': 'EXP-8', 'factors': {'machine_count': 10, 'worker_count': 3, 'utilization': 0.95}},

    # Combo 7: A3B1C3 (M=20, W=5, U=0.95) - Few workers, high load (negative correlation!)
    {'scenario_id': 'EXP-9', 'factors': {'machine_count': 20, 'worker_count': 5, 'utilization': 0.95}},
    # Combo 8: A3B2C1 (M=20, W=8, U=0.75) - Medium workers, medium load
    {'scenario_id': 'EXP-10', 'factors': {'machine_count': 20, 'worker_count': 8, 'utilization': 0.75}},
    # Combo 9: A3B3C2 (M=20, W=12, U=0.85) - Many workers, medium load
    {'scenario_id': 'EXP-11', 'factors': {'machine_count': 20, 'worker_count': 12, 'utilization': 0.85}},
    # Combo 12: A3 negative correlation (M=20, W=5, U=0.75) - Few workers, low load (compare with EXP-7)
    {'scenario_id': 'EXP-12', 'factors': {'machine_count': 20, 'worker_count': 5, 'utilization': 0.75}},
     
    
]

def _build_scenario_parameters(scenario):
   
    return {        
            'machine_count':scenario['factors']['machine_count'],
            'worker_count': scenario['factors']['worker_count'],
            'utilization': scenario['factors']['utilization'],
            'process_time_range': FIXED_PARAMETERS['process_time_range'],
            'failure_rate_mean': FIXED_PARAMETERS['failure_rate_mean'],
            'repair_time_mean': FIXED_PARAMETERS['repair_time_mean'],
            'repair_heterogeneity_range':FIXED_PARAMETERS['repair_heterogeneity_range'],
            'loading_range': FIXED_PARAMETERS['loading_range'],
            'unloading_range': FIXED_PARAMETERS['unloading_range'],
            'worker_heterogeneity_range':FIXED_PARAMETERS['worker_heterogeneity_range'],
            'machine_heterogeneity_range':FIXED_PARAMETERS['machine_heterogeneity_range']
        }
       
    


def get_all_scenarios():
    """Get complete parameter list for all scenarios"""
    scenarios = []
    for scenario in SCENARIO_DEFINITIONS:
        scenarios.append({
            'scenario_id': scenario['scenario_id'],
            'parameters': _build_scenario_parameters(scenario)
        })
    return scenarios
