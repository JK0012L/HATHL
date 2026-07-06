# experiment_sensitivity.py - Heterogeneity Parameter Sensitivity Analysis Scenario Definitions

import numpy as np

# ========== Fixed Parameters (consistent across all scenarios) ==========
FIXED_PARAMETERS = {
    'process_time_range': (10, 20),      # Processing time range
    'failure_rate_mean': 1000,           # Baseline mean failure interval
    'repair_time_mean': 5,               # Baseline mean repair time
    'loading_range': [0.1, 0.2],         # Loading time range
    'unloading_range': [0.05, 0.12],      # Unloading time range
}

# ========== Baseline Scenario Parameters (fixed M=10, W=5, U=0.85) ==========
BASE_SCENARIO = {
    'machine_count': 10,
    'worker_count': 5,
    'utilization': 0.95,
    'process_time_range': FIXED_PARAMETERS['process_time_range'],
    'failure_rate_mean': FIXED_PARAMETERS['failure_rate_mean'],
    'repair_time_mean': FIXED_PARAMETERS['repair_time_mean'],
    'loading_range': FIXED_PARAMETERS['loading_range'],
    'unloading_range': FIXED_PARAMETERS['unloading_range'],
}


def _build_sensitivity_parameters(heterogeneity_config):
    """
    Build scenario parameters from heterogeneity configuration
    Convert single values to ranges with identical bounds for algorithm compatibility
    
    Args:
        heterogeneity_config: Dictionary containing worker_heterogeneity, machine_heterogeneity, 
                              repair_heterogeneity
    
    Returns:
        dict: Complete scenario parameters (ranges with identical bounds)
    """
    params = BASE_SCENARIO.copy()
    
    # Key: Convert single value to a list with identical bounds (compatible with algorithm code)
    worker_val = heterogeneity_config['worker_heterogeneity']
    machine_val = heterogeneity_config['machine_heterogeneity']
    repair_val = heterogeneity_config['repair_heterogeneity']
    
    params['worker_heterogeneity_range'] = [worker_val, worker_val]
    params['machine_heterogeneity_range'] = [machine_val, machine_val]
    params['repair_heterogeneity_range'] = [repair_val, repair_val]
    
    return params


# ========== Worker Heterogeneity Sensitivity Analysis (9 scenarios) ==========
WORKER_HETEROGENEITY_SCENARIOS = {
    'W-HET-1': {'worker_heterogeneity': 0.05, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-2': {'worker_heterogeneity': 0.10, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-3': {'worker_heterogeneity': 0.15, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-4': {'worker_heterogeneity': 0.20, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-5': {'worker_heterogeneity': 0.25, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-6': {'worker_heterogeneity': 0.30, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-7': {'worker_heterogeneity': 0.40, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-8': {'worker_heterogeneity': 0.50, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
    'W-HET-9': {'worker_heterogeneity': 0.60, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.5},
}

# ========== Machine Heterogeneity Sensitivity Analysis (9 scenarios) ==========
MACHINE_HETEROGENEITY_SCENARIOS = {
    'M-HET-1': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.05, 'repair_heterogeneity': 0.5},
    'M-HET-2': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.10, 'repair_heterogeneity': 0.5},
    'M-HET-3': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.15, 'repair_heterogeneity': 0.5},
    'M-HET-4': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.20, 'repair_heterogeneity': 0.5},
    'M-HET-5': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.25, 'repair_heterogeneity': 0.5},
    'M-HET-6': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.30, 'repair_heterogeneity': 0.5},
    'M-HET-7': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.40, 'repair_heterogeneity': 0.5},
    'M-HET-8': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.50, 'repair_heterogeneity': 0.5},
    'M-HET-9': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.60, 'repair_heterogeneity': 0.5},
}

# ========== Repair Heterogeneity Sensitivity Analysis (9 scenarios) ==========
REPAIR_HETEROGENEITY_SCENARIOS = {
    'R-HET-1': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.05},
    'R-HET-2': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.10},
    'R-HET-3': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.15},
    'R-HET-4': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.20},
    'R-HET-5': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.30},
    'R-HET-6': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.50},
    'R-HET-7': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.70},
    'R-HET-8': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 0.85},
    'R-HET-9': {'worker_heterogeneity': 0.35, 'machine_heterogeneity': 0.3, 'repair_heterogeneity': 1.00},
}

# ========== All Sensitivity Experiment Scenarios Summary ==========
ALL_SENSITIVITY_SCENARIOS = {
    'worker': WORKER_HETEROGENEITY_SCENARIOS,
    'machine': MACHINE_HETEROGENEITY_SCENARIOS,
    'repair': REPAIR_HETEROGENEITY_SCENARIOS,
}


def get_worker_heterogeneity_scenarios():
    """Get all scenarios for worker heterogeneity sensitivity analysis"""
    scenarios = []
    for scenario_id, het_config in WORKER_HETEROGENEITY_SCENARIOS.items():
        scenarios.append({
            'scenario_id': scenario_id,
            'parameters': _build_sensitivity_parameters(het_config),
            'sensitivity_type': 'worker',
            'sensitivity_value': het_config['worker_heterogeneity']
        })
    return scenarios


def get_machine_heterogeneity_scenarios():
    """Get all scenarios for machine heterogeneity sensitivity analysis"""
    scenarios = []
    for scenario_id, het_config in MACHINE_HETEROGENEITY_SCENARIOS.items():
        scenarios.append({
            'scenario_id': scenario_id,
            'parameters': _build_sensitivity_parameters(het_config),
            'sensitivity_type': 'machine',
            'sensitivity_value': het_config['machine_heterogeneity']
        })
    return scenarios


def get_repair_heterogeneity_scenarios():
    """Get all scenarios for repair heterogeneity sensitivity analysis"""
    scenarios = []
    for scenario_id, het_config in REPAIR_HETEROGENEITY_SCENARIOS.items():
        scenarios.append({
            'scenario_id': scenario_id,
            'parameters': _build_sensitivity_parameters(het_config),
            'sensitivity_type': 'repair',
            'sensitivity_value': het_config['repair_heterogeneity']
        })
    return scenarios

def get_sensitivity_scenarios_by_type(sensitivity_type):
    """
    Get scenarios by sensitivity type
    
    Args:
        sensitivity_type: 'worker', 'machine', or 'repair'
    
    Returns:
        list: List of scenarios for the corresponding type
    """
    if sensitivity_type == 'worker':
        return get_worker_heterogeneity_scenarios()
    elif sensitivity_type == 'machine':
        return get_machine_heterogeneity_scenarios()
    elif sensitivity_type == 'repair':
        return get_repair_heterogeneity_scenarios()
    else:
        raise ValueError(f"Unknown sensitivity type: {sensitivity_type}, valid options: 'worker', 'machine', 'repair'")
    
def get_all_sensitivity_scenarios():
    """Get all sensitivity analysis scenarios"""
    all_scenarios = []
    all_scenarios.extend(get_worker_heterogeneity_scenarios())
    all_scenarios.extend(get_machine_heterogeneity_scenarios())
    all_scenarios.extend(get_repair_heterogeneity_scenarios())
    return all_scenarios

# ========== Usage Example ==========
if __name__ == "__main__":
  
    
    # Validate parameter format
    scenarios = get_all_sensitivity_scenarios()
    print(f"\nValidate parameter format (upper and lower bounds identical):")
    for scenario in scenarios[:3]:  # Only check first 3
        params = scenario['parameters']
        print(f"  {scenario['scenario_id']}:")
        print(f"    worker_heterogeneity_range: {params['worker_heterogeneity_range']}")
        print(f"    machine_heterogeneity_range: {params['machine_heterogeneity_range']}")
        print(f"    repair_heterogeneity_range: {params['repair_heterogeneity_range']}")