# static_scheduling.py - Static Scheduling Decoding and Evaluation Function Library
# Used for NSGA-II and other static multi-objective optimization algorithms, maintaining logic fully consistent with dynamic simulation

import numpy as np
import math
from typing import List, Dict, Tuple, Optional

# ============================================================================
# Global Constants (fully consistent with dynamic simulation)
# ============================================================================

# Fatigue accumulation rates
PHYSICAL_FATIGUE_RATE = 0.05
MENTAL_FATIGUE_RATE = 0.08

# Fatigue recovery rates
PHYSICAL_RECOVERY_RATE = 0.03
MENTAL_RECOVERY_RATE = 0.02

# Walking speed
WALKING_SPEED = 1.2

# Walking fatigue coefficient (physical fatigue only)
WALKING_FATIGUE_COEF = 0.8

# Loading fatigue intensity
LOADING_PHYSICAL_INTENSITY = 1.2
LOADING_MENTAL_INTENSITY = 0.7

# Unloading fatigue intensity
UNLOADING_PHYSICAL_INTENSITY = 1.1
UNLOADING_MENTAL_INTENSITY = 0.6

# Repair fatigue intensity
REPAIR_PHYSICAL_INTENSITY = 0.5
REPAIR_MENTAL_INTENSITY = 1.5

# Maximum load history length
MAX_LOAD_HISTORY = 100

# Processing time slice
PROCESSING_TIME_SLICE = 10.0


# ============================================================================
# Part 1: Core Decoding Functions
# ============================================================================

def decode_schedule(individual: List[int], problem) -> Dict:
    """
    Decode chromosome to generate scheduling plan (complete static version)
    Maintains logic fully consistent with dynamic simulation
    
    Key mechanisms:
    1. All jobs are initially in their first machine's queue
    2. Machine load = sum of current operation processing times for all jobs in queue
    3. Record machine load once after each operation completion
    4. Record objective value snapshot at each job completion moment
    5. Worker fatigue uses exponential accumulation + recovery model
    
    Args:
        individual: Chromosome with operation encoding
        problem: JobShopProblem instance
    
    Returns:
        Scheduling result dictionary
    """
    num_jobs = problem.num_jobs
    num_machines = problem.num_machines
    num_workers = problem.num_workers
    machine_positions = problem.machine_positions
    
    # ========== Initialize worker states ==========
    worker_states = {}
    for w_idx in range(num_workers):
        w_data = problem.workers_data[w_idx]
        worker_states[w_idx] = {
            'worker_idx': w_idx,
            'available_time': 0.0,
            'position': w_data['initial_position'],
            'physical_fatigue': w_data['initial_physical_fatigue'],
            'mental_fatigue': w_data['initial_mental_fatigue'],
            'total_fatigue': w_data['initial_physical_fatigue'] + w_data['initial_mental_fatigue'],
            'last_update_time': 0.0,
            'physical_capacity': w_data.get('physical_capacity', 0.5),
            'environmental_stress': w_data.get('environmental_stress', 0.3),
            'efficiency_matrix': w_data['efficiency_matrix'],
            'total_walking_distance': 0.0,
            'total_loading_time': 0.0,
            'total_unloading_time': 0.0,
        }
    
    # ========== Initialize machine states ==========
    machine_states = {}
    for m_idx in range(num_machines):
        machine_states[m_idx] = {
            'machine_idx': m_idx,
            'available_time': 0.0,
            'queue': [],                # Job indices in the queue
            'current_pt': [],           # Processing time of current operation for each job in queue
            'load_history': [],         # Instantaneous load history
            'total_breakdown_time': 0.0,
            'breakdown_count': 0,
        }
    
    # ========== Key: Initially place jobs in first machine queue ==========
    # Consistent with initial_job_assignment and new_job_arrival in dynamic simulation
    for job_idx in range(num_jobs):
        job_data = problem.jobs_data[job_idx]
        first_machine = job_data['machine_sequence'][0]
        first_pt = job_data['processing_times'][0]
        
        machine_states[first_machine]['queue'].append(job_idx)
        machine_states[first_machine]['current_pt'].append(first_pt)
    
    # Record initial machine load (state before operations start)
    for m_idx in range(num_machines):
        initial_load = sum(machine_states[m_idx]['current_pt'])
        machine_states[m_idx]['load_history'].append(initial_load)
    
    # ========== Initialize job progress ==========
    job_op_progress = {j: 0 for j in range(num_jobs)}  # Number of completed operations
    job_completion_time = {j: 0.0 for j in range(num_jobs)}  # Job available time (initially 0)
    job_operation_records = {j: [] for j in range(num_jobs)}
    
    # Record worker load history
    worker_load_history = {w: [] for w in range(num_workers)}
    
    # Record snapshots at each job completion moment
    job_completion_snapshots = []
    
    # ========== Process each gene (operation) in the chromosome ==========
    # Each gene in the chromosome represents: select job job_idx's next operation for processing
    for gene_idx, job_idx in enumerate(individual):
        op_idx = job_op_progress[job_idx]  # Current operation index to process
        
        if op_idx >= num_machines:
            continue  # Job has completed all operations
        
        job_data = problem.jobs_data[job_idx]
        machine = job_data['machine_sequence'][op_idx]
        processing_time = job_data['processing_times'][op_idx]
        loading_time = job_data['loading_times'][op_idx]
        unloading_time = job_data['unloading_times'][op_idx]
        
        # ========== Determine operation start time ==========
        job_ready_time = job_completion_time[job_idx]
        machine_ready_time = machine_states[machine]['available_time']
        current_time = max(job_ready_time, machine_ready_time)
        
        # ========== Handle machine breakdown ==========
        is_broken, remaining_breakdown = _check_machine_breakdown(machine, current_time, problem.machine_breakdowns)
        if is_broken:
            current_time += remaining_breakdown
            machine_states[machine]['total_breakdown_time'] += remaining_breakdown
            machine_states[machine]['breakdown_count'] += 1
        
        # ========== Loading operation ==========
        if loading_time > 0:
            _recover_all_workers_fatigue(worker_states, current_time)
            
            selected_worker = _select_worker_for_machine(machine, current_time, worker_states, machine_positions)
            
            if selected_worker is None:
                return {'valid': False, 'error': f'Operation {gene_idx}: No available loading worker'}
            
            w_state = worker_states[selected_worker]
            
            distance = _calculate_walking_distance(w_state['position'], machine, machine_positions)
            walking_time = distance / WALKING_SPEED / 60
            w_state['total_walking_distance'] += distance
            
            actual_start = max(current_time, w_state['available_time']) + walking_time
            if actual_start > current_time:
                current_time = actual_start
            
            efficiency = w_state['efficiency_matrix'][machine]
            actual_loading_time = loading_time / efficiency if efficiency > 0 else loading_time
            
            _update_fatigue_from_walking(w_state, distance)
            _update_fatigue_from_loading(w_state, actual_loading_time, machine)
            _update_mental_fatigue_from_handling(w_state, actual_loading_time, machine, 'load')
            
            w_state['position'] = machine
            w_state['available_time'] = current_time + actual_loading_time
            w_state['total_loading_time'] += actual_loading_time
            
            _record_worker_load(worker_load_history, selected_worker, w_state['total_fatigue'])
            
            current_time = w_state['available_time']
        
        # ========== Processing operation ==========
        process_start = current_time
        remaining_pt = processing_time
        
        while remaining_pt > 0:
            is_broken, remaining_breakdown = _check_machine_breakdown(machine, current_time, problem.machine_breakdowns)
            if is_broken:
                current_time += remaining_breakdown
                machine_states[machine]['total_breakdown_time'] += remaining_breakdown
                machine_states[machine]['breakdown_count'] += 1
            else:
                time_slice = min(remaining_pt, PROCESSING_TIME_SLICE)
                current_time += time_slice
                remaining_pt -= time_slice
        
        process_end = current_time
        
        # ========== Unloading operation ==========
        if unloading_time > 0:
            _recover_all_workers_fatigue(worker_states, current_time)
            
            selected_worker = _select_worker_for_machine(machine, current_time, worker_states, machine_positions)
            
            if selected_worker is None:
                return {'valid': False, 'error': f'Operation {gene_idx}: No available unloading worker'}
            
            w_state = worker_states[selected_worker]
            
            distance = _calculate_walking_distance(w_state['position'], machine, machine_positions)
            walking_time = distance / WALKING_SPEED / 60
            w_state['total_walking_distance'] += distance
            
            actual_start = max(current_time, w_state['available_time']) + walking_time
            if actual_start > current_time:
                current_time = actual_start
            
            efficiency = w_state['efficiency_matrix'][machine]
            actual_unloading_time = unloading_time / efficiency if efficiency > 0 else unloading_time
            
            _update_fatigue_from_walking(w_state, distance)
            _update_fatigue_from_unloading(w_state, actual_unloading_time, machine)
            _update_mental_fatigue_from_handling(w_state, actual_unloading_time, machine, 'unload')
            
            w_state['position'] = machine
            w_state['available_time'] = current_time + actual_unloading_time
            w_state['total_unloading_time'] += actual_unloading_time
            
            _record_worker_load(worker_load_history, selected_worker, w_state['total_fatigue'])
            
            current_time = w_state['available_time']
        
        # ========== Operation complete: Remove job from current machine queue ==========
        if job_idx in machine_states[machine]['queue']:
            pos = machine_states[machine]['queue'].index(job_idx)
            machine_states[machine]['queue'].pop(pos)
            machine_states[machine]['current_pt'].pop(pos)
        
        # ========== Transfer job to next machine (if more operations remain) ==========
        next_op_idx = op_idx + 1
        if next_op_idx < num_machines:
            next_machine = job_data['machine_sequence'][next_op_idx]
            next_pt = job_data['processing_times'][next_op_idx]
            
            machine_states[next_machine]['queue'].append(job_idx)
            machine_states[next_machine]['current_pt'].append(next_pt)
        
        # ========== Record machine load (fully consistent with record_machine_load in dynamic simulation after_operation) ==========
        for m_idx in range(num_machines):
            load = sum(machine_states[m_idx]['current_pt'])
            machine_states[m_idx]['load_history'].append(load)
            if len(machine_states[m_idx]['load_history']) > MAX_LOAD_HISTORY:
                machine_states[m_idx]['load_history'] = machine_states[m_idx]['load_history'][-MAX_LOAD_HISTORY:]
        
        # ========== Update state ==========
        job_completion_time[job_idx] = current_time
        machine_states[machine]['available_time'] = current_time
        job_op_progress[job_idx] += 1
        
        job_operation_records[job_idx].append({
            'op_idx': op_idx,
            'machine': machine,
            'process_start': process_start,
            'process_end': process_end,
        })
        
        # ========== Check if job is completed, record snapshot ==========
        if job_op_progress[job_idx] == num_machines:
            arrival = 0.0  # All jobs arrive at time 0
            flow_time = current_time - arrival
            
            machine_balance = _calculate_machine_balance(machine_states, num_machines)
            worker_balance = _calculate_worker_balance(worker_load_history, num_workers)
            
            job_completion_snapshots.append({
                'job_idx': job_idx,
                'completion_time': current_time,
                'flow_time': flow_time,
                'machine_balance': machine_balance,
                'worker_balance': worker_balance
            })
    
    # ========== Check validity ==========
    valid = all(job_op_progress[j] == num_machines for j in range(num_jobs))
    
    if not valid:
        incomplete_jobs = [j for j in range(num_jobs) if job_op_progress[j] < num_machines]
        return {'valid': False, 'error': f'Incomplete jobs: {incomplete_jobs}'}
    
    # ========== Calculate cumulative objective values ==========
    total_flow_time = sum(s['flow_time'] for s in job_completion_snapshots)
    total_machine_balance = sum(s['machine_balance'] for s in job_completion_snapshots)
    total_worker_balance = sum(s['worker_balance'] for s in job_completion_snapshots)
    
    return {
        'valid': True,
        'makespan': max(job_completion_time.values()),
        'total_flow_time': total_flow_time,
        'total_machine_balance': total_machine_balance,
        'total_worker_balance': total_worker_balance,
        'job_completion_snapshots': job_completion_snapshots,
        'flow_times': [s['flow_time'] for s in job_completion_snapshots],
        'machine_balances': [s['machine_balance'] for s in job_completion_snapshots],
        'worker_balances': [s['worker_balance'] for s in job_completion_snapshots],
        'job_completion_times': job_completion_time,
        'job_operation_records': job_operation_records,
    }


# ============================================================================
# Part 2: Objective Function Evaluation
# ============================================================================

def evaluate_objectives(schedule: Dict) -> Tuple[float, float, float]:
    """Evaluate three objective functions (fully consistent with dynamic simulation bit=1)"""
    if not schedule.get('valid', False):
        return (float('inf'), float('inf'), float('inf'))
    
    return (
        schedule.get('total_flow_time', float('inf')),
        schedule.get('total_machine_balance', float('inf')),
        schedule.get('total_worker_balance', float('inf'))
    )


# ============================================================================
# Part 3: Worker Fatigue Calculation Functions (fully consistent with agent_worker.py)
# ============================================================================

def _update_fatigue_core(worker_state: Dict, duration: float, machine_idx: int,
                          fatigue_type: str, intensity: float) -> None:
    if duration <= 0:
        return
    
    if machine_idx >= 0:
        efficiency = worker_state['efficiency_matrix'][machine_idx]
        perceived_duration = duration / efficiency if efficiency > 0 else duration
    else:
        perceived_duration = duration
    
    if fatigue_type == 'physical':
        rate = PHYSICAL_FATIGUE_RATE
        factor = worker_state['physical_capacity']
    else:
        rate = MENTAL_FATIGUE_RATE
        factor = worker_state['environmental_stress']
    
    delta = 1 - math.exp(-rate * perceived_duration * factor * intensity)
    
    if fatigue_type == 'physical':
        worker_state['physical_fatigue'] = 1 - (1 - worker_state['physical_fatigue']) * (1 - delta)
    else:
        worker_state['mental_fatigue'] = 1 - (1 - worker_state['mental_fatigue']) * (1 - delta)
    
    worker_state['total_fatigue'] = worker_state['physical_fatigue'] + worker_state['mental_fatigue']
    worker_state['last_update_time'] += duration


def _update_fatigue_from_walking(worker_state: Dict, distance: float) -> None:
    walking_time = distance / WALKING_SPEED
    _update_fatigue_core(worker_state, walking_time, -1, 'physical', WALKING_FATIGUE_COEF)


def _update_fatigue_from_loading(worker_state: Dict, duration: float, machine_idx: int) -> None:
    _update_fatigue_core(worker_state, duration, machine_idx, 'physical', LOADING_PHYSICAL_INTENSITY)


def _update_fatigue_from_unloading(worker_state: Dict, duration: float, machine_idx: int) -> None:
    _update_fatigue_core(worker_state, duration, machine_idx, 'physical', UNLOADING_PHYSICAL_INTENSITY)


def _update_mental_fatigue_from_handling(worker_state: Dict, duration: float,machine_idx: int, operation_type: str) -> None:
    intensity = LOADING_MENTAL_INTENSITY if operation_type == 'load' else UNLOADING_MENTAL_INTENSITY
    _update_fatigue_core(worker_state, duration, machine_idx, 'mental', intensity)


def _recover_all_workers_fatigue(worker_states: Dict, current_time: float) -> None:
    for w_state in worker_states.values():
        if w_state['last_update_time'] < current_time:
            idle_duration = current_time - w_state['last_update_time']
            if idle_duration > 0:
                w_state['physical_fatigue'] *= math.exp(-PHYSICAL_RECOVERY_RATE * idle_duration)
                w_state['mental_fatigue'] *= math.exp(-MENTAL_RECOVERY_RATE * idle_duration)
                w_state['total_fatigue'] = w_state['physical_fatigue'] + w_state['mental_fatigue']
            w_state['last_update_time'] = current_time


# ============================================================================
# Part 4: Helper Functions
# ============================================================================

def _check_machine_breakdown(machine_idx: int, time: float,machine_breakdowns: Dict) -> Tuple[bool, float]:
    breakdowns = machine_breakdowns.get(machine_idx, [])
    for b in breakdowns:
        if isinstance(b, dict):
            start, end = b['start'], b['end']
        else:
            start, end = b.start_time, b.end_time
        if start <= time < end:
            return True, end - time
    return False, 0


def _calculate_walking_distance(from_machine: int, to_machine: int,machine_positions: Dict = None) -> float:
    if machine_positions is None:
        from_pos = (-5.0, 0.0) if from_machine == -1 else (from_machine * 5.0, 0.0)
        to_pos = (-5.0, 0.0) if to_machine == -1 else (to_machine * 5.0, 0.0)
    else:
        from_pos = machine_positions.get(from_machine, (-5.0, 0.0))
        to_pos = machine_positions.get(to_machine, (-5.0, 0.0))
    
    return math.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)


def _select_worker_for_machine(machine_idx: int, current_time: float,worker_states: Dict, machine_positions: Dict = None) -> Optional[int]:
    best_worker = None
    best_score = float('inf')
    
    for w_idx, w_state in worker_states.items():
        efficiency = w_state['efficiency_matrix'][machine_idx]
        if efficiency <= 0:
            continue
        
        distance = _calculate_walking_distance(w_state['position'], machine_idx, machine_positions)
        walking_time = distance / WALKING_SPEED / 60
        
        worker_ready = w_state['available_time'] + walking_time
        wait_time = max(0.0, worker_ready - current_time)
        fatigue = w_state['total_fatigue']
        
        score = ((wait_time + 0.01) * (fatigue + 0.01)) / (efficiency + 0.01)
        
        if score < best_score:
            best_score = score
            best_worker = w_idx
    
    return best_worker


def _record_worker_load(worker_load_history: Dict, worker_idx: int, load_value: float) -> None:
    worker_load_history[worker_idx].append(load_value)
    if len(worker_load_history[worker_idx]) > MAX_LOAD_HISTORY:
        worker_load_history[worker_idx] = worker_load_history[worker_idx][-MAX_LOAD_HISTORY:]


def _calculate_machine_balance(machine_states: Dict, num_machines: int) -> float:
    avg_loads = []
    for m_idx in range(num_machines):
        history = machine_states[m_idx]['load_history'][-MAX_LOAD_HISTORY:]
        avg_loads.append(float(np.mean(history)) if history else 0.0)
    
    if len(avg_loads) > 1:
        mean_load = np.mean(avg_loads)
        if mean_load < 1e-6:
            return 0.0
        return float(np.std(avg_loads) / mean_load)
    return 0.0


def _calculate_worker_balance(worker_load_history: Dict, num_workers: int) -> float:
    avg_loads = []
    for w_idx in range(num_workers):
        history = worker_load_history[w_idx][-MAX_LOAD_HISTORY:]
        avg_loads.append(float(np.mean(history)) if history else 0.0)
    
    if len(avg_loads) > 1:
        mean_load = np.mean(avg_loads)
        if mean_load < 1e-6:
            return 0.0
        return float(np.std(avg_loads) / mean_load)
    return 0.0


# ============================================================================
# Part 5: Comparison Analysis Helper Functions
# ============================================================================

def calculate_hypervolume(points: List[Tuple[float, float, float]],reference_point: Tuple[float, float, float]) -> float:
    if not points:
        return 0.0
    
    points = np.array(points)
    ref = np.array(reference_point)
    
    valid_mask = np.all(points <= ref, axis=1)
    if not np.any(valid_mask):
        return 0.0
    
    valid_points = points[valid_mask]
    sorted_indices = np.argsort(valid_points[:, 0])
    sorted_points = valid_points[sorted_indices]
    
    hv = 0.0
    for i, p in enumerate(sorted_points):
        if i == 0:
            volume = (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
        else:
            prev = sorted_points[i-1]
            volume = (ref[0] - p[0]) * (ref[1] - min(p[1], prev[1])) * (ref[2] - min(p[2], prev[2]))
        hv += max(0, volume)
    
    return hv


def calculate_coverage(pareto_a: List[Tuple[float, float, float]],pareto_b: List[Tuple[float, float, float]]) -> float:
    if not pareto_b:
        return 0.0
    
    def dominates(a, b):
        return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))
    
    dominated_count = sum(1 for b in pareto_b if any(dominates(a, b) for a in pareto_a))
    return dominated_count / len(pareto_b)


def normalize_objectives(objectives_list: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not objectives_list:
        return []
    
    arr = np.array(objectives_list)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    
    ranges = max_vals - min_vals
    ranges[ranges < 1e-10] = 1.0
    
    normalized = (arr - min_vals) / ranges
    return [tuple(row) for row in normalized]