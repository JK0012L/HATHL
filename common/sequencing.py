# sequencing.py - Common Dual-Resource Scheduling Rules (first 8)
# The same rule can select either jobs or workers

import random
import numpy as np
import torch

# ========== Utility Functions ==========

def _safe_tensor_to_numpy(data):
    """Safely convert tensor to numpy array"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def _get_argmin_with_random_tie(values):
    """Get the index of the minimum value, randomly choose if multiple equal values exist"""
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = np.array(values)
    
    if values_np.ndim > 1:
        values_np = values_np.flatten()
    
    if len(values_np) == 0:
        return 0
    
    min_value = np.min(values_np)
    min_indices = np.where(values_np == min_value)[0]
    return random.choice(min_indices)

def _get_argmax_with_random_tie(values):
    """Get the index of the maximum value, randomly choose if multiple equal values exist"""
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = np.array(values)
    
    if values_np.ndim > 1:
        values_np = values_np.flatten()
    
    if len(values_np) == 0:
        return 0
    
    max_value = np.max(values_np)
    max_indices = np.where(values_np == max_value)[0]
    return random.choice(max_indices)


def SPT(data, bit='job'):
    """
    Shortest Processing Time First
    Rule logic: Select the job with the shortest current operation processing time
    Optimization objective: Minimize Makespan
    """
    current_pt = _safe_tensor_to_numpy(data.get('current_pt', []))
    if len(current_pt) == 0:
        return 0
    return _get_argmin_with_random_tie(current_pt)


def LWKR(data, bit='job'):
    """
    Least Work Remaining First
    Rule logic: Select the job with the least remaining processing time (including loading/unloading)
    Optimization objective: Minimize Makespan, accelerate job completion
    """
    remaining_work = _safe_tensor_to_numpy(data.get('remaining_work', []))
    if len(remaining_work) == 0:
        return 0
    return _get_argmin_with_random_tie(remaining_work)


def WINQ(data, bit='job'):
    """
    Work In Next Queue First
    Rule logic: Select the job whose next machine queue has the smallest workload
    Optimization objective: Balance machine loads, reduce machine idle time
    """
    winq = _safe_tensor_to_numpy(data.get('winq', []))
    if len(winq) == 0:
        return 0
    return _get_argmin_with_random_tie(winq)


def SRO(data, bit='job'):
    """
    Shortest Remaining Operations First
    Rule logic: Select the job with the fewest remaining operations
    Optimization objective: Complete jobs quickly, free up machines
    """
    remaining_ops = _safe_tensor_to_numpy(data.get('remaining_ops', []))
    if len(remaining_ops) == 0:
        return 0
    return _get_argmin_with_random_tie(remaining_ops)


def NPT(data, bit='job'):
    """
    Next Processing Time First
    Rule logic: Select the job with the shortest next operation processing time
    Optimization objective: Balance downstream machine loads
    """
    next_pt = _safe_tensor_to_numpy(data.get('next_pt', []))
    if len(next_pt) == 0:
        return 0
    return _get_argmin_with_random_tie(next_pt)


def RAND(data, operation_type=None, bit='worker'):
    """
    Random Selection
    Randomly select candidate (for exploration)
    """
    n_candidates = len(data.get('current_time', [])) or len(data.get('efficiency', []))
    return random.randint(0, max(0, n_candidates - 1))