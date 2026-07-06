# cfunctions.py - Common Scheduling Function Library
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from typing import List, Tuple, Dict
import random
avg_rwd_rule = [0]
avg_rwd_alpha = [0]

def plot_machine_gantt(machine_records, machine_idx, save_path=None):
    """
    Plot Gantt chart for a single machine
    """
    if not machine_records:
        print(f"Machine {machine_idx} has no records")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    job_ids = list(set(r['job_idx'] for r in machine_records))
    job_colors = {}
    for job_id in job_ids:
        hue = (job_id * 0.618033988749895) % 1.0
        job_colors[job_id] = plt.cm.hsv(hue)
    
    y_pos = 0
    y_labels = []
    
    for record in machine_records:
        job_idx = record['job_idx']
        op_idx = record['op_idx']
        
        if record['loading_start'] is not None and record['loading_end'] is not None:
            loading_start = record['loading_start']
            loading_end = record['loading_end']
            loading_duration = loading_end - loading_start
            ax.barh(y_pos, loading_duration, left=loading_start,
                    height=0.3, color='lightgreen', edgecolor='black', linewidth=0.5)
            ax.text(loading_start + loading_duration/2, y_pos + 0.15,
                   f'L-{record["loading_worker"]}',
                   ha='center', va='center', fontsize=8)
        
        if record['process_start'] is not None and record['process_end'] is not None:
            process_start = record['process_start']
            process_end = record['process_end']
            process_duration = process_end - process_start
            ax.barh(y_pos, process_duration, left=process_start,
                    height=0.3, color=job_colors[job_idx], edgecolor='black', linewidth=0.5)
            ax.text(process_start + process_duration/2, y_pos + 0.15,
                   f'J{job_idx}-Op{op_idx}',
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        if record['unloading_start'] is not None and record['unloading_end'] is not None:
            unloading_start = record['unloading_start']
            unloading_end = record['unloading_end']
            unloading_duration = unloading_end - unloading_start
            ax.barh(y_pos, unloading_duration, left=unloading_start,
                    height=0.3, color='lightcoral', edgecolor='black', linewidth=0.5)
            ax.text(unloading_start + unloading_duration/2, y_pos + 0.15,
                   f'U-{record["unloading_worker"]}',
                   ha='center', va='center', fontsize=8)
        
        y_labels.append(f"J{job_idx}-Op{op_idx}")
        y_pos += 1
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)
    ax.set_title(f'Machine {machine_idx} Gantt Chart', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Loading'),
        mpatches.Patch(color='lightcoral', label='Unloading'),
        mpatches.Patch(color='gray', label='Processing')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gantt chart saved: {save_path}")
    else:
        plt.show()

def plot_all_machines_gantt(all_machine_records, num_machines, save_dir=None):
    for m_idx in range(num_machines):
        if m_idx < len(all_machine_records) and all_machine_records[m_idx]:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'machine_{m_idx}_gantt.png')
            plot_machine_gantt(all_machine_records[m_idx], m_idx, save_path)
        else:
            print(f"Machine {m_idx} has no data")

def plot_combined_gantt(all_machine_records, num_machines, save_path=None):
    fig, axes = plt.subplots(num_machines, 1, figsize=(16, 4 * num_machines))
    if num_machines == 1:
        axes = [axes]
    
    all_jobs = []
    for m_records in all_machine_records:
        for r in m_records:
            all_jobs.append(r['job_idx'])
    job_ids = list(set(all_jobs))
    job_colors = {}
    for job_id in job_ids:
        hue = (job_id * 0.618033988749895) % 1.0
        job_colors[job_id] = plt.cm.hsv(hue)
    
    for m_idx, ax in enumerate(axes):
        records = all_machine_records[m_idx] if m_idx < len(all_machine_records) else []
        
        if not records:
            ax.text(0.5, 0.5, f'Machine {m_idx} has no data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Machine {m_idx}')
            continue
        
        y_pos = 0
        y_labels = []
        
        for record in records:
            job_idx = record['job_idx']
            op_idx = record['op_idx']
            
            if record['loading_start'] is not None and record['loading_end'] is not None:
                loading_start = record['loading_start']
                loading_end = record['loading_end']
                ax.barh(y_pos, loading_end - loading_start, left=loading_start,
                        height=0.25, color='lightgreen', edgecolor='black', linewidth=0.5)
            
            if record['process_start'] is not None and record['process_end'] is not None:
                process_start = record['process_start']
                process_end = record['process_end']
                ax.barh(y_pos, process_end - process_start, left=process_start,
                        height=0.25, color=job_colors[job_idx], edgecolor='black', linewidth=0.5)
                ax.text(process_start + (process_end - process_start)/2, y_pos + 0.125,
                       f'J{job_idx}-Op{op_idx}', ha='center', va='center', fontsize=7)
            
            if record['unloading_start'] is not None and record['unloading_end'] is not None:
                unloading_start = record['unloading_start']
                unloading_end = record['unloading_end']
                ax.barh(y_pos, unloading_end - unloading_start, left=unloading_start,
                        height=0.25, color='lightcoral', edgecolor='black', linewidth=0.5)
            
            y_labels.append(f"Op{op_idx} J{job_idx}")
            y_pos += 1
        
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Machine {m_idx}')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined Gantt chart saved: {save_path}")
    else:
        plt.show()

def add_job(self, pt, loading, unloading):
    self.pt_list.append(pt)
    self.loading_time_list.append(loading)
    self.unloading_time_list.append(unloading)
    self.current_pt.append(pt[self.m_idx] if len(pt) > self.m_idx else 0.0)
    self.current_loading.append(loading[self.m_idx] if len(loading) > self.m_idx else 0.0)
    self.current_unloading.append(unloading[self.m_idx] if len(unloading) > self.m_idx else 0.0)

def remove_job(self, index):
    del self.pt_list[index]
    del self.loading_time_list[index]
    del self.unloading_time_list[index]
    self.current_pt = [x[self.m_idx] for x in self.pt_list]
    self.current_loading = [x[self.m_idx] for x in self.loading_time_list]
    self.current_unloading = [x[self.m_idx] for x in self.unloading_time_list]

def before_operation(self):
    winq_vals = list(map(float, self.winq))
    self.before_op_winq_chosen = (winq_vals[self.position] if self.position < len(winq_vals) else 0.0)

def after_operation(self, timebit):
    """Handle job transfer or completion after processing on current machine"""
    
    # ========== Record operation completion info ==========
    if self.position < len(self.current_pt) and self.position < len(self.current_loading) and self.position < len(self.current_unloading):
        op_total_time = self.current_pt[self.position]
        op_total_time += self.current_loading[self.position]
        op_total_time += self.current_unloading[self.position]
    
    wait_time = self.env.now - self.before_op_time - op_total_time
    self.job_creator.record_op_start(self.job_idx, self.cur_ops - 1, self.before_op_time)
    self.job_creator.record_op_completion(self.job_idx, self.cur_ops - 1, self.env.now, wait_time)
    self.record_machine_load()
    if len(self.sequence_list[self.position]):        
        remaining_ptl = self.remaining_pt_list.pop(self.position)
        remaining_ptl.pop(0)  # Remove current operation's processing time
        remaining_loading = self.remaining_loading_times.pop(self.position)
        remaining_loading.pop(0)  # Remove current operation's loading time
        
        remaining_unloading = self.remaining_unloading_times.pop(self.position)
        remaining_unloading.pop(0)  # Remove current operation's unloading time
        
        # Get next machine index
        next_ma = self.sequence_list[self.position][0]
        
        # Cache data to transfer (complete list for add_job)
        transferred_pt = self.pt_list[self.position]
        transferred_loading = self.loading_time_list[self.position]
        transferred_unloading = self.unloading_time_list[self.position]
        
        # Remove job info from current machine
        remove_job(self, self.position)
        
        # Add job to next machine's queue
        self.m_list[next_ma].queue.append(self.queue.pop(self.position))
        
        # Transfer operation sequence (remove current operation)
        popped_sequence = self.sequence_list.pop(self.position)
        popped_sequence.pop(0)
        self.m_list[next_ma].sequence_list.append(popped_sequence)
        
        # Transfer remaining processing time
        self.m_list[next_ma].remaining_pt_list.append(remaining_ptl)
                
        self.m_list[next_ma].remaining_loading_times.append(remaining_loading)
        self.m_list[next_ma].remaining_unloading_times.append(remaining_unloading)        
      
        # Add job info to target machine
        add_job(self.m_list[next_ma], pt=transferred_pt, 
               loading=transferred_loading, 
               unloading=transferred_unloading)
        # Update estimates and fatigue
        self.job_creator.update_job_estimates(self.job_idx, self.env.now, self.worker_manager, bit=0)

        # Trigger target machine's sufficient_stock event
        try:
            self.m_list[next_ma].sufficient_stock.succeed()
        except:
            pass
    else:
        # ========== Job completion processing ==========
        
        # Remove job info from current machine
        del self.queue[self.position]
        del self.sequence_list[self.position]
        del self.remaining_pt_list[self.position]
        
        del self.remaining_loading_times[self.position]
        del self.remaining_unloading_times[self.position]
        
        remove_job(self, self.position)
        # Update estimates and fatigue
        self.job_creator.update_job_estimates(self.job_idx, self.env.now, self.worker_manager, bit=1)
        # Update system statistics
        self.job_creator.in_system_job_no -= 1
        
        if hasattr(self,'sqc_brain') and self.sqc_brain.train == True:
            # Update external archive
            estimated_solution = self.job_creator.get_estimated_solution(self.job_idx)
            archive_updated = self.sqc_brain.multi_obj_manager.update_archive(estimated_solution)
            if archive_updated:
                self.sqc_brain.multi_obj_manager.generate_preference_samples()
    
    # ========== Update system state ==========
    state_update_all(self)
    
    # ========== Complete job decision experience storage ==========
    # Job decision samples are completed after job processing finishes
    complete_experience(self, timebit)

def get_machine_load_imbalance(machine_list):
    """
    Get inter-machine load coefficient of variation (Objective 2)
    Formula: CV = std(avg load per machine) / mean(avg load per machine)
    """
    avg_loads = []
    for m in machine_list:
        avg_load = m.get_average_load()
        avg_loads.append(avg_load)
    
    mean_load = np.mean(avg_loads)
    if mean_load < 1e-6:
        return 0.0
    std_load = np.std(avg_loads)
    return float(std_load / mean_load)

def state_update_all(self):
    self.current_pt = [x[self.m_idx] for x in self.pt_list]
    self.current_loading = [x[self.m_idx] if len(x) > self.m_idx else 0.0 for x in self.loading_time_list]
    self.current_unloading = [x[self.m_idx] if len(x) > self.m_idx else 0.0 for x in self.unloading_time_list]
    env_now = float(self.env.now)
    
    self.cumulative_pt = sum(self.current_pt)
    self.cumulative_pt += sum(self.current_loading)
    self.cumulative_pt += sum(self.current_unloading)
    
    self.available_time = env_now + self.cumulative_pt
    
    self.remaining_job_pt = [sum(x) for x in self.remaining_pt_list]
    if len(self.remaining_job_pt) != len(self.current_pt):
        print("ad")
    self.remaining_loading = [sum(x[1:]) if len(x) > 1 else 0.0 for x in self.loading_time_list]
    self.remaining_unloading = [sum(x[1:]) if len(x) > 1 else 0.0 for x in self.unloading_time_list]
    
    self.next_pt = []
    for i in range(len(self.remaining_pt_list)):
        next_pt_val = float(self.remaining_pt_list[i][0]) if self.remaining_pt_list[i] else 0.0
        if len(self.loading_time_list[i]) > 1:
            next_pt_val += float(self.loading_time_list[i][1])
        if len(self.unloading_time_list[i]) > 1:
            next_pt_val += float(self.unloading_time_list[i][1])
        self.next_pt.append(next_pt_val)
    
    self.completion_rate = [max(0.0, (self.no_ops - len(x) - 1) / self.no_ops) for x in self.remaining_pt_list]
    
    self.winq = []
    for seq in self.sequence_list:
        if seq:
            next_machine = self.m_list[seq[0]]
            winq_val = float(next_machine.cumulative_pt) if hasattr(next_machine, 'cumulative_pt') else 0.0
            self.winq.append(winq_val)
        else:
            self.winq.append(0.0)
    
    if len(self.winq) != len(self.queue):
        if len(self.winq) < len(self.queue):
            self.winq.extend([0.0] * (len(self.queue) - len(self.winq)))
        else:
            self.winq = self.winq[:len(self.queue)]

def sequencing_data_generation(self):
    """Generate core data needed for scheduling decisions (simplified version)"""
    self.sequencing_data = {
        'current_pt': np.array(self.current_pt, dtype=np.float32),
        'remaining_no_op': np.array([len(x) for x in self.remaining_pt_list], dtype=np.int32),
        'completion_rate': np.array(self.completion_rate, dtype=np.float32),
        'queue': np.array(self.queue, dtype=np.int32),
        'queue_size': len(self.queue),
        'machine_idx': self.m_idx,
        'in_system': self.job_creator.in_system_job_no,
    }
    return self.sequencing_data

def state_multi_channel(self, sqc_data):
    """
    Job scheduling state features - 15 dimensions
    Divided into 4 groups by value range: ratio features, count features, time features, difference features
    """
    
    # ========== Extract data ==========
    current_pt = sqc_data.get('current_pt', np.array([]))
    remaining_no_op = sqc_data.get('remaining_no_op', np.array([]))
    completion_rate = sqc_data.get('completion_rate', np.array([]))
    queue = sqc_data.get('queue', np.array([]))
    queue_size = sqc_data.get('queue_size', 0)
    m_idx = sqc_data.get('machine_idx', 0)
    in_system = sqc_data.get('in_system', 0)
    
    # Get system machine data (via self.m_list)
    machine_loads = [len(m.queue) for m in self.m_list] if hasattr(self, 'm_list') else [0]
    machine_workloads = [m.cumulative_run_time for m in self.m_list] if hasattr(self, 'm_list') else [0]
    current_workload = self.cumulative_run_time if hasattr(self, 'cumulative_run_time') else 0
    
    # ========== Group 1: Ratio features (5 dims, range 0-1) ==========
    
    # 1. Current machine load ratio
    total_workload = sum(machine_workloads) if machine_workloads else 1
    load_ratio = current_workload / (total_workload + 1e-6)
    
    # 2. Current machine queue ratio
    total_queue = sum(machine_loads) if machine_loads else 1
    queue_ratio = queue_size / (total_queue + 1e-6)
    
    # 3. Average completion rate
    avg_completion = np.mean(completion_rate) if len(completion_rate) > 0 else 0
    
    # 4. Machine load balance (coefficient of variation)
    if len(machine_workloads) > 1:
        workload_std = np.std(machine_workloads)
        workload_mean = np.mean(machine_workloads)
        load_balance = workload_std / (workload_mean + 1e-6)
    else:
        load_balance = 0
    
    # 5. Processing time coefficient of variation
    if len(current_pt) > 1:
        pt_cv = np.std(current_pt) / (np.mean(current_pt) + 1e-6)
    else:
        pt_cv = 0
    
    ratio_features = [
        float(load_ratio),      # 1. Current machine load ratio
        float(queue_ratio),     # 2. Current machine queue ratio
        float(avg_completion),  # 3. Average completion rate
        float(load_balance),    # 4. Machine load balance
        float(pt_cv),           # 5. Processing time coefficient of variation
    ]
    
    # ========== Group 2: Count features (4 dims, range 0~large integer) ==========
    
    avg_remaining_ops = np.mean(remaining_no_op) if len(remaining_no_op) > 0 else 0
    max_remaining_ops = np.max(remaining_no_op) if len(remaining_no_op) > 0 else 0
    
    count_features = [
        float(queue_size),          # 6. Current queue length
        float(in_system),           # 7. Total jobs in system
        float(avg_remaining_ops),   # 8. Average remaining operations
        float(max_remaining_ops),   # 9. Max remaining operations
    ]
    
    # ========== Group 3: Time features (4 dims, range 0~large real) ==========
    
    avg_current_pt = np.mean(current_pt) if len(current_pt) > 0 else 0
    max_current_pt = np.max(current_pt) if len(current_pt) > 0 else 0
    
    time_features = [
        float(avg_current_pt),      # 10. Average current processing time
        float(max_current_pt),      # 11. Max current processing time
        float(current_workload),    # 12. Current machine cumulative workload
        float(np.mean(machine_workloads)) if machine_workloads else 0,  # 13. System average workload
    ]
    
    # ========== Group 4: Difference features (2 dims, range -∞~+∞) ==========
    
    # 14. Current machine load deviation from system average
    avg_workload = np.mean(machine_workloads) if machine_workloads else 0
    load_deviation = current_workload - avg_workload
    
    # 15. Current queue deviation from system average
    avg_queue = np.mean(machine_loads) if machine_loads else 0
    queue_deviation = queue_size - avg_queue
    
    diff_features = [
        float(load_deviation),      # 14. Current machine load deviation
        float(queue_deviation),     # 15. Current queue deviation
    ]
    
    # ========== Merge all features ==========
    all_features = ratio_features + count_features + time_features + diff_features
    
    # Ensure correct dimensions (5+4+4+2=15 dims)
    assert len(all_features) == 15, f"Feature dimension error: {len(all_features)}, expected 15 dims"
    
    # Convert to tensor
    s_t = torch.tensor(all_features, dtype=torch.float32)
    s_t = torch.nan_to_num(s_t, nan=0.0, posinf=1.0, neginf=-1.0)
    s_t = s_t.requires_grad_(True)
    
    return s_t

def complete_experience(self, timebit):
    """
    Complete experience storage, maintaining balance between two sample types
    """
    try:
        incomplete_exp = self.job_creator.incomplete_rep_memo[self.m_idx].pop(timebit)
        s_t, a_rule, reward, preference, job_weights = incomplete_exp
        
        local_data = sequencing_data_generation(self)
        s_next_t = state_multi_channel(self, local_data)
        weights_next = self.sqc_brain.multi_obj_manager.calculate_job_importance(
                self.job_creator, self.queue)
        complete_exp = [s_t, a_rule, reward, s_next_t, preference, job_weights, weights_next]
        self.sqc_brain.trajectory_buffer.append(complete_exp) 
        if len(self.sqc_brain.trajectory_buffer) > self.sqc_brain.trajectory_buffer_size:
            self.sqc_brain.trajectory_buffer = self.sqc_brain.trajectory_buffer[-self.sqc_brain.trajectory_buffer_size:]
        
    except (KeyError, Exception):
        pass

def build_experience(self, timebit, m_idx, s_t, a_rule, reward, preference, job_weights, bit='job'):
   
    
    self.job_creator.incomplete_rep_memo[m_idx][timebit] = [s_t, a_rule, reward, preference, job_weights]

def adapt_for_job_rule(self):
    current_pt = np.array(self.current_pt, dtype=np.float32) if hasattr(self, 'current_pt') else np.array([])
    remaining_work = np.array(self.remaining_job_pt, dtype=np.float32) if hasattr(self, 'remaining_job_pt') else np.array([])
    remaining_ops = np.array([len(x) for x in self.remaining_pt_list], dtype=np.int32) if hasattr(self, 'remaining_pt_list') else np.array([])
    winq = np.array(self.winq, dtype=np.float32) if hasattr(self, 'winq') else np.array([])
    next_pt = np.array(self.next_pt, dtype=np.float32) if hasattr(self, 'next_pt') else np.array([])
    
    if hasattr(self, 'remaining_loading') and len(self.remaining_loading) == len(remaining_work):
        remaining_work += np.array(self.remaining_loading, dtype=np.float32)
    if hasattr(self, 'remaining_unloading') and len(self.remaining_unloading) == len(remaining_work):
        remaining_work += np.array(self.remaining_unloading, dtype=np.float32)
    
    n = len(self.queue)
    if len(current_pt) != n:
        current_pt = current_pt[:n] if len(current_pt) >= n else np.pad(current_pt, (0, n - len(current_pt)), 'constant')
    if len(remaining_work) != n:
        remaining_work = remaining_work[:n] if len(remaining_work) >= n else np.pad(remaining_work, (0, n - len(remaining_work)), 'constant')
    if len(remaining_ops) != n:
        remaining_ops = remaining_ops[:n] if len(remaining_ops) >= n else np.pad(remaining_ops, (0, n - len(remaining_ops)), 'constant')
    if len(winq) != n:
        winq = winq[:n] if len(winq) >= n else np.pad(winq, (0, n - len(winq)), 'constant')
    if len(next_pt) != n:
        next_pt = next_pt[:n] if len(next_pt) >= n else np.pad(next_pt, (0, n - len(next_pt)), 'constant')
    
    return {
        'current_pt': current_pt,
        'remaining_work': remaining_work,
        'remaining_ops': remaining_ops,
        'winq': winq,
        'next_pt': next_pt,
    }

 # cfunctions.py - Add the following functions (append at end of file)
