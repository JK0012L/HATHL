# SMS_EMOA.py - S-metric Selection based Evolutionary Multi-Objective Optimization Algorithm

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time
from dataclasses import dataclass

# Import static scheduling functions (consistent with other algorithms)
from common.static_scheduling import decode_schedule, evaluate_objectives


@dataclass
class Solution:
    """Solution data structure"""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Optional[Dict] = None
    hv_contribution: float = 0.0
    rank: int = 0
    dom_count: int = 0


class JobShopProblem:
    """Job shop scheduling problem definition - receives deterministic parameters from frozen scenario"""
    
    def __init__(self, static_instance: Dict = None, **kwargs):
        """
        Initialize problem instance
        
        Args:
            static_instance: Frozen scenario instance from dynamic simulation
            **kwargs: Fallback random generation parameters
        """
        if static_instance is not None:
            self._load_from_static_instance(static_instance)
        else:
            self._generate_random_instance(kwargs)
    
    def _load_from_static_instance(self, static_instance: Dict):
        """Load deterministic instance from frozen scenario"""
        self.num_jobs = static_instance['num_jobs']
        self.num_machines = static_instance['num_machines']
        self.num_workers = static_instance['num_workers']
        self.walking_speed = static_instance.get('walking_speed', 1.2)
        self.physical_recovery_rate = static_instance.get('physical_recovery_rate', 0.03)
        self.mental_recovery_rate = static_instance.get('mental_recovery_rate', 0.02)
        self.machine_positions = static_instance.get('machine_positions', {})
        
        # Load job data
        self.jobs_data = []
        for job in static_instance['jobs']:
            self.jobs_data.append({
                'job_id': job['job_idx'],
                'arrival_time': job['arrival_time'],
                'machine_sequence': job['route'],
                'processing_times': job['processing_times'],
                'loading_times': job['loading_times'],
                'unloading_times': job['unloading_times'],
                'total_processing_time': sum(job['processing_times']),
                'total_loading_time': sum(job['loading_times']),
                'total_unloading_time': sum(job['unloading_times'])
            })
        
        # Sort by job_id
        self.jobs_data.sort(key=lambda x: x['job_id'])
        
        # Load machine breakdown data
        self.machine_breakdowns = {}
        for machine in static_instance['machines']:
            self.machine_breakdowns[machine['machine_idx']] = machine['breakdowns']
        
        # Load worker initial state
        self.workers_data = []
        for worker in static_instance['workers']:
            self.workers_data.append({
                'worker_idx': worker['worker_idx'],
                'initial_position': worker['initial_position'],
                'efficiency_matrix': worker['efficiency_matrix'],
                'initial_physical_fatigue': worker['initial_physical_fatigue'],
                'initial_mental_fatigue': worker['initial_mental_fatigue'],
                'physical_capacity': worker.get('physical_capacity', 0.5),
                'environmental_stress': worker.get('environmental_stress', 0.3)
            })
        
        # Precompute basic job data
        self.job_arrival_times = np.array([job['arrival_time'] for job in self.jobs_data])
        self.job_total_processing = np.array([job['total_processing_time'] for job in self.jobs_data])
    
    def _generate_random_instance(self, kwargs):
        """Fallback random instance generation (when no frozen scenario is available)"""
        self.num_jobs = kwargs.get('num_jobs', 50)
        self.num_machines = kwargs.get('num_machines', 5)
        self.num_workers = kwargs.get('num_workers', 3)
        self.walking_speed = 1.2
        self.physical_recovery_rate = 0.03
        self.mental_recovery_rate = 0.02
        
        # Generate default machine positions
        self.machine_positions = {i: (i * 5.0, 0.0) for i in range(self.num_machines)}
        
        self.jobs_data = []
        for job_id in range(self.num_jobs):
            arrival_time = 0  # All jobs arrive at time 0
            machine_sequence = random.sample(range(self.num_machines), self.num_machines)
            processing_times = [random.randint(5, 15) for _ in range(self.num_machines)]
            loading_times = [round(random.uniform(0.1, 0.2), 2) for _ in range(self.num_machines)]
            unloading_times = [round(random.uniform(0.05, 0.12), 2) for _ in range(self.num_machines)]
            
            self.jobs_data.append({
                'job_id': job_id,
                'arrival_time': arrival_time,
                'machine_sequence': machine_sequence,
                'processing_times': processing_times,
                'loading_times': loading_times,
                'unloading_times': unloading_times,
                'total_processing_time': sum(processing_times),
                'total_loading_time': sum(loading_times),
                'total_unloading_time': sum(unloading_times)
            })
        
        self.machine_breakdowns = {m: [] for m in range(self.num_machines)}
        self.workers_data = []
        for w_idx in range(self.num_workers):
            efficiency_matrix = [round(random.uniform(0.8, 1.2), 2) for _ in range(self.num_machines)]
            self.workers_data.append({
                'worker_idx': w_idx,
                'initial_position': -1,
                'efficiency_matrix': efficiency_matrix,
                'initial_physical_fatigue': 0.0,
                'initial_mental_fatigue': 0.0,
                'physical_capacity': 0.5,
                'environmental_stress': 0.3
            })
        
        self.job_arrival_times = np.array([job['arrival_time'] for job in self.jobs_data])
        self.job_total_processing = np.array([job['total_processing_time'] for job in self.jobs_data])


class SMS_EMOAScheduler:
    """SMS-EMOA (S-metric Selection Evolutionary Multi-Objective Optimization Algorithm) based job shop scheduling optimizer"""
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=100,
                 crossover_prob=0.8, mutation_prob=0.15,
                 use_fast_hv=True, ref_point=None):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # SMS-EMOA specific parameters
        self.use_fast_hv = use_fast_hv
        
        # Reference point settings
        if ref_point is None:
            self.ref_point = None
            self.adaptive_ref_point = True
        else:
            self.ref_point = ref_point
            self.adaptive_ref_point = False
        
        # Cache
        self._hv_cache = {}
        self._hv_contrib_cache = {}
        
        # Algorithm state
        self.population = []
        self.archive = []
        self.convergence_data = []
        self.hv_history = []
        
        # Precomputed constants
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.num_machines
        self.individual_length = self.num_jobs * self.num_machines
    
    def create_individual(self) -> List[int]:
        """Create individual - operation-based encoding"""
        individual = []
        for job_id in range(self.num_jobs):
            individual.extend([job_id] * self.num_machines)
        random.shuffle(individual)
        return individual
    
    def repair_individual(self, individual: List[int]) -> List[int]:
        """Repair individual to ensure correct appearance count for each job"""
        job_counts = defaultdict(int)
        for gene in individual:
            job_counts[gene] += 1
        
        repaired = individual[:]
        
        for job_id in range(self.num_jobs):
            expected = self.num_machines
            count = job_counts.get(job_id, 0)
            
            if count < expected:
                for _ in range(expected - count):
                    repaired.append(job_id)
            elif count > expected:
                indices_to_remove = [i for i, g in enumerate(repaired) if g == job_id]
                for idx in sorted(indices_to_remove[expected:], reverse=True):
                    repaired.pop(idx)
        
        return repaired
    
    def evaluate_individual(self, individual: List[int]) -> Tuple[Optional[Solution], Optional[Dict]]:
        """Evaluate individual"""
        # Repair individual
        individual = self.repair_individual(individual)
        
        # Use unified decoding function
        schedule = decode_schedule(individual, self.problem)
        
        if not schedule.get('valid', False):
            return None, None
        
        # Use unified objective value computation function
        objectives = evaluate_objectives(schedule)
        
        solution = Solution(
            individual=individual,
            objectives=objectives,
            schedule=schedule
        )
        
        return solution, schedule
    
    def dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if obj1 dominates obj2"""
        # Handle infinity
        if any(np.isinf(o) for o in obj1) or any(np.isinf(o) for o in obj2):
            return False
        
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better
    
    def fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sorting"""
        if not solutions:
            return []
        
        n = len(solutions)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                if self.dominates(solutions[i].objectives, solutions[j].objectives):
                    dominated_solutions[i].append(j)
                elif self.dominates(solutions[j].objectives, solutions[i].objectives):
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(solutions[i])
                solutions[i].rank = 0
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for solution in fronts[current_front]:
                idx = solutions.index(solution)
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        solutions[dominated_idx].rank = current_front + 1
                        next_front.append(solutions[dominated_idx])
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def _update_reference_point(self, population: List[Solution]) -> List[float]:
        """Dynamically update reference point"""
        if not population:
            return [1000.0, 1.0, 1.0]
        
        objectives = np.array([sol.objectives for sol in population])
        max_vals = np.max(objectives, axis=0)
        # Set reference point to 1.1x of max values to ensure margin
        ref_point = max_vals * 1.1
        return ref_point.tolist()
    
    def _is_boundary_solution(self, sol: Solution, front: List[Solution]) -> bool:
        """Check if it is a boundary solution"""
        objectives = np.array([s.objectives for s in front])
        for i in range(3):
            if sol.objectives[i] == np.min(objectives[:, i]) or \
               sol.objectives[i] == np.max(objectives[:, i]):
                return True
        return False
    
    def _calculate_hv_contribution(self, front: List[Solution], ref_point: List[float]) -> List[float]:
        """Calculate HV contribution of each solution in the front"""
        if len(front) == 0:
            return []
        
        n = len(front)
        if n == 1:
            return [float('inf')]
        
        # Compute total HV
        total_hv = self._calculate_hypervolume(front, ref_point)
        
        # Compute contribution of each solution (deletion method)
        contributions = []
        for i in range(n):
            front_without = front[:i] + front[i+1:]
            hv_without = self._calculate_hypervolume(front_without, ref_point)
            contributions.append(total_hv - hv_without)
        
        return contributions
    
    def _calculate_hypervolume(self, solutions: List[Solution], ref_point=None) -> float:
        """Compute hypervolume"""
        if not solutions:
            return 0.0
        
        if ref_point is None:
            if self.adaptive_ref_point:
                ref_point = self._update_reference_point(solutions)
            else:
                ref_point = self.ref_point
        
        # Get non-dominated solutions
        fronts = self.fast_non_dominated_sort(solutions)
        non_dominated = fronts[0] if fronts else []
        
        if not non_dominated:
            return 0.0
        
        obj_array = np.array([sol.objectives for sol in non_dominated])
        ref_array = np.array(ref_point)
        
        # Sort by first objective
        sorted_indices = np.argsort(obj_array[:, 0])
        sorted_objs = obj_array[sorted_indices]
        
        hv = 0.0
        for i in range(len(sorted_objs)):
            if i == 0:
                volume = np.prod(ref_array - sorted_objs[i])
            else:
                delta_f1 = sorted_objs[i-1, 0] - sorted_objs[i, 0]
                if delta_f1 <= 0:
                    continue
                proj_area = (ref_array[1] - sorted_objs[i, 1]) * (ref_array[2] - sorted_objs[i, 2])
                volume = delta_f1 * proj_area
            hv += max(0, volume)
        
        return hv
    
    def _sms_emoa_selection(self, population: List[Solution], offspring: Solution) -> List[Solution]:
        """SMS-EMOA steady-state selection"""
        # Merge population
        combined = population + [offspring]
        
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined)
        
        # Find worst front
        worst_front = fronts[-1]
        
        # Compute reference point
        ref_point = self._update_reference_point(combined)
        
        # Select individual to remove
        if len(worst_front) == 1:
            to_remove = worst_front[0]
        else:
            # Compute HV contributions
            hv_contributions = self._calculate_hv_contribution(worst_front, ref_point)
            
            # Find solution with minimum contribution (skip boundary solutions)
            non_boundary_indices = [i for i, sol in enumerate(worst_front)
                                   if not self._is_boundary_solution(sol, worst_front)]
            
            if non_boundary_indices:
                non_boundary_contribs = [hv_contributions[i] for i in non_boundary_indices]
                min_idx = non_boundary_indices[np.argmin(non_boundary_contribs)]
            else:
                min_idx = np.argmin(hv_contributions)
            
            to_remove = worst_front[min_idx]
        
        # Remove selected individual
        new_population = [sol for sol in combined if sol is not to_remove]
        
        return new_population
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover"""
        if random.random() > self.crossover_prob:
            return parent1[:]
        
        size = len(parent1)
        if size <= 1:
            return parent1[:]
        
        cx1, cx2 = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[cx1:cx2+1] = parent1[cx1:cx2+1]
        
        remaining_genes = []
        for g in parent2:
            if g not in child[cx1:cx2+1]:
                remaining_genes.append(g)
            elif parent1[cx1:cx2+1].count(g) < parent2.count(g):
                remaining_genes.append(g)
        
        if len(remaining_genes) != (size - (cx2 - cx1 + 1)):
            return self.repair_individual(parent1[:])
        
        fill_index = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining_genes[fill_index]
                fill_index += 1
        
        return self.repair_individual(child)
    
    def mutation(self, individual: List[int]) -> List[int]:
        """Swap mutation"""
        if random.random() > self.mutation_prob:
            return individual[:]
        
        mutated = individual[:]
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def tournament_selection(self, population: List[Solution], tournament_size=2) -> Solution:
        """Tournament selection"""
        participants = random.sample(population, min(tournament_size, len(population)))
        
        # Select by rank and HV contribution
        best = participants[0]
        for sol in participants[1:]:
            if sol.rank < best.rank:
                best = sol
            elif sol.rank == best.rank and sol.hv_contribution > best.hv_contribution:
                best = sol
        
        return best
    
    def update_archive(self, archive: List[Solution], new_solutions: List[Solution], max_size: int) -> List[Solution]:
        """Update archive"""
        all_solutions = archive + new_solutions
        
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(all_solutions)
        
        new_archive = []
        remaining = max_size
        
        for front in fronts:
            if len(front) <= remaining:
                new_archive.extend(front)
                remaining -= len(front)
            else:
                # Select using HV contribution
                ref_point = self._update_reference_point(front)
                hv_contributions = self._calculate_hv_contribution(front, ref_point)
                sorted_indices = np.argsort(hv_contributions)[::-1]
                for i in range(remaining):
                    new_archive.append(front[sorted_indices[i]])
                break
        
        return new_archive
    
    def run(self) -> Dict:
        """Run SMS-EMOA algorithm"""
        start_time = time.time()
        print(f"SMS-EMOA Initialization: Population size={self.pop_size}, Max generations={self.max_gen}")
        
        # Initialize population
        print("Initializing population...")
        self.population = []
        for _ in range(self.pop_size):
            individual = self.create_individual()
            solution, _ = self.evaluate_individual(individual)
            if solution is not None:
                self.population.append(solution)
        
        # Supplement insufficient individuals
        while len(self.population) < self.pop_size:
            individual = self.create_individual()
            solution, _ = self.evaluate_individual(individual)
            if solution is not None:
                self.population.append(solution)
        
        # Initialize non-dominated sorting
        self.fast_non_dominated_sort(self.population)
        
        # Initialize archive
        self.archive = self.population.copy()
        
        # Compute initial HV
        initial_hv = self._calculate_hypervolume(self.archive)
        self.hv_history.append(initial_hv)
        print(f"Initial hypervolume: {initial_hv:.4f}")
        
        # Record convergence data
        self.convergence_data = []
        best_solutions = []
        
        # Main evolution loop
        print("\nStarting evolution (steady-state selection)...")
        for generation in range(self.max_gen):
            # Select parent
            parent = self.tournament_selection(self.population)
            
            # Crossover and mutation
            if random.random() < self.crossover_prob:
                # Select second parent
                parent2 = self.tournament_selection(self.population)
                child_ind = self.crossover(parent.individual, parent2.individual)
            else:
                child_ind = parent.individual[:]
            
            child_ind = self.mutation(child_ind)
            
            # Evaluate offspring
            child_sol, _ = self.evaluate_individual(child_ind)
            
            if child_sol is not None:
                # SMS-EMOA steady-state selection
                self.population = self._sms_emoa_selection(self.population, child_sol)
                
                # Update archive
                self.archive = self.update_archive(self.archive, [child_sol], self.pop_size)
            
            # Compute current HV
            current_hv = self._calculate_hypervolume(self.archive)
            self.hv_history.append(current_hv)
            
            # Record convergence data
            avg_obj1 = np.mean([sol.objectives[0] for sol in self.population])
            self.convergence_data.append(avg_obj1)
            
            # Record best front
            fronts = self.fast_non_dominated_sort(self.population)
            if fronts:
                best_solutions.append([sol.objectives for sol in fronts[0]])
            
            # Output progress
            if generation % 20 == 0 or generation == self.max_gen - 1:
                archive_size = len(self.archive)
                pareto_size = len(fronts[0]) if fronts else 0
                elapsed = time.time() - start_time
                
                print(f"  Generation {generation:3d}: "
                      f"HV={current_hv:.4f}, "
                      f"Archive={archive_size:3d}, "
                      f"Front={pareto_size:3d}, "
                      f"Avg Obj1={avg_obj1:8.2f}, "
                      f"Elapsed={elapsed:6.2f}s")
        
        # Return results
        pareto_front = [sol.objectives for sol in self.archive]
        
        elapsed_time = time.time() - start_time
        
        print(f"SMS-EMOA Complete: Final front solution count={len(pareto_front)}, Final HV={self.hv_history[-1]:.4f}")
        
        return {
            'pareto_front': pareto_front,
            'convergence_data': self.convergence_data,
            'hv_history': self.hv_history,
            'best_solutions': best_solutions,
            'archive': [(sol.individual, sol.objectives) for sol in self.archive],
            'elapsed_time': elapsed_time,
            'final_population': self.population,
            'final_hv': self.hv_history[-1]
        }