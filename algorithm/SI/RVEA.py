# RVEA.py - Reference Vector Guided Evolutionary Algorithm (RVEA) for Multi-Objective Optimization

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time
from dataclasses import dataclass
from math import acos, pi
from common.static_scheduling import decode_schedule, evaluate_objectives


@dataclass
class Solution:
    """Solution data structure"""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Optional[Dict] = None
    scalar_value: float = float('inf')
    subregion_id: int = -1


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


class RVEAScheduler:
    """Reference Vector Guided Evolutionary Algorithm (RVEA) based job shop scheduling optimizer"""
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=100,
                 crossover_prob=0.8, mutation_prob=0.1,
                 alpha=2.0, fr=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # RVEA specific parameters
        self.alpha = alpha
        self.fr = fr
        self.num_objectives = 3
        
        # Reference vectors
        self.reference_vectors = None
        self.original_reference_vectors = None
        
        # Ideal point and nadir point
        self.ideal_point = None
        self.nadir_point = None
        
        # Population
        self.population = []
        self.archive = []
        
        # Convergence tracking
        self.convergence_data = []
        
        # Precomputed constants
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.num_machines
        self.individual_length = self.num_jobs * self.num_machines
    
    def generate_reference_vectors(self) -> np.ndarray:
        """Generate uniformly distributed reference vectors"""
        M = self.num_objectives
        
        # Use Das-Dennis method to generate uniformly distributed weight vectors
        # For 3 objectives, H1 and H2 parameters
        H1 = 13  # Boundary layer
        H2 = 2   # Inner layer
        
        vectors = []
        
        # Boundary vectors
        for i in range(H1 + 1):
            for j in range(H1 + 1 - i):
                k = H1 - i - j
                v = np.array([i, j, k], dtype=np.float64) / H1
                v += 1e-6
                v = v / np.linalg.norm(v)
                vectors.append(v)
        
        # If insufficient quantity, supplement with random vectors
        if len(vectors) < self.pop_size:
            num_needed = self.pop_size - len(vectors)
            for _ in range(num_needed):
                v = np.random.dirichlet([1, 1, 1])
                v = v / np.linalg.norm(v)
                vectors.append(v)
        
        vectors = np.array(vectors[:self.pop_size], dtype=np.float64)
        
        return vectors
    
    def update_reference_vectors(self, generation: int, objectives: List[Tuple[float, float, float]]):
        """Adjust reference vectors based on objective value ranges"""
        if generation % max(1, int(self.fr * self.max_gen)) != 0:
            return
        
        if len(objectives) == 0:
            return
        
        obj_array = np.array(objectives, dtype=np.float64)
        
        # Compute objective value ranges
        min_obj = np.min(obj_array, axis=0)
        max_obj = np.max(obj_array, axis=0)
        ranges = max_obj - min_obj
        ranges = np.maximum(ranges, 1.0)
        
        # Adjust reference vectors
        for i in range(len(self.reference_vectors)):
            scaled_vector = self.original_reference_vectors[i] * ranges
            norm = np.linalg.norm(scaled_vector)
            if norm > 1e-10:
                self.reference_vectors[i] = scaled_vector / norm
            else:
                self.reference_vectors[i] = self.original_reference_vectors[i].copy()
    
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
        dominated_count = [0] * n
        dominates_list = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(solutions[i].objectives, solutions[j].objectives):
                    dominates_list[i].append(j)
                elif self.dominates(solutions[j].objectives, solutions[i].objectives):
                    dominated_count[i] += 1
        
        fronts = [[]]
        for i in range(n):
            if dominated_count[i] == 0:
                fronts[0].append(solutions[i])
        
        current_level = 0
        while fronts[current_level]:
            next_front = []
            for sol in fronts[current_level]:
                idx = solutions.index(sol)
                for j in dominates_list[idx]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(solutions[j])
            current_level += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def partition_population(self, objective_vectors: np.ndarray, reference_vectors: np.ndarray) -> List[List[int]]:
        """Partition population to each reference vector"""
        num_ref_vectors = len(reference_vectors)
        num_solutions = len(objective_vectors)
        
        subpopulations = [[] for _ in range(num_ref_vectors)]
        
        for sol_idx in range(num_solutions):
            obj_vec = objective_vectors[sol_idx]
            max_cos = -float('inf')
            best_ref_idx = 0
            
            for ref_idx, ref_vec in enumerate(reference_vectors):
                norm_obj = np.linalg.norm(obj_vec)
                norm_ref = np.linalg.norm(ref_vec)
                
                if norm_obj < 1e-10 or norm_ref < 1e-10:
                    cos_theta = 0.0
                else:
                    cos_theta = np.dot(obj_vec, ref_vec) / (norm_obj * norm_ref)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                
                if cos_theta > max_cos:
                    max_cos = cos_theta
                    best_ref_idx = ref_idx
            
            subpopulations[best_ref_idx].append(sol_idx)
        
        return subpopulations
    
    def calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        return acos(cos_theta)
    
    def calculate_apd(self, objective_vector: np.ndarray, reference_vector: np.ndarray,
                      generation: int) -> float:
        """Calculate Angle Penalized Distance (APD)"""
        # Compute angle with reference vector
        angle = self.calculate_angle(objective_vector, reference_vector)
        
        # Compute distance to origin
        distance = np.linalg.norm(objective_vector)
        
        # Compute penalty term
        t = generation / self.max_gen
        P = self.num_objectives * (t ** self.alpha) * (angle / (pi / 2))
        
        # APD
        apd = (1.0 + P) * distance
        
        return apd
    
    def reference_vector_guided_selection(self, population: List[Solution],
                                           objective_values: List[Tuple[float, float, float]],
                                           generation: int) -> List[Solution]:
        """Reference vector guided selection"""
        if len(population) <= self.pop_size:
            return population[:]
        
        obj_array = np.array(objective_values, dtype=np.float64)
        
        # Update ideal point
        if self.ideal_point is None:
            self.ideal_point = np.min(obj_array, axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point, np.min(obj_array, axis=0))
        
        # Translate objective values
        translated_obj = obj_array - self.ideal_point
        translated_obj = np.maximum(translated_obj, 0.0)
        
        # Partition population
        subpopulations = self.partition_population(translated_obj, self.reference_vectors)
        
        # Select best solution from each subpopulation
        selected_indices = []
        
        for ref_idx, subpop in enumerate(subpopulations):
            if not subpop:
                continue
            
            ref_vec = self.reference_vectors[ref_idx]
            
            # Compute APD and select minimum
            best_idx = subpop[0]
            best_apd = float('inf')
            
            for sol_idx in subpop:
                apd = self.calculate_apd(translated_obj[sol_idx], ref_vec, generation)
                if apd < best_apd:
                    best_apd = apd
                    best_idx = sol_idx
            
            selected_indices.append(best_idx)
        
        # If insufficient solutions selected, supplement
        if len(selected_indices) < self.pop_size:
            all_indices = set(range(len(population)))
            remaining = list(all_indices - set(selected_indices))
            
            # Supplement with non-dominated sorting
            remaining_solutions = [population[i] for i in remaining]
            fronts = self.fast_non_dominated_sort(remaining_solutions)
            
            for front in fronts:
                if not front:
                    continue
                for sol in front:
                    idx = population.index(sol)
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        if len(selected_indices) >= self.pop_size:
                            break
                if len(selected_indices) >= self.pop_size:
                    break
        
        # Return new population
        new_population = [population[i] for i in selected_indices[:self.pop_size]]
        
        return new_population
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover"""
        if random.random() > self.crossover_prob:
            return parent1[:], parent2[:]
        
        size = len(parent1)
        cx1 = random.randint(0, size - 2)
        cx2 = random.randint(cx1 + 1, size - 1)
        
        def ox_crossover(p1, p2):
            child = [-1] * size
            child[cx1:cx2+1] = p1[cx1:cx2+1]
            
            remaining_genes = []
            for g in p2:
                if g not in child[cx1:cx2+1]:
                    remaining_genes.append(g)
                elif p1[cx1:cx2+1].count(g) < p2.count(g):
                    remaining_genes.append(g)
            
            if len(remaining_genes) != (size - (cx2 - cx1 + 1)):
                return p1[:]
            
            fill_index = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining_genes[fill_index]
                    fill_index += 1
            return child
        
        child1 = ox_crossover(parent1, parent2)
        child2 = ox_crossover(parent2, parent1)
        
        child1 = self.repair_individual(child1)
        child2 = self.repair_individual(child2)
        
        return child1, child2
    
    def mutation(self, individual: List[int]) -> List[int]:
        """Swap mutation"""
        if random.random() > self.mutation_prob:
            return individual[:]
        
        mutated = individual[:]
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def update_archive(self, solutions: List[Solution]):
        """Update archive"""
        for sol in solutions:
            # Check if dominated by solutions in archive
            is_dominated = False
            for archived in self.archive:
                if self.dominates(archived.objectives, sol.objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove solutions dominated by new solution
                self.archive = [a for a in self.archive 
                               if not self.dominates(sol.objectives, a.objectives)]
                self.archive.append(sol)
    
    def run(self) -> Dict:
        """Run RVEA algorithm"""
        start_time = time.time()
        print(f"RVEA Initialization: Population size={self.pop_size}, Max generations={self.max_gen}")
        
        # Generate reference vectors
        print("Generating reference vectors...")
        self.reference_vectors = self.generate_reference_vectors()
        self.original_reference_vectors = self.reference_vectors.copy()
        
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
        
        # Initialize archive
        self.archive = []
        self.update_archive(self.population)
        
        # Initialize ideal point
        obj_array = np.array([sol.objectives for sol in self.population], dtype=np.float64)
        self.ideal_point = np.min(obj_array, axis=0)
        
        # Record convergence data
        self.convergence_data = []
        best_solutions = []
        
        # Main evolution loop
        print("\nStarting evolution...")
        for generation in range(self.max_gen):
            # Update reference vectors
            objectives = [sol.objectives for sol in self.population]
            self.update_reference_vectors(generation, objectives)
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                parents = random.sample(self.population, min(2, len(self.population)))
                if len(parents) < 2:
                    continue
                
                parent1 = parents[0].individual
                parent2 = parents[1].individual
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                sol1, _ = self.evaluate_individual(child1)
                sol2, _ = self.evaluate_individual(child2)
                
                if sol1 is not None:
                    offspring.append(sol1)
                if sol2 is not None and len(offspring) < self.pop_size:
                    offspring.append(sol2)
            
            # Merge parent and offspring
            combined_population = self.population + offspring
            
            # Reference vector guided selection
            combined_objectives = [sol.objectives for sol in combined_population]
            self.population = self.reference_vector_guided_selection(
                combined_population, combined_objectives, generation
            )
            
            # Update archive
            self.update_archive(self.population)
            
            # Update ideal point
            obj_array = np.array([sol.objectives for sol in self.population], dtype=np.float64)
            self.ideal_point = np.minimum(self.ideal_point, np.min(obj_array, axis=0))
            
            # Record convergence data
            avg_obj1 = np.mean([sol.objectives[0] for sol in self.population])
            self.convergence_data.append(avg_obj1)
            
            # Record best front
            if self.archive:
                best_solutions.append([sol.objectives for sol in self.archive])
            
            # Output progress
            if generation % 20 == 0 or generation == self.max_gen - 1:
                archive_size = len(self.archive)
                elapsed = time.time() - start_time
                
                # Compute non-dominated front size
                fronts = self.fast_non_dominated_sort(self.population)
                pareto_size = len(fronts[0]) if fronts else 0
                
                print(f"  Generation {generation:3d}: "
                      f"Archive={archive_size:3d}, "
                      f"Front={pareto_size:3d}, "
                      f"Avg Obj1={avg_obj1:8.2f}, "
                      f"Elapsed={elapsed:6.2f}s")
        
        # Return results
        pareto_front = [sol.objectives for sol in self.archive]
        
        elapsed_time = time.time() - start_time
        
        print(f"RVEA Complete: Final front solution count={len(pareto_front)}")
        
        return {
            'pareto_front': pareto_front,
            'convergence_data': self.convergence_data,
            'best_solutions': best_solutions,
            'archive': [(sol.individual, sol.objectives) for sol in self.archive],
            'elapsed_time': elapsed_time,
            'final_population': self.population
        }