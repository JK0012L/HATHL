import numpy as np
import random
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from dataclasses import dataclass
import itertools
from scipy.spatial import KDTree
from common.static_scheduling import decode_schedule, evaluate_objectives



def is_dominated(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Check if obj1 dominates obj2"""
    # Check if completely identical (with small tolerance)
    if np.allclose(obj1, obj2, atol=0.01):
        return True
    
    # Check dominance relationship
    all_not_worse = True
    at_least_one_better = False
    
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            all_not_worse = False
            break
        if obj1[i] < obj2[i]:
            at_least_one_better = True
    
    return all_not_worse and at_least_one_better


@dataclass
class Solution:
    """Solution data structure"""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Dict = None
    weight_vector: np.ndarray = None
    scalar_value: float = float('inf')
    subregion_id: int = -1
    _hash: int = None
    
    def __post_init__(self):
        self._hash = hash(tuple(self.individual))
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, Solution):
            return False
        return tuple(self.individual) == tuple(other.individual)


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
            arrival_time = random.randint(0, 20)
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


class MOEA_DDScheduler:
    """MOEA/DD-based job shop scheduling optimizer"""
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=100,
                 crossover_prob=0.8, mutation_prob=0.1,
                 decomposition_method='tchebycheff', neighborhood_size=20,
                 delta=0.9, theta=5.0):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.decomposition_method = decomposition_method
        self.neighborhood_size = min(neighborhood_size, pop_size)
        self.delta = delta
        self.theta = theta
        
        # MOEA/DD specific variables
        self.weight_vectors = None
        self.neighbors = None
        self.ideal_point = None
        self.nadir_point = None
        self.population = []
        self.archive = set()
        self.archive_list = []
        
        # Subregion related information
        self.subregions = []
        self.nondomination_levels = []
        
        # Cache
        self._scalar_cache = {}
        
        # Convergence tracking
        self.convergence_data = []
        
        # Precomputed constants
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.num_machines
        self.individual_length = self.num_jobs * self.num_machines
    
    def generate_weight_vectors(self, N: int, m: int = 3) -> np.ndarray:
        """Generate uniformly distributed weight vectors"""
        if m == 3:
            # For 3 objectives, use Das-Dennis method
            H = int(np.ceil(N ** (1 / (m - 1)))) - 1
            H = max(H, 1)
            
            weight_vectors = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    w = np.array([i, j, k], dtype=np.float64) / H
                    w += 1e-6
                    w /= np.sum(w)
                    weight_vectors.append(w)
            
            # If insufficient quantity, randomly supplement
            if len(weight_vectors) < N:
                num_needed = N - len(weight_vectors)
                random_vectors = np.random.rand(num_needed, m)
                random_vectors = random_vectors / np.sum(random_vectors, axis=1, keepdims=True)
                weight_vectors.extend(random_vectors.tolist())
            
            return np.array(weight_vectors[:N], dtype=np.float64)
        else:
            # General case
            weight_vectors = []
            for _ in range(N):
                w = np.random.dirichlet([1] * m)
                weight_vectors.append(w)
            return np.array(weight_vectors, dtype=np.float64)
    
    def compute_neighbors(self, weight_vectors: np.ndarray, T: int) -> List[List[int]]:
        """Compute neighbors for each weight vector"""
        n = len(weight_vectors)
        if n <= T:
            return [list(range(n)) for _ in range(n)]
        
        # Use KDTree for fast nearest neighbor search
        tree = KDTree(weight_vectors)
        neighbors = []
        
        distances, indices = tree.query(weight_vectors, k=T + 1)
        
        for i in range(n):
            neighbor_indices = [idx for idx in indices[i] if idx != i][:T]
            neighbors.append(neighbor_indices)
        
        return neighbors
    
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
                # Add missing genes
                for _ in range(expected - count):
                    repaired.append(job_id)
            elif count > expected:
                # Remove excess genes
                indices_to_remove = [i for i, g in enumerate(repaired) if g == job_id]
                # Delete from back to front to avoid index shifting
                for idx in sorted(indices_to_remove[expected:], reverse=True):
                    repaired.pop(idx)
        
        return repaired
    
    def evaluate_individual(self, individual: List[int]) -> Tuple[Solution, Dict]:
        """Evaluate individual, return Solution object and scheduling result"""
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
    
    def update_reference_points(self, solutions: List[Solution]):
        """Update ideal point and nadir point"""
        if not solutions:
            return
        
        objectives = np.array([sol.objectives for sol in solutions], dtype=np.float64)
        
        if self.ideal_point is None:
            self.ideal_point = np.min(objectives, axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point, np.min(objectives, axis=0))
        
        if self.nadir_point is None:
            self.nadir_point = np.max(objectives, axis=0)
        else:
            self.nadir_point = np.maximum(self.nadir_point, np.max(objectives, axis=0))
    
    def scalarizing_function(self, objectives: np.ndarray, weight_vector: np.ndarray,
                            method: str = 'tchebycheff') -> float:
        """Scalarizing function"""
        # Cache key
        cache_key = (tuple(objectives), tuple(weight_vector), method)
        if cache_key in self._scalar_cache:
            return self._scalar_cache[cache_key]
        
        # Ensure reference points are initialized
        if self.ideal_point is None or self.nadir_point is None:
            # Use raw objective values
            if method == 'tchebycheff':
                result = float(np.max(weight_vector * np.abs(objectives)))
            elif method == 'weighted_sum':
                result = float(np.sum(weight_vector * objectives))
            elif method == 'pbi':
                weight_norm = np.linalg.norm(weight_vector)
                if weight_norm < 1e-10:
                    weight_norm = 1e-10
                d1 = np.abs(np.dot(objectives, weight_vector)) / weight_norm
                weight_norm_vec = weight_vector / weight_norm
                d2 = np.linalg.norm(objectives - d1 * weight_norm_vec)
                result = float(d1 + self.theta * d2)
            else:
                result = float(np.sum(weight_vector * objectives))
            
            self._scalar_cache[cache_key] = result
            return result
        
        # Normalize
        norm_denominator = self.nadir_point - self.ideal_point
        norm_denominator = np.maximum(norm_denominator, 1.0)
        norm_objectives = (objectives - self.ideal_point) / norm_denominator
        
        if method == 'tchebycheff':
            weighted = weight_vector * np.abs(norm_objectives)
            result = float(np.max(weighted))
        elif method == 'weighted_sum':
            result = float(np.sum(weight_vector * norm_objectives))
        elif method == 'pbi':
            weight_norm = np.linalg.norm(weight_vector)
            if weight_norm < 1e-10:
                weight_norm = 1e-10
            d1 = np.abs(np.dot(norm_objectives, weight_vector)) / weight_norm
            weight_norm_vec = weight_vector / weight_norm
            d2 = np.linalg.norm(norm_objectives - d1 * weight_norm_vec)
            result = float(d1 + self.theta * d2)
        else:
            result = float(np.sum(weight_vector * norm_objectives))
        
        self._scalar_cache[cache_key] = result
        return result
    
    def associate_solution_to_subregion(self, solution: Solution) -> int:
        """Associate solution to subregion"""
        if solution.subregion_id != -1:
            return solution.subregion_id
        
        objectives = np.array(solution.objectives, dtype=np.float64)
        
        # Normalize
        if self.ideal_point is not None and self.nadir_point is not None:
            norm_denominator = self.nadir_point - self.ideal_point
            norm_denominator = np.maximum(norm_denominator, 1.0)
            obj_norm = (objectives - self.ideal_point) / norm_denominator
        else:
            obj_norm = objectives
        
        # Compute angular distance to each weight vector
        dot_products = np.dot(self.weight_vectors, obj_norm)
        norm_obj = np.linalg.norm(obj_norm)
        norm_weights = np.linalg.norm(self.weight_vectors, axis=1)
        
        cos_thetas = dot_products / (norm_obj * norm_weights + 1e-10)
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
        angles = np.arccos(cos_thetas)
        
        best_idx = np.argmin(angles)
        solution.subregion_id = best_idx
        
        return best_idx
    
    def update_subregions_association(self):
        """Update subregion associations"""
        self.subregions = [[] for _ in range(len(self.weight_vectors))]
        
        for sol in self.population:
            subregion_id = self.associate_solution_to_subregion(sol)
            self.subregions[subregion_id].append(sol)
    
    def _is_dominated(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if obj1 dominates obj2"""
        obj1_arr = np.array(obj1)
        obj2_arr = np.array(obj2)
        return is_dominated(obj1_arr, obj2_arr)
    
    def fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sorting"""
        if not solutions:
            return []
        
        n = len(solutions)
        dominated_count = np.zeros(n, dtype=np.int32)
        dominates = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._is_dominated(solutions[i].objectives, solutions[j].objectives):
                    dominates[i].append(j)
                elif self._is_dominated(solutions[j].objectives, solutions[i].objectives):
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
                for j in dominates[idx]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(solutions[j])
            current_level += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def sbx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Simulated binary crossover"""
        if random.random() > self.crossover_prob:
            return parent1[:]
        
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        
        return self.repair_individual(child)
    
    def polynomial_mutation(self, individual: List[int]) -> List[int]:
        """Polynomial mutation"""
        if random.random() > self.mutation_prob:
            return individual[:]
        
        mutated = individual[:]
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def mating_selection(self, subproblem_idx: int) -> Tuple[List[int], List[int]]:
        """Mating selection"""
        if random.random() < self.delta:
            # Select from neighbors
            neighbor_indices = self.neighbors[subproblem_idx]
            
            candidate_indices = []
            for neighbor_idx in neighbor_indices:
                if self.subregions[neighbor_idx]:
                    for sol in self.subregions[neighbor_idx]:
                        try:
                            candidate_indices.append(self.population.index(sol))
                        except ValueError:
                            continue
            
            if len(candidate_indices) >= 2:
                parents = np.random.choice(candidate_indices, 2, replace=False)
                return self.population[parents[0]].individual, self.population[parents[1]].individual
        
        # Randomly select from entire population
        parents = random.sample(self.population, 2)
        return parents[0].individual, parents[1].individual
    
    def get_subregion_niche_count(self, subregion_id: int) -> int:
        """Get number of solutions in subregion"""
        if subregion_id < 0 or subregion_id >= len(self.subregions):
            return 0
        return len(self.subregions[subregion_id])
    
    def _locate_worst_solution(self, solutions: List[Solution], subregion_id: int = None) -> Solution:
        """Locate worst solution"""
        if not solutions:
            return None
        
        if subregion_id is not None:
            candidates = [sol for sol in solutions if sol.subregion_id == subregion_id]
            if not candidates:
                candidates = solutions
        else:
            candidates = solutions
        
        if not candidates:
            return None
        
        # Compute PBI values
        pbi_values = []
        for sol in candidates:
            pbi_value = self.scalarizing_function(
                np.array(sol.objectives),
                self.weight_vectors[sol.subregion_id],
                'pbi'
            )
            pbi_values.append(pbi_value)
        
        max_idx = np.argmax(pbi_values)
        return candidates[max_idx]
    
    def _get_most_crowded_subregion(self, solutions: List[Solution] = None) -> int:
        """Get most crowded subregion"""
        if solutions is None:
            solutions = self.population
        
        subregion_counts = defaultdict(int)
        for sol in solutions:
            subregion_counts[sol.subregion_id] += 1
        
        if not subregion_counts:
            return -1
        
        max_count = max(subregion_counts.values())
        candidate_subregions = [rid for rid, cnt in subregion_counts.items() if cnt == max_count]
        
        if len(candidate_subregions) == 1:
            return candidate_subregions[0]
        
        # Compute PBI sum
        pbi_sums = []
        for rid in candidate_subregions:
            pbi_sum = 0
            for sol in solutions:
                if sol.subregion_id == rid:
                    pbi_sum += self.scalarizing_function(
                        np.array(sol.objectives),
                        self.weight_vectors[rid],
                        'pbi'
                    )
            pbi_sums.append(pbi_sum)
        
        return candidate_subregions[np.argmax(pbi_sums)]
    
    def update_population(self, subproblem_idx: int, offspring: Solution) -> bool:
        """Update population"""
        # Associate subregion
        subregion_id = self.associate_solution_to_subregion(offspring)
        
        # Add to population
        if offspring not in self.population:
            self.population.append(offspring)
        else:
            return False
        
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(self.population)
        self.nondomination_levels = fronts
        
        num_levels = len(fronts)
        last_level = fronts[-1] if num_levels > 0 else []
        
        removed = False
        
        if num_levels == 1:
            # All solutions are in first level
            crowded_subregion = self._get_most_crowded_subregion()
            worst = self._locate_worst_solution(self.population, crowded_subregion)
            if worst and worst in self.population:
                self.population.remove(worst)
                removed = True
        else:
            # Multiple levels, select worst solution from the last level
            if len(last_level) == 1:
                xl = last_level[0]
                xl_count = self.get_subregion_niche_count(xl.subregion_id)
                
                if xl_count > 1:
                    if xl in self.population:
                        self.population.remove(xl)
                        removed = True
                else:
                    crowded_subregion = self._get_most_crowded_subregion()
                    worst = self._locate_worst_solution(self.population, crowded_subregion)
                    if worst and worst in self.population:
                        self.population.remove(worst)
                        removed = True
            else:
                crowded_subregion = self._get_most_crowded_subregion(last_level)
                if crowded_subregion != -1:
                    subregion_count = self.get_subregion_niche_count(crowded_subregion)
                    if subregion_count > 1:
                        worst = self._locate_worst_solution(last_level, crowded_subregion)
                        if worst and worst in self.population:
                            self.population.remove(worst)
                            removed = True
                    else:
                        crowded_subregion = self._get_most_crowded_subregion()
                        worst = self._locate_worst_solution(self.population, crowded_subregion)
                        if worst and worst in self.population:
                            self.population.remove(worst)
                            removed = True
        
        # Ensure population size
        while len(self.population) > self.pop_size:
            self.population.pop()
            removed = True
        
        return removed
    
    def run(self) -> Dict:
        """Run MOEA/DD algorithm"""
        start_time = time.time()
        print(f"MOEA/DD Initialization: Population size={self.pop_size}, Max generations={self.max_gen}")
        
        # Generate weight vectors and neighbors
        print("Generating weight vectors and neighbor structure...")
        self.weight_vectors = self.generate_weight_vectors(self.pop_size, m=3)
        self.neighbors = self.compute_neighbors(self.weight_vectors, self.neighborhood_size)
        
        # Initialize population
        print("Initializing population...")
        self.population = []
        self.ideal_point = None
        self.nadir_point = None
        
        for i in range(self.pop_size):
            individual = self.create_individual()
            solution, _ = self.evaluate_individual(individual)
            
            if solution is not None:
                solution.weight_vector = self.weight_vectors[i]
                solution.scalar_value = self.scalarizing_function(
                    np.array(solution.objectives),
                    solution.weight_vector,
                    self.decomposition_method
                )
                self.population.append(solution)
        
        # Supplement insufficient individuals
        while len(self.population) < self.pop_size:
            individual = self.create_individual()
            solution, _ = self.evaluate_individual(individual)
            if solution is not None:
                i = len(self.population)
                solution.weight_vector = self.weight_vectors[i]
                solution.scalar_value = self.scalarizing_function(
                    np.array(solution.objectives),
                    solution.weight_vector,
                    self.decomposition_method
                )
                self.population.append(solution)
        
        # Update reference points
        self.update_reference_points(self.population)
        
        # Recompute scalarizing values
        for sol in self.population:
            sol.scalar_value = self.scalarizing_function(
                np.array(sol.objectives),
                sol.weight_vector,
                self.decomposition_method
            )
        
        # Initialize subregion associations
        self.update_subregions_association()
        
        # Initialize non-dominated levels
        self.nondomination_levels = self.fast_non_dominated_sort(self.population)
        
        # Initialize archive
        self.archive = set(self.population)
        
        # Main evolution loop
        print("\nStarting evolution...")
        self.convergence_data = []
        
        for generation in range(self.max_gen):
            offspring_solutions = []
            
            for i in range(self.pop_size):
                # Mating selection
                parent1, parent2 = self.mating_selection(i)
                
                # Crossover and mutation
                child = self.sbx_crossover(parent1, parent2)
                child = self.polynomial_mutation(child)
                
                # Evaluate offspring
                child_solution, _ = self.evaluate_individual(child)
                
                if child_solution is not None:
                    child_solution.weight_vector = self.weight_vectors[i]
                    child_solution.scalar_value = self.scalarizing_function(
                        np.array(child_solution.objectives),
                        child_solution.weight_vector,
                        self.decomposition_method
                    )
                    offspring_solutions.append(child_solution)
            
            # Update reference points
            if offspring_solutions:
                self.update_reference_points(offspring_solutions)
            
            # Update population
            update_count = 0
            for i, child_solution in enumerate(offspring_solutions):
                if self.update_population(i, child_solution):
                    update_count += 1
            
            # Periodically update subregions
            if generation % 5 == 0:
                self.update_subregions_association()
                self.archive.update(offspring_solutions)
            
            # Record convergence data
            scalar_values = [sol.scalar_value for sol in self.population]
            avg_scalar = np.mean(scalar_values) if scalar_values else 0.0
            self.convergence_data.append(avg_scalar)
            
            # Output progress
            if generation % 20 == 0 or generation == self.max_gen - 1:
                archive_size = len(self.archive)
                pareto_size = len(self.nondomination_levels[0]) if self.nondomination_levels else 0
                elapsed = time.time() - start_time
                
                print(f"  Generation {generation:3d}: "
                      f"Archive={archive_size:3d}, "
                      f"Front={pareto_size:3d}, "
                      f"Avg Scalar={avg_scalar:8.4f}, "
                      f"Updates={update_count:3d}, "
                      f"Elapsed={elapsed:6.2f}s")
        
        # Return results
        self.archive_list = list(self.archive)
        pareto_front = [sol.objectives for sol in self.archive_list]
        
        elapsed_time = time.time() - start_time
        
        print(f"MOEA/DD Complete: Final front solution count={len(pareto_front)}")
        
        return {
            'pareto_front': pareto_front,
            'convergence_data': self.convergence_data,
            'archive': [(sol.individual, sol.objectives) for sol in self.archive_list],
            'elapsed_time': elapsed_time,
            'weight_vectors': self.weight_vectors,
            'neighbors': self.neighbors,
            'ideal_point': self.ideal_point,
            'nadir_point': self.nadir_point
        }