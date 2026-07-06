# NSGA2.py - NSGA-II-based Multi-Objective Optimization Algorithm

import numpy as np
import random
from typing import List, Tuple, Dict
from common.static_scheduling import decode_schedule, evaluate_objectives


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


class NSGA2Scheduler:
    """NSGA-II-based multi-objective job shop scheduling optimizer"""
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=100, 
                 crossover_prob=0.8, mutation_prob=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Statistics
        self.best_objectives_history = []
        self.avg_objectives_history = []
    
    def create_individual(self) -> List[int]:
        """Create individual - operation-based encoding"""
        individual = []
        for job_id in range(self.problem.num_jobs):
            individual.extend([job_id] * self.problem.num_machines)
        random.shuffle(individual)
        return individual
    
    def dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if solution 1 dominates solution 2 (all objectives minimized)"""
        # Handle infinity
        if any(np.isinf(o) for o in obj1) or any(np.isinf(o) for o in obj2):
            return False
        
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better
    
    def fast_non_dominated_sort(self, population_with_objectives):
        """Fast non-dominated sorting"""
        if not population_with_objectives:
            return [[]]
        
        population_size = len(population_with_objectives)
        S = [[] for _ in range(population_size)]
        n = [0] * population_size
        fronts = [[]]
        
        for i in range(population_size):
            ind1, obj1 = population_with_objectives[i]
            for j in range(population_size):
                if i == j:
                    continue
                ind2, obj2 = population_with_objectives[j]
                if self.dominates(obj1, obj2):
                    S[i].append(j)
                elif self.dominates(obj2, obj1):
                    n[i] += 1
            
            if n[i] == 0:
                fronts[0].append((ind1, obj1))
        
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for solution in fronts[i]:
                ind, obj = solution
                idx = None
                for k in range(population_size):
                    if (population_with_objectives[k][0] == ind and 
                        population_with_objectives[k][1] == obj):
                        idx = k
                        break
                
                if idx is not None:
                    for dominated_idx in S[idx]:
                        n[dominated_idx] -= 1
                        if n[dominated_idx] == 0:
                            next_front.append(population_with_objectives[dominated_idx])
            
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts
    
    def crowding_distance_assignment(self, front):
        """Compute crowding distance"""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        num_solutions = len(front)
        distances = [0.0] * num_solutions
        
        for m in range(3):  # 3 objectives
            # Sort by the m-th objective
            sorted_front = sorted(front, key=lambda x: x[1][m])
            
            distances[0] = float('inf')
            distances[-1] = float('inf')
            
            min_obj = sorted_front[0][1][m]
            max_obj = sorted_front[-1][1][m]
            obj_range = max_obj - min_obj if max_obj > min_obj else 1.0
            
            for i in range(1, num_solutions - 1):
                distances[i] += (sorted_front[i+1][1][m] - sorted_front[i-1][1][m]) / obj_range
        
        return distances
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order Crossover"""
        if random.random() > self.crossover_prob:
            return parent1[:], parent2[:]
        
        size = len(parent1)
        cx1 = random.randint(0, size - 2)
        cx2 = random.randint(cx1 + 1, size - 1)
        
        def ox_crossover(p1, p2):
            child = [-1] * size
            # Copy middle segment
            child[cx1:cx2+1] = p1[cx1:cx2+1]
            
            # Get remaining genes from second parent
            remaining_genes = []
            for g in p2:
                if g not in child[cx1:cx2+1]:
                    remaining_genes.append(g)
                elif p1[cx1:cx2+1].count(g) < p2.count(g):
                    # Handle duplicate genes
                    remaining_genes.append(g)
            
            if len(remaining_genes) != (size - (cx2 - cx1 + 1)):
                # If length does not match, return parent copy
                return p1[:]
            
            # Fill remaining positions
            fill_index = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining_genes[fill_index]
                    fill_index += 1
            return child
        
        child1 = ox_crossover(parent1, parent2)
        child2 = ox_crossover(parent2, parent1)
        
        return child1, child2
    
    def mutation(self, individual: List[int]) -> List[int]:
        """Swap mutation"""
        if random.random() > self.mutation_prob:
            return individual[:]
        
        mutated = individual[:]
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def repair_individual(self, individual: List[int]) -> List[int]:
        """Repair individual to ensure correct appearance count for each job"""
        job_counts = {j: 0 for j in range(self.problem.num_jobs)}
        for gene in individual:
            if gene in job_counts:
                job_counts[gene] += 1
        
        repaired = individual[:]
        for job_id, count in job_counts.items():
            expected = self.problem.num_machines
            if count < expected:
                # Add missing genes
                for _ in range(expected - count):
                    repaired.append(job_id)
            elif count > expected:
                # Remove excess genes
                indices_to_remove = [i for i, g in enumerate(repaired) if g == job_id]
                for idx in indices_to_remove[expected:]:
                    repaired.pop(idx)
        
        return repaired
    
    def evaluate_individual(self, individual: List[int]) -> Tuple[float, float, float]:
        """Evaluate individual, return three objective values"""
        # Repair individual
        individual = self.repair_individual(individual)
        
        # Decode and evaluate
        schedule = decode_schedule(individual, self.problem)
        if not schedule.get('valid', False):
            return (float('inf'), float('inf'), float('inf'))
        
        return evaluate_objectives(schedule)
    
    def run(self) -> Dict:
        """Run NSGA-II algorithm"""
        print(f"NSGA-II Initialization: Population size={self.pop_size}, Max generations={self.max_gen}")
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        best_solutions = []
        convergence_data = []
        
        for generation in range(self.max_gen):
            # Evaluate current population
            evaluated_population = []
            for ind in population:
                obj = self.evaluate_individual(ind)
                if not any(np.isinf(o) for o in obj):
                    evaluated_population.append((ind, obj))
            
            if not evaluated_population:
                print(f"  Generation {generation}: No valid solution, reinitializing")
                population = [self.create_individual() for _ in range(self.pop_size)]
                continue
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(evaluated_population)
            
            # Record convergence data
            if fronts and fronts[0]:
                front_objs = [obj for _, obj in fronts[0]]
                avg_obj1 = np.mean([obj[0] for obj in front_objs])
                avg_obj2 = np.mean([obj[1] for obj in front_objs])
                avg_obj3 = np.mean([obj[2] for obj in front_objs])
                convergence_data.append((avg_obj1, avg_obj2, avg_obj3))
                best_solutions.append(fronts[0])
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament selection
                candidates = random.sample(population, min(4, len(population)))
                parent1 = candidates[0]
                candidates = random.sample(population, min(4, len(population)))
                parent2 = candidates[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)
            
            # Merge parent and offspring
            combined_population = population + offspring
            
            # Evaluate combined population
            combined_evaluated = []
            for ind in combined_population:
                obj = self.evaluate_individual(ind)
                if not any(np.isinf(o) for o in obj):
                    combined_evaluated.append((ind, obj))
            
            if not combined_evaluated:
                population = [self.create_individual() for _ in range(self.pop_size)]
                continue
            
            # Non-dominated sorting
            combined_fronts = self.fast_non_dominated_sort(combined_evaluated)
            
            # Build new generation population
            new_population = []
            front_index = 0
            
            while front_index < len(combined_fronts) and len(new_population) < self.pop_size:
                current_front = combined_fronts[front_index]
                
                if len(new_population) + len(current_front) <= self.pop_size:
                    for ind, obj in current_front:
                        new_population.append(ind)
                else:
                    needed = self.pop_size - len(new_population)
                    if len(current_front) > 1:
                        crowding_distances = self.crowding_distance_assignment(current_front)
                        # Sort by crowding distance in descending order
                        sorted_indices = np.argsort(crowding_distances)[::-1]
                        for i in range(min(needed, len(sorted_indices))):
                            new_population.append(current_front[sorted_indices[i]][0])
                    else:
                        new_population.append(current_front[0][0])
                
                front_index += 1
            
            population = new_population[:self.pop_size]
            
            if generation % 20 == 0:
                front_count = len(fronts[0]) if fronts and fronts[0] else 0
                print(f"  Evolution generation {generation}: Front solution count={front_count}")
        
        # Return final Pareto front
        final_evaluated = []
        for ind in population:
            obj = self.evaluate_individual(ind)
            if not any(np.isinf(o) for o in obj):
                final_evaluated.append((ind, obj))
        
        fronts_final = self.fast_non_dominated_sort(final_evaluated)
        pareto_front = []
        if fronts_final and fronts_final[0]:
            pareto_front = [obj for _, obj in fronts_final[0]]
        
        print(f"NSGA-II Complete: Final front solution count={len(pareto_front)}")
        
        return {
            'pareto_front': pareto_front,
            'convergence_data': convergence_data,
            'best_solutions': best_solutions,
            'final_population': population
        }