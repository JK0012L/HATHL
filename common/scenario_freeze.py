# scenario_freeze.py - Scenario Freezer (In-Memory Interaction Version)

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class FrozenJob:
    """Frozen job parameters"""
    job_idx: int
    arrival_time: float
    route: List[int]                    # Operation sequence (machine index list)
    processing_times: List[float]       # Processing time for each operation
    loading_times: List[float]          # Loading time for each operation
    unloading_times: List[float]        # Unloading time for each operation


@dataclass
class BreakdownEvent:
    """Machine breakdown event"""
    start_time: float
    end_time: float
    duration: float


@dataclass
class FrozenMachine:
    """Frozen machine parameters"""
    machine_idx: int
    breakdowns: List[BreakdownEvent]    # Breakdown event list


@dataclass
class FrozenWorker:
    """Frozen worker initial state"""
    worker_idx: int
    initial_position: int               # Initial position (machine index, -1 means starting point)
    efficiency_matrix: List[float]      # Efficiency per machine
    initial_physical_fatigue: float
    initial_mental_fatigue: float
    physical_capacity: float = 0.5      # Physical capacity index (heterogeneity parameter)
    environmental_stress: float = 0.3   # Environmental stress index (heterogeneity parameter)


@dataclass
class FrozenScenario:
    """Frozen complete scenario"""
    scenario_id: str
    seed: int
    
    # System configuration
    num_machines: int
    num_workers: int
    num_jobs: int
    
    # Frozen data
    jobs: List[FrozenJob]               # By job index
    machines: List[FrozenMachine]       # By machine index
    workers: List[FrozenWorker]         # By worker index
    
    # Machine positions (for calculating walking distance)
    machine_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    
    # Global parameters
    walking_speed: float = 1.2
    physical_recovery_rate: float = 0.03
    mental_recovery_rate: float = 0.02
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (for static algorithm use)"""
        return {
            'scenario_id': self.scenario_id,
            'seed': self.seed,
            'num_machines': self.num_machines,
            'num_workers': self.num_workers,
            'num_jobs': self.num_jobs,
            'walking_speed': self.walking_speed,
            'physical_recovery_rate': self.physical_recovery_rate,
            'mental_recovery_rate': self.mental_recovery_rate,
            'machine_positions': self.machine_positions,
            'jobs': [
                {
                    'job_idx': j.job_idx,
                    'arrival_time': j.arrival_time,
                    'route': j.route,
                    'processing_times': j.processing_times,
                    'loading_times': j.loading_times,
                    'unloading_times': j.unloading_times
                }
                for j in self.jobs
            ],
            'machines': [
                {
                    'machine_idx': m.machine_idx,
                    'breakdowns': [
                        {'start': b.start_time, 'end': b.end_time, 'duration': b.duration}
                        for b in m.breakdowns
                    ]
                }
                for m in self.machines
            ],
            'workers': [
                {
                    'worker_idx': w.worker_idx,
                    'initial_position': w.initial_position,
                    'efficiency_matrix': w.efficiency_matrix,
                    'initial_physical_fatigue': w.initial_physical_fatigue,
                    'initial_mental_fatigue': w.initial_mental_fatigue,
                    'physical_capacity': w.physical_capacity,
                    'environmental_stress': w.environmental_stress
                }
                for w in self.workers
            ]
        }


class ScenarioFreezer:
    """
    Scenario freezer - collects data during dynamic simulation
    Only records data in memory, does not save to files
    """
    
    def __init__(self, scenario_id: str, seed: int, 
                 num_machines: int, num_workers: int):
        self.scenario_id = scenario_id
        self.seed = seed
        self.num_machines = num_machines
        self.num_workers = num_workers
        
        # Data containers
        self.jobs: List[FrozenJob] = []
        self.machine_breakdowns: Dict[int, List[BreakdownEvent]] = {}
        self.worker_initial_states: Dict[int, Dict] = {}
        
        # Temporary records (for associating with the currently recorded job)
        self._current_job_idx = None
        self._current_job_route = []
        self._current_job_pt = []
        self._current_job_loading = []
        self._current_job_unloading = []
    
    def start_job_recording(self, job_idx: int):
        """
        Start recording job parameters
        Called when a job arrives
        """
        self._current_job_idx = job_idx
        self._current_job_route = []
        self._current_job_pt = []
        self._current_job_loading = []
        self._current_job_unloading = []
    
    def record_job_operation(self, machine_idx: int, 
                              processing_time: float,
                              loading_time: float,
                              unloading_time: float):
        """
        Record parameters for one operation
        Called when generating each operation of a job
        """
        self._current_job_route.append(machine_idx)
        self._current_job_pt.append(processing_time)
        self._current_job_loading.append(loading_time)
        self._current_job_unloading.append(unloading_time)
    
    def finish_job_recording(self, arrival_time: float):
        """
        Finish recording job
        Called at the end of job arrival process
        """
        if self._current_job_idx is not None:
            job = FrozenJob(
                job_idx=self._current_job_idx,
                arrival_time=arrival_time,
                route=self._current_job_route.copy(),
                processing_times=self._current_job_pt.copy(),
                loading_times=self._current_job_loading.copy(),
                unloading_times=self._current_job_unloading.copy()
            )
            self.jobs.append(job)
            
            # Clear temporary records
            self._current_job_idx = None
            self._current_job_route = []
            self._current_job_pt = []
            self._current_job_loading = []
            self._current_job_unloading = []
    
    def record_breakdown(self, machine_idx: int, start_time: float, end_time: float):
        """
        Record machine breakdown event
        Called after machine breakdown is repaired
        """
        if machine_idx not in self.machine_breakdowns:
            self.machine_breakdowns[machine_idx] = []
        
        self.machine_breakdowns[machine_idx].append(
            BreakdownEvent(start_time=start_time, end_time=end_time, 
                          duration=end_time - start_time)
        )
    
    def record_worker_initial_state(self, worker_idx: int, position: int,
                                     efficiency_matrix: List[float],
                                     physical_fatigue: float, mental_fatigue: float,
                                     physical_capacity: float = 0.5,
                                     environmental_stress: float = 0.3):
        """
        Record worker initial state
        Called during simulation initialization
        """
        self.worker_initial_states[worker_idx] = {
            'initial_position': position,
            'efficiency_matrix': efficiency_matrix.copy() if efficiency_matrix else [],
            'initial_physical_fatigue': physical_fatigue,
            'initial_mental_fatigue': mental_fatigue,
            'physical_capacity': physical_capacity,
            'environmental_stress': environmental_stress
        }
    
    def _get_default_machine_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get default machine positions (linear layout)"""
        positions = {}
        for i in range(self.num_machines):
            positions[i] = (i * 5.0, 0.0)
        return positions
    
    def freeze(self) -> FrozenScenario:
        """
        Generate frozen scenario
        Called after simulation ends, returns a FrozenScenario object
        """
        # Organize machine data
        machines = []
        for m_idx in range(self.num_machines):
            breakdowns = self.machine_breakdowns.get(m_idx, [])
            # Sort by start time
            breakdowns.sort(key=lambda x: x.start_time)
            machines.append(FrozenMachine(
                machine_idx=m_idx,
                breakdowns=breakdowns
            ))
        
        # Organize worker data
        workers = []
        for w_idx in range(self.num_workers):
            state = self.worker_initial_states.get(w_idx, {})
            workers.append(FrozenWorker(
                worker_idx=w_idx,
                initial_position=state.get('initial_position', -1),
                efficiency_matrix=state.get('efficiency_matrix', []),
                initial_physical_fatigue=state.get('initial_physical_fatigue', 0.0),
                initial_mental_fatigue=state.get('initial_mental_fatigue', 0.0),
                physical_capacity=state.get('physical_capacity', 0.5),
                environmental_stress=state.get('environmental_stress', 0.3)
            ))
        
        # Sort by job index
        self.jobs.sort(key=lambda x: x.job_idx)
        
        return FrozenScenario(
            scenario_id=self.scenario_id,
            seed=self.seed,
            num_machines=self.num_machines,
            num_workers=self.num_workers,
            num_jobs=len(self.jobs),
            jobs=self.jobs,
            machines=machines,
            workers=workers,
            machine_positions=self._get_default_machine_positions()
        )
    
    def get_job_count(self) -> int:
        """Get the number of recorded jobs"""
        return len(self.jobs)
    
    def get_machine_breakdown_count(self, machine_idx: int) -> int:
        """Get the number of breakdowns for a specified machine"""
        return len(self.machine_breakdowns.get(machine_idx, []))
    
    def reset(self):
        """Reset all records (for multiple simulations)"""
        self.jobs = []
        self.machine_breakdowns = {}
        self.worker_initial_states = {}
        self._current_job_idx = None
        self._current_job_route = []
        self._current_job_pt = []
        self._current_job_loading = []
        self._current_job_unloading = []