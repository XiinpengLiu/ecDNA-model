"""
ecDNA Copy-Number Kinetics Model - Ogata Thinning Simulation
=============================================================
Implements Section 7: Exact event-driven simulation with bounded intensities.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from cell import Cell, CellPopulation
from dynamics import JumpIntensities, apply_flow, apply_transition
from division import DivisionKernel
import config as cfg


@dataclass
class SimulationResult:
    """Container for simulation results."""
    times: List[float] = field(default_factory=list)
    population_sizes: List[int] = field(default_factory=list)
    ecdna_means: List[float] = field(default_factory=list)
    ecdna_stds: List[float] = field(default_factory=list)
    state_compositions: List[Dict] = field(default_factory=list)
    events: List[Tuple] = field(default_factory=list)
    sister_correlations: List[float] = field(default_factory=list)


class OgataThinningSimulator:
    """
    Exact simulation using Ogata thinning algorithm (Section 7.3).
    
    The algorithm:
    1. Compute dominating bound r̄
    2. Sample Δ ~ Exp(r̄)
    3. Propagate along deterministic flow
    4. Evaluate true intensity r(z(t'); u(t'))
    5. Accept with probability r/r̄
    6. If accepted, choose channel and apply transition
    """
    
    def __init__(self, 
                 drug_schedule: Dict = None,
                 env_schedule: callable = None,
                 seed: int = None):
        """
        Initialize simulator.
        
        Args:
            drug_schedule: Time-dependent drug concentrations
            env_schedule: Function E(t) -> int for deterministic environment switching
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed or cfg.RANDOM_SEED)
        self.drug_schedule = drug_schedule or cfg.DRUG_SCHEDULE
        self.env_schedule = env_schedule  # None means no deterministic env switching
        self.intensities = JumpIntensities(self.drug_schedule)
        self.division_kernel = DivisionKernel(self.rng, self.drug_schedule)
        
        # Compute global dominating bound
        self.r_bar = self.intensities.compute_dominating_bound()
    
    # -------------------------------------------------------------------------
    # Section 7.3: Ogata Thinning for Single Cell
    # -------------------------------------------------------------------------
    
    def sample_next_event(self, cell: Cell, t: float, t_max: float) -> Tuple[Optional[float], Optional[str], Optional[dict]]:
        """
        Sample next event time and type for a single cell using Ogata thinning.
        
        Returns:
            (event_time, channel_type, params) or (None, None, None) if no event before t_max
        """
        current_t = t
        temp_cell = cell.copy()  # work on copy for flow propagation
        
        while current_t < t_max:
            # Step 1-2: Sample candidate time from dominating process
            delta = self.rng.exponential(1.0 / self.r_bar)
            candidate_t = current_t + delta
            
            if candidate_t >= t_max:
                # No event before t_max
                return None, None, None
            
            # Step 3: Propagate along deterministic flow
            apply_flow(temp_cell, delta)
            
            # Step 4: Evaluate true intensity
            channels, total_rate = self.intensities.compute_all_rates(temp_cell, candidate_t)
            
            # Step 5: Accept/reject
            accept_prob = total_rate / self.r_bar
            
            if self.rng.random() < accept_prob:
                # Accepted! Choose channel
                if total_rate > 0:
                    rates = [r for _, _, r in channels]
                    channel_idx = self.rng.choice(len(channels), p=np.array(rates) / total_rate)
                    channel_type, params, _ = channels[channel_idx]
                    
                    return candidate_t, channel_type, params
            
            # Rejected, continue from candidate_t
            current_t = candidate_t
        
        return None, None, None
    
    # -------------------------------------------------------------------------
    # Section 7.4: Population Simulation
    # -------------------------------------------------------------------------
    
    def simulate(self, 
                 population: CellPopulation = None,
                 t_max: float = None,
                 record_interval: float = None,
                 max_pop: int = None,
                 verbose: bool = True) -> SimulationResult:
        """
        Run population simulation.
        
        Args:
            population: Initial population (created if None)
            t_max: Maximum simulation time
            record_interval: Interval for recording time series
            max_pop: Maximum population size
            verbose: Print progress
            
        Returns:
            SimulationResult with time series and events
        """
        # Defaults
        t_max = t_max or cfg.T_MAX
        record_interval = record_interval or cfg.RECORD_INTERVAL
        max_pop = max_pop or cfg.MAX_POP_SIZE
        
        # Initialize population
        if population is None:
            population = CellPopulation(self.rng)
            population.initialize(cfg.N_INIT)
        
        # Results container
        result = SimulationResult()
        
        # Current time
        t = 0.0
        next_record = record_interval  # First record after interval (t=0 recorded below)
        
        # Apply initial env_schedule if provided
        if self.env_schedule is not None:
            e_new = self.env_schedule(t)
            for cell in population.cells:
                cell.e = e_new
        
        # Record initial state at t=0
        self._record_state(result, t, population)
        
        if verbose:
            print(f"Starting simulation: {population.size()} cells, t_max={t_max}")
        
        # Main simulation loop
        event_count = 0
        
        while t < t_max and population.size() > 0:
            # Check population size limit
            if population.size() >= max_pop:
                if verbose:
                    print(f"Population limit reached at t={t:.2f}")
                break
            
            # Find next event across all cells
            next_event_t = np.inf
            next_cell = None
            next_channel = None
            next_params = None
            
            for cell in population.cells:
                event_t, channel, params = self.sample_next_event(cell, t, t_max)
                if event_t is not None and event_t < next_event_t:
                    next_event_t = event_t
                    next_cell = cell
                    next_channel = channel
                    next_params = params
            
            if next_cell is None:
                # No more events before t_max, advance to t_max for final records
                next_event_t = t_max
            
            # Record at scheduled times BEFORE advancing to event
            # (advance to each record time, record, then continue)
            while next_record <= next_event_t and next_record <= t_max:
                delta_to_record = next_record - t
                if delta_to_record > 0:
                    for cell in population.cells:
                        apply_flow(cell, delta_to_record)
                    # Update environment at record time
                    if self.env_schedule is not None:
                        e_new = self.env_schedule(next_record)
                        for cell in population.cells:
                            cell.e = e_new
                    t = next_record
                self._record_state(result, next_record, population)
                next_record += record_interval
            
            if next_cell is None:
                # No event, we've advanced to t_max
                break
            
            # Advance all cells from current t to event time
            delta = next_event_t - t
            if delta > 0:
                for cell in population.cells:
                    apply_flow(cell, delta)
            
            t = next_event_t
            
            # Update environment at event time
            if self.env_schedule is not None:
                e_new = self.env_schedule(t)
                for cell in population.cells:
                    cell.e = e_new
            
            # Process event
            self._process_event(population, next_cell, next_channel, next_params, t, result)
            event_count += 1
            
            # Progress update
            if verbose and event_count % 1000 == 0:
                print(f"t={t:.2f}, pop={population.size()}, events={event_count}")
        
        # Final recording if not already at a record time
        if not result.times or result.times[-1] < t:
            self._record_state(result, t, population)
        
        if verbose:
            print(f"Simulation complete: t={t:.2f}, pop={population.size()}, events={event_count}")
        
        return result
    
    def _process_event(self, population: CellPopulation, cell: Cell, 
                       channel: str, params: dict, t: float, result: SimulationResult):
        """Process a single event."""
        
        if channel == "division":
            # Division: remove parent, add two daughters
            daughter1, daughter2 = self.division_kernel.divide(cell, t)
            
            population.remove_cell(cell)
            population.add_cell(daughter1)
            population.add_cell(daughter2)
            
            # Log event
            population.log_event(t, "division", cell.cell_id, {
                "d1_id": daughter1.cell_id,
                "d2_id": daughter2.cell_id,
                "parent_k": cell.k.tolist(),
                "d1_k": daughter1.k.tolist(),
                "d2_k": daughter2.k.tolist(),
            })
            
            # Record sister correlation
            from division import compute_sister_correlation
            corr = compute_sister_correlation(daughter1.k, daughter2.k)
            result.sister_correlations.append(corr)
            
        elif channel == "death":
            # Death: remove cell
            population.remove_cell(cell)
            population.log_event(t, "death", cell.cell_id, {"k": cell.k.tolist()})
            
        else:
            # State transition
            apply_transition(cell, channel, params)
            population.log_event(t, channel, cell.cell_id, params)
    
    def _record_state(self, result: SimulationResult, t: float, population: CellPopulation):
        """Record current population state."""
        result.times.append(t)
        summary = population.get_summary()
        result.population_sizes.append(summary["n"])
        result.ecdna_means.append(summary.get("ecdna_mean", 0))
        result.ecdna_stds.append(summary.get("ecdna_std", 0))
        result.state_compositions.append(summary)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_simulation(t_max: float = None, 
                   n_init: int = None,
                   drug_schedule: Dict = None,
                   env_schedule: callable = None,
                   seed: int = None,
                   verbose: bool = True) -> SimulationResult:
    """
    Run a simulation with default or custom parameters.
    
    Args:
        t_max: Maximum simulation time
        n_init: Initial population size
        drug_schedule: Drug concentration functions
        env_schedule: Function E(t) -> int for deterministic environment switching
        seed: Random seed
        verbose: Print progress
        
    Returns:
        SimulationResult
    """
    # Create simulator
    sim = OgataThinningSimulator(drug_schedule=drug_schedule, env_schedule=env_schedule, seed=seed)
    
    # Create population
    rng = np.random.default_rng(seed or cfg.RANDOM_SEED)
    pop = CellPopulation(rng)
    pop.initialize(n_init or cfg.N_INIT)
    
    # Run
    return sim.simulate(
        population=pop,
        t_max=t_max or cfg.T_MAX,
        verbose=verbose
    )
