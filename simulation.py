"""
ecDNA Copy-Number Kinetics Model - Ogata Thinning Simulation
Exact event-driven simulation with bounded intensities.
"""

import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from cell import Cell, CellPopulation
from dynamics import JumpIntensities, apply_flow, apply_transition, lazy_apply_flow, batch_lazy_apply_flow
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
    ecdna_distributions: List[np.ndarray] = field(default_factory=list)  # Full ecDNA distribution at each time point
    fitness_snapshots: List[List[Dict]] = field(default_factory=list)  # Per-cell fitness data at each time point

    def save_as_csv(self, base_dir: str):
        """Save simulation results to CSV files in the specified directory."""
        import csv
        import json
        from pathlib import Path
        
        dir_path = Path(base_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary Time Series
        with open(dir_path / 'time_series_summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'population_size', 'ecdna_mean', 'ecdna_std'])
            for t, n, m, s in zip(self.times, self.population_sizes, self.ecdna_means, self.ecdna_stds):
                writer.writerow([t, n, m, s])
                
        # 2. ecDNA Distributions (Long format)
        with open(dir_path / 'ecdna_distributions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'cell_index', 'ecdna_copy_number'])
            for t, dist in zip(self.times, self.ecdna_distributions):
                if len(dist) > 0:
                    for idx, val in enumerate(dist):
                        writer.writerow([t, idx, val])

        # 3. Fitness Landscape Snapshots
        with open(dir_path / 'fitness_landscape.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'cell_index', 'ecdna', 'cycle', 'sen', 'expr', 'div_rate', 'death_rate', 'net_rate'])
            for t, snapshot in zip(self.times, self.fitness_snapshots):
                for idx, cell_data in enumerate(snapshot):
                    writer.writerow([
                        t, idx, 
                        cell_data.get('ecdna'), 
                        cell_data.get('cycle'), 
                        cell_data.get('sen'), 
                        cell_data.get('expr'),
                        cell_data.get('div_rate'),
                        cell_data.get('death_rate'),
                        cell_data.get('net_rate')
                    ])

        # 4. Events Log (Detailed with pre/post states)
        with open(dir_path / 'events_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time', 'event_type', 'cell_id',
                'pre_e', 'pre_c', 'pre_s', 'pre_x', 'pre_k', 'pre_a',
                'post_e', 'post_c', 'post_s', 'post_x', 'post_k', 'post_a',
                'd1_id', 'd1_e', 'd1_c', 'd1_s', 'd1_x', 'd1_k', 'd1_a',
                'd2_id', 'd2_e', 'd2_c', 'd2_s', 'd2_x', 'd2_k', 'd2_a',
            ])
            for event in self.events:
                t, etype, cid, details = event
                pre = details.get('state_pre', {})
                post = details.get('state_post', {})
                d1 = details.get('d1_state', {})
                d2 = details.get('d2_state', {})
                writer.writerow([
                    t, etype, cid,
                    pre.get('e'), pre.get('c'), pre.get('s'), pre.get('x'), 
                    json.dumps(pre.get('k')), pre.get('a'),
                    post.get('e') if post else None, post.get('c') if post else None,
                    post.get('s') if post else None, post.get('x') if post else None,
                    json.dumps(post.get('k')) if post else None, post.get('a') if post else None,
                    details.get('d1_id'), d1.get('e'), d1.get('c'), d1.get('s'), d1.get('x'),
                    json.dumps(d1.get('k')) if d1 else None, d1.get('a'),
                    details.get('d2_id'), d2.get('e'), d2.get('c'), d2.get('s'), d2.get('x'),
                    json.dumps(d2.get('k')) if d2 else None, d2.get('a'),
                ])

        # 5. Sister Correlations
        with open(dir_path / 'sister_correlations.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['correlation_coefficient'])
            for val in self.sister_correlations:
                writer.writerow([val])
                
        print(f"Simulation results saved to CSV files in {dir_path}")


class OgataThinningSimulator:
    """
    Exact simulation using Ogata thinning algorithm
    
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
    
    
    # Ogata Thinning for Single Cell
    
    
    def sample_next_event(self, cell: Cell, t_start: float, t_max: float) -> Tuple[Optional[float], Optional[str], Optional[dict]]:
        """
        Sample next event time and type for a single cell using Ogata thinning.
        
        Note: This method works on a COPY of the cell to simulate flow propagation
        without modifying the original cell's state.
        
        Args:
            cell: Cell to sample event for (will not be modified)
            t_start: Start time for sampling (should match cell.last_update_time)
            t_max: Maximum time horizon
        
        Returns:
            (event_time, channel_type, params) or (None, None, None) if no event before t_max
        """
        current_t = t_start
        temp_cell = cell.copy()  # work on copy for flow propagation
        
        while current_t < t_max:
            # Step 1-2: Sample candidate time from dominating process
            delta = self.rng.exponential(1.0 / self.r_bar)
            candidate_t = current_t + delta
            
            if candidate_t >= t_max:
                # No event before t_max
                return None, None, None
            
            # Step 3: Propagate temp_cell along deterministic flow
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


    # Population Simulation with Lazy Flow

    def simulate(self, 
                 population: CellPopulation = None,
                 t_max: float = None,
                 record_interval: float = None,
                 max_pop: int = None,
                 verbose: bool = True) -> SimulationResult:
        """
        Run population simulation with lazy flow updates.
        
        Key optimization: Cells are only updated (flow applied) when:
        1. Their event is about to be processed
        2. A record time is reached (all cells synchronized)
        
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
        
        # Initialize last_update_time for all cells
        for cell in population.cells:
            cell.last_update_time = 0.0
        
        # Results container
        result = SimulationResult()
        
        # Current simulation time (logical clock)
        t = 0.0
        next_record = record_interval  # First record after interval (t=0 recorded below)
        last_env = None
        
        # Apply initial env_schedule if provided
        if self.env_schedule is not None:
            last_env = self.env_schedule(t)
            for cell in population.cells:
                cell.e = last_env
        
        # Record initial state at t=0
        self._record_state(result, t, population)
        
        if verbose:
            print(f"Starting simulation: {population.size()} cells, t_max={t_max}")
        
        # Prepare event heap: (event_time, cell_id, channel, params, version)
        event_heap: List[Tuple[float, int, str, dict, int]] = []
        cell_versions: Dict[int, int] = {}
        cell_lookup: Dict[int, Cell] = {cell.cell_id: cell for cell in population.cells}
        
        def schedule_cell_event(cell: Cell, from_time: float):
            """Schedule next event for a cell starting from from_time."""
            cell_id = cell.cell_id
            event_t, channel, params = self.sample_next_event(cell, from_time, t_max)
            version = cell_versions.get(cell_id, 0) + 1
            cell_versions[cell_id] = version
            if event_t is not None:
                heapq.heappush(event_heap, (event_t, cell_id, channel, params, version))
        
        def resample_all_events(from_time: float):
            """Resample events for all cells from given time."""
            event_heap.clear()
            for cell in population.cells:
                schedule_cell_event(cell, from_time)
        
        # Initialize heap with all cells
        resample_all_events(t)
        
        # Main simulation loop
        event_count = 0
        
        while t < t_max and population.size() > 0:
            # Check population size limit
            if population.size() >= max_pop:
                if verbose:
                    print(f"Population limit reached at t={t:.2f}")
                break
            
            # Find next valid event
            next_event_t = np.inf
            next_cell = None
            next_channel = None
            next_params = None
            
            while event_heap:
                event_t, cell_id, channel, params, version = heapq.heappop(event_heap)
                if cell_versions.get(cell_id) != version:
                    continue  # Stale event, skip
                cell = cell_lookup.get(cell_id)
                if cell is None:
                    continue  # Cell removed, skip
                next_event_t = event_t
                next_cell = cell
                next_channel = channel
                next_params = params
                break
            
            if next_cell is None:
                # No more events before t_max
                next_event_t = t_max
            
            # Handle record times before the next event
            env_changed = False
            while next_record <= min(next_event_t, t_max):
                # Synchronize all cells to record time (lazy update)
                batch_lazy_apply_flow(population.cells, next_record)
                
                # Update environment at record time
                if self.env_schedule is not None:
                    e_new = self.env_schedule(next_record)
                    if e_new != last_env:
                        for cell in population.cells:
                            cell.e = e_new
                        last_env = e_new
                        env_changed = True
                
                # Update logical time
                t = next_record
                
                # Record state
                self._record_state(result, next_record, population)
                next_record += record_interval
                
                if env_changed:
                    # Environment changed: resample all events from current time
                    resample_all_events(t)
                    break
            
            if env_changed:
                continue  # Restart loop with new events
            
            if next_cell is None:
                # No event, we've recorded up to t_max
                break
            
            # Process the event
            # Only update the cell that has the event (lazy flow)
            lazy_apply_flow(next_cell, next_event_t)
            
            # Update environment at event time if needed
            if self.env_schedule is not None:
                e_new = self.env_schedule(next_event_t)
                if e_new != last_env:
                    # Environment change at event time
                    # Update all cells' environment (but not their flow state)
                    for cell in population.cells:
                        cell.e = e_new
                    last_env = e_new
                    env_changed = True
            
            # Update logical time
            t = next_event_t
            
            # Process event
            self._process_event(population, next_cell, next_channel, next_params, t, result)
            event_count += 1
            
            # Update lookup/cache after event
            if next_channel == "division":
                # Parent removed, two daughters added
                cell_lookup.pop(next_cell.cell_id, None)
                cell_versions.pop(next_cell.cell_id, None)
                # Get new daughters (last two added)
                new_cells = population.cells[-2:]
                for new_cell in new_cells:
                    cell_lookup[new_cell.cell_id] = new_cell
                    new_cell.last_update_time = t  # Daughters start at current time
            elif next_channel == "death":
                # Cell removed
                cell_lookup.pop(next_cell.cell_id, None)
                cell_versions.pop(next_cell.cell_id, None)
            else:
                # Cell state changed, update lookup
                cell_lookup[next_cell.cell_id] = next_cell
            
            # Reschedule events
            if env_changed:
                # Environment changed: resample all events
                resample_all_events(t)
            else:
                if next_channel == "division":
                    # Schedule events for new daughters
                    for new_cell in population.cells[-2:]:
                        schedule_cell_event(new_cell, t)
                elif next_channel == "death":
                    pass  # Cell is gone, no new event
                else:
                    # Reschedule for the cell that just had an event
                    schedule_cell_event(next_cell, t)
            
            # Progress update
            if verbose and event_count % 1000 == 0:
                print(f"t={t:.2f}, pop={population.size()}, events={event_count}")
        
        # Final recording if not already at a record time
        if result.times and result.times[-1] < t:
            batch_lazy_apply_flow(population.cells, t)
            self._record_state(result, t, population)
        
        # Copy events to result for lineage analysis
        result.events = population.events.copy()
        
        if verbose:
            print(f"Simulation complete: t={t:.2f}, pop={population.size()}, events={event_count}")
        
        return result
    
    def _process_event(self, population: CellPopulation, cell: Cell, 
                       channel: str, params: dict, t: float, result: SimulationResult):
        """Process a single event with full state tracking."""
        
        # Capture pre-event state
        state_pre = cell.get_state_dict()
        
        if channel == "division":
            # Division: remove parent, add two daughters
            daughter1, daughter2 = self.division_kernel.divide(cell, t)
            
            population.remove_cell(cell)
            population.add_cell(daughter1)
            population.add_cell(daughter2)
            
            # Log event with full states
            population.log_event(t, "division", cell.cell_id, {
                "state_pre": state_pre,
                "state_post": None,  # Parent no longer exists
                "d1_id": daughter1.cell_id,
                "d2_id": daughter2.cell_id,
                "d1_state": daughter1.get_state_dict(),
                "d2_state": daughter2.get_state_dict(),
            })
            
            # Record sister correlation
            from division import compute_sister_correlation
            corr = compute_sister_correlation(daughter1.k, daughter2.k)
            result.sister_correlations.append(corr)
            
        elif channel == "death":
            # Death: remove cell
            population.remove_cell(cell)
            population.log_event(t, "death", cell.cell_id, {
                "state_pre": state_pre,
                "state_post": None,  # Cell no longer exists
            })
            
        else:
            # State transition (cycle, sen, expr, ecdna_gain, ecdna_loss)
            apply_transition(cell, channel, params)
            state_post = cell.get_state_dict()
            population.log_event(t, channel, cell.cell_id, {
                "state_pre": state_pre,
                "state_post": state_post,
                **params,  # Include transition-specific params (e.g., j for ecdna)
            })
    
    def _record_state(self, result: SimulationResult, t: float, population: CellPopulation):
        """Record current population state."""
        result.times.append(t)
        summary = population.get_summary()
        result.population_sizes.append(summary["n"])
        result.ecdna_means.append(summary.get("ecdna_mean", 0))
        result.ecdna_stds.append(summary.get("ecdna_std", 0))
        result.state_compositions.append(summary)
        # Record full ecDNA distribution for heterogeneity analysis
        if population.cells:
            ecdna_dist = np.array([c.total_ecdna() for c in population.cells])
            result.ecdna_distributions.append(ecdna_dist)
            
            # Record per-cell fitness data for fitness landscape analysis
            fitness_data = []
            for cell in population.cells:
                k_total = cell.total_ecdna()
                div_rate = self.intensities.division_hazard(cell, t, k_total=k_total)
                death_rate = self.intensities.death_hazard(cell, t)
                fitness_data.append({
                    'ecdna': k_total,
                    'cycle': cell.c,
                    'sen': cell.s,
                    'expr': cell.x,
                    'div_rate': div_rate,
                    'death_rate': death_rate,
                    'net_rate': div_rate - death_rate,
                })
            result.fitness_snapshots.append(fitness_data)
        else:
            result.ecdna_distributions.append(np.array([]))
            result.fitness_snapshots.append([])


def run_simulation(t_max: float = None, 
                   n_init: int = None,
                   drug_schedule: Dict = None,
                   env_schedule: callable = None,
                   seed: int = None,
                   max_pop: int = None,
                   verbose: bool = True) -> SimulationResult:
    """
    Run a simulation with default or custom parameters.
    
    Args:
        t_max: Maximum simulation time
        n_init: Initial population size
        drug_schedule: Drug concentration functions
        env_schedule: Function E(t) -> int for deterministic environment switching
        seed: Random seed
        max_pop: Maximum population size (termination condition)
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
        max_pop=max_pop,
        verbose=verbose
    )
