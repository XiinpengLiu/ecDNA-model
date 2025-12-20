"""
ecDNA Kinetic Model - Comprehensive Implementation
==================================================
A stochastic model for cell population dynamics with ecDNA copy number variation.

Cell state Z(t) = (E, M, K, Y, A):
- E: environment/stage label
- M: discrete regulatory state  
- K: ecDNA copy numbers (vector)
- Y: continuous phenomic state (vector)
- A: age since last division
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Dict, Any
from scipy.stats import binom
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class CellState:
    """Single cell state at time t."""
    e: int = 0                      # Environment label
    m: int = 0                      # Regulatory state (0 to M-1)
    k: np.ndarray = field(default_factory=lambda: np.array([0]))  # ecDNA copy numbers
    y: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Phenomic state
    a: float = 0.0                  # Age since last division
    
    def copy(self) -> 'CellState':
        return CellState(
            e=self.e, m=self.m,
            k=self.k.copy(), y=self.y.copy(),
            a=self.a
        )
    
    @property
    def discrete_type(self) -> Tuple[int, int, Tuple[int, ...]]:
        """Return discrete type i = (e, m, k)."""
        return (self.e, self.m, tuple(self.k))


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    dt: float = 0.01              # Time step
    t_max: float = 100.0          # Maximum simulation time
    max_cells: int = 10000        # Maximum cell population
    seed: Optional[int] = None    # Random seed
    track_lineage: bool = False   # Track cell lineage


# =============================================================================
# Model Parameters Interface
# =============================================================================

class ModelParameters(ABC):
    """Abstract base class for model parameters."""
    
    def __init__(self, n_environments: int = 1, n_regulatory: int = 1,
                 n_ecdna_species: int = 1, n_phenotype_dims: int = 1):
        self.n_env = n_environments
        self.n_reg = n_regulatory
        self.n_ecdna = n_ecdna_species
        self.n_pheno = n_phenotype_dims
    
    # --- Phenomic dynamics (SDE) ---
    def drift(self, state: CellState) -> np.ndarray:
        """Drift f_i(y, a) for phenomic SDE. Default: no drift."""
        return np.zeros(self.n_pheno)
    
    def diffusion(self, state: CellState) -> np.ndarray:
        """Diffusion matrix Sigma_i(y, a). Default: no diffusion."""
        return np.zeros((self.n_pheno, self.n_pheno))
    
    # --- Terminal event hazards ---
    @abstractmethod
    def lambda_div(self, state: CellState) -> float:
        """Division hazard rate."""
        pass
    
    @abstractmethod
    def lambda_death(self, state: CellState) -> float:
        """Death hazard rate."""
        pass
    
    # --- Regulatory switching (CTMC for M) ---
    def q_m_rates(self, state: CellState) -> np.ndarray:
        """
        Transition rates for regulatory state M.
        Returns array of shape (n_reg,) with rate to switch to each state.
        Rate to current state should be 0.
        """
        return np.zeros(self.n_reg)
    
    # --- Environment switching (CTMC for E) ---
    def omega_e_rates(self, state: CellState) -> np.ndarray:
        """Transition rates for environment E."""
        return np.zeros(self.n_env)
    
    # --- Inter-division ecDNA dynamics ---
    def mu_gain(self, state: CellState, j: int) -> float:
        """Gain rate for ecDNA species j."""
        return 0.0
    
    def mu_loss(self, state: CellState, j: int) -> float:
        """Per-copy loss rate for ecDNA species j."""
        return 0.0
    
    # --- Division kernel ---
    def sample_amplification(self, state: CellState) -> np.ndarray:
        """Sample amplification vector A. Default: no extra amplification."""
        return np.zeros(self.n_ecdna, dtype=int)
    
    def post_segregation_loss_prob(self, state: CellState, j: int, 
                                    k_daughter_j: int) -> float:
        """Post-segregation loss probability ell for species j."""
        return 0.0
    
    def sample_daughter_m(self, parent: CellState, 
                          k_daughter: np.ndarray) -> int:
        """Sample regulatory state for daughter. Default: inherit."""
        return parent.m
    
    def sample_daughter_y(self, parent: CellState, m_daughter: int,
                          k_daughter: np.ndarray) -> np.ndarray:
        """Sample phenotype for daughter. Default: inherit with noise."""
        return parent.y.copy()


# =============================================================================
# Default/Example Parameter Sets
# =============================================================================

class SimpleEcDNAParameters(ModelParameters):
    """
    Simple ecDNA model with:
    - Age-dependent division (Erlang-like)
    - Constant death rate
    - ecDNA copy number affects fitness
    """
    
    def __init__(self, 
                 div_rate: float = 0.05,
                 death_rate: float = 0.01,
                 optimal_copies: float = 10.0,
                 fitness_width: float = 5.0,
                 n_ecdna_species: int = 1):
        super().__init__(n_ecdna_species=n_ecdna_species)
        self.div_rate = div_rate
        self.death_rate = death_rate
        self.optimal_copies = optimal_copies
        self.fitness_width = fitness_width
    
    def _fitness_modifier(self, k_total: float) -> float:
        """Gaussian fitness function centered at optimal copies."""
        return np.exp(-0.5 * ((k_total - self.optimal_copies) / self.fitness_width)**2)
    
    def lambda_div(self, state: CellState) -> float:
        k_total = np.sum(state.k)
        # Age-dependent (increases with age) + fitness modifier
        age_factor = 1 - np.exp(-0.1 * state.a)  # Approaches 1 as age increases
        return self.div_rate * age_factor * self._fitness_modifier(k_total)
    
    def lambda_death(self, state: CellState) -> float:
        k_total = np.sum(state.k)
        # Higher death if too many or too few copies
        fitness = self._fitness_modifier(k_total)
        return self.death_rate * (2 - fitness)  # Inverse fitness effect


class FullEcDNAParameters(ModelParameters):
    """
    Full-featured ecDNA model with all dynamics.
    """
    
    def __init__(self,
                 n_environments: int = 1,
                 n_regulatory: int = 2,
                 n_ecdna_species: int = 1,
                 n_phenotype_dims: int = 1,
                 # Division/death
                 base_div_rate: float = 0.05,
                 base_death_rate: float = 0.01,
                 # Regulatory switching
                 reg_switch_rate: float = 0.01,
                 # ecDNA dynamics
                 ecdna_gain_rate: float = 0.001,
                 ecdna_loss_rate: float = 0.0005,
                 # Phenotype dynamics
                 drift_strength: float = -0.1,
                 diffusion_strength: float = 0.05,
                 # Division kernel
                 amp_rate: float = 0.1,
                 loss_prob: float = 0.01,
                 phenotype_inheritance_noise: float = 0.1):
        
        super().__init__(n_environments, n_regulatory, 
                        n_ecdna_species, n_phenotype_dims)
        
        self.base_div_rate = base_div_rate
        self.base_death_rate = base_death_rate
        self.reg_switch_rate = reg_switch_rate
        self.ecdna_gain_rate = ecdna_gain_rate
        self.ecdna_loss_rate = ecdna_loss_rate
        self.drift_strength = drift_strength
        self.diffusion_strength = diffusion_strength
        self.amp_rate = amp_rate
        self.loss_prob = loss_prob
        self.phenotype_noise = phenotype_inheritance_noise
    
    def drift(self, state: CellState) -> np.ndarray:
        """OU process drift toward 0."""
        return self.drift_strength * state.y
    
    def diffusion(self, state: CellState) -> np.ndarray:
        return self.diffusion_strength * np.eye(self.n_pheno)
    
    def lambda_div(self, state: CellState) -> float:
        # Regulatory state affects division
        reg_factor = 1.0 + 0.5 * state.m / max(1, self.n_reg - 1)
        age_factor = 1 - np.exp(-0.1 * state.a)
        return self.base_div_rate * reg_factor * age_factor
    
    def lambda_death(self, state: CellState) -> float:
        return self.base_death_rate
    
    def q_m_rates(self, state: CellState) -> np.ndarray:
        """Uniform switching between regulatory states."""
        rates = np.full(self.n_reg, self.reg_switch_rate / max(1, self.n_reg - 1))
        rates[state.m] = 0.0
        return rates
    
    def mu_gain(self, state: CellState, j: int) -> float:
        return self.ecdna_gain_rate
    
    def mu_loss(self, state: CellState, j: int) -> float:
        return self.ecdna_loss_rate
    
    def sample_amplification(self, state: CellState) -> np.ndarray:
        """Poisson amplification."""
        return np.random.poisson(self.amp_rate * state.k)
    
    def post_segregation_loss_prob(self, state: CellState, j: int,
                                    k_daughter_j: int) -> float:
        return self.loss_prob
    
    def sample_daughter_y(self, parent: CellState, m_daughter: int,
                          k_daughter: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.phenotype_noise, self.n_pheno)
        return parent.y + noise


# =============================================================================
# Simulation Engine
# =============================================================================

class EcDNASimulator:
    """
    Stochastic simulator for ecDNA cell population dynamics.
    Uses hybrid tau-leaping / direct method approach.
    """
    
    def __init__(self, params: ModelParameters, config: SimulationConfig):
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)
    
    def _update_phenotype(self, state: CellState, dt: float) -> None:
        """Euler-Maruyama update for phenotype SDE."""
        drift = self.params.drift(state)
        sigma = self.params.diffusion(state)
        noise = self.rng.standard_normal(self.params.n_pheno)
        state.y = state.y + drift * dt + sigma @ noise * np.sqrt(dt)
    
    def _compute_rates(self, state: CellState) -> Dict[str, Any]:
        """Compute all event rates for a cell."""
        rates = {
            'div': self.params.lambda_div(state),
            'death': self.params.lambda_death(state),
            'm_switch': self.params.q_m_rates(state),
            'e_switch': self.params.omega_e_rates(state),
            'k_gain': np.array([self.params.mu_gain(state, j) 
                               for j in range(self.params.n_ecdna)]),
            'k_loss': np.array([self.params.mu_loss(state, j) * state.k[j]
                               for j in range(self.params.n_ecdna)])
        }
        rates['total'] = (rates['div'] + rates['death'] + 
                         np.sum(rates['m_switch']) + np.sum(rates['e_switch']) +
                         np.sum(rates['k_gain']) + np.sum(rates['k_loss']))
        return rates
    
    def _sample_event(self, rates: Dict[str, Any]) -> str:
        """Sample which event occurs given rates."""
        total = rates['total']
        if total <= 0:
            return 'none'
        
        u = self.rng.random() * total
        cumsum = 0
        
        cumsum += rates['div']
        if u < cumsum:
            return 'div'
        
        cumsum += rates['death']
        if u < cumsum:
            return 'death'
        
        for m_new in range(self.params.n_reg):
            cumsum += rates['m_switch'][m_new]
            if u < cumsum:
                return f'm_switch_{m_new}'
        
        for e_new in range(self.params.n_env):
            cumsum += rates['e_switch'][e_new]
            if u < cumsum:
                return f'e_switch_{e_new}'
        
        for j in range(self.params.n_ecdna):
            cumsum += rates['k_gain'][j]
            if u < cumsum:
                return f'k_gain_{j}'
        
        for j in range(self.params.n_ecdna):
            cumsum += rates['k_loss'][j]
            if u < cumsum:
                return f'k_loss_{j}'
        
        return 'none'
    
    def _divide_cell(self, parent: CellState) -> Tuple[CellState, CellState]:
        """Execute division and generate two daughters."""
        # Step 1: Amplification
        amp = self.params.sample_amplification(parent)
        k_tilde = 2 * parent.k + amp
        
        # Step 2: Binomial segregation
        k1 = np.array([self.rng.binomial(k_tilde[j], 0.5) 
                       for j in range(len(k_tilde))])
        k2 = k_tilde - k1
        
        # Step 3: Post-segregation loss
        daughters_k = []
        for k_r in [k1, k2]:
            k_star = np.zeros_like(k_r)
            for j in range(len(k_r)):
                loss_prob = self.params.post_segregation_loss_prob(parent, j, k_r[j])
                k_star[j] = self.rng.binomial(k_r[j], 1 - loss_prob)
            daughters_k.append(k_star)
        
        # Step 4 & 5: Sample daughter states
        daughters = []
        for k_d in daughters_k:
            m_d = self.params.sample_daughter_m(parent, k_d)
            y_d = self.params.sample_daughter_y(parent, m_d, k_d)
            daughter = CellState(
                e=parent.e, m=m_d, k=k_d, y=y_d, a=0.0
            )
            daughters.append(daughter)
        
        return daughters[0], daughters[1]
    
    def simulate_single_cell(self, initial: CellState, 
                             t_max: Optional[float] = None) -> List[Dict]:
        """
        Simulate a single cell lineage until death or t_max.
        Returns trajectory as list of state snapshots.
        """
        t_max = t_max or self.config.t_max
        dt = self.config.dt
        
        state = initial.copy()
        t = 0.0
        trajectory = [{'t': t, 'state': state.copy()}]
        
        while t < t_max:
            # Update phenotype (SDE)
            self._update_phenotype(state, dt)
            
            # Update age
            state.a += dt
            t += dt
            
            # Compute rates and check for events
            rates = self._compute_rates(state)
            
            # Probability of event in dt (approximate)
            p_event = 1 - np.exp(-rates['total'] * dt)
            
            if self.rng.random() < p_event:
                event = self._sample_event(rates)
                
                if event == 'death':
                    trajectory.append({'t': t, 'state': state.copy(), 'event': 'death'})
                    break
                elif event == 'div':
                    trajectory.append({'t': t, 'state': state.copy(), 'event': 'div'})
                    break
                elif event.startswith('m_switch_'):
                    state.m = int(event.split('_')[-1])
                elif event.startswith('e_switch_'):
                    state.e = int(event.split('_')[-1])
                elif event.startswith('k_gain_'):
                    j = int(event.split('_')[-1])
                    state.k[j] += 1
                elif event.startswith('k_loss_'):
                    j = int(event.split('_')[-1])
                    state.k[j] = max(0, state.k[j] - 1)
            
            trajectory.append({'t': t, 'state': state.copy()})
        
        return trajectory
    
    def simulate_population(self, initial_cells: List[CellState],
                           t_max: Optional[float] = None,
                           record_interval: float = 1.0) -> Dict:
        """
        Simulate a population of cells.
        Returns population snapshots at regular intervals.
        """
        t_max = t_max or self.config.t_max
        dt = self.config.dt
        
        cells = [c.copy() for c in initial_cells]
        t = 0.0
        next_record = 0.0
        
        history = {
            'times': [],
            'population_size': [],
            'mean_k': [],
            'std_k': [],
            'mean_age': [],
            'mean_y': [],
            'k_distribution': [],
            'm_distribution': []
        }
        
        def record_snapshot():
            if not cells:
                return
            history['times'].append(t)
            history['population_size'].append(len(cells))
            
            k_vals = np.array([np.sum(c.k) for c in cells])
            history['mean_k'].append(np.mean(k_vals))
            history['std_k'].append(np.std(k_vals))
            history['mean_age'].append(np.mean([c.a for c in cells]))
            
            y_vals = np.array([c.y for c in cells])
            history['mean_y'].append(np.mean(y_vals, axis=0))
            history['k_distribution'].append(k_vals.copy())
            
            m_counts = np.bincount([c.m for c in cells], minlength=self.params.n_reg)
            history['m_distribution'].append(m_counts / len(cells))
        
        record_snapshot()
        next_record += record_interval
        
        while t < t_max and len(cells) > 0:
            if len(cells) > self.config.max_cells:
                # Subsample to prevent explosion
                indices = self.rng.choice(len(cells), self.config.max_cells, replace=False)
                cells = [cells[i] for i in indices]
                warnings.warn(f"Population capped at {self.config.max_cells}")
            
            new_cells = []
            cells_to_remove = []
            
            for idx, cell in enumerate(cells):
                # Update phenotype
                self._update_phenotype(cell, dt)
                cell.a += dt
                
                # Check events
                rates = self._compute_rates(cell)
                p_event = 1 - np.exp(-rates['total'] * dt)
                
                if self.rng.random() < p_event:
                    event = self._sample_event(rates)
                    
                    if event == 'death':
                        cells_to_remove.append(idx)
                    elif event == 'div':
                        d1, d2 = self._divide_cell(cell)
                        cells_to_remove.append(idx)
                        new_cells.extend([d1, d2])
                    elif event.startswith('m_switch_'):
                        cell.m = int(event.split('_')[-1])
                    elif event.startswith('e_switch_'):
                        cell.e = int(event.split('_')[-1])
                    elif event.startswith('k_gain_'):
                        j = int(event.split('_')[-1])
                        cell.k[j] += 1
                    elif event.startswith('k_loss_'):
                        j = int(event.split('_')[-1])
                        cell.k[j] = max(0, cell.k[j] - 1)
            
            # Remove dead/divided cells (reverse order)
            for idx in sorted(cells_to_remove, reverse=True):
                cells.pop(idx)
            
            # Add new cells
            cells.extend(new_cells)
            
            t += dt
            
            if t >= next_record:
                record_snapshot()
                next_record += record_interval
        
        return history


# =============================================================================
# Parameter Sweep Utilities
# =============================================================================

class ParameterSweep:
    """Utility for running parameter sweeps."""
    
    @staticmethod
    def sweep_1d(param_name: str, param_values: np.ndarray,
                 base_params: Dict, param_class: type,
                 initial_cells: List[CellState],
                 config: SimulationConfig,
                 n_replicates: int = 3,
                 metric_fn: Optional[Callable] = None) -> Dict:
        """
        Sweep over a single parameter.
        
        Args:
            param_name: Parameter to vary
            param_values: Values to test
            base_params: Base parameter dictionary
            param_class: Parameter class to instantiate
            initial_cells: Initial cell population
            config: Simulation config
            n_replicates: Number of replicates per value
            metric_fn: Function(history) -> scalar metric
        
        Returns:
            Dictionary with sweep results
        """
        if metric_fn is None:
            metric_fn = lambda h: h['population_size'][-1] if h['population_size'] else 0
        
        results = {
            'param_name': param_name,
            'param_values': param_values,
            'metrics': [],
            'metrics_mean': [],
            'metrics_std': [],
            'histories': []
        }
        
        for val in param_values:
            params_dict = base_params.copy()
            params_dict[param_name] = val
            
            rep_metrics = []
            rep_histories = []
            
            for rep in range(n_replicates):
                params = param_class(**params_dict)
                rep_config = SimulationConfig(
                    dt=config.dt, t_max=config.t_max, max_cells=config.max_cells,
                    seed=config.seed + rep if config.seed else None
                )
                sim = EcDNASimulator(params, rep_config)
                
                history = sim.simulate_population(
                    [c.copy() for c in initial_cells]
                )
                
                metric = metric_fn(history)
                rep_metrics.append(metric)
                rep_histories.append(history)
            
            results['metrics'].append(rep_metrics)
            results['metrics_mean'].append(np.mean(rep_metrics))
            results['metrics_std'].append(np.std(rep_metrics))
            results['histories'].append(rep_histories)
        
        return results
    
    @staticmethod
    def sweep_2d(param1_name: str, param1_values: np.ndarray,
                 param2_name: str, param2_values: np.ndarray,
                 base_params: Dict, param_class: type,
                 initial_cells: List[CellState],
                 config: SimulationConfig,
                 n_replicates: int = 1,
                 metric_fn: Optional[Callable] = None) -> Dict:
        """2D parameter sweep."""
        if metric_fn is None:
            metric_fn = lambda h: h['population_size'][-1] if h['population_size'] else 0
        
        results = {
            'param1_name': param1_name,
            'param1_values': param1_values,
            'param2_name': param2_name,
            'param2_values': param2_values,
            'metrics': np.zeros((len(param1_values), len(param2_values)))
        }
        
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                params_dict = base_params.copy()
                params_dict[param1_name] = v1
                params_dict[param2_name] = v2
                
                rep_metrics = []
                for rep in range(n_replicates):
                    params = param_class(**params_dict)
                    rep_config = SimulationConfig(
                        dt=config.dt, t_max=config.t_max, max_cells=config.max_cells,
                        seed=config.seed + rep if config.seed else None
                    )
                    sim = EcDNASimulator(params, rep_config)
                    history = sim.simulate_population([c.copy() for c in initial_cells])
                    rep_metrics.append(metric_fn(history))
                
                results['metrics'][i, j] = np.mean(rep_metrics)
        
        return results


# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_growth_rate(history: Dict, start_frac: float = 0.2) -> float:
    """Estimate exponential growth rate from population history."""
    times = np.array(history['times'])
    pop = np.array(history['population_size'])
    
    # Use latter portion of simulation
    start_idx = int(len(times) * start_frac)
    if start_idx >= len(times) - 2:
        return 0.0
    
    t = times[start_idx:]
    n = pop[start_idx:]
    
    # Filter positive values
    mask = n > 0
    if np.sum(mask) < 2:
        return 0.0
    
    # Log-linear regression
    log_n = np.log(n[mask])
    t_masked = t[mask]
    
    if len(t_masked) < 2:
        return 0.0
    
    # Simple slope estimation
    slope = (log_n[-1] - log_n[0]) / (t_masked[-1] - t_masked[0])
    return slope


def compute_extinction_probability(histories: List[Dict]) -> float:
    """Compute extinction probability from replicate histories."""
    n_extinct = sum(1 for h in histories if h['population_size'][-1] == 0)
    return n_extinct / len(histories)


if __name__ == "__main__":
    # Quick test
    params = SimpleEcDNAParameters(div_rate=0.1, death_rate=0.02)
    config = SimulationConfig(dt=0.1, t_max=50, seed=42)
    sim = EcDNASimulator(params, config)
    
    initial = [CellState(k=np.array([10])) for _ in range(10)]
    history = sim.simulate_population(initial)
    
    print(f"Final population: {history['population_size'][-1]}")
    print(f"Mean ecDNA copies: {history['mean_k'][-1]:.2f}")
