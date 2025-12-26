"""
ecDNA Copy-Number Kinetics Model - Treatment / Intervention Module
===================================================================
Implements Section 8: In silico trials with time-dependent interventions.
"""

import numpy as np
from typing import Dict, Callable, List, Optional
from dataclasses import dataclass
from simulation import OgataThinningSimulator, SimulationResult, run_simulation
from cell import CellPopulation
import config as cfg


@dataclass
class TreatmentProtocol:
    """
    Defines a treatment schedule for in silico trials.
    
    Attributes:
        name: Protocol name
        drug_schedules: Dict mapping drug name to concentration function u(t)
        env_schedule: Optional function E(t) for environment switching
        duration: Total trial duration
    """
    name: str
    drug_schedules: Dict[str, Callable[[float], float]]
    env_schedule: Optional[Callable[[float], int]] = None
    duration: float = 100.0


# =============================================================================
# Pre-defined Treatment Protocols
# =============================================================================

def constant_dose(drug_name: str, concentration: float, start: float = 0, end: float = np.inf):
    """Create constant-dose schedule."""
    def schedule(t):
        return concentration if start <= t < end else 0.0
    return {drug_name: schedule}


def pulsed_dose(drug_name: str, concentration: float, 
                on_duration: float, off_duration: float, 
                n_cycles: int = 5, start: float = 0):
    """Create pulsed (on-off) schedule."""
    cycle_length = on_duration + off_duration
    
    def schedule(t):
        if t < start:
            return 0.0
        t_rel = (t - start) % cycle_length
        cycle_num = int((t - start) / cycle_length)
        if cycle_num >= n_cycles:
            return 0.0
        return concentration if t_rel < on_duration else 0.0
    
    return {drug_name: schedule}


def ramped_dose(drug_name: str, max_conc: float, ramp_duration: float, start: float = 0):
    """Create ramped (escalating) schedule."""
    def schedule(t):
        if t < start:
            return 0.0
        t_rel = t - start
        if t_rel < ramp_duration:
            return max_conc * (t_rel / ramp_duration)
        return max_conc
    
    return {drug_name: schedule}


def combination_therapy(schedules: List[Dict[str, Callable]]) -> Dict[str, Callable]:
    """Combine multiple drug schedules."""
    combined = {}
    for sched in schedules:
        combined.update(sched)
    return combined


# =============================================================================
# Pre-built Protocols
# =============================================================================

PROTOCOLS = {
    "untreated": TreatmentProtocol(
        name="Untreated control",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 0.0,
            "senolytic": lambda t: 0.0,
            "ecdna_destabilizer": lambda t: 0.0,
        },
        duration=100.0
    ),
    
    "cdk_inhibitor_continuous": TreatmentProtocol(
        name="CDK inhibitor continuous",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 1.0 if t >= 10 else 0.0,
            "senolytic": lambda t: 0.0,
            "ecdna_destabilizer": lambda t: 0.0,
        },
        duration=100.0
    ),
    
    "senolytic_late": TreatmentProtocol(
        name="Senolytic at day 50",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 0.0,
            "senolytic": lambda t: 1.0 if t >= 50 else 0.0,
            "ecdna_destabilizer": lambda t: 0.0,
        },
        duration=100.0
    ),
    
    "ecdna_targeting": TreatmentProtocol(
        name="ecDNA destabilizer",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 0.0,
            "senolytic": lambda t: 0.0,
            "ecdna_destabilizer": lambda t: 1.5 if t >= 10 else 0.0,
        },
        duration=100.0
    ),
    
    "combination_sequential": TreatmentProtocol(
        name="CDK then senolytic",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 1.0 if 10 <= t < 50 else 0.0,
            "senolytic": lambda t: 1.0 if t >= 50 else 0.0,
            "ecdna_destabilizer": lambda t: 0.0,
        },
        duration=100.0
    ),
    
    "treatment_with_env": TreatmentProtocol(
        name="Treatment phase protocol",
        drug_schedules={
            "cell_cycle_inhibitor": lambda t: 1.0 if t >= 20 else 0.0,
            "senolytic": lambda t: 0.0,
            "ecdna_destabilizer": lambda t: 0.5 if t >= 20 else 0.0,
        },
        env_schedule=lambda t: 1 if t >= 20 else 0,  # Switch to treatment env at t=20
        duration=100.0
    ),
}


# =============================================================================
# In Silico Trial Runner
# =============================================================================

class InSilicoTrial:
    """
    Run in silico treatment trials with replicates.
    """
    
    def __init__(self, base_seed: int = None):
        self.base_seed = base_seed or cfg.RANDOM_SEED
        self.results = {}
    
    def run_protocol(self, 
                     protocol: TreatmentProtocol,
                     n_replicates: int = 1,
                     n_init: int = None,
                     verbose: bool = True) -> List[SimulationResult]:
        """
        Run a treatment protocol with multiple replicates.
        
        Args:
            protocol: TreatmentProtocol to run
            n_replicates: Number of replicate simulations
            n_init: Initial population size
            verbose: Print progress
            
        Returns:
            List of SimulationResult for each replicate
        """
        results = []
        
        for rep in range(n_replicates):
            seed = self.base_seed + rep
            
            if verbose:
                print(f"\n=== {protocol.name} - Replicate {rep+1}/{n_replicates} ===")
            
            result = run_simulation(
                t_max=protocol.duration,
                n_init=n_init or cfg.N_INIT,
                drug_schedule=protocol.drug_schedules,
                env_schedule=protocol.env_schedule,  # Now passes env_schedule
                seed=seed,
                verbose=verbose
            )
            results.append(result)
        
        self.results[protocol.name] = results
        return results
    
    def compare_protocols(self, 
                          protocol_names: List[str],
                          n_replicates: int = 3,
                          n_init: int = None,
                          verbose: bool = True) -> Dict[str, List[SimulationResult]]:
        """
        Compare multiple protocols.
        
        Args:
            protocol_names: List of protocol names from PROTOCOLS
            n_replicates: Number of replicates per protocol
            n_init: Initial population size
            verbose: Print progress
            
        Returns:
            Dict mapping protocol name to list of results
        """
        all_results = {}
        
        for name in protocol_names:
            if name not in PROTOCOLS:
                print(f"Warning: Protocol '{name}' not found, skipping")
                continue
            
            protocol = PROTOCOLS[name]
            results = self.run_protocol(protocol, n_replicates, n_init, verbose)
            all_results[name] = results
        
        return all_results
    
    def summarize_results(self, protocol_name: str) -> Dict:
        """
        Summarize results for a protocol across replicates.
        """
        if protocol_name not in self.results:
            return {}
        
        results = self.results[protocol_name]
        
        # Extract final values
        final_pops = [r.population_sizes[-1] for r in results]
        final_ecdna = [r.ecdna_means[-1] for r in results if r.ecdna_means]
        
        return {
            "n_replicates": len(results),
            "final_pop_mean": np.mean(final_pops),
            "final_pop_std": np.std(final_pops),
            "final_ecdna_mean": np.mean(final_ecdna) if final_ecdna else 0,
            "final_ecdna_std": np.std(final_ecdna) if final_ecdna else 0,
        }


# =============================================================================
# Response Metrics
# =============================================================================

def compute_growth_rate(result: SimulationResult, window: float = 10.0) -> float:
    """
    Compute exponential growth rate from population trajectory.
    """
    times = np.array(result.times)
    pops = np.array(result.population_sizes)
    
    # Use late window
    mask = times >= (times[-1] - window)
    if np.sum(mask) < 2:
        return 0.0
    
    t_win = times[mask]
    p_win = pops[mask]
    
    # Log-linear fit
    log_p = np.log(p_win + 1)
    coeffs = np.polyfit(t_win, log_p, 1)
    
    return coeffs[0]  # slope = growth rate


def compute_ecdna_dynamics(result: SimulationResult) -> Dict:
    """
    Compute ecDNA dynamics metrics.
    """
    times = np.array(result.times)
    means = np.array(result.ecdna_means)
    
    if len(means) < 2:
        return {"ecdna_trend": 0, "ecdna_final": 0}
    
    # Linear trend
    coeffs = np.polyfit(times, means, 1)
    
    return {
        "ecdna_trend": coeffs[0],
        "ecdna_final": means[-1],
        "ecdna_max": np.max(means),
    }


def compute_sister_correlation_stats(result: SimulationResult) -> Dict:
    """
    Compute sister correlation statistics.
    """
    corrs = result.sister_correlations
    
    if not corrs:
        return {"sister_corr_mean": 0, "sister_corr_std": 0}
    
    return {
        "sister_corr_mean": np.mean(corrs),
        "sister_corr_std": np.std(corrs),
        "n_divisions": len(corrs),
    }
