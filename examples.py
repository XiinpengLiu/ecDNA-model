"""
Examples for the event-driven ecDNA simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ecdna_model import (
    CellState,
    SimulationConfig,
    EcDNASimulator,
    ModelParameters,
    ParameterSweep,
    compute_growth_rate,
    compute_extinction_probability,
    OUParameters,
)
from visualization import plot_simulation_summary

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_default_model() -> ModelParameters:
    ou = OUParameters(mean=np.zeros(1), rate=np.ones(1) * 0.5, diffusion=np.ones(1) * 0.2)
    return ModelParameters(ou_params=ou, k_max=np.array([40]), div_rate_max=0.15, death_rate_max=0.05)


def example_population():
    params = make_default_model()
    config = SimulationConfig(t_max=30.0, seed=123, record_interval=1.0)
    initial_cells = [CellState(k=np.array([np.random.poisson(8)])) for _ in range(20)]
    sim = EcDNASimulator(params, config)
    history = sim.simulate_population(initial_cells)
    fig = plot_simulation_summary(history)
    fig.suptitle("Event-driven ecDNA population", y=1.02)
    fig.savefig(OUTPUT_DIR / "example_population.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Final population: {history['population_size'][-1] if history['population_size'] else 0}")
    print(f"Growth rate: {compute_growth_rate(history):.4f}")
    return history


def example_lineage():
    params = make_default_model()
    config = SimulationConfig(t_max=15.0, seed=7, record_interval=1.0)
    sim = EcDNASimulator(params, config)
    initial = CellState(k=np.array([10]))
    trajectory = sim.simulate_single_cell_lineage(initial)
    times = [s["t"] for s in trajectory]
    k_vals = [snap["state"].k[0] for snap in trajectory]
    plt.figure(figsize=(6, 3))
    plt.plot(times, k_vals, "-o")
    plt.xlabel("Time")
    plt.ylabel("ecDNA copies")
    plt.title("Single lineage (thinning-driven)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "example_lineage.png", dpi=150)
    plt.close()
    print("Lineage events:", [snap.get("event", "continuation") for snap in trajectory])
    return trajectory


def example_sweep():
    params = make_default_model()
    config = SimulationConfig(t_max=10.0, seed=21, record_interval=1.0)
    initial_cells = [CellState(k=np.array([5])) for _ in range(5)]
    sweep = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=np.linspace(0.05, 0.2, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=2,
        metric_fn=lambda h: h["population_size"][-1] if h["population_size"] else 0,
    )
    print("Sweep means:", sweep["metrics_mean"])
    return sweep


if __name__ == "__main__":
    setup = example_population()
    example_lineage()
    example_sweep()
    print("Extinction prob placeholder: ", compute_extinction_probability([setup]))
