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
    compute_extinction_probability_ci,
    compute_extinction_summary,
    compute_survival_curve,
    compute_time_to_threshold,
    OUParameters,
)
from visualization import (
    plot_simulation_summary,
    plot_thinning_acceptance_rate,
    plot_event_channel_composition,
    plot_bound_ratio_over_time,
    plot_survival_curve,
    plot_extinction_time_hist,
    plot_extinction_probability_ci,
    plot_population_fan_chart,
    plot_sweep_1d,
    plot_sweep_2d,
    plot_mean_ecdna_by_species,
    plot_ecdna_heatmaps_by_species,
    plot_k_species_joint,
)
from model_atlas import (
    plot_event_raster,
    plot_channel_stack,
    plot_acceptance_history,
    plot_proposal_histogram,
    plot_waiting_time_hist,
    plot_hazard_vs_age,
    plot_hazard_vs_k,
    plot_hazard_heatmap,
    plot_gain_loss_hazard,
    plot_amplification_distribution,
    plot_segregation_distribution,
    plot_daughter_scatter,
)

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_default_model() -> ModelParameters:
    ou = OUParameters(mean=np.zeros(1), rate=np.ones(1) * 0.5, diffusion=np.ones(1) * 0.2)
    return ModelParameters(ou_params=ou, k_max=np.array([40]), div_rate_max=0.15, death_rate_max=0.05)

def sample_initial_cells(sim: EcDNASimulator, n_cells: int, k_means: np.ndarray):
    k_means = np.atleast_1d(k_means)
    cells = []
    for _ in range(n_cells):
        k = np.array([sim.rng.poisson(mu) for mu in k_means], dtype=int)
        cells.append(CellState(k=k))
    return cells


def example_population():
    params = make_default_model()
    config = SimulationConfig(t_max=30.0, seed=123, record_interval=1.0)
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, 20, np.array([8]))
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
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, 5, np.array([5]))
    sweep = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=np.linspace(0.05, 0.2, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=3,
        metric_fn=compute_growth_rate,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_sweep_1d(sweep, ax=ax, ylabel="Growth Rate")
    fig.savefig(OUTPUT_DIR / "example_sweep_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sweep_ext = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=np.linspace(0.05, 0.2, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=5,
        metric_fn=lambda h: 1.0 if any(n == 0 for n in h.get("population_size", [])) else 0.0,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_sweep_1d(sweep_ext, ax=ax, ylabel="Extinction Probability")
    fig.savefig(OUTPUT_DIR / "example_sweep_extinction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sweep_k = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=np.linspace(0.05, 0.2, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=3,
        metric_fn=lambda h: h["mean_k"][-1] if h.get("mean_k") else 0.0,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_sweep_1d(sweep_k, ax=ax, ylabel="Final Mean ecDNA")
    fig.savefig(OUTPUT_DIR / "example_sweep_mean_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sweep_2d = ParameterSweep.sweep_2d(
        param1_name="div_rate_max",
        param1_values=np.linspace(0.08, 0.2, 4),
        param2_name="death_rate_max",
        param2_values=np.linspace(0.03, 0.12, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=3,
        metric_fn=compute_growth_rate,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_sweep_2d(sweep_2d, ax=ax, log_scale=False)
    fig.savefig(OUTPUT_DIR / "example_sweep_2d_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    threshold = 40
    sweep_t = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=np.linspace(0.05, 0.2, 4),
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=3,
        metric_fn=lambda h: compute_time_to_threshold(h, threshold) or config.t_max,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_sweep_1d(sweep_t, ax=ax, ylabel=f"Time to Pop>={threshold}")
    fig.savefig(OUTPUT_DIR / "example_sweep_time_to_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return sweep


def example_thinning_diagnostics():
    ou = OUParameters(mean=np.zeros(1), rate=np.ones(1) * 0.4, diffusion=np.ones(1) * 0.2)
    params = ModelParameters(
        ou_params=ou,
        k_max=np.array([40]),
        div_rate_max=0.12,
        death_rate_max=0.05,
        n_reg=3,
        n_env=2,
        reg_switch_max=0.05,
        env_switch_max=0.03,
    )
    config = SimulationConfig(t_max=20.0, seed=33, record_interval=1.0, check_bounds=True)
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, 20, np.array([6]))
    history = sim.simulate_population(initial_cells)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plot_thinning_acceptance_rate(sim.event_log, ax=axes[0])
    plot_event_channel_composition(sim.event_log, ax=axes[1])
    plot_bound_ratio_over_time(sim.event_log, ax=axes[2])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_thinning_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return history, sim.event_log


def example_extinction_analysis():
    params = make_default_model()
    params.death_rate_max = 0.2
    config = SimulationConfig(t_max=25.0, seed=101, record_interval=1.0)
    histories = []
    for rep in range(20):
        rep_config = SimulationConfig(
            t_max=config.t_max,
            record_interval=config.record_interval,
            seed=(config.seed + rep) if config.seed is not None else None,
            max_cells=config.max_cells,
        )
        sim = EcDNASimulator(params, rep_config)
        initial_cells = sample_initial_cells(sim, 10, np.array([4]))
        histories.append(sim.simulate_population(initial_cells))

    summary = compute_extinction_summary(histories)
    times, survival = compute_survival_curve(summary)
    p, ci = compute_extinction_probability_ci(histories)
    ext_times = [t for t in summary["t_extinction"] if t is not None]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plot_survival_curve(times, survival, ax=axes[0])
    plot_extinction_time_hist(ext_times, ax=axes[1])
    plot_extinction_probability_ci(p, ci, ax=axes[2])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_extinction_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Extinction probability: {p:.2f} (CI: {ci[0]:.2f}-{ci[1]:.2f})")
    return histories


def example_replicate_variability():
    params = make_default_model()
    config = SimulationConfig(t_max=20.0, seed=202, record_interval=1.0)
    histories = []
    for rep in range(15):
        rep_config = SimulationConfig(
            t_max=config.t_max,
            record_interval=config.record_interval,
            seed=(config.seed + rep) if config.seed is not None else None,
            max_cells=config.max_cells,
        )
        sim = EcDNASimulator(params, rep_config)
        initial_cells = sample_initial_cells(sim, 10, np.array([6]))
        histories.append(sim.simulate_population(initial_cells))
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_population_fan_chart(histories, ax=ax, ci=0.8)
    fig.savefig(OUTPUT_DIR / "example_replicate_variability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return histories


def example_multi_ecdna():
    ou = OUParameters(mean=np.zeros(1), rate=np.ones(1) * 0.4, diffusion=np.ones(1) * 0.2)
    params = ModelParameters(
        ou_params=ou,
        n_ecdna=2,
        k_max=np.array([30, 20]),
        div_rate_max=0.12,
        death_rate_max=0.04,
        gain_rate_max=0.015,
        loss_rate_max=0.02,
    )
    config = SimulationConfig(t_max=20.0, seed=77, record_interval=1.0)
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, 20, np.array([6, 3]))
    history = sim.simulate_population(initial_cells)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_mean_ecdna_by_species(history, ax=ax, show_std=True)
    fig.savefig(OUTPUT_DIR / "example_multi_ecdna_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_ecdna_heatmaps_by_species(history, max_k=25, time_resolution=40)
    if fig is not None:
        fig.savefig(OUTPUT_DIR / "example_multi_ecdna_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_k_species_joint(history, species=(0, 1), ax=ax)
    fig.savefig(OUTPUT_DIR / "example_multi_ecdna_joint.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return history


def example_model_atlas():
    params = make_default_model()
    params.n_reg = 3
    params.n_env = 2
    params.reg_switch_max = 0.05
    params.env_switch_max = 0.03
    params.gain_rate_max = 0.015
    params.loss_rate_max = 0.02
    config = SimulationConfig(t_max=12.0, seed=301, record_interval=1.0, check_bounds=True)
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, 15, np.array([6]))
    sim.simulate_population(initial_cells)
    event_log = sim.event_log

    fig, ax = plt.subplots(figsize=(7, 2.5))
    plot_event_raster(event_log, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_event_raster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    plot_channel_stack(event_log, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_channel_stack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    plot_acceptance_history(event_log, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_acceptance_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_proposal_histogram(event_log, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_proposal_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_waiting_time_hist(event_log, ax=ax, bins=20)
    fig.savefig(OUTPUT_DIR / "atlas_waiting_time_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_hazard_vs_age(params, k_value=[10], ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_hazard_vs_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_hazard_vs_k(params, k_range=np.arange(0, 41), ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_hazard_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_hazard_heatmap(params, k_range=np.arange(0, 41), a_grid=np.linspace(0, 5, 40), kind="div", ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_hazard_heatmap_div.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_hazard_heatmap(params, k_range=np.arange(0, 41), a_grid=np.linspace(0, 5, 40), kind="death", ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_hazard_heatmap_death.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_gain_loss_hazard(params, k_range=np.arange(0, 41), ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_gain_loss_hazard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    parent = CellState(k=np.array([10]), y=params.ou_params.mean.copy())
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_amplification_distribution(params, parent, n_samples=250, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_amplification_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_segregation_distribution(params, parent, n_samples=250, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_segregation_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_daughter_scatter(params, parent, n_samples=250, ax=ax)
    fig.savefig(OUTPUT_DIR / "atlas_daughter_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup = example_population()
    example_lineage()
    example_sweep()
    example_thinning_diagnostics()
    example_extinction_analysis()
    example_replicate_variability()
    example_multi_ecdna()
    example_model_atlas()
    print("Extinction prob: ", compute_extinction_probability([setup]))
