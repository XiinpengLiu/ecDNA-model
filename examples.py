"""
Examples for the event-driven ecDNA simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config as global_config

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

OUTPUT_DIR = Path(global_config.EXAMPLES["output_dir"])
OUTPUT_DIR.mkdir(exist_ok=True)

def make_default_model() -> ModelParameters:
    cfg = global_config.EXAMPLES["default_model"]
    ou = OUParameters(mean=cfg["ou_mean"].copy(), rate=cfg["ou_rate"].copy(), diffusion=cfg["ou_diffusion"].copy())
    return ModelParameters(
        ou_params=ou,
        k_max=cfg["k_max"].copy(),
        div_rate_max=cfg["div_rate_max"],
        death_rate_max=cfg["death_rate_max"],
    )

def sample_initial_cells(sim: EcDNASimulator, n_cells: int, k_means: np.ndarray):
    k_means = np.atleast_1d(k_means)
    cells = []
    for _ in range(n_cells):
        k = np.array([sim.rng.poisson(mu) for mu in k_means], dtype=int)
        cells.append(CellState(k=k))
    return cells


def example_population():
    cfg = global_config.EXAMPLES["population"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
    history = sim.simulate_population(initial_cells)
    fig = plot_simulation_summary(history)
    fig.suptitle("Event-driven ecDNA population", y=cfg["summary_title_y"])
    fig.savefig(OUTPUT_DIR / "example_population.png", dpi=plot_cfg["dpi"], bbox_inches=plot_cfg["bbox_tight"])
    plt.close(fig)
    print(f"Final population: {history['population_size'][-1] if history['population_size'] else 0}")
    print(f"Growth rate: {compute_growth_rate(history):.4f}")
    return history


def example_lineage():
    cfg = global_config.EXAMPLES["lineage"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    sim = EcDNASimulator(params, config)
    initial = CellState(k=cfg["initial_k"].copy())
    trajectory = sim.simulate_single_cell_lineage(initial)
    times = [s["t"] for s in trajectory]
    k_vals = [snap["state"].k[0] for snap in trajectory]
    plt.figure(figsize=cfg["figsize"])
    plt.plot(times, k_vals, "-o")
    plt.xlabel("Time")
    plt.ylabel("ecDNA copies")
    plt.title("Single lineage (thinning-driven)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "example_lineage.png", dpi=plot_cfg["dpi"])
    plt.close()
    print("Lineage events:", [snap.get("event", "continuation") for snap in trajectory])
    return trajectory


def example_sweep():
    cfg = global_config.EXAMPLES["sweep"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
    sweep = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=cfg["param_values"],
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=cfg["n_replicates_growth"],
        metric_fn=compute_growth_rate,
    )
    fig, ax = plt.subplots(figsize=cfg["figsize_1d"])
    plot_sweep_1d(sweep, ax=ax, ylabel="Growth Rate")
    fig.savefig(OUTPUT_DIR / "example_sweep_growth.png", dpi=plot_cfg["dpi"], bbox_inches=plot_cfg["bbox_tight"])
    plt.close(fig)

    sweep_ext = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=cfg["param_values"],
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=cfg["n_replicates_extinction"],
        metric_fn=lambda h: 1.0 if any(n == 0 for n in h.get("population_size", [])) else 0.0,
    )
    fig, ax = plt.subplots(figsize=cfg["figsize_1d"])
    plot_sweep_1d(sweep_ext, ax=ax, ylabel="Extinction Probability")
    fig.savefig(OUTPUT_DIR / "example_sweep_extinction.png", dpi=plot_cfg["dpi"], bbox_inches=plot_cfg["bbox_tight"])
    plt.close(fig)

    sweep_k = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=cfg["param_values"],
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=cfg["n_replicates_k"],
        metric_fn=lambda h: h["mean_k"][-1] if h.get("mean_k") else 0.0,
    )
    fig, ax = plt.subplots(figsize=cfg["figsize_1d"])
    plot_sweep_1d(sweep_k, ax=ax, ylabel="Final Mean ecDNA")
    fig.savefig(OUTPUT_DIR / "example_sweep_mean_k.png", dpi=plot_cfg["dpi"], bbox_inches=plot_cfg["bbox_tight"])
    plt.close(fig)

    sweep_2d = ParameterSweep.sweep_2d(
        param1_name="div_rate_max",
        param1_values=cfg["param1_values"],
        param2_name="death_rate_max",
        param2_values=cfg["param2_values"],
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=cfg["n_replicates_2d"],
        metric_fn=compute_growth_rate,
    )
    fig, ax = plt.subplots(figsize=cfg["figsize_2d"])
    plot_sweep_2d(sweep_2d, ax=ax, log_scale=False)
    fig.savefig(OUTPUT_DIR / "example_sweep_2d_growth.png", dpi=plot_cfg["dpi"], bbox_inches=plot_cfg["bbox_tight"])
    plt.close(fig)

    threshold = cfg["threshold"]
    sweep_t = ParameterSweep.sweep_1d(
        param_name="div_rate_max",
        param_values=cfg["param_values"],
        base_params=params.__dict__,
        param_class=ModelParameters,
        initial_cells=initial_cells,
        config=config,
        n_replicates=cfg["n_replicates_threshold"],
        metric_fn=lambda h: compute_time_to_threshold(h, threshold) or config.t_max,
    )
    fig, ax = plt.subplots(figsize=cfg["figsize_1d"])
    plot_sweep_1d(sweep_t, ax=ax, ylabel=f"Time to Pop>={threshold}")
    fig.savefig(
        OUTPUT_DIR / "example_sweep_time_to_threshold.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)
    return sweep


def example_thinning_diagnostics():
    cfg = global_config.EXAMPLES["thinning"]
    plot_cfg = global_config.PLOT_DEFAULTS
    ou = OUParameters(mean=cfg["ou_mean"].copy(), rate=cfg["ou_rate"].copy(), diffusion=cfg["ou_diffusion"].copy())
    params = ModelParameters(
        ou_params=ou,
        k_max=cfg["k_max"].copy(),
        div_rate_max=cfg["div_rate_max"],
        death_rate_max=cfg["death_rate_max"],
        n_reg=cfg["n_reg"],
        n_env=cfg["n_env"],
        reg_switch_max=cfg["reg_switch_max"],
        env_switch_max=cfg["env_switch_max"],
    )
    config = SimulationConfig(
        t_max=cfg["t_max"],
        seed=cfg["seed"],
        record_interval=cfg["record_interval"],
        check_bounds=cfg["check_bounds"],
    )
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
    history = sim.simulate_population(initial_cells)

    fig, axes = plt.subplots(1, 3, figsize=cfg["figsize"])
    plot_thinning_acceptance_rate(sim.event_log, ax=axes[0])
    plot_event_channel_composition(sim.event_log, ax=axes[1])
    plot_bound_ratio_over_time(sim.event_log, ax=axes[2])
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "example_thinning_diagnostics.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)
    return history, sim.event_log


def example_extinction_analysis():
    cfg = global_config.EXAMPLES["extinction"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    params.death_rate_max = cfg["death_rate_max"]
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    histories = []
    for rep in range(cfg["n_replicates"]):
        rep_config = SimulationConfig(
            t_max=config.t_max,
            record_interval=config.record_interval,
            seed=(config.seed + rep) if config.seed is not None else None,
            max_cells=config.max_cells,
        )
        sim = EcDNASimulator(params, rep_config)
        initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
        histories.append(sim.simulate_population(initial_cells))

    summary = compute_extinction_summary(histories)
    times, survival = compute_survival_curve(summary)
    p, ci = compute_extinction_probability_ci(histories)
    ext_times = [t for t in summary["t_extinction"] if t is not None]

    fig, axes = plt.subplots(1, 3, figsize=cfg["figsize"])
    plot_survival_curve(times, survival, ax=axes[0])
    plot_extinction_time_hist(ext_times, ax=axes[1])
    plot_extinction_probability_ci(p, ci, ax=axes[2])
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "example_extinction_analysis.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)
    print(f"Extinction probability: {p:.2f} (CI: {ci[0]:.2f}-{ci[1]:.2f})")
    return histories


def example_replicate_variability():
    cfg = global_config.EXAMPLES["replicate"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    histories = []
    for rep in range(cfg["n_replicates"]):
        rep_config = SimulationConfig(
            t_max=config.t_max,
            record_interval=config.record_interval,
            seed=(config.seed + rep) if config.seed is not None else None,
            max_cells=config.max_cells,
        )
        sim = EcDNASimulator(params, rep_config)
        initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
        histories.append(sim.simulate_population(initial_cells))
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    plot_population_fan_chart(histories, ax=ax, ci=cfg["ci"])
    fig.savefig(
        OUTPUT_DIR / "example_replicate_variability.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)
    return histories


def example_multi_ecdna():
    cfg = global_config.EXAMPLES["multi_ecdna"]
    plot_cfg = global_config.PLOT_DEFAULTS
    ou = OUParameters(mean=cfg["ou_mean"].copy(), rate=cfg["ou_rate"].copy(), diffusion=cfg["ou_diffusion"].copy())
    params = ModelParameters(
        ou_params=ou,
        n_ecdna=cfg["n_ecdna"],
        k_max=cfg["k_max"].copy(),
        div_rate_max=cfg["div_rate_max"],
        death_rate_max=cfg["death_rate_max"],
        gain_rate_max=cfg["gain_rate_max"],
        loss_rate_max=cfg["loss_rate_max"],
    )
    config = SimulationConfig(t_max=cfg["t_max"], seed=cfg["seed"], record_interval=cfg["record_interval"])
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
    history = sim.simulate_population(initial_cells)

    fig, ax = plt.subplots(figsize=cfg["figsize_mean"])
    plot_mean_ecdna_by_species(history, ax=ax, show_std=True)
    fig.savefig(
        OUTPUT_DIR / "example_multi_ecdna_mean.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig = plot_ecdna_heatmaps_by_species(
        history,
        max_k=cfg["heatmap_max_k"],
        time_resolution=cfg["heatmap_time_resolution"],
    )
    if fig is not None:
        fig.savefig(
            OUTPUT_DIR / "example_multi_ecdna_heatmaps.png",
            dpi=plot_cfg["dpi"],
            bbox_inches=plot_cfg["bbox_tight"],
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_joint"])
    plot_k_species_joint(history, species=(0, 1), ax=ax)
    fig.savefig(
        OUTPUT_DIR / "example_multi_ecdna_joint.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)
    return history


def example_model_atlas():
    cfg = global_config.EXAMPLES["atlas"]
    plot_cfg = global_config.PLOT_DEFAULTS
    params = make_default_model()
    params.n_reg = cfg["n_reg"]
    params.n_env = cfg["n_env"]
    params.reg_switch_max = cfg["reg_switch_max"]
    params.env_switch_max = cfg["env_switch_max"]
    params.gain_rate_max = cfg["gain_rate_max"]
    params.loss_rate_max = cfg["loss_rate_max"]
    config = SimulationConfig(
        t_max=cfg["t_max"],
        seed=cfg["seed"],
        record_interval=cfg["record_interval"],
        check_bounds=cfg["check_bounds"],
    )
    sim = EcDNASimulator(params, config)
    initial_cells = sample_initial_cells(sim, cfg["n_cells"], cfg["k_means"])
    sim.simulate_population(initial_cells)
    event_log = sim.event_log

    fig, ax = plt.subplots(figsize=cfg["figsize_event_raster"])
    plot_event_raster(event_log, ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_event_raster.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_channel_stack"])
    plot_channel_stack(event_log, ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_channel_stack.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_acceptance_history"])
    plot_acceptance_history(event_log, ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_acceptance_history.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_proposal_hist"])
    plot_proposal_histogram(event_log, ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_proposal_hist.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_waiting_time"])
    plot_waiting_time_hist(event_log, ax=ax, bins=cfg["waiting_time_bins"])
    fig.savefig(
        OUTPUT_DIR / "atlas_waiting_time_hist.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_hazard_vs_age"])
    plot_hazard_vs_age(params, k_value=cfg["hazard_k_value"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_hazard_vs_age.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_hazard_vs_k"])
    plot_hazard_vs_k(params, k_range=cfg["hazard_k_range"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_hazard_vs_k.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_hazard_heatmap"])
    plot_hazard_heatmap(
        params,
        k_range=cfg["hazard_k_range"],
        a_grid=cfg["hazard_age_grid"],
        kind="div",
        ax=ax,
    )
    fig.savefig(
        OUTPUT_DIR / "atlas_hazard_heatmap_div.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_hazard_heatmap"])
    plot_hazard_heatmap(
        params,
        k_range=cfg["hazard_k_range"],
        a_grid=cfg["hazard_age_grid"],
        kind="death",
        ax=ax,
    )
    fig.savefig(
        OUTPUT_DIR / "atlas_hazard_heatmap_death.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_gain_loss"])
    plot_gain_loss_hazard(params, k_range=cfg["hazard_k_range"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_gain_loss_hazard.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    parent = CellState(k=cfg["parent_k"].copy(), y=params.ou_params.mean.copy())
    fig, ax = plt.subplots(figsize=cfg["figsize_amplification"])
    plot_amplification_distribution(params, parent, n_samples=cfg["draw_samples"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_amplification_dist.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_segregation"])
    plot_segregation_distribution(params, parent, n_samples=cfg["draw_samples"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_segregation_dist.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["figsize_daughter_scatter"])
    plot_daughter_scatter(params, parent, n_samples=cfg["draw_samples"], ax=ax)
    fig.savefig(
        OUTPUT_DIR / "atlas_daughter_scatter.png",
        dpi=plot_cfg["dpi"],
        bbox_inches=plot_cfg["bbox_tight"],
    )
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
