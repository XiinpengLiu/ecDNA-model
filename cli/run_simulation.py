"""
Main entry point for the ecDNA simulation program.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ecdna_model import config as cfg
from ecdna_model.analysis.plotting import (
    plot_event_summary,
    plot_lineage_state_paths,
    plot_observation_proxies,
    plot_results,
)
from ecdna_model.analysis.treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts
from ecdna_model.core.simulation import run_simulation


def main() -> None:
    output_dir = Path("results_v4")
    output_dir.mkdir(exist_ok=True)

    base_params = cfg.DEFAULT_MODEL_PARAMETERS
    simulation_params = replace(
        base_params.simulation,
        time_unit="week",
        t_max=10.0,
        record_times=tuple(float(week) for week in range(1, 11)),
        target_population_size=None,
        max_pop_size=200000,
        record_full_snapshots=True,
        record_events=True,
    )
    params = replace(base_params, simulation=simulation_params)

    result = run_simulation(
        params=params,
        n_init=80,
        seed=42,
        verbose=True,
    )

    growth_rate = compute_growth_rate(result)
    copy_trends = compute_bulk_copy_trends(result)
    terminal_counts = compute_terminal_event_counts(result)

    print("=" * 64)
    print("ecDNA simulation summary")
    print("=" * 64)
    print(
        "Simulation limits: "
        f"time_unit={params.simulation.time_unit}, "
        f"t_max={params.simulation.t_max:.1f}, "
        f"record_times={params.simulation.record_times}, "
        f"target_population_size={params.simulation.target_population_size}, "
        f"max_pop_size={params.simulation.max_pop_size}"
    )
    print(f"Stop reason: {result.stop_reason} at t={result.stop_time:.2f}")
    print(f"Final population size: {result.population_sizes[-1]}")
    print(f"Estimated late growth rate: {growth_rate:.4f}")
    print(
        "Bulk ecDNA trends: "
        f"MYC={copy_trends['MYC']:.4f}, "
        f"CDK4={copy_trends['CDK4']:.4f}, "
        f"PDGFRA={copy_trends['PDGFRA']:.4f}"
    )
    print(
        "Terminal events: "
        f"division={terminal_counts['division']}, "
        f"death={terminal_counts['death']}"
    )

    result.save_as_csv(output_dir / "simulation_data")
    plot_results(result, save_path=output_dir / "simulation_summary.png")
    plot_observation_proxies(result, save_path=output_dir / "observation_proxies.png")
    plot_event_summary(result, save_path=output_dir / "event_summary.png")
    plot_lineage_state_paths(result, save_path=output_dir / "lineage_state_paths.png")

    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
