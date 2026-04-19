"""
Main entry point for the ecDNA v4 simulation program.
"""

from pathlib import Path

from plotting import plot_event_summary, plot_lineage_state_paths, plot_observation_proxies, plot_results
from simulation import run_simulation
from treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts


def main() -> None:
    output_dir = Path("results_v4")
    output_dir.mkdir(exist_ok=True)

    t_max = 72000
    target_population_size = 2000
    max_pop_size = 200000
    record_interval = 1.0

    result = run_simulation(
        t_max=t_max,
        n_init=80,
        record_interval=record_interval,
        target_population_size=target_population_size,
        max_pop_size=max_pop_size,
        seed=42,
        verbose=True,
    )

    growth_rate = compute_growth_rate(result)
    copy_trends = compute_bulk_copy_trends(result)
    terminal_counts = compute_terminal_event_counts(result)

    print("=" * 64)
    print("ecDNA v4 simulation summary")
    print("=" * 64)
    print(
        "Simulation limits: "
        f"t_max={t_max:.1f}, "
        f"record_interval={record_interval:.1f}, "
        f"target_population_size={target_population_size}, "
        f"max_pop_size={max_pop_size}"
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
