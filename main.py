"""
ecDNA Copy-Number Kinetics Model - Main Script
Run simulations and generate results.
"""

from pathlib import Path

# Import model components
import config as cfg
from cell import Cell, CellPopulation
from simulation import run_simulation, OgataThinningSimulator
from treatment import (
    InSilicoTrial, PROTOCOLS,
    compute_growth_rate, compute_ecdna_dynamics, compute_sister_correlation_stats
)
from plotting import (
    plot_results, compare_treatments, 
    plot_ecdna_distribution_evolution, plot_heterogeneity_metrics,
    plot_ecdna_positive_fraction, plot_lineage_tree,
    plot_muller_ecdna, plot_muller_comparison, plot_fitness_landscape,
    plot_lineage_state_trajectory, plot_event_summary,
    plot_grouped_ecdna_violin
)



def main():
    """Main entry point."""
    print("=" * 60)
    print("ecDNA Copy-Number Kinetics Model")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Single untreated simulation
    print("\n--- Example 1: Untreated Simulation ---")
    
    result = run_simulation(
        t_max=100.0,
        n_init=100,
        max_pop=2000,
        seed=42,
        verbose=True
    )
    
    # Compute metrics
    growth_rate = compute_growth_rate(result)
    ecdna_metrics = compute_ecdna_dynamics(result)
    sister_stats = compute_sister_correlation_stats(result)
    
    print(f"\nResults Summary:")
    print(f"  Final population: {result.population_sizes[-1]}")
    print(f"  Growth rate: {growth_rate:.4f}")
    print(f"  Final ecDNA mean: {ecdna_metrics['ecdna_final']:.2f}")
    print(f"  ecDNA trend: {ecdna_metrics['ecdna_trend']:.4f}")
    print(f"  Sister correlation: {sister_stats['sister_corr_mean']:.3f} ± {sister_stats['sister_corr_std']:.3f}")
    print(f"  Total divisions: {sister_stats['n_divisions']}")
    
    # Save raw data to CSVs
    result.save_as_csv(output_dir / "untreated_simulation_data")
    
    # Plot and save
    plot_results(result, title="Untreated Simulation",  
                 save_path=output_dir / "untreated_simulation.pdf")
    
    # Plot ecDNA distribution evolution (intratumoral heterogeneity)
    print("\n--- ecDNA Heterogeneity Analysis ---")
    
    # Combined view: violin + ECDF + quantile trajectories
    plot_ecdna_distribution_evolution(
        result, 
        mode='combined',
        n_time_points=8,
        title="ecDNA Copy Number Distribution Evolution",
        save_path=output_dir / "ecdna_distribution_combined.pdf"
    )
    
    # Ridge plot (alternative visualization)
    plot_ecdna_distribution_evolution(
        result,
        mode='ridge',
        n_time_points=10,
        title="ecDNA Distribution Evolution (Ridge Plot)",
        save_path=output_dir / "ecdna_distribution_ridge.pdf"
    )
    
    # Heterogeneity metrics over time
    plot_heterogeneity_metrics(
        result,
        title="ecDNA Heterogeneity Metrics",
        save_path=output_dir / "ecdna_heterogeneity_metrics.pdf"
    )
    
    # ecDNA+ fraction and high-copy subpopulation over time
    plot_ecdna_positive_fraction(
        result,
        threshold_high=20,
        title="ecDNA+ and High-Copy Subpopulation Dynamics",
        save_path=output_dir / "ecdna_positive_fraction.pdf"
    )
    
    # Alternative: use 95th percentile of initial distribution as threshold
    plot_ecdna_positive_fraction(
        result,
        use_quantile=95,
        title="ecDNA+ and Extreme Subpopulation (95th percentile threshold)",
        save_path=output_dir / "ecdna_positive_fraction_quantile.pdf"
    )
    
    # Lineage tree showing ecDNA inheritance patterns
    plot_lineage_tree(
        result,
        n_lineages=4,
        max_depth=5,
        title="ecDNA Inheritance: Non-Mendelian Segregation",
        save_path=output_dir / "ecdna_lineage_tree.pdf"
    )
    
    # Muller plot showing clonal dynamics by ecDNA copy number
    plot_muller_ecdna(
        result,
        title="ecDNA Clonal Dynamics (Muller Plot)",
        save_path=output_dir / "ecdna_muller_plot.pdf"
    )
    
    # Fitness landscape: ecDNA vs division/death rates
    plot_fitness_landscape(
        result,
        rate_type='both',
        title="ecDNA-Fitness Landscape",
        save_path=output_dir / "ecdna_fitness_landscape.pdf"
    )
    
    # Lineage state trajectory: trace how states evolve along lineages
    plot_lineage_state_trajectory(
        result,
        n_lineages=5,
        max_events=40,
        title="Lineage State Trajectories",
        save_path=output_dir / "lineage_state_trajectory.pdf"
    )
    
    # Event summary: distribution of event types and rates
    plot_event_summary(
        result,
        title="Event Type Distribution and Dynamics",
        save_path=output_dir / "event_summary.pdf"
    )

    # Grouped violin plots: ecDNA distribution by cell state
    if result.fitness_snapshots:
        # Plot 1: All cells
        plot_grouped_ecdna_violin(
            result.fitness_snapshots[-1], 
            min_copy=0, 
            title="ecDNA Distribution by Cell State (All Cells)",
            save_path=output_dir / "grouped_violin_all.pdf"
        )
        
        # Plot 2: High copy subpopulation (>= 90th percentile)
        import numpy as np
        all_ecdna_values = [d['ecdna'] for d in result.fitness_snapshots[-1]]
        if all_ecdna_values:
            p90 = np.percentile(all_ecdna_values, 90)
            plot_grouped_ecdna_violin(
                result.fitness_snapshots[-1], 
                min_copy=p90, 
                title=f"ecDNA Distribution by Cell State (High Copy >= {p90:.1f} [90%ile])",
                save_path=output_dir / "grouped_violin_high_copy.pdf"
            )

    # Example 2: Treatment Comparison
    # print("\n--- Example 2: Treatment Comparison ---")
    
    # trial = InSilicoTrial(base_seed=42)
    
    # # Run comparison
    # protocols_to_compare = ["untreated", "cdk_inhibitor_continuous", "ecdna_targeting"]
    
    # results_dict = trial.compare_protocols(
    #     protocol_names=protocols_to_compare,
    #     n_replicates=2,
    #     n_init=50,
    #     verbose=True
    # )
    
    # # Summarize
    # print("\n--- Summary ---")
    # for name in protocols_to_compare:
    #     summary = trial.summarize_results(name)
    #     print(f"\n{name}:")
    #     print(f"  Final pop: {summary.get('final_pop_mean', 0):.1f} ± {summary.get('final_pop_std', 0):.1f}")
    #     print(f"  Final ecDNA: {summary.get('final_ecdna_mean', 0):.2f} ± {summary.get('final_ecdna_std', 0):.2f}")
    
    # # Compare plots and save
    # compare_treatments(results_dict, save_path=output_dir / "treatment_comparison.pdf")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print(f"Figures saved to: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
