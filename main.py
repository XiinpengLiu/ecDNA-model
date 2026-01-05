"""
ecDNA Copy-Number Kinetics Model - Main Script
Run simulations and generate results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import model components
import config as cfg
from cell import Cell, CellPopulation
from simulation import run_simulation, OgataThinningSimulator
from treatment import (
    InSilicoTrial, PROTOCOLS,
    compute_growth_rate, compute_ecdna_dynamics, compute_sister_correlation_stats
)


def plot_results(result, title="Simulation Results", save_path=None):
    """Plot simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Population size
    ax = axes[0, 0]
    ax.plot(result.times, result.population_sizes, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Size')
    ax.set_title('Population Dynamics')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # ecDNA mean
    ax = axes[0, 1]
    ax.plot(result.times, result.ecdna_means, 'r-', linewidth=2)
    ax.fill_between(result.times, 
                    np.array(result.ecdna_means) - np.array(result.ecdna_stds),
                    np.array(result.ecdna_means) + np.array(result.ecdna_stds),
                    alpha=0.3, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA Copy Number')
    ax.set_title('ecDNA Dynamics')
    ax.grid(True, alpha=0.3)
    
    # State composition over time (cycle phases)
    ax = axes[1, 0]
    if result.state_compositions and 'cycle_dist' in result.state_compositions[0]:
        cycle_names = ['G0', 'G1', 'S', 'G2M']
        cycle_data = np.array([s.get('cycle_dist', [0]*4) for s in result.state_compositions])
        for i, name in enumerate(cycle_names):
            if i < cycle_data.shape[1]:
                ax.plot(result.times, cycle_data[:, i], label=name, linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_title('Cell Cycle Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Sister correlations histogram
    ax = axes[1, 1]
    if result.sister_correlations:
        ax.hist(result.sister_correlations, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(result.sister_correlations), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(result.sister_correlations):.3f}')
        ax.set_xlabel('Sister Correlation')
        ax.set_ylabel('Count')
        ax.set_title('Sister ecDNA Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def compare_treatments(results_dict, save_path=None):
    """Compare multiple treatment results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Population dynamics
    ax = axes[0]
    for name, results in results_dict.items():
        for i, result in enumerate(results):
            alpha = 0.3 if i > 0 else 1.0
            label = name if i == 0 else None
            ax.plot(result.times, result.population_sizes, alpha=alpha, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Size')
    ax.set_title('Population Dynamics by Treatment')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ecDNA dynamics
    ax = axes[1]
    for name, results in results_dict.items():
        for i, result in enumerate(results):
            alpha = 0.3 if i > 0 else 1.0
            label = name if i == 0 else None
            ax.plot(result.times, result.ecdna_means, alpha=alpha, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA')
    ax.set_title('ecDNA Dynamics by Treatment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


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
        t_max=50.0,
        n_init=50,
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
    
    # Plot and save
    plot_results(result, title="Untreated Simulation", 
                 save_path=output_dir / "untreated_simulation.png")
    
    # Example 2: Treatment comparison
    print("\n--- Example 2: Treatment Comparison ---")
    
    trial = InSilicoTrial(base_seed=42)
    
    # Run comparison
    protocols_to_compare = ["untreated", "cdk_inhibitor_continuous", "ecdna_targeting"]
    
    results_dict = trial.compare_protocols(
        protocol_names=protocols_to_compare,
        n_replicates=2,
        n_init=50,
        verbose=True
    )
    
    # Summarize
    print("\n--- Summary ---")
    for name in protocols_to_compare:
        summary = trial.summarize_results(name)
        print(f"\n{name}:")
        print(f"  Final pop: {summary.get('final_pop_mean', 0):.1f} ± {summary.get('final_pop_std', 0):.1f}")
        print(f"  Final ecDNA: {summary.get('final_ecdna_mean', 0):.2f} ± {summary.get('final_ecdna_std', 0):.2f}")
    
    # Compare plots and save
    compare_treatments(results_dict, save_path=output_dir / "treatment_comparison.png")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print(f"Figures saved to: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
