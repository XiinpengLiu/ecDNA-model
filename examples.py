"""
ecDNA Kinetic Model - Example Usage and Parameter Sweeps
========================================================
Demonstrates the full capabilities of the ecDNA simulation framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ecdna_model import (
    CellState, SimulationConfig, EcDNASimulator,
    SimpleEcDNAParameters, FullEcDNAParameters,
    ParameterSweep, compute_growth_rate, compute_extinction_probability
)
from visualization import (
    setup_style, plot_simulation_summary, plot_replicate_comparison,
    plot_sweep_1d, plot_sweep_2d, plot_single_cell_trajectory,
    plot_population_dynamics, plot_mean_ecdna, plot_ecdna_heatmap
)

# Output directory for figures
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def example_1_basic_simulation():
    """
    Example 1: Basic population simulation with simple ecDNA model.
    """
    print("=" * 60)
    print("Example 1: Basic Population Simulation")
    print("=" * 60)
    
    # Set up parameters
    params = SimpleEcDNAParameters(
        div_rate=0.08,
        death_rate=0.02,
        optimal_copies=15,
        fitness_width=8.0
    )
    
    config = SimulationConfig(
        dt=0.1,
        t_max=100,
        max_cells=5000,
        seed=42
    )
    
    # Create initial population with variable ecDNA
    initial_cells = [
        CellState(k=np.array([k0])) 
        for k0 in np.random.poisson(10, 50)
    ]
    
    # Run simulation
    sim = EcDNASimulator(params, config)
    history = sim.simulate_population(initial_cells, record_interval=1.0)
    
    # Print summary statistics
    print(f"Initial population: {len(initial_cells)}")
    print(f"Final population: {history['population_size'][-1]}")
    print(f"Initial mean ecDNA: {np.mean([c.k[0] for c in initial_cells]):.2f}")
    print(f"Final mean ecDNA: {history['mean_k'][-1]:.2f}")
    print(f"Growth rate: {compute_growth_rate(history):.4f}")
    
    # Plot summary
    fig = plot_simulation_summary(history)
    fig.suptitle("Example 1: Basic ecDNA Population Dynamics", y=1.02)
    fig.savefig(OUTPUT_DIR / "example1_basic_simulation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example1_basic_simulation.png'}")
    return history


def example_2_single_cell_tracking():
    """
    Example 2: Track individual cell trajectories.
    """
    print("\n" + "=" * 60)
    print("Example 2: Single Cell Trajectory Tracking")
    print("=" * 60)
    
    params = SimpleEcDNAParameters(
        div_rate=0.1,
        death_rate=0.03,
        optimal_copies=10,
        fitness_width=5.0
    )
    
    config = SimulationConfig(dt=0.05, t_max=50, seed=123)
    sim = EcDNASimulator(params, config)
    
    # Track several single cells
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for i in range(6):
        initial = CellState(k=np.array([np.random.poisson(10)]))
        trajectory = sim.simulate_single_cell(initial, t_max=50)
        
        ax = axes[i // 3, i % 3]
        times = [snap['t'] for snap in trajectory]
        k_vals = [snap['state'].k[0] for snap in trajectory]
        
        ax.plot(times, k_vals, 'b-', lw=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('ecDNA copies')
        ax.set_title(f'Cell {i+1} (k₀={initial.k[0]})')
        
        # Mark terminal event
        if 'event' in trajectory[-1]:
            event = trajectory[-1]['event']
            color = 'red' if event == 'death' else 'green'
            marker = 'x' if event == 'death' else 'o'
            ax.scatter([times[-1]], [k_vals[-1]], c=color, marker=marker, 
                      s=100, zorder=5)
            ax.annotate(event, (times[-1], k_vals[-1]), fontsize=8)
    
    plt.tight_layout()
    fig.suptitle("Example 2: Single Cell Trajectories", y=1.02)
    fig.savefig(OUTPUT_DIR / "example2_single_cell.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example2_single_cell.png'}")


def example_3_full_model():
    """
    Example 3: Full model with regulatory switching, phenotype dynamics,
    and inter-division ecDNA jumps.
    """
    print("\n" + "=" * 60)
    print("Example 3: Full Model with All Features")
    print("=" * 60)
    
    params = FullEcDNAParameters(
        n_regulatory=3,
        n_phenotype_dims=1,
        base_div_rate=0.06,
        base_death_rate=0.015,
        reg_switch_rate=0.02,
        ecdna_gain_rate=0.005,
        ecdna_loss_rate=0.002,
        drift_strength=-0.05,
        diffusion_strength=0.1,
        amp_rate=0.2,
        loss_prob=0.02,
        phenotype_inheritance_noise=0.15
    )
    
    config = SimulationConfig(
        dt=0.1,
        t_max=150,
        max_cells=3000,
        seed=42
    )
    
    # Initial cells with varied states
    initial_cells = []
    for _ in range(30):
        initial_cells.append(CellState(
            m=np.random.randint(0, 3),
            k=np.array([np.random.poisson(8)]),
            y=np.array([np.random.randn() * 0.5])
        ))
    
    sim = EcDNASimulator(params, config)
    history = sim.simulate_population(initial_cells, record_interval=2.0)
    
    print(f"Final population: {history['population_size'][-1]}")
    print(f"Final mean ecDNA: {history['mean_k'][-1]:.2f} ± {history['std_k'][-1]:.2f}")
    print(f"Growth rate: {compute_growth_rate(history):.4f}")
    
    # Custom plot for full model
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Population dynamics
    axes[0, 0].semilogy(history['times'], history['population_size'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Population (log)')
    axes[0, 0].set_title('Population Growth')
    
    # ecDNA dynamics
    axes[0, 1].plot(history['times'], history['mean_k'], 'g-', lw=2)
    axes[0, 1].fill_between(
        history['times'],
        np.array(history['mean_k']) - np.array(history['std_k']),
        np.array(history['mean_k']) + np.array(history['std_k']),
        alpha=0.3, color='green'
    )
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('ecDNA copies')
    axes[0, 1].set_title('Mean ecDNA (±1 SD)')
    
    # ecDNA heatmap
    k_all = [k for dist in history['k_distribution'] for k in dist if len(dist) > 0]
    if k_all:
        max_k = min(int(np.percentile(k_all, 98)), 60)
        plot_ecdna_heatmap(history, ax=axes[0, 2], max_k=max_k)
    
    # Regulatory states
    m_dist = np.array(history['m_distribution'])
    for m in range(params.n_reg):
        axes[1, 0].plot(history['times'], m_dist[:, m], label=f'M={m}', lw=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].set_title('Regulatory State Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Mean phenotype
    y_means = np.array([y[0] for y in history['mean_y']])
    axes[1, 1].plot(history['times'], y_means, 'm-', lw=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Mean Phenotype Y')
    axes[1, 1].set_title('Phenotype Evolution')
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Final ecDNA distribution
    if history['k_distribution'] and len(history['k_distribution'][-1]) > 0:
        k_final = history['k_distribution'][-1]
        axes[1, 2].hist(k_final, bins=30, density=True, alpha=0.7, color='teal')
        axes[1, 2].axvline(np.mean(k_final), color='red', linestyle='--', 
                          label=f'Mean={np.mean(k_final):.1f}')
        axes[1, 2].set_xlabel('ecDNA copies')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Final ecDNA Distribution')
        axes[1, 2].legend()
    
    plt.tight_layout()
    fig.suptitle("Example 3: Full Model Dynamics", y=1.02)
    fig.savefig(OUTPUT_DIR / "example3_full_model.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example3_full_model.png'}")
    return history


def example_4_parameter_sweep_1d():
    """
    Example 4: 1D parameter sweep - Division rate effect.
    """
    print("\n" + "=" * 60)
    print("Example 4: 1D Parameter Sweep (Division Rate)")
    print("=" * 60)
    
    div_rates = np.linspace(0.02, 0.15, 8)
    
    base_params = {
        'death_rate': 0.02,
        'optimal_copies': 12,
        'fitness_width': 6.0
    }
    
    initial_cells = [CellState(k=np.array([10])) for _ in range(30)]
    config = SimulationConfig(dt=0.1, t_max=80, max_cells=3000, seed=0)
    
    # Growth rate metric
    def growth_metric(h):
        return compute_growth_rate(h)
    
    # Final population metric
    def pop_metric(h):
        return h['population_size'][-1] if h['population_size'] else 0
    
    # Final mean ecDNA
    def ecdna_metric(h):
        return h['mean_k'][-1] if h['mean_k'] else 0
    
    print("Running parameter sweep...")
    results_growth = ParameterSweep.sweep_1d(
        'div_rate', div_rates, base_params, SimpleEcDNAParameters,
        initial_cells, config, n_replicates=3, metric_fn=growth_metric
    )
    
    results_pop = ParameterSweep.sweep_1d(
        'div_rate', div_rates, base_params, SimpleEcDNAParameters,
        initial_cells, config, n_replicates=3, metric_fn=pop_metric
    )
    
    results_ecdna = ParameterSweep.sweep_1d(
        'div_rate', div_rates, base_params, SimpleEcDNAParameters,
        initial_cells, config, n_replicates=3, metric_fn=ecdna_metric
    )
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    plot_sweep_1d(results_growth, ax=axes[0], ylabel='Growth Rate', color='blue')
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero growth')
    axes[0].legend()
    
    plot_sweep_1d(results_pop, ax=axes[1], ylabel='Final Population', 
                  log_y=True, color='green')
    
    plot_sweep_1d(results_ecdna, ax=axes[2], ylabel='Final Mean ecDNA', color='purple')
    
    plt.tight_layout()
    fig.suptitle("Example 4: Division Rate Parameter Sweep", y=1.02)
    fig.savefig(OUTPUT_DIR / "example4_sweep_1d.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example4_sweep_1d.png'}")
    
    # Print results table
    print("\nResults Table:")
    print("-" * 50)
    print(f"{'Div Rate':>10} | {'Growth Rate':>12} | {'Final Pop':>10}")
    print("-" * 50)
    for i, dr in enumerate(div_rates):
        print(f"{dr:>10.3f} | {results_growth['metrics_mean'][i]:>12.4f} | "
              f"{results_pop['metrics_mean'][i]:>10.0f}")


def example_5_parameter_sweep_2d():
    """
    Example 5: 2D parameter sweep - Division vs Death rate phase diagram.
    """
    print("\n" + "=" * 60)
    print("Example 5: 2D Parameter Sweep (Division vs Death)")
    print("=" * 60)
    
    div_rates = np.linspace(0.03, 0.12, 10)
    death_rates = np.linspace(0.01, 0.06, 10)
    
    base_params = {
        'optimal_copies': 10,
        'fitness_width': 5.0
    }
    
    initial_cells = [CellState(k=np.array([10])) for _ in range(20)]
    config = SimulationConfig(dt=0.15, t_max=60, max_cells=2000, seed=0)
    
    print("Running 2D sweep (this may take a few minutes)...")
    results = ParameterSweep.sweep_2d(
        'div_rate', div_rates,
        'death_rate', death_rates,
        base_params, SimpleEcDNAParameters,
        initial_cells, config, n_replicates=2,
        metric_fn=lambda h: compute_growth_rate(h)
    )
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Growth rate heatmap
    im1 = axes[0].imshow(results['metrics'], aspect='auto', origin='lower',
                         extent=[death_rates[0], death_rates[-1],
                                div_rates[0], div_rates[-1]],
                         cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    plt.colorbar(im1, ax=axes[0], label='Growth Rate')
    axes[0].set_xlabel('Death Rate')
    axes[0].set_ylabel('Division Rate')
    axes[0].set_title('Growth Rate Phase Diagram')
    
    # Add zero-growth contour
    cs = axes[0].contour(death_rates, div_rates, results['metrics'], 
                         levels=[0], colors='black', linewidths=2)
    axes[0].clabel(cs, inline=True, fmt='r=0')
    
    # Extinction boundary (where growth < 0)
    extinct_mask = results['metrics'] < 0
    axes[1].imshow(extinct_mask.astype(float), aspect='auto', origin='lower',
                   extent=[death_rates[0], death_rates[-1],
                          div_rates[0], div_rates[-1]],
                   cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_xlabel('Death Rate')
    axes[1].set_ylabel('Division Rate')
    axes[1].set_title('Extinction Risk (Red = High)')
    
    # Add diagonal reference line
    min_val = min(div_rates[0], death_rates[0])
    max_val = max(div_rates[-1], death_rates[-1])
    for ax in axes:
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.tight_layout()
    fig.suptitle("Example 5: Division vs Death Rate Phase Diagram", y=1.02)
    fig.savefig(OUTPUT_DIR / "example5_sweep_2d.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example5_sweep_2d.png'}")


def example_6_fitness_landscape():
    """
    Example 6: ecDNA fitness landscape exploration.
    """
    print("\n" + "=" * 60)
    print("Example 6: ecDNA Fitness Landscape")
    print("=" * 60)
    
    # Sweep optimal ecDNA copies
    optimal_copies_values = np.array([5, 10, 15, 20, 25])
    fitness_widths = np.array([3, 5, 8, 12])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Different fitness optima
    ax = axes[0, 0]
    for opt in optimal_copies_values:
        params = SimpleEcDNAParameters(
            div_rate=0.08, death_rate=0.02,
            optimal_copies=opt, fitness_width=6.0
        )
        config = SimulationConfig(dt=0.1, t_max=100, max_cells=3000, seed=42)
        
        initial = [CellState(k=np.array([10])) for _ in range(30)]
        sim = EcDNASimulator(params, config)
        history = sim.simulate_population(initial)
        
        ax.plot(history['times'], history['mean_k'], label=f'Opt={opt}', lw=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA')
    ax.set_title('Effect of Optimal Copy Number')
    ax.legend()
    ax.axhline(10, color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Panel B: Different fitness widths (selection strength)
    ax = axes[0, 1]
    for width in fitness_widths:
        params = SimpleEcDNAParameters(
            div_rate=0.08, death_rate=0.02,
            optimal_copies=15, fitness_width=width
        )
        config = SimulationConfig(dt=0.1, t_max=100, max_cells=3000, seed=42)
        
        initial = [CellState(k=np.array([10])) for _ in range(30)]
        sim = EcDNASimulator(params, config)
        history = sim.simulate_population(initial)
        
        ax.plot(history['times'], history['mean_k'], 
               label=f'σ={width} ({"strong" if width < 5 else "weak"})', lw=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA')
    ax.set_title('Effect of Selection Strength (σ)')
    ax.legend()
    
    # Panel C: ecDNA variance over time
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(fitness_widths)))
    for width, color in zip(fitness_widths, colors):
        params = SimpleEcDNAParameters(
            div_rate=0.08, death_rate=0.02,
            optimal_copies=15, fitness_width=width
        )
        config = SimulationConfig(dt=0.1, t_max=100, max_cells=3000, seed=42)
        
        initial = [CellState(k=np.array([10])) for _ in range(30)]
        sim = EcDNASimulator(params, config)
        history = sim.simulate_population(initial)
        
        ax.plot(history['times'], history['std_k'], 
               label=f'σ={width}', lw=2, color=color)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('ecDNA Std Dev')
    ax.set_title('ecDNA Heterogeneity Over Time')
    ax.legend()
    
    # Panel D: Final distributions comparison
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for (opt, color) in zip([5, 10, 15, 20], colors):
        params = SimpleEcDNAParameters(
            div_rate=0.08, death_rate=0.02,
            optimal_copies=opt, fitness_width=6.0
        )
        config = SimulationConfig(dt=0.1, t_max=80, max_cells=3000, seed=42)
        
        initial = [CellState(k=np.array([10])) for _ in range(30)]
        sim = EcDNASimulator(params, config)
        history = sim.simulate_population(initial)
        
        if history['k_distribution'] and len(history['k_distribution'][-1]) > 0:
            k_final = history['k_distribution'][-1]
            ax.hist(k_final, bins=25, alpha=0.4, color=color, 
                   density=True, label=f'Opt={opt}')
            ax.axvline(np.mean(k_final), color=color, linestyle='--', lw=2)
    
    ax.set_xlabel('ecDNA Copy Number')
    ax.set_ylabel('Density')
    ax.set_title('Final ecDNA Distributions')
    ax.legend()
    
    plt.tight_layout()
    fig.suptitle("Example 6: ecDNA Fitness Landscape Analysis", y=1.02)
    fig.savefig(OUTPUT_DIR / "example6_fitness_landscape.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example6_fitness_landscape.png'}")


def example_7_replicate_variability():
    """
    Example 7: Assess stochastic variability across replicates.
    """
    print("\n" + "=" * 60)
    print("Example 7: Stochastic Variability Analysis")
    print("=" * 60)
    
    params = SimpleEcDNAParameters(
        div_rate=0.06,
        death_rate=0.025,
        optimal_copies=12,
        fitness_width=6.0
    )
    
    n_replicates = 10
    histories = []
    
    for rep in range(n_replicates):
        config = SimulationConfig(
            dt=0.1, t_max=120, max_cells=3000,
            seed=100 + rep
        )
        
        initial = [CellState(k=np.array([10])) for _ in range(25)]
        sim = EcDNASimulator(params, config)
        history = sim.simulate_population(initial)
        histories.append(history)
        
        print(f"  Replicate {rep+1}: Final pop = {history['population_size'][-1]}, "
              f"Mean ecDNA = {history['mean_k'][-1]:.2f}")
    
    # Plot replicate comparison
    fig = plot_replicate_comparison(histories, figsize=(14, 4))
    fig.suptitle("Example 7: Variability Across Replicates", y=1.02)
    fig.savefig(OUTPUT_DIR / "example7_replicates.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Summary statistics
    final_pops = [h['population_size'][-1] for h in histories]
    final_ecdna = [h['mean_k'][-1] for h in histories]
    growth_rates = [compute_growth_rate(h) for h in histories]
    
    print("\nSummary Statistics (n={} replicates):".format(n_replicates))
    print(f"  Final population: {np.mean(final_pops):.1f} ± {np.std(final_pops):.1f}")
    print(f"  Final mean ecDNA: {np.mean(final_ecdna):.2f} ± {np.std(final_ecdna):.2f}")
    print(f"  Growth rate: {np.mean(growth_rates):.4f} ± {np.std(growth_rates):.4f}")
    print(f"  Extinction probability: {compute_extinction_probability(histories):.2%}")
    
    print(f"Figure saved to {OUTPUT_DIR / 'example7_replicates.png'}")


def example_8_treatment_scenario():
    """
    Example 8: Simulating treatment effects via parameter changes.
    """
    print("\n" + "=" * 60)
    print("Example 8: Treatment Scenario Simulation")
    print("=" * 60)
    
    # Pre-treatment population
    params_pre = SimpleEcDNAParameters(
        div_rate=0.08, death_rate=0.015,
        optimal_copies=15, fitness_width=8.0
    )
    
    config = SimulationConfig(dt=0.1, t_max=50, max_cells=5000, seed=42)
    initial = [CellState(k=np.array([np.random.poisson(12)])) for _ in range(20)]
    
    sim_pre = EcDNASimulator(params_pre, config)
    history_pre = sim_pre.simulate_population(initial)
    
    # Get cells at end of pre-treatment
    print(f"Pre-treatment final population: {history_pre['population_size'][-1]}")
    
    # Post-treatment (increased death, selection for high ecDNA)
    params_post = SimpleEcDNAParameters(
        div_rate=0.04,  # Reduced proliferation
        death_rate=0.08,  # Increased death (treatment)
        optimal_copies=25,  # Selection favors higher copies (resistance)
        fitness_width=5.0
    )
    
    # Sample surviving cells from pre-treatment distribution
    if history_pre['k_distribution'] and len(history_pre['k_distribution'][-1]) > 0:
        k_dist_final = history_pre['k_distribution'][-1]
        n_survivors = min(len(k_dist_final), 100)
        survivor_k = np.random.choice(k_dist_final, n_survivors, replace=True)
        post_initial = [CellState(k=np.array([k])) for k in survivor_k]
    else:
        post_initial = [CellState(k=np.array([15])) for _ in range(50)]
    
    config_post = SimulationConfig(dt=0.1, t_max=100, max_cells=5000, seed=43)
    sim_post = EcDNASimulator(params_post, config_post)
    history_post = sim_post.simulate_population(post_initial)
    
    print(f"Post-treatment final population: {history_post['population_size'][-1]}")
    
    # Plot treatment timeline
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Combined population timeline
    ax = axes[0, 0]
    t_pre = history_pre['times']
    t_post = np.array(history_post['times']) + t_pre[-1]
    ax.plot(t_pre, history_pre['population_size'], 'b-', lw=2, label='Pre-treatment')
    ax.plot(t_post, history_post['population_size'], 'r-', lw=2, label='Post-treatment')
    ax.axvline(t_pre[-1], color='black', linestyle='--', lw=2, label='Treatment start')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_yscale('log')
    ax.set_title('Population Dynamics')
    ax.legend()
    
    # ecDNA dynamics
    ax = axes[0, 1]
    ax.plot(t_pre, history_pre['mean_k'], 'b-', lw=2, label='Pre-treatment')
    ax.fill_between(t_pre, 
                    np.array(history_pre['mean_k']) - np.array(history_pre['std_k']),
                    np.array(history_pre['mean_k']) + np.array(history_pre['std_k']),
                    alpha=0.3, color='blue')
    ax.plot(t_post, history_post['mean_k'], 'r-', lw=2, label='Post-treatment')
    ax.fill_between(t_post,
                    np.array(history_post['mean_k']) - np.array(history_post['std_k']),
                    np.array(history_post['mean_k']) + np.array(history_post['std_k']),
                    alpha=0.3, color='red')
    ax.axvline(t_pre[-1], color='black', linestyle='--', lw=2)
    ax.axhline(params_pre['optimal_copies'] if isinstance(params_pre, dict) else 15, 
               color='blue', linestyle=':', alpha=0.5)
    ax.axhline(25, color='red', linestyle=':', alpha=0.5, label='Post-opt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA')
    ax.set_title('ecDNA Evolution')
    ax.legend()
    
    # Pre-treatment final distribution
    ax = axes[1, 0]
    if history_pre['k_distribution'] and len(history_pre['k_distribution'][-1]) > 0:
        ax.hist(history_pre['k_distribution'][-1], bins=25, alpha=0.7, 
               color='blue', density=True, label='Pre-treatment end')
    ax.axvline(15, color='blue', linestyle='--', label='Optimal (pre)')
    ax.set_xlabel('ecDNA Copy Number')
    ax.set_ylabel('Density')
    ax.set_title('Pre-treatment Final Distribution')
    ax.legend()
    
    # Post-treatment final distribution
    ax = axes[1, 1]
    if history_post['k_distribution'] and len(history_post['k_distribution'][-1]) > 0:
        ax.hist(history_post['k_distribution'][-1], bins=25, alpha=0.7,
               color='red', density=True, label='Post-treatment end')
    ax.axvline(25, color='red', linestyle='--', label='Optimal (post)')
    ax.set_xlabel('ecDNA Copy Number')
    ax.set_ylabel('Density')
    ax.set_title('Post-treatment Final Distribution')
    ax.legend()
    
    plt.tight_layout()
    fig.suptitle("Example 8: Treatment Response Simulation", y=1.02)
    fig.savefig(OUTPUT_DIR / "example8_treatment.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {OUTPUT_DIR / 'example8_treatment.png'}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    setup_style()
    np.random.seed(42)
    
    print("\n" + "=" * 60)
    print("ecDNA Kinetic Model - Example Usage Suite")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_simulation()
    example_2_single_cell_tracking()
    example_3_full_model()
    example_4_parameter_sweep_1d()
    example_5_parameter_sweep_2d()
    example_6_fitness_landscape()
    example_7_replicate_variability()
    example_8_treatment_scenario()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print(f"Figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
