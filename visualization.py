"""
Visualization module for ecDNA kinetic model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, List, Optional, Tuple
import warnings


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# Population Dynamics Plots
# =============================================================================

def plot_population_dynamics(history: Dict, ax: Optional[plt.Axes] = None,
                             log_scale: bool = True, **kwargs) -> plt.Axes:
    """Plot population size over time."""
    if ax is None:
        _, ax = plt.subplots()
    
    ax.plot(history['times'], history['population_size'], **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Size')
    if log_scale and min(history['population_size']) > 0:
        ax.set_yscale('log')
    ax.set_title('Population Dynamics')
    return ax


def plot_mean_ecdna(history: Dict, ax: Optional[plt.Axes] = None,
                    show_std: bool = True, **kwargs) -> plt.Axes:
    """Plot mean ecDNA copy number over time."""
    if ax is None:
        _, ax = plt.subplots()
    
    times = history['times']
    mean_k = history['mean_k']
    
    line, = ax.plot(times, mean_k, **kwargs)
    
    if show_std and 'std_k' in history:
        std_k = history['std_k']
        color = line.get_color()
        ax.fill_between(times, 
                        np.array(mean_k) - np.array(std_k),
                        np.array(mean_k) + np.array(std_k),
                        alpha=0.3, color=color)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean ecDNA Copy Number')
    ax.set_title('ecDNA Dynamics')
    return ax


def plot_ecdna_distribution_over_time(history: Dict, 
                                       time_points: Optional[List[float]] = None,
                                       ax: Optional[plt.Axes] = None,
                                       max_k: Optional[int] = None) -> plt.Axes:
    """Plot ecDNA distribution at selected time points."""
    if ax is None:
        _, ax = plt.subplots()
    
    times = np.array(history['times'])
    k_dists = history['k_distribution']
    
    if time_points is None:
        # Select 5 evenly spaced time points
        indices = np.linspace(0, len(times)-1, 5, dtype=int)
    else:
        indices = [np.argmin(np.abs(times - t)) for t in time_points]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for idx, color in zip(indices, colors):
        k_vals = k_dists[idx]
        if len(k_vals) > 0:
            bins = np.arange(0, max(k_vals) + 2) - 0.5 if max_k is None else np.arange(0, max_k + 2) - 0.5
            ax.hist(k_vals, bins=bins, alpha=0.5, color=color, 
                   label=f't={times[idx]:.1f}', density=True)
    
    ax.set_xlabel('ecDNA Copy Number')
    ax.set_ylabel('Density')
    ax.set_title('ecDNA Distribution Over Time')
    ax.legend()
    return ax


def plot_ecdna_heatmap(history: Dict, ax: Optional[plt.Axes] = None,
                       max_k: int = 50, time_resolution: int = 50) -> plt.Axes:
    """Plot ecDNA distribution as heatmap over time."""
    if ax is None:
        _, ax = plt.subplots()
    
    times = np.array(history['times'])
    k_dists = history['k_distribution']
    
    # Create time bins
    time_indices = np.linspace(0, len(times)-1, min(time_resolution, len(times)), dtype=int)
    
    # Create distribution matrix
    dist_matrix = np.zeros((max_k + 1, len(time_indices)))
    
    for j, idx in enumerate(time_indices):
        k_vals = k_dists[idx]
        if len(k_vals) > 0:
            counts, _ = np.histogram(k_vals, bins=np.arange(0, max_k + 2))
            counts = counts / (np.sum(counts) + 1e-10)  # Normalize
            dist_matrix[:, j] = counts
    
    im = ax.imshow(dist_matrix, aspect='auto', origin='lower',
                   extent=[times[time_indices[0]], times[time_indices[-1]], 0, max_k],
                   cmap='viridis')
    
    plt.colorbar(im, ax=ax, label='Density')
    ax.set_xlabel('Time')
    ax.set_ylabel('ecDNA Copy Number')
    ax.set_title('ecDNA Distribution Heatmap')
    return ax


def plot_regulatory_state_dynamics(history: Dict, 
                                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot regulatory state proportions over time."""
    if ax is None:
        _, ax = plt.subplots()
    
    times = history['times']
    m_dists = np.array(history['m_distribution'])
    
    n_states = m_dists.shape[1] if len(m_dists.shape) > 1 else 1
    
    for m in range(n_states):
        ax.plot(times, m_dists[:, m], label=f'State {m}')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')
    ax.set_title('Regulatory State Distribution')
    ax.legend()
    ax.set_ylim(0, 1)
    return ax


# =============================================================================
# Parameter Sweep Plots
# =============================================================================

def plot_sweep_1d(results: Dict, ax: Optional[plt.Axes] = None,
                  log_x: bool = False, log_y: bool = False,
                  ylabel: str = 'Metric', **kwargs) -> plt.Axes:
    """Plot results from 1D parameter sweep."""
    if ax is None:
        _, ax = plt.subplots()
    
    x = results['param_values']
    y = results['metrics_mean']
    yerr = results['metrics_std']
    
    ax.errorbar(x, y, yerr=yerr, capsize=3, marker='o', **kwargs)
    
    ax.set_xlabel(results['param_name'])
    ax.set_ylabel(ylabel)
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    ax.set_title(f'{ylabel} vs {results["param_name"]}')
    return ax


def plot_sweep_2d(results: Dict, ax: Optional[plt.Axes] = None,
                  log_scale: bool = False, cmap: str = 'viridis') -> plt.Axes:
    """Plot results from 2D parameter sweep as heatmap."""
    if ax is None:
        _, ax = plt.subplots()
    
    metrics = results['metrics']
    
    if log_scale:
        metrics = np.log10(metrics + 1)
    
    im = ax.imshow(metrics, aspect='auto', origin='lower', cmap=cmap,
                   extent=[results['param2_values'][0], results['param2_values'][-1],
                          results['param1_values'][0], results['param1_values'][-1]])
    
    plt.colorbar(im, ax=ax, label='Metric')
    ax.set_xlabel(results['param2_name'])
    ax.set_ylabel(results['param1_name'])
    ax.set_title('2D Parameter Sweep')
    return ax


# =============================================================================
# Summary Dashboard
# =============================================================================

def plot_simulation_summary(history: Dict, figsize: Tuple = (14, 10)) -> plt.Figure:
    """Create a comprehensive summary dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Population dynamics
    plot_population_dynamics(history, ax=axes[0, 0])
    
    # Mean ecDNA with std
    plot_mean_ecdna(history, ax=axes[0, 1], show_std=True)
    
    # ecDNA distribution over time
    plot_ecdna_distribution_over_time(history, ax=axes[0, 2])
    
    # ecDNA heatmap
    if history['k_distribution'] and len(history['k_distribution'][0]) > 0:
        max_k = int(np.percentile([max(d) if len(d) > 0 else 0 
                                   for d in history['k_distribution']], 95))
        max_k = max(max_k, 10)
        plot_ecdna_heatmap(history, ax=axes[1, 0], max_k=max_k)
    
    # Regulatory state dynamics
    if len(history['m_distribution']) > 0 and len(history['m_distribution'][0]) > 1:
        plot_regulatory_state_dynamics(history, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Single regulatory state', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Regulatory States')
    
    # Mean age over time
    ax = axes[1, 2]
    ax.plot(history['times'], history['mean_age'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Age')
    ax.set_title('Mean Cell Age')
    
    plt.tight_layout()
    return fig


def plot_replicate_comparison(histories: List[Dict], 
                               figsize: Tuple = (12, 4)) -> plt.Figure:
    """Compare multiple simulation replicates."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, color) in enumerate(zip(histories, colors)):
        # Population
        axes[0].plot(history['times'], history['population_size'],
                    alpha=0.7, color=color, label=f'Rep {i+1}')
        
        # Mean ecDNA
        axes[1].plot(history['times'], history['mean_k'],
                    alpha=0.7, color=color)
        
        # Final distribution
        if history['k_distribution'] and len(history['k_distribution'][-1]) > 0:
            k_final = history['k_distribution'][-1]
            axes[2].hist(k_final, alpha=0.3, color=color, density=True, bins=20)
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Population')
    axes[0].set_title('Population Size')
    axes[0].legend(fontsize=8)
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mean ecDNA')
    axes[1].set_title('Mean ecDNA Copy Number')
    
    axes[2].set_xlabel('ecDNA Copy Number')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Final ecDNA Distribution')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Lineage / Trajectory Plots
# =============================================================================

def plot_single_cell_trajectory(trajectory: List[Dict], 
                                 figsize: Tuple = (12, 4)) -> plt.Figure:
    """Plot trajectory of a single cell."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    times = [snap['t'] for snap in trajectory]
    k_vals = [np.sum(snap['state'].k) for snap in trajectory]
    ages = [snap['state'].a for snap in trajectory]
    
    # ecDNA copy number
    axes[0].plot(times, k_vals, 'b-', lw=1.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('ecDNA Copy Number')
    axes[0].set_title('ecDNA Over Time')
    
    # Age (sawtooth pattern if dividing)
    axes[1].plot(times, ages, 'g-', lw=1.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Age')
    axes[1].set_title('Cell Age')
    
    # Mark terminal event if any
    final = trajectory[-1]
    if 'event' in final:
        event = final['event']
        color = 'red' if event == 'death' else 'orange'
        marker = 'x' if event == 'death' else '*'
        for ax in axes[:2]:
            ax.scatter([final['t']], [ax.get_lines()[0].get_ydata()[-1]], 
                      c=color, marker=marker, s=100, zorder=5, 
                      label=event.capitalize())
            ax.legend()
    
    # Phenotype (if present and non-trivial)
    if trajectory[0]['state'].y.size > 0:
        y_vals = np.array([snap['state'].y for snap in trajectory])
        if y_vals.shape[1] == 1:
            axes[2].plot(times, y_vals[:, 0], 'm-', lw=1.5)
            axes[2].set_ylabel('Phenotype')
        else:
            for dim in range(min(y_vals.shape[1], 3)):
                axes[2].plot(times, y_vals[:, dim], label=f'Y{dim}', lw=1.5)
            axes[2].legend()
            axes[2].set_ylabel('Phenotype')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Phenotype Dynamics')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test with dummy data
    setup_style()
    
    # Create dummy history
    np.random.seed(42)
    n_points = 100
    history = {
        'times': np.linspace(0, 100, n_points),
        'population_size': np.exp(0.02 * np.linspace(0, 100, n_points)) * 100,
        'mean_k': 10 + 2 * np.sin(0.1 * np.linspace(0, 100, n_points)),
        'std_k': np.ones(n_points) * 2,
        'mean_age': 10 + np.random.randn(n_points) * 0.5,
        'mean_y': [np.array([0.0])] * n_points,
        'k_distribution': [np.random.poisson(10, int(100 * np.exp(0.02 * t))) 
                          for t in np.linspace(0, 100, n_points)],
        'm_distribution': [np.array([0.6, 0.4])] * n_points
    }
    
    fig = plot_simulation_summary(history)
    plt.savefig('test_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
