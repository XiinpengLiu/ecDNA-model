"""
Plotting utilities for ecDNA Copy-Number Kinetics Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection


# ============================================================================
# Publication-quality style settings
# ============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })


# Nature/Cell-inspired color palette
PALETTE = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#E74C3C',    # Coral red
    'tertiary': '#3498DB',     # Sky blue
    'quaternary': '#27AE60',   # Emerald green
    'quinary': '#9B59B6',      # Purple
    'light_gray': '#ECF0F1',
    'dark_gray': '#7F8C8D',
    'gradient': ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#E74C3C'],  # Blue to red
}

# Quantile colors - harmonious gradient from cool to warm
QUANTILE_COLORS = {
    50: '#3498DB',   # Blue - median
    75: '#27AE60',   # Green
    90: '#F39C12',   # Orange
    95: '#E74C3C',   # Coral red
    99: '#8E44AD',   # Deep purple
}


# ============================================================================
# ecDNA Distribution Evolution Plot (Violin/ECDF/Quantile bands)
# ============================================================================

def plot_ecdna_distribution_evolution(result, 
                                       mode='violin',
                                       time_points=None,
                                       n_time_points=8,
                                       title="ecDNA Copy Number Distribution Evolution",
                                       save_path=None,
                                       figsize=(14, 10)):
    """
    Plot single-cell ecDNA copy number distribution evolution over time.
    
    This visualization reveals intratumoral heterogeneity driven by uneven 
    segregation during cell division - a key feature of ecDNA dynamics.
    
    Args:
        result: SimulationResult object with ecdna_distributions
        mode: 'violin', 'ecdf', 'ridge', or 'combined' (default: 'violin')
        time_points: Specific time indices to plot (if None, auto-select)
        n_time_points: Number of time points to show if auto-selecting
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    # Validate data
    if not hasattr(result, 'ecdna_distributions') or not result.ecdna_distributions:
        print("Warning: No ecDNA distribution data available. Run simulation with updated code.")
        return None
    
    # Select time points to plot
    n_total = len(result.times)
    if time_points is None:
        # Auto-select evenly spaced time points
        indices = np.linspace(0, n_total - 1, min(n_time_points, n_total), dtype=int)
    else:
        indices = np.array(time_points)
    
    times = [result.times[i] for i in indices]
    distributions = [result.ecdna_distributions[i] for i in indices]
    
    # Filter out empty distributions
    valid_data = [(t, d) for t, d in zip(times, distributions) if len(d) > 0]
    if not valid_data:
        print("Warning: All distributions are empty.")
        return None
    
    times, distributions = zip(*valid_data)
    
    if mode == 'combined':
        fig = _plot_combined_distribution(result, times, distributions, title, figsize)
    elif mode == 'violin':
        fig = _plot_violin_distribution(times, distributions, title, figsize)
    elif mode == 'ecdf':
        fig = _plot_ecdf_distribution(times, distributions, title, figsize)
    elif mode == 'ridge':
        fig = _plot_ridge_distribution(times, distributions, title, figsize)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'violin', 'ecdf', 'ridge', or 'combined'.")
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def _plot_combined_distribution(result, times, distributions, title, figsize):
    """Create a combined multi-panel figure with violin, ECDF, and quantile trajectories."""
    fig = plt.figure(figsize=figsize)
    
    # Create grid: 2 rows, 2 columns with custom heights
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.5, 1],
                          hspace=0.3, wspace=0.25)
    
    # Panel A: Violin plots over time
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_violin_panel(ax1, times, distributions)
    ax1.set_title('A  Single-cell distribution over time', loc='left', fontweight='bold', fontsize=11)
    
    # Panel B: ECDF comparison
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_ecdf_panel(ax2, times, distributions)
    ax2.set_title('B  Cumulative distribution', loc='left', fontweight='bold', fontsize=11)
    
    # Panel C: Quantile trajectories (full time series)
    ax3 = fig.add_subplot(gs[1, :])
    _draw_quantile_trajectory_panel(ax3, result)
    ax3.set_title('C  Quantile trajectories', loc='left', fontweight='bold', fontsize=11)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    return fig


def _draw_violin_panel(ax, times, distributions):
    """Draw violin plot panel."""
    # Create custom colormap for time progression
    n_times = len(times)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_times))
    
    # Prepare data for violin plot
    positions = np.arange(n_times)
    
    # Draw violins manually for better control
    for i, (t, dist) in enumerate(zip(times, distributions)):
        if len(dist) < 2:
            continue
            
        # Compute KDE
        try:
            kde = stats.gaussian_kde(dist, bw_method='scott')
            y_range = np.linspace(max(0, dist.min() - 5), dist.max() + 5, 200)
            density = kde(y_range)
            
            # Normalize density for visualization
            density = density / density.max() * 0.4
            
            # Draw violin shape
            ax.fill_betweenx(y_range, positions[i] - density, positions[i] + density,
                            color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Add quantile markers
            quantiles = np.percentile(dist, [25, 50, 75])
            ax.scatter([positions[i]] * 3, quantiles, color='white', s=15, zorder=5, 
                      edgecolor=colors[i], linewidth=1)
            ax.scatter([positions[i]], [quantiles[1]], color='white', s=30, zorder=6,
                      edgecolor='black', linewidth=1.5)
            
        except Exception:
            # Fallback: simple box representation
            ax.boxplot([dist], positions=[positions[i]], widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.7))
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels([f't={t:.1f}' for t in times], rotation=45, ha='right')
    ax.set_ylabel('ecDNA copy number')
    ax.set_xlabel('Time')
    
    # Add subtle grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def _draw_ecdf_panel(ax, times, distributions):
    """Draw ECDF panel with time-colored curves."""
    n_times = len(times)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_times))
    
    for i, (t, dist) in enumerate(zip(times, distributions)):
        if len(dist) < 2:
            continue
        
        # Compute ECDF
        sorted_data = np.sort(dist)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot ECDF
        ax.step(sorted_data, ecdf, where='post', color=colors[i], 
                linewidth=2, alpha=0.8, label=f't={t:.1f}')
    
    ax.set_xlabel('ecDNA copy number')
    ax.set_ylabel('Cumulative probability')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')


def _draw_quantile_trajectory_panel(ax, result):
    """Draw quantile trajectories over full time series."""
    times = np.array(result.times)
    
    # Compute quantiles at each time point
    quantiles_to_plot = [50, 75, 90, 95, 99]
    quantile_data = {q: [] for q in quantiles_to_plot}
    
    valid_times = []
    for i, dist in enumerate(result.ecdna_distributions):
        if len(dist) > 0:
            valid_times.append(result.times[i])
            for q in quantiles_to_plot:
                quantile_data[q].append(np.percentile(dist, q))
    
    valid_times = np.array(valid_times)
    
    # Plot quantile bands (shaded regions between quantiles)
    quantile_pairs = [(50, 75), (75, 90), (90, 95), (95, 99)]
    fill_colors = ['#E8F4FD', '#D4EDDA', '#FFF3CD', '#F8D7DA']
    fill_alphas = [0.6, 0.5, 0.4, 0.3]
    
    for (q_low, q_high), color, alpha in zip(quantile_pairs, fill_colors, fill_alphas):
        ax.fill_between(valid_times, quantile_data[q_low], quantile_data[q_high],
                       color=color, alpha=alpha, linewidth=0)
    
    # Plot quantile lines
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    for q, ls in zip(quantiles_to_plot, line_styles):
        ax.plot(valid_times, quantile_data[q], 
                color=QUANTILE_COLORS[q], linewidth=2, linestyle=ls,
                label=f'{q}th percentile')
    
    # Add mean line
    means = [np.mean(d) if len(d) > 0 else np.nan for d in result.ecdna_distributions]
    valid_means = [m for m, d in zip(means, result.ecdna_distributions) if len(d) > 0]
    ax.plot(valid_times, valid_means, color=PALETTE['primary'], linewidth=2.5,
            linestyle='-', label='Mean', alpha=0.9)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('ecDNA copy number')
    ax.legend(loc='upper left', ncol=3, framealpha=0.9, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(valid_times[0], valid_times[-1])


def _plot_violin_distribution(times, distributions, title, figsize):
    """Create standalone violin plot figure."""
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.6))
    _draw_violin_panel(ax, times, distributions)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_ecdf_distribution(times, distributions, title, figsize):
    """Create standalone ECDF plot figure."""
    fig, ax = plt.subplots(figsize=(figsize[0] * 0.6, figsize[1] * 0.6))
    _draw_ecdf_panel(ax, times, distributions)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_ridge_distribution(times, distributions, title, figsize):
    """Create ridge plot (joy plot) showing distribution evolution."""
    fig, ax = plt.subplots(figsize=figsize)
    
    n_times = len(times)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_times))
    
    # Determine global x-range
    all_data = np.concatenate([d for d in distributions if len(d) > 0])
    x_min, x_max = all_data.min(), all_data.max()
    x_range = np.linspace(x_min - 5, x_max + 5, 300)
    
    # Vertical spacing
    overlap = 0.7
    
    for i, (t, dist) in enumerate(zip(times, distributions)):
        if len(dist) < 2:
            continue
        
        try:
            kde = stats.gaussian_kde(dist, bw_method='scott')
            density = kde(x_range)
            
            # Normalize
            density = density / density.max()
            
            # Offset for ridge effect
            baseline = i * (1 - overlap)
            
            # Fill
            ax.fill_between(x_range, baseline, baseline + density * 0.8,
                           color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.8)
            
            # Add time label
            ax.text(x_min - 2, baseline + 0.3, f't={t:.1f}', 
                   fontsize=9, ha='right', va='center')
            
        except Exception:
            continue
    
    ax.set_xlabel('ecDNA copy number')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(x_min - 10, x_max + 10)
    
    plt.tight_layout()
    return fig


def plot_heterogeneity_metrics(result, 
                                title="ecDNA Heterogeneity Analysis",
                                save_path=None,
                                figsize=(12, 8)):
    """
    Plot metrics quantifying ecDNA heterogeneity over time.
    
    Includes:
    - Coefficient of variation (CV)
    - Gini coefficient
    - Tail weight (fraction of cells above various thresholds)
    - Shannon entropy of binned distribution
    
    Args:
        result: SimulationResult object
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not hasattr(result, 'ecdna_distributions') or not result.ecdna_distributions:
        print("Warning: No ecDNA distribution data available.")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    valid_times = []
    cv_values = []
    gini_values = []
    tail_fractions = {90: [], 95: [], 99: []}
    entropy_values = []
    
    for i, dist in enumerate(result.ecdna_distributions):
        if len(dist) < 2:
            continue
        
        valid_times.append(result.times[i])
        
        # Coefficient of variation
        cv = np.std(dist) / np.mean(dist) if np.mean(dist) > 0 else 0
        cv_values.append(cv)
        
        # Gini coefficient
        gini = _compute_gini(dist)
        gini_values.append(gini)
        
        # Tail fractions (cells above percentile thresholds of initial distribution)
        if i == 0:
            initial_percentiles = {q: np.percentile(dist, q) for q in [90, 95, 99]}
        for q in [90, 95, 99]:
            if i == 0:
                tail_fractions[q].append(1 - q/100)
            else:
                frac = np.mean(dist > initial_percentiles.get(q, np.percentile(dist, q)))
                tail_fractions[q].append(frac)
        
        # Shannon entropy (binned)
        entropy = _compute_entropy(dist)
        entropy_values.append(entropy)
    
    valid_times = np.array(valid_times)
    
    # Panel A: CV over time
    ax = axes[0, 0]
    ax.plot(valid_times, cv_values, color=PALETTE['primary'], linewidth=2.5)
    ax.fill_between(valid_times, 0, cv_values, color=PALETTE['tertiary'], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('A  Dispersion (CV)', loc='left', fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Gini coefficient
    ax = axes[0, 1]
    ax.plot(valid_times, gini_values, color=PALETTE['secondary'], linewidth=2.5)
    ax.fill_between(valid_times, 0, gini_values, color=PALETTE['secondary'], alpha=0.15)
    ax.set_xlabel('Time')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('B  Inequality (Gini)', loc='left', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    
    # Panel C: Tail fractions
    ax = axes[1, 0]
    for q, color in [(90, PALETTE['quaternary']), (95, PALETTE['secondary']), (99, PALETTE['quinary'])]:
        ax.plot(valid_times, tail_fractions[q], linewidth=2, 
                label=f'>{q}th initial percentile', color=color)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of cells')
    ax.set_title('C  Tail expansion', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, None)
    
    # Panel D: Shannon entropy
    ax = axes[1, 1]
    ax.plot(valid_times, entropy_values, color=PALETTE['quinary'], linewidth=2.5)
    ax.fill_between(valid_times, 0, entropy_values, color=PALETTE['quinary'], alpha=0.15)
    ax.set_xlabel('Time')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('D  Distribution entropy', loc='left', fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def _compute_gini(data):
    """Compute Gini coefficient for a distribution."""
    data = np.array(data)
    if len(data) == 0 or np.sum(data) == 0:
        return 0
    sorted_data = np.sort(data)
    n = len(data)
    cumsum = np.cumsum(sorted_data)
    return (2 * np.sum((np.arange(1, n+1) * sorted_data)) - (n + 1) * np.sum(sorted_data)) / (n * np.sum(sorted_data))


def _compute_entropy(data, n_bins=30):
    """Compute Shannon entropy of binned distribution."""
    if len(data) < 2:
        return 0
    hist, _ = np.histogram(data, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    if len(hist) == 0:
        return 0
    # Normalize to probability
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-10))


def plot_muller_ecdna(result,
                       bins=None,
                       bin_labels=None,
                       title="ecDNA Clonal Dynamics (Muller Plot)",
                       save_path=None,
                       figsize=(12, 6)):
    """
    Plot Muller-style stacked area chart showing ecDNA copy number bin dynamics.
    
    Visualizes clonal sweeps and selection: high-copy subpopulations can rapidly
    expand under selective pressure, explaining why ecDNA is more prevalent in
    metastatic and heavily-treated tumors.
    
    Args:
        result: SimulationResult object with ecdna_distributions
        bins: List of bin edges (default: [0, 1, 6, 11, 21, inf] for 0, 1-5, 6-10, 11-20, >20)
        bin_labels: Labels for each bin
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not hasattr(result, 'ecdna_distributions') or not result.ecdna_distributions:
        print("Warning: No ecDNA distribution data available.")
        return None
    
    # Default bins: 0, 1-5, 6-10, 11-20, >20
    if bins is None:
        bins = [0, 1, 6, 11, 21, np.inf]
    if bin_labels is None:
        bin_labels = ['0', '1-5', '6-10', '11-20', '>20']
    
    # Compute bin fractions at each time point
    valid_times = []
    bin_fractions = {label: [] for label in bin_labels}
    
    for i, dist in enumerate(result.ecdna_distributions):
        if len(dist) == 0:
            continue
        valid_times.append(result.times[i])
        
        # Count cells in each bin
        total = len(dist)
        for j, label in enumerate(bin_labels):
            lower = bins[j]
            upper = bins[j + 1]
            if j == 0:  # First bin: exactly 0
                count = np.sum(dist == 0)
            else:
                count = np.sum((dist >= lower) & (dist < upper))
            bin_fractions[label].append(count / total)
    
    valid_times = np.array(valid_times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Muller-style color palette (cool to warm gradient)
    colors = [
        '#4575B4',  # Blue - no ecDNA
        '#91BFDB',  # Light blue
        '#FEE090',  # Light yellow
        '#FC8D59',  # Orange
        '#D73027',  # Red - high ecDNA
    ]
    
    # Prepare data for stackplot
    y_data = [bin_fractions[label] for label in bin_labels]
    
    # Draw stacked area (Muller plot)
    ax.stackplot(valid_times, *y_data, labels=bin_labels, colors=colors, alpha=0.85)
    
    # Add subtle edge lines between areas
    cumsum = np.zeros(len(valid_times))
    for i, (label, data) in enumerate(zip(bin_labels, y_data)):
        cumsum = cumsum + np.array(data)
        if i < len(bin_labels) - 1:
            ax.plot(valid_times, cumsum, color='white', linewidth=0.5, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of cells')
    ax.set_ylim(0, 1)
    ax.set_xlim(valid_times[0], valid_times[-1])
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title='ecDNA copies',
              framealpha=0.9, fontsize=9)
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.yaxis.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def plot_muller_comparison(results_dict,
                            bins=None,
                            bin_labels=None,
                            title="ecDNA Clonal Dynamics Comparison",
                            save_path=None,
                            figsize=(14, 8)):
    """
    Compare Muller plots across multiple conditions (e.g., treatments).
    
    Args:
        results_dict: Dict of {condition_name: SimulationResult}
        bins: Bin edges for ecDNA copy number
        bin_labels: Labels for bins
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    n_conditions = len(results_dict)
    if n_conditions == 0:
        return None
    
    # Default bins
    if bins is None:
        bins = [0, 1, 6, 11, 21, np.inf]
    if bin_labels is None:
        bin_labels = ['0', '1-5', '6-10', '11-20', '>20']
    
    # Colors
    colors = ['#4575B4', '#91BFDB', '#FEE090', '#FC8D59', '#D73027']
    
    # Create subplots
    fig, axes = plt.subplots(1, n_conditions, figsize=figsize, sharey=True)
    if n_conditions == 1:
        axes = [axes]
    
    for ax, (name, result) in zip(axes, results_dict.items()):
        if not hasattr(result, 'ecdna_distributions') or not result.ecdna_distributions:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name)
            continue
        
        # Compute bin fractions
        valid_times = []
        bin_fractions = {label: [] for label in bin_labels}
        
        for i, dist in enumerate(result.ecdna_distributions):
            if len(dist) == 0:
                continue
            valid_times.append(result.times[i])
            total = len(dist)
            for j, label in enumerate(bin_labels):
                lower = bins[j]
                upper = bins[j + 1]
                if j == 0:
                    count = np.sum(dist == 0)
                else:
                    count = np.sum((dist >= lower) & (dist < upper))
                bin_fractions[label].append(count / total)
        
        if not valid_times:
            continue
        
        valid_times = np.array(valid_times)
        y_data = [bin_fractions[label] for label in bin_labels]
        
        # Draw
        ax.stackplot(valid_times, *y_data, labels=bin_labels, colors=colors, alpha=0.85)
        
        # Subtle edges
        cumsum = np.zeros(len(valid_times))
        for i, data in enumerate(y_data):
            cumsum = cumsum + np.array(data)
            if i < len(bin_labels) - 1:
                ax.plot(valid_times, cumsum, color='white', linewidth=0.5, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_xlim(valid_times[0], valid_times[-1])
        ax.set_title(name, fontweight='bold', fontsize=11)
        ax.yaxis.grid(True, alpha=0.2, linestyle='--')
    
    axes[0].set_ylabel('Fraction of cells')
    
    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95),
               title='ecDNA copies', framealpha=0.9, fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def plot_fitness_landscape(result,
                            time_indices=None,
                            rate_type='both',
                            title="ecDNA-Fitness Landscape",
                            save_path=None,
                            figsize=(14, 10)):
    """
    Plot ecDNA vs cell fitness (division/death hazard) showing the fitness landscape.
    
    Visualizes how ecDNA copy number affects cell fate probabilities - the core
    biological question of whether ecDNA provides net proliferative advantage
    and at what copy number the benefit-cost tradeoff peaks.
    
    Args:
        result: SimulationResult object with fitness_snapshots
        time_indices: List of time indices to plot (if None, uses last time point)
        rate_type: 'division', 'death', 'net', or 'both' (division and death on same plot)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not hasattr(result, 'fitness_snapshots') or not result.fitness_snapshots:
        print("Warning: No fitness snapshot data available.")
        return None
    
    # Select time points
    if time_indices is None:
        # Use last time point with data
        time_indices = [len(result.fitness_snapshots) - 1]
    
    # Collect all data from selected time points
    all_data = []
    for idx in time_indices:
        if idx < len(result.fitness_snapshots):
            all_data.extend(result.fitness_snapshots[idx])
    
    if not all_data:
        print("Warning: No fitness data at selected time points.")
        return None
    
    # Extract arrays
    ecdna = np.array([d['ecdna'] for d in all_data])
    div_rates = np.array([d['div_rate'] for d in all_data])
    death_rates = np.array([d['death_rate'] for d in all_data])
    net_rates = np.array([d['net_rate'] for d in all_data])
    cycles = np.array([d['cycle'] for d in all_data])
    
    # Cell cycle colors
    cycle_names = ['G0', 'G1', 'S', 'G2/M']
    cycle_colors = ['#95A5A6', '#3498DB', '#27AE60', '#E74C3C']
    
    if rate_type == 'both':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Panel A: Division rate vs ecDNA (colored by cycle)
        ax = axes[0, 0]
        for c in range(4):
            mask = cycles == c
            if np.sum(mask) > 0:
                ax.scatter(ecdna[mask], div_rates[mask], c=cycle_colors[c], 
                          alpha=0.4, s=20, label=cycle_names[c], edgecolors='none')
        
        # Add smoothed trend line
        if len(ecdna) > 10:
            _add_smooth_trend(ax, ecdna, div_rates, color=PALETTE['primary'])
        
        ax.set_xlabel('ecDNA copy number')
        ax.set_ylabel('Division hazard rate')
        ax.set_title('A  Division rate vs ecDNA', loc='left', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, title='Cell cycle')
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        
        # Panel B: Death rate vs ecDNA
        ax = axes[0, 1]
        for c in range(4):
            mask = cycles == c
            if np.sum(mask) > 0:
                ax.scatter(ecdna[mask], death_rates[mask], c=cycle_colors[c],
                          alpha=0.4, s=20, label=cycle_names[c], edgecolors='none')
        
        if len(ecdna) > 10:
            _add_smooth_trend(ax, ecdna, death_rates, color=PALETTE['secondary'])
        
        ax.set_xlabel('ecDNA copy number')
        ax.set_ylabel('Death hazard rate')
        ax.set_title('B  Death rate vs ecDNA', loc='left', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, title='Cell cycle')
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        
        # Panel C: Net rate (division - death) vs ecDNA
        ax = axes[1, 0]
        for c in range(4):
            mask = cycles == c
            if np.sum(mask) > 0:
                ax.scatter(ecdna[mask], net_rates[mask], c=cycle_colors[c],
                          alpha=0.4, s=20, label=cycle_names[c], edgecolors='none')
        
        if len(ecdna) > 10:
            _add_smooth_trend(ax, ecdna, net_rates, color=PALETTE['quinary'])
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('ecDNA copy number')
        ax.set_ylabel('Net growth rate (div - death)')
        ax.set_title('C  Net fitness vs ecDNA', loc='left', fontweight='bold')
        ax.legend(loc='best', fontsize=8, title='Cell cycle')
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        
        # Panel D: 2D density / marginal histogram
        ax = axes[1, 1]
        _plot_fitness_density(ax, ecdna, net_rates)
        ax.set_xlabel('ecDNA copy number')
        ax.set_ylabel('Net growth rate')
        ax.set_title('D  Fitness landscape density', loc='left', fontweight='bold')
        
    else:
        # Single panel for specified rate type
        fig, ax = plt.subplots(figsize=(figsize[0]*0.6, figsize[1]*0.5))
        
        if rate_type == 'division':
            rates = div_rates
            ylabel = 'Division hazard rate'
            trend_color = PALETTE['primary']
        elif rate_type == 'death':
            rates = death_rates
            ylabel = 'Death hazard rate'
            trend_color = PALETTE['secondary']
        else:  # net
            rates = net_rates
            ylabel = 'Net growth rate'
            trend_color = PALETTE['quinary']
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        for c in range(4):
            mask = cycles == c
            if np.sum(mask) > 0:
                ax.scatter(ecdna[mask], rates[mask], c=cycle_colors[c],
                          alpha=0.4, s=20, label=cycle_names[c], edgecolors='none')
        
        if len(ecdna) > 10:
            _add_smooth_trend(ax, ecdna, rates, color=trend_color)
        
        ax.set_xlabel('ecDNA copy number')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=9, title='Cell cycle')
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def _add_smooth_trend(ax, x, y, color, n_bins=20):
    """Add smoothed trend line using binned means with confidence bands."""
    x = np.array(x)
    y = np.array(y)
    
    # Filter valid data
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    
    if len(x) < 10:
        return
    
    # Bin the data
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.sum(mask) >= 3:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(y[mask]))
            bin_stds.append(np.std(y[mask]) / np.sqrt(np.sum(mask)))  # SEM
    
    if len(bin_centers) < 3:
        return
    
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    # Plot trend line with confidence band
    ax.plot(bin_centers, bin_means, color=color, linewidth=2.5, zorder=10)
    ax.fill_between(bin_centers, bin_means - 1.96*bin_stds, bin_means + 1.96*bin_stds,
                    color=color, alpha=0.2, zorder=5)


def _plot_fitness_density(ax, ecdna, net_rates):
    """Plot 2D density of ecDNA vs net fitness."""
    from scipy.stats import gaussian_kde
    
    # Filter valid data
    valid = np.isfinite(ecdna) & np.isfinite(net_rates)
    x, y = ecdna[valid], net_rates[valid]
    
    if len(x) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return
    
    try:
        # Compute 2D KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method='scott')
        
        # Create grid
        x_grid = np.linspace(x.min(), x.max(), 50)
        y_grid = np.linspace(y.min(), y.max(), 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Plot density
        ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd', alpha=0.8)
        ax.contour(X, Y, Z, levels=5, colors='white', linewidths=0.5, alpha=0.5)
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
    except Exception:
        # Fallback: scatter plot
        ax.scatter(x, y, alpha=0.3, s=10, c=PALETTE['tertiary'])
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)


def plot_lineage_tree(result,
                       n_lineages=5,
                       max_depth=6,
                       title="ecDNA Inheritance Lineage Tree",
                       save_path=None,
                       figsize=(14, 10)):
    """
    Plot lineage tree showing ecDNA as a heritable but stochastically drifting trait.
    
    Visualizes non-Mendelian inheritance: same ancestor can produce vastly different
    descendants due to random segregation at division - a key mechanism driving
    ecDNA-mediated rapid evolution and heterogeneity.
    
    Args:
        result: SimulationResult object with events
        n_lineages: Number of founder lineages to trace
        max_depth: Maximum tree depth (generations)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not result.events:
        print("Warning: No events recorded. Cannot plot lineage tree.")
        return None
    
    # Build lineage data structure from division events
    divisions = {}
    deaths = set()
    cell_ecdna = {}  # cell_id -> total ecDNA at division/death
    
    for time, event_type, cell_id, details in result.events:
        if event_type == "division":
            parent_state = details.get("state_pre", {})
            d1_state = details.get("d1_state", {})
            d2_state = details.get("d2_state", {})
            parent_k = np.array(parent_state.get("k", details.get("parent_k", [0])))
            d1_k = np.array(d1_state.get("k", details.get("d1_k", [0])))
            d2_k = np.array(d2_state.get("k", details.get("d2_k", [0])))
            d1_id = details.get("d1_id")
            d2_id = details.get("d2_id")
            
            divisions[cell_id] = {
                "time": time,
                "parent_ecdna": int(np.sum(parent_k)),
                "d1_id": d1_id,
                "d2_id": d2_id,
                "d1_ecdna": int(np.sum(d1_k)),
                "d2_ecdna": int(np.sum(d2_k)),
            }
            cell_ecdna[cell_id] = int(np.sum(parent_k))
            cell_ecdna[d1_id] = int(np.sum(d1_k))
            cell_ecdna[d2_id] = int(np.sum(d2_k))
            
        elif event_type == "death":
            deaths.add(cell_id)
            state_pre = details.get("state_pre", {})
            k = np.array(state_pre.get("k", details.get("k", [0])))
            cell_ecdna[cell_id] = int(np.sum(k))
    
    if not divisions:
        print("Warning: No division events found.")
        return None
    
    # Find founder cells (cells that divided but have no recorded parent division)
    all_daughters = set()
    for d in divisions.values():
        all_daughters.add(d["d1_id"])
        all_daughters.add(d["d2_id"])
    
    founders = [cid for cid in divisions.keys() if cid not in all_daughters]
    founders = sorted(founders)[:n_lineages]  # Take first n founders
    
    if not founders:
        print("Warning: No founder cells found.")
        return None
    
    # Build trees recursively
    def build_tree(cell_id, depth=0):
        """Build tree structure recursively."""
        node = {
            "id": cell_id,
            "ecdna": cell_ecdna.get(cell_id, 0),
            "depth": depth,
            "divided": cell_id in divisions,
            "died": cell_id in deaths,
            "children": []
        }
        if cell_id in divisions and depth < max_depth:
            div = divisions[cell_id]
            node["div_time"] = div["time"]
            node["children"] = [
                build_tree(div["d1_id"], depth + 1),
                build_tree(div["d2_id"], depth + 1)
            ]
        return node
    
    trees = [build_tree(f) for f in founders]
    
    # Determine color scale
    all_ecdna = list(cell_ecdna.values())
    vmin, vmax = min(all_ecdna), max(all_ecdna)
    if vmax == vmin:
        vmax = vmin + 1
    
    # Create figure
    fig, axes = plt.subplots(1, len(trees), figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Color map
    cmap = plt.cm.RdYlBu_r  # Red (high) - Yellow - Blue (low)
    
    def draw_tree(ax, tree, x_center=0.5, x_width=1.0, y=1.0, y_step=0.15):
        """Recursively draw tree on axes."""
        # Node color based on ecDNA
        norm_ecdna = (tree["ecdna"] - vmin) / (vmax - vmin)
        color = cmap(norm_ecdna)
        
        # Node size based on fate
        if tree["divided"]:
            size = 300
            marker = 'o'
        elif tree["died"]:
            size = 150
            marker = 'X'
        else:
            size = 200
            marker = 's'  # Still alive
        
        # Draw node
        ax.scatter([x_center], [y], c=[color], s=size, marker=marker,
                   edgecolor='white', linewidth=1, zorder=3)
        
        # Add ecDNA label
        ax.annotate(str(tree["ecdna"]), (x_center, y), 
                    textcoords="offset points", xytext=(0, -20),
                    ha='center', fontsize=7, color=PALETTE['dark_gray'])
        
        # Draw children
        if tree["children"]:
            n_children = len(tree["children"])
            child_width = x_width / n_children
            for i, child in enumerate(tree["children"]):
                child_x = x_center - x_width/2 + child_width * (i + 0.5)
                child_y = y - y_step
                
                # Draw edge
                ax.plot([x_center, child_x], [y, child_y], 
                        color=PALETTE['dark_gray'], linewidth=1, alpha=0.6, zorder=1)
                
                # Recurse
                draw_tree(ax, child, child_x, child_width * 0.9, child_y, y_step)
    
    # Draw each tree
    for i, (ax, tree) in enumerate(zip(axes, trees)):
        draw_tree(ax, tree)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        ax.set_title(f'Founder {tree["id"]}\n(ecDNA={tree["ecdna"]})', fontsize=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                        fraction=0.03, pad=0.08, aspect=40)
    cbar.set_label('ecDNA copy number', fontsize=10)
    
    # Add legend for markers
    legend_elements = [
        plt.scatter([], [], c='gray', s=200, marker='o', label='Divided'),
        plt.scatter([], [], c='gray', s=100, marker='X', label='Died'),
        plt.scatter([], [], c='gray', s=150, marker='s', label='Alive/Untracked'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def plot_ecdna_positive_fraction(result,
                                  threshold_high=20,
                                  use_quantile=None,
                                  title="ecDNA+ Cell Fraction Over Time",
                                  save_path=None,
                                  figsize=(10, 6)):
    """
    Plot ecDNA+ cell fraction and high-copy subpopulation fraction over time.
    
    Tracks the emergence and expansion of ecDNA-carrying cells and extreme 
    high-copy subclones - key indicators of tumor aggressiveness (Turner 2017, Nature).
    
    Args:
        result: SimulationResult object with ecdna_distributions
        threshold_high: Absolute threshold for "high ecDNA" (default: 20)
        use_quantile: If set (e.g., 95), use this percentile of initial distribution as threshold
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not hasattr(result, 'ecdna_distributions') or not result.ecdna_distributions:
        print("Warning: No ecDNA distribution data available.")
        return None
    
    # Compute fractions at each time point
    valid_times = []
    frac_positive = []      # P(ecDNA > 0)
    frac_high = []          # P(ecDNA >= threshold)
    
    # Determine threshold
    initial_dist = result.ecdna_distributions[0]
    if use_quantile is not None and len(initial_dist) > 0:
        threshold = np.percentile(initial_dist, use_quantile)
        threshold_label = f'{use_quantile}th percentile of t=0'
    else:
        threshold = threshold_high
        threshold_label = f'copy number ≥ {threshold}'
    
    for i, dist in enumerate(result.ecdna_distributions):
        if len(dist) == 0:
            continue
        valid_times.append(result.times[i])
        frac_positive.append(np.mean(dist > 0))
        frac_high.append(np.mean(dist >= threshold))
    
    valid_times = np.array(valid_times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ecDNA+ fraction
    ax.plot(valid_times, frac_positive, color=PALETTE['tertiary'], linewidth=2.5,
            label='ecDNA+ (copy number > 0)', marker='o', markersize=4, markevery=max(1, len(valid_times)//10))
    ax.fill_between(valid_times, 0, frac_positive, color=PALETTE['tertiary'], alpha=0.15)
    
    # Plot high-copy fraction
    ax.plot(valid_times, frac_high, color=PALETTE['secondary'], linewidth=2.5,
            label=f'High ecDNA ({threshold_label})', marker='s', markersize=4, markevery=max(1, len(valid_times)//10))
    ax.fill_between(valid_times, 0, frac_high, color=PALETTE['secondary'], alpha=0.15)
    
    # Styling
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction of cells')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(valid_times[0], valid_times[-1])
    ax.legend(loc='best', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(title, fontweight='bold', fontsize=12)
    
    # Add annotation for final values
    ax.annotate(f'{frac_positive[-1]:.1%}', xy=(valid_times[-1], frac_positive[-1]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color=PALETTE['tertiary'])
    ax.annotate(f'{frac_high[-1]:.1%}', xy=(valid_times[-1], frac_high[-1]),
                xytext=(5, -10), textcoords='offset points', fontsize=9, color=PALETTE['secondary'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


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
    
    plt.close(fig)


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
    
    plt.close(fig)


# ============================================================================
# Lineage State Trajectory Visualization
# ============================================================================

def plot_lineage_state_trajectory(result,
                                   n_lineages=5,
                                   max_events=50,
                                   title="Lineage State Trajectories",
                                   save_path=None,
                                   figsize=(16, 12)):
    """
    Plot state trajectories along cell lineages showing how states evolve.
    
    Visualizes the stochastic jumps in cell state (cycle, senescence, expression, 
    ecDNA copy number) along individual lineages - revealing the dynamics of 
    non-genetic heterogeneity and ecDNA inheritance.
    
    Args:
        result: SimulationResult with detailed events
        n_lineages: Number of lineages to trace
        max_events: Maximum events per lineage to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not result.events:
        print("Warning: No events recorded.")
        return None
    
    # Build lineage chains from events
    lineages = _build_lineage_chains(result.events, n_lineages, max_events)
    
    if not lineages:
        print("Warning: Could not build any lineages.")
        return None
    
    # Create figure with 4 rows: ecDNA, cycle, senescence, expression
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # State labels
    cycle_labels = ['G0', 'G1', 'S', 'G2/M']
    sen_labels = ['Normal', 'Pre-sen', 'Senescent']
    expr_labels = ['Basal', 'Activated']
    
    # Colors for different lineages
    lineage_colors = plt.cm.tab10(np.linspace(0, 1, min(n_lineages, 10)))
    
    for i, (lineage, color) in enumerate(zip(lineages, lineage_colors)):
        times = [e['time'] for e in lineage]
        
        # Panel A: ecDNA copy number
        ecdna_vals = [sum(e['k']) if e['k'] else 0 for e in lineage]
        axes[0].step(times, ecdna_vals, where='post', color=color, 
                     linewidth=2, alpha=0.8, label=f'Lineage {i+1}')
        axes[0].scatter(times, ecdna_vals, color=color, s=20, alpha=0.6, zorder=5)
        
        # Panel B: Cell cycle
        cycle_vals = [e['c'] for e in lineage]
        axes[1].step(times, cycle_vals, where='post', color=color, linewidth=2, alpha=0.8)
        axes[1].scatter(times, cycle_vals, color=color, s=20, alpha=0.6, zorder=5)
        
        # Panel C: Senescence
        sen_vals = [e['s'] for e in lineage]
        axes[2].step(times, sen_vals, where='post', color=color, linewidth=2, alpha=0.8)
        axes[2].scatter(times, sen_vals, color=color, s=20, alpha=0.6, zorder=5)
        
        # Panel D: Expression
        expr_vals = [e['x'] for e in lineage]
        axes[3].step(times, expr_vals, where='post', color=color, linewidth=2, alpha=0.8)
        axes[3].scatter(times, expr_vals, color=color, s=20, alpha=0.6, zorder=5)
    
    # Style panels
    axes[0].set_ylabel('ecDNA copies')
    axes[0].set_title('A  ecDNA Copy Number', loc='left', fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[0].yaxis.grid(True, alpha=0.3, linestyle='--')
    
    axes[1].set_ylabel('Cycle phase')
    axes[1].set_yticks(range(len(cycle_labels)))
    axes[1].set_yticklabels(cycle_labels)
    axes[1].set_title('B  Cell Cycle Phase', loc='left', fontweight='bold')
    axes[1].yaxis.grid(True, alpha=0.3, linestyle='--')
    
    axes[2].set_ylabel('Sen. state')
    axes[2].set_yticks(range(len(sen_labels)))
    axes[2].set_yticklabels(sen_labels)
    axes[2].set_title('C  Senescence Status', loc='left', fontweight='bold')
    axes[2].yaxis.grid(True, alpha=0.3, linestyle='--')
    
    axes[3].set_ylabel('Expr. program')
    axes[3].set_yticks(range(len(expr_labels)))
    axes[3].set_yticklabels(expr_labels)
    axes[3].set_xlabel('Time')
    axes[3].set_title('D  Expression Program', loc='left', fontweight='bold')
    axes[3].yaxis.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def plot_event_summary(result,
                        title="Event Type Distribution",
                        save_path=None,
                        figsize=(14, 8)):
    """
    Plot summary of all event types over time.
    
    Shows event frequency, timing, and state changes - useful for understanding
    the relative rates of different processes (division, death, transitions).
    
    Args:
        result: SimulationResult with events
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    
    if not result.events:
        print("Warning: No events recorded.")
        return None
    
    # Categorize events
    event_types = ['division', 'death', 'cycle', 'sen', 'expr', 'ecdna_gain', 'ecdna_loss']
    event_colors = {
        'division': PALETTE['quaternary'],
        'death': PALETTE['secondary'],
        'cycle': PALETTE['tertiary'],
        'sen': PALETTE['quinary'],
        'expr': '#F39C12',
        'ecdna_gain': '#2ECC71',
        'ecdna_loss': '#E74C3C',
    }
    
    # Collect event times by type
    event_times = {et: [] for et in event_types}
    for t, etype, cid, details in result.events:
        if etype in event_times:
            event_times[etype].append(t)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel A: Event counts (bar chart)
    ax = axes[0, 0]
    counts = [len(event_times[et]) for et in event_types]
    bars = ax.bar(event_types, counts, color=[event_colors[et] for et in event_types], 
                  edgecolor='white', linewidth=1)
    ax.set_ylabel('Count')
    ax.set_title('A  Total Event Counts', loc='left', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Panel B: Event rate over time (stacked area)
    ax = axes[0, 1]
    t_max = max(result.times) if result.times else 1
    n_bins = 20
    bin_edges = np.linspace(0, t_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    event_rates = {}
    for et in event_types:
        if event_times[et]:
            hist, _ = np.histogram(event_times[et], bins=bin_edges)
            event_rates[et] = hist / (t_max / n_bins)  # Rate per time unit
        else:
            event_rates[et] = np.zeros(n_bins)
    
    # Stack plot
    y_stack = np.vstack([event_rates[et] for et in event_types])
    ax.stackplot(bin_centers, y_stack, labels=event_types,
                 colors=[event_colors[et] for et in event_types], alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Events per time unit')
    ax.set_title('B  Event Rate Over Time', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    # Panel C: ecDNA change distribution (for gain/loss events)
    ax = axes[1, 0]
    ecdna_changes = []
    for t, etype, cid, details in result.events:
        if etype in ['ecdna_gain', 'ecdna_loss']:
            pre = details.get('state_pre', {})
            post = details.get('state_post', {})
            if pre and post and pre.get('k') and post.get('k'):
                delta = sum(post['k']) - sum(pre['k'])
                ecdna_changes.append(delta)
    
    if ecdna_changes:
        ax.hist(ecdna_changes, bins=30, color=PALETTE['tertiary'], 
                edgecolor='white', alpha=0.8)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('ecDNA change per event')
        ax.set_ylabel('Count')
    else:
        ax.text(0.5, 0.5, 'No ecDNA gain/loss events', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
    ax.set_title('C  ecDNA Change Distribution', loc='left', fontweight='bold')
    
    # Panel D: Division ecDNA partitioning
    ax = axes[1, 1]
    d1_ecdna, d2_ecdna = [], []
    for t, etype, cid, details in result.events:
        if etype == 'division':
            d1_state = details.get('d1_state', {})
            d2_state = details.get('d2_state', {})
            if d1_state.get('k') and d2_state.get('k'):
                d1_ecdna.append(sum(d1_state['k']))
                d2_ecdna.append(sum(d2_state['k']))
    
    if d1_ecdna:
        ax.scatter(d1_ecdna, d2_ecdna, alpha=0.4, s=20, c=PALETTE['primary'], edgecolors='none')
        max_val = max(max(d1_ecdna), max(d2_ecdna)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Equal partition')
        ax.set_xlabel('Daughter 1 ecDNA')
        ax.set_ylabel('Daughter 2 ecDNA')
        ax.legend(loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No division events', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
    ax.set_title('D  Division ecDNA Partitioning', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def _build_lineage_chains(events, n_lineages, max_events):
    """Build lineage chains from event log by tracing cell ancestry."""
    # Index events by cell_id
    cell_events = {}  # cell_id -> list of (time, event_type, details)
    parent_map = {}   # child_id -> parent_id
    
    for t, etype, cid, details in events:
        if cid not in cell_events:
            cell_events[cid] = []
        
        # Store state at this event
        state = details.get('state_pre', {}) or details.get('state_post', {})
        cell_events[cid].append({
            'time': t,
            'type': etype,
            'c': state.get('c', 0),
            's': state.get('s', 0),
            'x': state.get('x', 0),
            'k': state.get('k', [0]),
        })
        
        # Track parent-child relationships
        if etype == 'division':
            d1_id = details.get('d1_id')
            d2_id = details.get('d2_id')
            if d1_id is not None:
                parent_map[d1_id] = cid
                # Add initial state for daughter 1
                d1_state = details.get('d1_state', {})
                if d1_id not in cell_events:
                    cell_events[d1_id] = []
                cell_events[d1_id].append({
                    'time': t,
                    'type': 'birth',
                    'c': d1_state.get('c', 0),
                    's': d1_state.get('s', 0),
                    'x': d1_state.get('x', 0),
                    'k': d1_state.get('k', [0]),
                })
            if d2_id is not None:
                parent_map[d2_id] = cid
                d2_state = details.get('d2_state', {})
                if d2_id not in cell_events:
                    cell_events[d2_id] = []
                cell_events[d2_id].append({
                    'time': t,
                    'type': 'birth',
                    'c': d2_state.get('c', 0),
                    's': d2_state.get('s', 0),
                    'x': d2_state.get('x', 0),
                    'k': d2_state.get('k', [0]),
                })
    
    # Find cells with longest event histories (most interesting lineages)
    cell_ids_by_events = sorted(cell_events.keys(), 
                                 key=lambda cid: len(cell_events[cid]), 
                                 reverse=True)
    
    lineages = []
    for cid in cell_ids_by_events[:n_lineages]:
        # Build lineage by tracing back through parents
        chain = []
        current_id = cid
        visited = set()
        
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            if current_id in cell_events:
                chain = cell_events[current_id] + chain
            current_id = parent_map.get(current_id)
        
        # Limit to max_events
        if len(chain) > max_events:
            chain = chain[-max_events:]
        
        if chain:
            lineages.append(chain)
    
    return lineages
