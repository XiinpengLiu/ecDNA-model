"""
Visualization module for ecDNA kinetic model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, List, Optional, Tuple
import warnings


def _event_channel(event: str) -> str:
    if event is None:
        return "other"
    if event.startswith("k_gain"):
        return "k_gain"
    if event.startswith("k_loss"):
        return "k_loss"
    if event.startswith("m_switch"):
        return "m_switch"
    if event.startswith("e_switch"):
        return "e_switch"
    return event if event in {"div", "death"} else "other"


def _time_bins(event_log: List[Dict], time_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    times = np.array([e["t"] for e in event_log]) if event_log else np.array([])
    if times.size == 0:
        return np.array([]), np.array([])
    t_min = float(times.min())
    t_max = float(times.max())
    if t_min == t_max:
        t_max = t_min + 1.0
    edges = np.linspace(t_min, t_max, time_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _select_time_indices(times: np.ndarray, time_points: Optional[List[float]], n_default: int = 5) -> List[int]:
    if times.size == 0:
        return []
    if time_points is None:
        return list(np.linspace(0, len(times) - 1, min(n_default, len(times)), dtype=int))
    return [int(np.argmin(np.abs(times - t))) for t in time_points]


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
    
    indices = _select_time_indices(times, time_points)
    
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


def plot_ecdna_heatmaps_by_species(history: Dict, max_k: Optional[int] = None, time_resolution: int = 50) -> Optional[plt.Figure]:
    """Plot ecDNA distribution heatmaps for each species."""
    k_mats = history.get("k_matrix", [])
    if not k_mats:
        warnings.warn("k_matrix missing from history.")
        return None
    n_species = k_mats[0].shape[1] if k_mats[0].size > 0 else 0
    if n_species == 0:
        warnings.warn("No ecDNA species data available.")
        return None
    times = np.array(history["times"])
    time_indices = np.linspace(0, len(times) - 1, min(time_resolution, len(times)), dtype=int)
    if n_species == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_species, figsize=(4 * n_species, 4), sharey=True)
        axes = list(axes)
    for j in range(n_species):
        max_k_j = max_k
        if max_k_j is None:
            max_k_j = 0
            for idx in time_indices:
                if k_mats[idx].size > 0:
                    max_k_j = max(max_k_j, int(np.max(k_mats[idx][:, j])))
            max_k_j = max(max_k_j, 10)
        dist_matrix = np.zeros((max_k_j + 1, len(time_indices)))
        for col, idx in enumerate(time_indices):
            if k_mats[idx].size == 0:
                continue
            counts, _ = np.histogram(k_mats[idx][:, j], bins=np.arange(0, max_k_j + 2))
            counts = counts / (np.sum(counts) + 1e-10)
            dist_matrix[:, col] = counts
        im = axes[j].imshow(
            dist_matrix,
            aspect="auto",
            origin="lower",
            extent=[times[time_indices[0]], times[time_indices[-1]], 0, max_k_j],
            cmap="viridis",
        )
        axes[j].set_title(f"Species {j}")
        axes[j].set_xlabel("Time")
        if j == 0:
            axes[j].set_ylabel("ecDNA Copy Number")
        fig.colorbar(im, ax=axes[j], label="Density")
    plt.tight_layout()
    return fig


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


def plot_mean_ecdna_by_species(history: Dict, ax: Optional[plt.Axes] = None,
                               show_std: bool = True) -> plt.Axes:
    """Plot mean ecDNA per species over time."""
    if ax is None:
        _, ax = plt.subplots()
    if "mean_k_species" not in history:
        warnings.warn("mean_k_species missing from history.")
        return ax
    times = np.array(history["times"])
    means = np.array(history["mean_k_species"])
    stds = np.array(history.get("std_k_species", [])) if show_std else None
    for j in range(means.shape[1]):
        line, = ax.plot(times, means[:, j], label=f"Species {j}")
        if show_std and stds.size:
            color = line.get_color()
            ax.fill_between(times, means[:, j] - stds[:, j], means[:, j] + stds[:, j], alpha=0.2, color=color)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean ecDNA")
    ax.set_title("Mean ecDNA by Species")
    ax.legend()
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
    
    x = results.get('param_values', results.get('values'))
    y = results['metrics_mean']
    yerr = results['metrics_std']
    
    ax.errorbar(x, y, yerr=yerr, capsize=3, marker='o', **kwargs)
    
    ax.set_xlabel(results.get('param_name', results.get('param', 'parameter')))
    ax.set_ylabel(ylabel)
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    param_label = results.get("param_name", results.get("param", "parameter"))
    ax.set_title(f'{ylabel} vs {param_label}')
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


def plot_population_fan_chart(histories: List[Dict], ax: Optional[plt.Axes] = None,
                              ci: float = 0.9, color: str = "C0") -> plt.Axes:
    """Plot mean population trajectory with a quantile band."""
    if ax is None:
        _, ax = plt.subplots()
    if not histories:
        return ax
    max_len = max(len(h["times"]) for h in histories)
    # Use the time grid from the longest trajectory (record intervals are shared).
    longest_history = max(histories, key=lambda h: len(h["times"]))
    times = np.array(longest_history["times"])

    pops = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        pop = np.asarray(h["population_size"], dtype=float)
        pops[i, : len(pop)] = pop

    mean_pop = np.nanmean(pops, axis=0)
    lower = np.nanquantile(pops, (1 - ci) / 2.0, axis=0)
    upper = np.nanquantile(pops, 1 - (1 - ci) / 2.0, axis=0)
    ax.fill_between(times, lower, upper, color=color, alpha=0.25, label=f"{int(ci*100)}% band")
    ax.plot(times, mean_pop, color=color, lw=2, label="Mean")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population Size")
    ax.set_title("Population Trajectory")
    ax.legend()
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
# Event-driven diagnostics
# =============================================================================

def plot_thinning_acceptance_rate(event_log: List[Dict], ax: Optional[plt.Axes] = None,
                                  time_bins: int = 30, plot_reject: bool = False) -> plt.Axes:
    """Plot thinning acceptance rate (or reject fraction) over time."""
    if ax is None:
        _, ax = plt.subplots()
    edges, centers = _time_bins(event_log, time_bins)
    if centers.size == 0:
        return ax
    accepted = np.zeros_like(centers)
    proposed = np.zeros_like(centers)
    for entry in event_log:
        idx = np.searchsorted(edges, entry["t"], side="right") - 1
        if idx < 0 or idx >= len(centers):
            continue
        accepted[idx] += 1
        proposed[idx] += max(1, entry.get("n_proposed", 1))
    rate = np.divide(accepted, proposed, out=np.zeros_like(accepted), where=proposed > 0)
    series = 1.0 - rate if plot_reject else rate
    ax.plot(centers, series, lw=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reject Fraction" if plot_reject else "Acceptance Rate")
    ax.set_title("Thinning Efficiency")
    ax.set_ylim(0, 1)
    return ax


def plot_event_channel_composition(event_log: List[Dict], ax: Optional[plt.Axes] = None,
                                   time_bins: int = 30, normalize: bool = True) -> plt.Axes:
    """Plot event channel counts (or proportions) as a stacked area chart."""
    if ax is None:
        _, ax = plt.subplots()
    edges, centers = _time_bins(event_log, time_bins)
    if centers.size == 0:
        return ax
    channels = ["div", "death", "k_gain", "k_loss", "m_switch", "e_switch", "other"]
    counts = np.zeros((len(channels), len(centers)))
    for entry in event_log:
        idx = np.searchsorted(edges, entry["t"], side="right") - 1
        if idx < 0 or idx >= len(centers):
            continue
        chan = _event_channel(entry.get("event", "other"))
        counts[channels.index(chan), idx] += 1
    if normalize:
        denom = counts.sum(axis=0)
        denom[denom == 0] = 1.0
        counts = counts / denom
    ax.stackplot(centers, counts, labels=channels, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title("Event Channel Composition")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    return ax


def plot_bound_ratio_over_time(event_log: List[Dict], ax: Optional[plt.Axes] = None,
                               time_bins: int = 30) -> plt.Axes:
    """Plot max r_total / bar_r over time bins."""
    if ax is None:
        _, ax = plt.subplots()
    edges, centers = _time_bins(event_log, time_bins)
    if centers.size == 0:
        return ax
    max_ratio = np.zeros_like(centers)
    for entry in event_log:
        idx = np.searchsorted(edges, entry["t"], side="right") - 1
        if idx < 0 or idx >= len(centers):
            continue
        max_ratio[idx] = max(max_ratio[idx], float(entry.get("bound_ratio", 0.0)))
    ax.plot(centers, max_ratio, lw=2)
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("max(r_total / bar_r)")
    ax.set_title("Bound Tightness")
    return ax


def plot_inter_event_time_distribution(event_log: List[Dict], ax: Optional[plt.Axes] = None,
                                       bins: int = 30) -> plt.Axes:
    """Plot distribution of inter-event times."""
    if ax is None:
        _, ax = plt.subplots()
    times = np.sort([e["t"] for e in event_log]) if event_log else np.array([])
    if times.size < 2:
        return ax
    dt = np.diff(times)
    ax.hist(dt, bins=bins, alpha=0.7)
    ax.set_xlabel("Inter-event Time")
    ax.set_ylabel("Count")
    ax.set_title("Inter-event Time Distribution")
    return ax


def plot_event_rate_decomposition_over_time(event_log: List[Dict], history: Optional[Dict] = None,
                                            ax: Optional[plt.Axes] = None, time_bins: int = 30) -> plt.Axes:
    """Plot empirical event rates by channel."""
    if ax is None:
        _, ax = plt.subplots()
    edges, centers = _time_bins(event_log, time_bins)
    if centers.size == 0:
        return ax
    channels = ["div", "death", "k_gain", "k_loss", "m_switch", "e_switch"]
    counts = np.zeros((len(channels), len(centers)))
    for entry in event_log:
        idx = np.searchsorted(edges, entry["t"], side="right") - 1
        if idx < 0 or idx >= len(centers):
            continue
        chan = _event_channel(entry.get("event", "other"))
        if chan in channels:
            counts[channels.index(chan), idx] += 1
    dt = edges[1:] - edges[:-1]
    rates = counts / dt
    if history is not None and history.get("times"):
        times = np.array(history["times"])
        pops = np.array(history["population_size"])
        pop_means = np.zeros_like(centers)
        for i, (t0, t1) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (times >= t0) & (times <= t1)
            if mask.any():
                pop_means[i] = np.mean(pops[mask])
        pop_means[pop_means == 0] = np.nan
        rates = rates / pop_means
    ax.stackplot(centers, rates, labels=channels, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate per time" + (" per cell" if history is not None else ""))
    ax.set_title("Event Rate Decomposition")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    return ax


# =============================================================================
# Phenotype and conditional plots
# =============================================================================

def plot_phenotype_distribution_over_time(history: Dict, time_points: Optional[List[float]] = None,
                                          ax: Optional[plt.Axes] = None, dim: int = 0,
                                          bins: int = 20) -> plt.Axes:
    """Plot phenotype distribution at selected time points."""
    if ax is None:
        _, ax = plt.subplots()
    if "y_values" not in history:
        warnings.warn("y_values missing from history.")
        return ax
    times = np.array(history["times"])
    indices = _select_time_indices(times, time_points)
    colors = plt.cm.plasma(np.linspace(0, 1, len(indices)))
    for idx, color in zip(indices, colors):
        y_vals = history["y_values"][idx]
        if y_vals.size == 0:
            continue
        y_slice = y_vals[:, dim] if y_vals.ndim > 1 else y_vals
        ax.hist(y_slice, bins=bins, density=True, alpha=0.5, color=color, label=f"t={times[idx]:.1f}")
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Density")
    ax.set_title("Phenotype Distribution Over Time")
    ax.legend()
    return ax


def plot_joint_k_y(history: Dict, time_point: Optional[float] = None, ax: Optional[plt.Axes] = None,
                   dim: int = 0, bins: int = 30) -> plt.Axes:
    """Plot joint k-y distribution at a selected time point."""
    if ax is None:
        _, ax = plt.subplots()
    if "y_values" not in history:
        warnings.warn("y_values missing from history.")
        return ax
    times = np.array(history["times"])
    indices = _select_time_indices(times, [time_point] if time_point is not None else None, n_default=1)
    if not indices:
        return ax
    idx = indices[0]
    k_vals = history["k_distribution"][idx]
    y_vals = history["y_values"][idx]
    if len(k_vals) == 0 or y_vals.size == 0:
        return ax
    y_slice = y_vals[:, dim] if y_vals.ndim > 1 else y_vals
    ax.hist2d(k_vals, y_slice, bins=bins, cmap="magma")
    ax.set_xlabel("ecDNA Copy Number")
    ax.set_ylabel("Phenotype")
    ax.set_title(f"k-y Joint at t={times[idx]:.1f}")
    return ax


def plot_corr_k_y_over_time(history: Dict, ax: Optional[plt.Axes] = None, dim: int = 0) -> plt.Axes:
    """Plot correlation between k and phenotype over time."""
    if ax is None:
        _, ax = plt.subplots()
    if "y_values" not in history:
        warnings.warn("y_values missing from history.")
        return ax
    times = np.array(history["times"])
    corrs = []
    for k_vals, y_vals in zip(history["k_distribution"], history["y_values"]):
        if len(k_vals) < 2 or y_vals.size == 0:
            corrs.append(np.nan)
            continue
        y_slice = y_vals[:, dim] if y_vals.ndim > 1 else y_vals
        corrs.append(float(np.corrcoef(k_vals, y_slice)[0, 1]))
    ax.plot(times, corrs, lw=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("corr(k, y)")
    ax.set_title("k-y Correlation Over Time")
    return ax


def plot_conditional_k_distribution(history: Dict, state: str = "m",
                                    time_point: Optional[float] = None,
                                    ax: Optional[plt.Axes] = None, bins: int = 20) -> plt.Axes:
    """Plot k distribution conditioned on m or e at one time point."""
    if ax is None:
        _, ax = plt.subplots()
    key = "m_values" if state == "m" else "e_values"
    if key not in history:
        warnings.warn(f"{key} missing from history.")
        return ax
    times = np.array(history["times"])
    indices = _select_time_indices(times, [time_point] if time_point is not None else None, n_default=1)
    if not indices:
        return ax
    idx = indices[0]
    k_vals = history["k_distribution"][idx]
    s_vals = history[key][idx]
    if len(k_vals) == 0:
        return ax
    for s in np.unique(s_vals):
        mask = s_vals == s
        ax.hist(k_vals[mask], bins=bins, alpha=0.5, density=True, label=f"{state}={int(s)}")
    ax.set_xlabel("ecDNA Copy Number")
    ax.set_ylabel("Density")
    ax.set_title(f"k | {state} at t={times[idx]:.1f}")
    ax.legend()
    return ax


def plot_mean_by_state_over_time(history: Dict, state: str = "m", variable: str = "k",
                                 dim: int = 0, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot E[k|state] or E[y|state] over time."""
    if ax is None:
        _, ax = plt.subplots()
    key = "m_values" if state == "m" else "e_values"
    if key not in history:
        warnings.warn(f"{key} missing from history.")
        return ax
    times = np.array(history["times"])
    state_vals = history[key]
    n_states = max((vals.max() for vals in state_vals if len(vals) > 0), default=-1) + 1
    if n_states <= 0:
        return ax
    means = np.full((len(times), n_states), np.nan)
    for i, vals in enumerate(state_vals):
        if len(vals) == 0:
            continue
        if variable == "y":
            y_vals = history["y_values"][i]
            data = y_vals[:, dim] if y_vals.ndim > 1 else y_vals
        else:
            data = history["k_distribution"][i]
        for s in range(n_states):
            mask = vals == s
            if mask.any():
                means[i, s] = float(np.mean(data[mask]))
    for s in range(n_states):
        ax.plot(times, means[:, s], label=f"{state}={s}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Mean {variable}")
    ax.set_title(f"Mean {variable} by {state} state")
    ax.legend()
    return ax


def plot_state_transition_heatmap(event_log: List[Dict], state: str = "m",
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot state transition counts as a heatmap."""
    if ax is None:
        _, ax = plt.subplots()
    prefix = "m_switch" if state == "m" else "e_switch"
    prev_key = "m_prev" if state == "m" else "e_prev"
    transitions = []
    for entry in event_log:
        event = entry.get("event", "")
        if not event.startswith(prefix):
            continue
        src = int(entry.get(prev_key, 0))
        dst = int(event.split("_")[-1])
        transitions.append((src, dst))
    if not transitions:
        return ax
    n_states = max(max(s for s, _ in transitions), max(d for _, d in transitions)) + 1
    mat = np.zeros((n_states, n_states))
    for src, dst in transitions:
        mat[src, dst] += 1
    im = ax.imshow(mat, origin="lower", cmap="viridis")
    ax.set_xlabel(f"{state}'")
    ax.set_ylabel(f"{state}")
    ax.set_title(f"{state} Transition Counts")
    plt.colorbar(im, ax=ax, label="Count")
    return ax


# =============================================================================
# Multi-species ecDNA plots
# =============================================================================

def plot_k_species_joint(history: Dict, species: Tuple[int, int] = (0, 1),
                         time_point: Optional[float] = None, ax: Optional[plt.Axes] = None,
                         bins: int = 30) -> plt.Axes:
    """Plot joint distribution of two ecDNA species."""
    if ax is None:
        _, ax = plt.subplots()
    if "k_matrix" not in history:
        warnings.warn("k_matrix missing from history.")
        return ax
    times = np.array(history["times"])
    indices = _select_time_indices(times, [time_point] if time_point is not None else None, n_default=1)
    if not indices:
        return ax
    idx = indices[0]
    k_mat = history["k_matrix"][idx]
    if k_mat.size == 0 or k_mat.shape[1] <= max(species):
        return ax
    ax.hist2d(k_mat[:, species[0]], k_mat[:, species[1]], bins=bins, cmap="viridis")
    ax.set_xlabel(f"K{species[0]}")
    ax.set_ylabel(f"K{species[1]}")
    ax.set_title(f"Joint K{species[0]}-K{species[1]} at t={times[idx]:.1f}")
    return ax


def plot_k_species_correlation_over_time(history: Dict, species: Tuple[int, int] = (0, 1),
                                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot correlation between two ecDNA species over time."""
    if ax is None:
        _, ax = plt.subplots()
    if "k_matrix" not in history:
        warnings.warn("k_matrix missing from history.")
        return ax
    times = np.array(history["times"])
    corrs = []
    for k_mat in history["k_matrix"]:
        if k_mat.size == 0 or k_mat.shape[1] <= max(species) or k_mat.shape[0] < 2:
            corrs.append(np.nan)
            continue
        corrs.append(float(np.corrcoef(k_mat[:, species[0]], k_mat[:, species[1]])[0, 1]))
    ax.plot(times, corrs, lw=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("corr(Ki, Kj)")
    ax.set_title(f"Species Correlation K{species[0]}-K{species[1]}")
    return ax


# =============================================================================
# Extinction analysis plots
# =============================================================================

def plot_survival_curve(times: np.ndarray, survival: np.ndarray,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot Kaplan-Meier style survival curve."""
    if ax is None:
        _, ax = plt.subplots()
    ax.step(times, survival, where="post", lw=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve")
    ax.set_ylim(0, 1)
    return ax


def plot_extinction_time_hist(extinction_times: List[float],
                              ax: Optional[plt.Axes] = None, bins: int = 20) -> plt.Axes:
    """Plot histogram of extinction times."""
    if ax is None:
        _, ax = plt.subplots()
    if extinction_times:
        ax.hist(extinction_times, bins=bins, alpha=0.7)
    ax.set_xlabel("Time to Extinction")
    ax.set_ylabel("Count")
    ax.set_title("Extinction Time Distribution")
    return ax


def plot_extinction_probability_ci(p: float, ci: Tuple[float, float],
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot extinction probability with confidence interval."""
    if ax is None:
        _, ax = plt.subplots()
    lower, upper = ci
    ax.errorbar([0], [p], yerr=[[p - lower], [upper - p]], fmt="o", capsize=4)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Extinction Probability")
    ax.set_title("Extinction Probability (CI)")
    ax.set_ylim(0, 1)
    return ax


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
