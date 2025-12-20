"""
Model Atlas visualisations for the ecDNA simulator.

Plots are grouped by component (hazards, thinning, division kernel, events).
The functions here are lightweight so they can be imported alongside
`visualization.py` without altering existing APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ecdna_model import CellState, ModelParameters


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _channel_rates(entry: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    rates = entry.get("rates_at_event", {})
    return (
        float(rates.get("div", 0.0)),
        float(rates.get("death", 0.0)),
        float(np.sum(rates.get("m_switch", 0.0))),
        float(np.sum(rates.get("e_switch", 0.0))),
        float(np.sum(rates.get("k_gain", 0.0))),
        float(np.sum(rates.get("k_loss", 0.0))),
    )


def _division_draw(params: ModelParameters, parent: CellState, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    amp = params.sample_amplification(parent, rng)
    k_tilde = 2 * parent.k + amp
    k1 = np.array([rng.binomial(int(k_tilde[j]), 0.5) for j in range(len(k_tilde))])
    k2 = k_tilde - k1
    daughters = []
    for k_r in (k1, k2):
        k_star = np.zeros_like(k_r)
        for j in range(len(k_r)):
            loss_prob = params.post_segregation_loss_prob(parent, j, int(k_r[j]))
            k_star[j] = rng.binomial(int(k_r[j]), 1 - loss_prob)
        daughters.append(k_star)
    return daughters[0], daughters[1]


# -----------------------------------------------------------------------------
# Event log views (raster, stacks, thinning diagnostics)
# -----------------------------------------------------------------------------


def plot_event_raster(event_log: List[Dict[str, Any]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if not event_log:
        ax.set_axis_off()
        return ax
    times = [e["t"] for e in event_log]
    labels = [e["event"] for e in event_log]
    unique = {name: idx for idx, name in enumerate(sorted(set(labels)))}
    for t, label in zip(times, labels):
        ax.vlines(t, unique[label], unique[label] + 0.8, color="tab:blue", linewidth=1.0)
    ax.set_yticks(list(unique.values()))
    ax.set_yticklabels(list(unique.keys()))
    ax.set_xlabel("Time")
    ax.set_title("Event raster")
    return ax


def plot_channel_stack(event_log: List[Dict[str, Any]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if not event_log:
        ax.set_axis_off()
        return ax
    times = [e["t"] for e in event_log]
    stacks = [[], [], [], [], [], []]
    for entry in event_log:
        rates = _channel_rates(entry)
        for idx, r in enumerate(rates):
            stacks[idx].append(r)
    labels = ["div", "death", "m_switch", "e_switch", "k_gain", "k_loss"]
    ax.stackplot(times, stacks, labels=labels, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
    ax.set_title("Channel intensities at events")
    ax.legend()
    return ax


def plot_acceptance_history(event_log: List[Dict[str, Any]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if not event_log:
        ax.set_axis_off()
        return ax
    times = [e["t"] for e in event_log]
    probs = [e.get("accept_prob", 0.0) for e in event_log]
    ax.plot(times, probs, marker="o", linestyle="-", label="r/bar_r")
    ax.set_xlabel("Time")
    ax.set_ylabel("Acceptance prob.")
    ax.set_ylim(0, 1.05)
    ax.set_title("Thinning acceptance trajectory")
    return ax


def plot_proposal_histogram(event_log: List[Dict[str, Any]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    counts = [e.get("n_proposals", 0) for e in event_log]
    ax.hist(counts, bins=np.arange(1, max(counts + [1]) + 2) - 0.5, color="tab:orange", edgecolor="black")
    ax.set_xlabel("Proposals until acceptance")
    ax.set_ylabel("Frequency")
    ax.set_title("Thinning proposal counts")
    return ax


def plot_waiting_time_hist(event_log: List[Dict[str, Any]], ax: Optional[plt.Axes] = None, bins: int = 20) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    deltas = [e.get("delta_last", 0.0) for e in event_log]
    ax.hist(deltas, bins=bins, color="tab:green", edgecolor="black", density=True)
    ax.set_xlabel(r"$\Delta t$ (accepted)")
    ax.set_ylabel("Density")
    ax.set_title("Accepted inter-event intervals")
    return ax


# -----------------------------------------------------------------------------
# Hazard structure views
# -----------------------------------------------------------------------------


def plot_hazard_vs_age(
    params: ModelParameters,
    k_value: Sequence[int],
    y_value: Optional[Sequence[float]] = None,
    a_grid: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if a_grid is None:
        a_grid = np.linspace(0, 5.0, 100)
    if y_value is None:
        y_value = params.ou_params.mean
    div_vals = []
    death_vals = []
    for a in a_grid:
        state = CellState(k=np.array(k_value, dtype=int), y=np.array(y_value, dtype=float), a=float(a))
        div_vals.append(params.lambda_div(state))
        death_vals.append(params.lambda_death(state))
    ax.plot(a_grid, div_vals, label="lambda_div")
    ax.plot(a_grid, death_vals, label="lambda_death")
    ax.set_xlabel("Age")
    ax.set_ylabel("Hazard")
    ax.set_title("Hazards vs age")
    ax.legend()
    return ax


def plot_hazard_vs_k(
    params: ModelParameters,
    k_range: np.ndarray,
    a_value: float = 3.0,
    y_value: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if y_value is None:
        y_value = params.ou_params.mean
    div_vals = []
    death_vals = []
    for k in k_range:
        state = CellState(k=np.array([k], dtype=int), y=np.array(y_value, dtype=float), a=a_value)
        div_vals.append(params.lambda_div(state))
        death_vals.append(params.lambda_death(state))
    ax.plot(k_range, div_vals, label="lambda_div")
    ax.plot(k_range, death_vals, label="lambda_death")
    ax.set_xlabel("Total ecDNA copies")
    ax.set_ylabel("Hazard")
    ax.set_title("Hazards vs copy number")
    ax.legend()
    return ax


def plot_hazard_heatmap(
    params: ModelParameters,
    k_range: np.ndarray,
    a_grid: np.ndarray,
    kind: str = "div",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    matrix = np.zeros((len(a_grid), len(k_range)))
    for i, a in enumerate(a_grid):
        for j, k in enumerate(k_range):
            state = CellState(k=np.array([k], dtype=int), y=params.ou_params.mean.copy(), a=float(a))
            matrix[i, j] = params.lambda_div(state) if kind == "div" else params.lambda_death(state)
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[k_range[0], k_range[-1], a_grid[0], a_grid[-1]],
        cmap="magma",
    )
    ax.set_xlabel("Total ecDNA copies")
    ax.set_ylabel("Age")
    ax.set_title(f"lambda_{kind} heatmap")
    plt.colorbar(im, ax=ax, label="Hazard")
    return ax


def plot_gain_loss_hazard(params: ModelParameters, k_range: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    gain = []
    loss = []
    for k in k_range:
        state = CellState(k=np.array([k], dtype=int), y=params.ou_params.mean.copy(), a=1.0)
        gain.append(params.mu_gain(state, 0))
        loss.append(params.mu_loss(state, 0) * k)
    ax.plot(k_range, gain, label="mu_gain")
    ax.plot(k_range, loss, label="mu_loss * k")
    ax.set_xlabel("ecDNA copies")
    ax.set_ylabel("Hazard")
    ax.set_title("Gain/loss hazards vs copy number")
    ax.legend()
    return ax


# -----------------------------------------------------------------------------
# Division kernel diagnostics
# -----------------------------------------------------------------------------


def plot_amplification_distribution(
    params: ModelParameters,
    parent_state: CellState,
    n_samples: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    rng = np.random.default_rng()
    amps = [params.sample_amplification(parent_state, rng) for _ in range(n_samples)]
    amps = np.array(amps)
    if amps.ndim == 1:
        amps = amps[:, None]
    for j in range(amps.shape[1]):
        ax.hist(amps[:, j], bins=np.arange(0, int(amps[:, j].max()) + 2) - 0.5, alpha=0.6, label=f"species {j}")
    ax.set_xlabel("Amplification count")
    ax.set_ylabel("Frequency")
    ax.set_title("Division amplification draws")
    ax.legend()
    return ax


def plot_segregation_distribution(
    params: ModelParameters,
    parent_state: CellState,
    n_samples: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    rng = np.random.default_rng()
    draws = []
    for _ in range(n_samples):
        k1, _ = _division_draw(params, parent_state, rng)
        draws.append(int(k1[0]))
    bins = np.arange(0, max(draws + [0]) + 2) - 0.5
    ax.hist(draws, bins=bins, density=True, alpha=0.7, label="simulated")
    k_parent = int(parent_state.k[0])
    x_vals = np.arange(0, 2 * k_parent + 1)
    pmf = [np.math.comb(2 * k_parent, x) * (0.5 ** (2 * k_parent)) for x in x_vals]
    ax.plot(x_vals, pmf, marker="o", linestyle="--", label="binomial ref")
    ax.set_xlabel("Daughter copy count")
    ax.set_ylabel("Density")
    ax.set_title("Segregation distribution")
    ax.legend()
    return ax


def plot_daughter_scatter(
    params: ModelParameters,
    parent_state: CellState,
    n_samples: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    rng = np.random.default_rng()
    d1_vals = []
    d2_vals = []
    for _ in range(n_samples):
        k1, k2 = _division_draw(params, parent_state, rng)
        d1_vals.append(int(k1[0]))
        d2_vals.append(int(k2[0]))
    ax.scatter(d1_vals, d2_vals, alpha=0.5, s=15)
    ax.plot([0, max(d1_vals + d2_vals)], [max(d1_vals + d2_vals), 0], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("K1 copies")
    ax.set_ylabel("K2 copies")
    ax.set_title("Daughter copy-number pairs")
    return ax

