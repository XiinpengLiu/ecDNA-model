"""
Minimal plotting utilities for the ecDNA model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from core.simulation import SimulationResult


STATE_COLORS = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")
SPECIES_COLORS = ("#1d4ed8", "#be123c", "#047857")
EVENT_COLORS = {
    "division": "#2563eb",
    "death": "#111827",
    "cycle": "#7c3aed",
    "ecDNA gain": "#16a34a",
    "ecDNA loss": "#f97316",
    "amplification": "#dc2626",
    "other": "#6b7280",
}


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_results(result: SimulationResult, title: str = "ecDNA simulation", save_path: str | Path | None = None) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    state_fractions = np.asarray(result.soft_state_fractions, dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, result.population_sizes, color="#1f2937", linewidth=2)
    axes[0, 0].set_title("Population size")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Cells")

    for idx, state_name in enumerate(cfg.STATE_NAMES):
        axes[0, 1].plot(times, state_fractions[:, idx], linewidth=2, label=state_name)
    axes[0, 1].set_title("Soft state fractions")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for idx, species in enumerate(cfg.SPECIES):
        axes[1, 0].plot(times, bulk[:, idx], linewidth=2, label=species)
    axes[1, 0].set_title("Bulk mean ecDNA copies")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Mean copies")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(times, result.mean_stress_scores, label="Stress score", linewidth=2)
    axes[1, 1].plot(times, result.mean_survival_scores, label="Survival score", linewidth=2)
    axes[1, 1].set_title("Latent stress and survival")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_observation_proxies(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    flow_fractions = np.asarray([snapshot["flow_fractions"] for snapshot in result.observations], dtype=float)
    qpcdr_means = np.asarray([snapshot["pooled_qpcdr_means"] for snapshot in result.observations], dtype=float)
    counts = np.asarray([snapshot["observed_count"] for snapshot in result.observations], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].stackplot(times, flow_fractions.T, labels=cfg.STATE_NAMES)
    axes[0].set_title("Observed flow fractions")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fraction")

    for idx, species in enumerate(cfg.SPECIES):
        axes[1].plot(times, qpcdr_means[:, idx], linewidth=2, label=species)
    axes[1].set_title("Sorted qPCDR proxy")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Observed signal")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(times, counts, color="#7c3aed", linewidth=2)
    axes[2].set_title("Observed cell count")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_event_summary(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    event_counts: dict[str, int] = {}
    for _, event_type, _, _ in result.events:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(event_counts.keys()) if event_counts else ["none"]
    values = [event_counts[key] for key in labels] if event_counts else [0]
    ax.bar(labels, values, color="#2563eb")
    ax.set_title("Event counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save(fig, save_path)


def _dominant_state_index(soft_state: list[float] | np.ndarray) -> int:
    composition = np.asarray(soft_state, dtype=float)
    cfg.require(composition.shape == (cfg.N_STATES,), "Expected a full 4-state soft-state vector.")
    return int(np.argmax(composition))


def _dominant_state_from_cell(cell: dict) -> int:
    if "dominant_state_index" in cell:
        return int(cell["dominant_state_index"])
    if "dominant_state" in cell:
        return int(cfg.STATE_INDEX[str(cell["dominant_state"])])
    return _dominant_state_index(cell["soft_state"])


def _state_count_matrix(result: SimulationResult) -> np.ndarray:
    counts = np.zeros((len(result.times), cfg.N_STATES), dtype=float)
    for idx, snapshot in enumerate(result.truth_snapshots):
        if "dominant_state_counts" in snapshot:
            counts[idx, :] = np.asarray(snapshot["dominant_state_counts"], dtype=float)
        else:
            counts[idx, :] = result.population_sizes[idx] * np.asarray(result.soft_state_fractions[idx], dtype=float)
    return counts


def _state_copy_mean_tensor(result: SimulationResult) -> np.ndarray:
    values = np.zeros((len(result.times), cfg.N_STATES, cfg.N_SPECIES), dtype=float)
    for time_idx, snapshot in enumerate(result.truth_snapshots):
        by_gate = snapshot.get("copy_means_by_gate", {})
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            values[time_idx, state_idx, :] = np.asarray(
                by_gate.get(state_name, np.zeros(cfg.N_SPECIES, dtype=float)),
                dtype=float,
            )
    return values


def _nonempty_snapshot_indices(result: SimulationResult) -> list[int]:
    return [idx for idx, snapshot in enumerate(result.cell_snapshots) if snapshot]


def _representative_snapshot_indices(result: SimulationResult, target_count: int = 3) -> list[int]:
    nonempty = _nonempty_snapshot_indices(result)
    if not nonempty:
        return []
    if len(nonempty) <= target_count:
        return nonempty
    raw_positions = np.rint(np.linspace(0, len(nonempty) - 1, num=target_count)).astype(int)
    return [nonempty[pos] for pos in sorted(set(raw_positions.tolist()))]


def _blank_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_population_state_counts(
    result: SimulationResult,
    title: str = "Population and state counts",
    save_path: str | Path | None = None,
) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    state_counts = _state_count_matrix(result)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, result.population_sizes, color="#111827", linewidth=2.5, label="Total")
    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        ax.plot(times, state_counts[:, state_idx], color=STATE_COLORS[state_idx], linewidth=2, label=state_name)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cells")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_state_fraction_trajectories(
    result: SimulationResult,
    title: str = "State fraction trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    state_fractions = np.asarray(result.soft_state_fractions, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(times, state_fractions.T, labels=cfg.STATE_NAMES, colors=STATE_COLORS, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save(fig, save_path)


def plot_ecdna_mean_trajectories(
    result: SimulationResult,
    title: str = "ecDNA copy-number trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)
    by_state = _state_copy_mean_tensor(result)

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4), sharex=True)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        ax.plot(times, bulk[:, species_idx], color="#111827", linewidth=2.5, linestyle="--", label="Bulk")
        for state_idx, state_name in enumerate(cfg.STATE_NAMES):
            ax.plot(
                times,
                by_state[:, state_idx, species_idx],
                color=STATE_COLORS[state_idx],
                linewidth=2,
                label=state_name,
            )
        ax.set_title(species_name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean copies")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_ecdna_copy_distributions(
    result: SimulationResult,
    title: str = "ecDNA copy-number distributions",
    save_path: str | Path | None = None,
) -> plt.Figure:
    snapshot_indices = _representative_snapshot_indices(result, target_count=3)
    n_rows = max(1, len(snapshot_indices))
    fig, axes = plt.subplots(n_rows, cfg.N_SPECIES, figsize=(14, 3.2 * n_rows), squeeze=False)

    if not snapshot_indices:
        for ax in axes.ravel():
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    for row_idx, snapshot_idx in enumerate(snapshot_indices):
        snapshot = result.cell_snapshots[snapshot_idx]
        time = float(result.times[snapshot_idx])
        copies = np.asarray([cell["copy_numbers"] for cell in snapshot], dtype=int)
        states = np.asarray([_dominant_state_from_cell(cell) for cell in snapshot], dtype=int)

        for species_idx, species_name in enumerate(cfg.SPECIES):
            ax = axes[row_idx, species_idx]
            max_copy = int(np.max(copies[:, species_idx])) if copies.size else 0
            bins = np.arange(0, max_copy + 2, dtype=float) - 0.5
            for state_idx, state_name in enumerate(cfg.STATE_NAMES):
                values = copies[states == state_idx, species_idx]
                if values.size == 0:
                    continue
                ax.hist(
                    values,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color=STATE_COLORS[state_idx],
                    label=state_name if row_idx == 0 and species_idx == 0 else None,
                )
            ax.set_title(f"{species_name}, t={time:.2f}")
            ax.set_xlabel("Copy number")
            ax.set_ylabel("Cells")

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def _event_category(event_type: str) -> str:
    if event_type == "division":
        return "division"
    if event_type == "death":
        return "death"
    if event_type.startswith("gain_"):
        return "ecDNA gain"
    if event_type.startswith("loss_"):
        return "ecDNA loss"
    if "_to_" in event_type:
        return "cycle"
    return "other"


def _amplified_copy_count(details: dict) -> int:
    if not {"state_pre", "daughter_one", "daughter_two"}.issubset(details):
        return 0
    mother = np.asarray(details["state_pre"]["copy_numbers"], dtype=int)
    daughter_one = np.asarray(details["daughter_one"]["copy_numbers"], dtype=int)
    daughter_two = np.asarray(details["daughter_two"]["copy_numbers"], dtype=int)
    amplification = daughter_one + daughter_two - 2 * mother
    return int(np.sum(np.maximum(amplification, 0)))


def plot_event_counts_by_window(
    result: SimulationResult,
    title: str = "Event counts by time window",
    save_path: str | Path | None = None,
) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    if times.size == 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        _blank_axis(ax, "No recorded times")
        return _save(fig, save_path)

    if times[0] <= 0.0:
        edges = times
    else:
        edges = np.concatenate(([0.0], times))
    if edges.size < 2:
        edges = np.asarray([0.0, float(times[-1])], dtype=float)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1.0

    categories = tuple(EVENT_COLORS.keys())
    counts = {category: np.zeros(edges.size - 1, dtype=float) for category in categories}
    for event_time, event_type, _cell_id, details in result.events:
        window_idx = int(np.searchsorted(edges, float(event_time), side="left") - 1)
        if window_idx < 0 or window_idx >= edges.size - 1:
            continue
        counts[_event_category(event_type)][window_idx] += 1.0
        if event_type == "division":
            counts["amplification"][window_idx] += _amplified_copy_count(details)

    mids = 0.5 * (edges[:-1] + edges[1:])
    widths = np.maximum(edges[1:] - edges[:-1], 1e-6)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(edges.size - 1, dtype=float)
    for category in categories:
        values = counts[category]
        if not np.any(values):
            continue
        ax.bar(mids, values, width=0.8 * widths, bottom=bottom, color=EVENT_COLORS[category], label=category)
        bottom += values
    ax.set_title(title)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def _state_transition_counts(result: SimulationResult) -> np.ndarray:
    transition_counts = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)

    previous_states: dict[int, int] | None = None
    for snapshot in result.cell_snapshots:
        current_states = {int(cell["cell_id"]): _dominant_state_from_cell(cell) for cell in snapshot}
        if previous_states is not None:
            for cell_id, previous_state in previous_states.items():
                current_state = current_states.get(cell_id)
                if current_state is not None:
                    transition_counts[previous_state, current_state] += 1.0
        if current_states:
            previous_states = current_states

    for _event_time, event_type, _cell_id, details in result.events:
        if event_type != "division" or "state_pre" not in details:
            continue
        mother_state = _dominant_state_index(details["state_pre"]["soft_state"])
        for daughter_key in ("daughter_one", "daughter_two"):
            daughter = details.get(daughter_key)
            if daughter is None:
                continue
            daughter_state = _dominant_state_index(daughter["soft_state"])
            transition_counts[mother_state, daughter_state] += 1.0
    return transition_counts


def plot_state_transition_heatmap(
    result: SimulationResult,
    title: str = "Dominant-state transition counts",
    save_path: str | Path | None = None,
) -> plt.Figure:
    transition_counts = _state_transition_counts(result)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(transition_counts, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES, rotation=30, ha="right")
    ax.set_yticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES)
    for source_idx in range(cfg.N_STATES):
        for target_idx in range(cfg.N_STATES):
            ax.text(
                target_idx,
                source_idx,
                f"{int(transition_counts[source_idx, target_idx])}",
                ha="center",
                va="center",
                color="#111827",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    return _save(fig, save_path)


def _sample_cell_ids_from_latest_snapshot(result: SimulationResult, n_cells: int) -> list[int]:
    _latest_time, latest_snapshot = _latest_nonempty_snapshot(result)
    ids = sorted(int(cell["cell_id"]) for cell in latest_snapshot)
    if len(ids) <= n_cells:
        return ids
    indices = np.rint(np.linspace(0, len(ids) - 1, num=n_cells)).astype(int)
    return [ids[idx] for idx in sorted(set(indices.tolist()))]


def plot_single_cell_trajectories(
    result: SimulationResult,
    n_cells: int = 12,
    title: str = "Representative single-cell trajectories",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    state_ax = fig.add_subplot(gs[0, :])
    copy_axes = [fig.add_subplot(gs[1, idx]) for idx in range(cfg.N_SPECIES)]

    if not _nonempty_snapshot_indices(result):
        _blank_axis(state_ax, "No cell snapshots recorded")
        for ax in copy_axes:
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    selected_ids = _sample_cell_ids_from_latest_snapshot(result, n_cells)
    times = np.asarray(result.times, dtype=float)
    states = np.full((len(selected_ids), len(times)), np.nan, dtype=float)
    copy_values = np.full((len(selected_ids), len(times), cfg.N_SPECIES), np.nan, dtype=float)
    row_by_cell = {cell_id: row_idx for row_idx, cell_id in enumerate(selected_ids)}

    for time_idx, snapshot in enumerate(result.cell_snapshots):
        for cell in snapshot:
            row_idx = row_by_cell.get(int(cell["cell_id"]))
            if row_idx is None:
                continue
            states[row_idx, time_idx] = _dominant_state_from_cell(cell)
            copy_values[row_idx, time_idx, :] = np.asarray(cell["copy_numbers"], dtype=float)

    cmap = ListedColormap(STATE_COLORS)
    cmap.set_bad("#f3f4f6")
    state_ax.imshow(np.ma.masked_invalid(states), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=cfg.N_STATES - 1)
    state_ax.set_title("Dominant state")
    state_ax.set_xlabel("Time")
    state_ax.set_ylabel("Cell")
    state_ax.set_xticks(np.arange(len(times)), labels=[f"{time:.1f}" for time in times], rotation=45, ha="right")
    state_ax.set_yticks(np.arange(len(selected_ids)), labels=[str(cell_id) for cell_id in selected_ids])

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", color=STATE_COLORS[idx], label=state_name)
        for idx, state_name in enumerate(cfg.STATE_NAMES)
    ]
    state_ax.legend(handles=legend_handles, frameon=False, ncols=4, fontsize=8, loc="upper right")

    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = copy_axes[species_idx]
        for row_idx in range(len(selected_ids)):
            ax.plot(times, copy_values[row_idx, :, species_idx], color=SPECIES_COLORS[species_idx], alpha=0.35, linewidth=1)
        ax.set_title(f"{species_name} copies")
        ax.set_xlabel("Time")
        ax.set_ylabel("Copy number")

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_division_inheritance(
    result: SimulationResult,
    title: str = "Mother-daughter ecDNA inheritance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    mother_values: list[list[int]] = []
    daughter_values: list[list[int]] = []
    for _event_time, event_type, _cell_id, details in result.events:
        if event_type != "division" or not {"state_pre", "daughter_one", "daughter_two"}.issubset(details):
            continue
        mother = np.asarray(details["state_pre"]["copy_numbers"], dtype=int)
        mother_values.append(mother.tolist())
        mother_values.append(mother.tolist())
        daughter_values.append(np.asarray(details["daughter_one"]["copy_numbers"], dtype=int).tolist())
        daughter_values.append(np.asarray(details["daughter_two"]["copy_numbers"], dtype=int).tolist())

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4))
    if not mother_values:
        for ax in axes:
            _blank_axis(ax, "No division events recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    mothers = np.asarray(mother_values, dtype=float)
    daughters = np.asarray(daughter_values, dtype=float)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        ax.scatter(
            mothers[:, species_idx],
            daughters[:, species_idx],
            s=18,
            alpha=0.45,
            color=SPECIES_COLORS[species_idx],
            edgecolors="none",
        )
        axis_max = float(max(np.max(mothers[:, species_idx]), np.max(daughters[:, species_idx]), 1.0))
        ax.plot([0, axis_max], [0, axis_max], color="#111827", linewidth=1, linestyle="--")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(species_name)
        ax.set_xlabel("Mother copies")
        ax.set_ylabel("Daughter copies")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_state_ecdna_heatmaps(
    result: SimulationResult,
    title: str = "State-ecDNA mean-copy heatmaps",
    save_path: str | Path | None = None,
) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    by_state = _state_copy_mean_tensor(result)

    fig, axes = plt.subplots(1, cfg.N_SPECIES, figsize=(14, 4), sharey=True)
    for species_idx, species_name in enumerate(cfg.SPECIES):
        ax = axes[species_idx]
        image = ax.imshow(by_state[:, :, species_idx].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(species_name)
        ax.set_xlabel("Time")
        ax.set_xticks(np.arange(len(times)), labels=[f"{time:.1f}" for time in times], rotation=45, ha="right")
        ax.set_yticks(np.arange(cfg.N_STATES), labels=cfg.STATE_NAMES)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("State")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_latent_phase_space(
    result: SimulationResult,
    title: str = "Latent burden-stress-survival phase space",
    save_path: str | Path | None = None,
    max_points: int = 4000,
) -> plt.Figure:
    rows: list[dict] = [cell for snapshot in result.cell_snapshots for cell in snapshot]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not rows:
        for ax in axes:
            _blank_axis(ax, "No cell snapshots recorded")
        fig.suptitle(title)
        fig.tight_layout()
        return _save(fig, save_path)

    if len(rows) > max_points:
        indices = np.rint(np.linspace(0, len(rows) - 1, num=max_points)).astype(int)
        rows = [rows[idx] for idx in sorted(set(indices.tolist()))]

    copies = np.asarray([cell["copy_numbers"] for cell in rows], dtype=float)
    burdens = np.log1p(copies) @ cfg.DEFAULT_MODEL_PARAMETERS.exposure.burden_weights
    stress = np.asarray([cell["stress_score"] for cell in rows], dtype=float)
    survival = np.asarray([cell["survival_score"] for cell in rows], dtype=float)
    division_hazard = np.asarray([cell.get("division_hazard", 0.0) for cell in rows], dtype=float)
    death_hazard = np.asarray([cell.get("death_hazard", 0.0) for cell in rows], dtype=float)
    states = np.asarray([_dominant_state_from_cell(cell) for cell in rows], dtype=int)

    def scaled_sizes(values: np.ndarray) -> np.ndarray:
        if values.size == 0 or float(np.max(values)) <= 0.0:
            return np.full(values.shape, 18.0, dtype=float)
        return 18.0 + 70.0 * values / float(np.max(values))

    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        mask = states == state_idx
        axes[0].scatter(
            burdens[mask],
            stress[mask],
            s=scaled_sizes(division_hazard[mask]),
            color=STATE_COLORS[state_idx],
            alpha=0.35,
            edgecolors="none",
            label=state_name,
        )
        axes[1].scatter(
            stress[mask],
            survival[mask],
            s=scaled_sizes(death_hazard[mask]),
            color=STATE_COLORS[state_idx],
            alpha=0.35,
            edgecolors="none",
        )

    axes[0].set_title("Burden vs stress, size=division hazard")
    axes[0].set_xlabel("ecDNA burden")
    axes[0].set_ylabel("Stress score")
    axes[1].set_title("Stress vs survival, size=death hazard")
    axes[1].set_xlabel("Stress score")
    axes[1].set_ylabel("Survival score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_single_run_diagnostic_suite(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        ("01_population_state_counts.png", plot_population_state_counts),
        ("02_state_fractions.png", plot_state_fraction_trajectories),
        ("03_ecdna_mean_trajectories.png", plot_ecdna_mean_trajectories),
        ("04_ecdna_copy_distributions.png", plot_ecdna_copy_distributions),
        ("05_event_counts_by_window.png", plot_event_counts_by_window),
        ("06_state_transition_heatmap.png", plot_state_transition_heatmap),
        ("07_single_cell_trajectories.png", plot_single_cell_trajectories),
        ("08_division_inheritance.png", plot_division_inheritance),
        ("09_state_ecdna_heatmaps.png", plot_state_ecdna_heatmaps),
        ("10_latent_phase_space.png", plot_latent_phase_space),
    )
    written: dict[str, Path] = {}
    for file_name, plot_func in plot_specs:
        path = output / file_name
        plot_func(result, save_path=path)
        written[file_name] = path
    return written


def _latest_nonempty_snapshot(result: SimulationResult) -> tuple[float, list[dict]]:
    for time, snapshot in zip(reversed(result.times), reversed(result.cell_snapshots)):
        if snapshot:
            return float(time), snapshot
    raise ValueError("Lineage state plot requires at least one non-empty cell snapshot.")


def _sample_terminal_cell_ids(snapshot: list[dict], n_terminal_cells: int) -> list[int]:
    cfg.require(n_terminal_cells > 0, "n_terminal_cells must be strictly positive.")
    terminal_ids = sorted(int(cell["cell_id"]) for cell in snapshot)
    cfg.require(bool(terminal_ids), "Cannot sample lineage paths from an empty snapshot.")
    if len(terminal_ids) <= n_terminal_cells:
        return terminal_ids

    sampled_indices: list[int] = []
    for raw_index in np.rint(np.linspace(0, len(terminal_ids) - 1, num=n_terminal_cells)).astype(int).tolist():
        if raw_index not in sampled_indices:
            sampled_indices.append(raw_index)
    sampled_indices.sort()
    return [terminal_ids[index] for index in sampled_indices]


def plot_lineage_state_paths(
    result: SimulationResult,
    n_terminal_cells: int = 8,
    title: str = "Sampled lineage state paths",
    save_path: str | Path | None = None,
) -> plt.Figure:
    cfg.require(bool(result.times), "Lineage state plot requires recorded simulation times.")
    latest_time, latest_snapshot = _latest_nonempty_snapshot(result)
    selected_terminal_ids = _sample_terminal_cell_ids(latest_snapshot, n_terminal_cells)

    initial_snapshot = result.cell_snapshots[0] if result.cell_snapshots else []
    parent_by_cell: dict[int, int | None] = {int(cell["cell_id"]): None for cell in initial_snapshot}
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    birth_times: dict[int, float] = {int(cell["cell_id"]): float(result.times[0]) for cell in initial_snapshot}
    division_times: dict[int, float] = {}
    death_times: dict[int, float] = {}

    for event_time, event_type, cell_id, details in result.events:
        event_time = float(event_time)
        parent_id = int(cell_id)
        if event_type == "division":
            division_times[parent_id] = event_time
            for daughter_key in ("daughter_one", "daughter_two"):
                daughter = details[daughter_key]
                daughter_id = int(daughter["cell_id"])
                cfg.require(
                    int(daughter["parent_id"]) == parent_id,
                    "Division event daughter must point back to the recorded parent cell.",
                )
                parent_by_cell[daughter_id] = parent_id
                birth_times[daughter_id] = event_time
                children_by_parent[parent_id].append(daughter_id)
        elif event_type == "death":
            death_times[parent_id] = event_time

    included_cells: set[int] = set()
    for terminal_id in selected_terminal_ids:
        current_id: int | None = terminal_id
        while current_id is not None and current_id not in included_cells:
            included_cells.add(current_id)
            current_id = parent_by_cell.get(current_id)

    cfg.require(bool(included_cells), "Lineage state plot requires at least one sampled lineage.")

    state_points: dict[int, dict[float, int]] = {cell_id: {} for cell_id in included_cells}
    for time, snapshot in zip(result.times, result.cell_snapshots):
        for cell in snapshot:
            cell_id = int(cell["cell_id"])
            if cell_id in included_cells:
                state_points[cell_id][float(time)] = _dominant_state_index(cell["soft_state"])

    for event_time, event_type, cell_id, details in result.events:
        event_time = float(event_time)
        cell_id = int(cell_id)
        if event_type == "division" and cell_id in included_cells:
            state_points[cell_id][event_time] = _dominant_state_index(details["state_pre"]["soft_state"])
            for daughter_key in ("daughter_one", "daughter_two"):
                daughter = details[daughter_key]
                daughter_id = int(daughter["cell_id"])
                if daughter_id in included_cells:
                    state_points[daughter_id][event_time] = _dominant_state_index(daughter["soft_state"])
        elif event_type == "death" and cell_id in included_cells:
            state_points[cell_id][event_time] = _dominant_state_index(details["state_pre"]["soft_state"])

    leaf_rank = {cell_id: idx for idx, cell_id in enumerate(selected_terminal_ids)}
    filtered_children = {
        cell_id: [child_id for child_id in children_by_parent.get(cell_id, []) if child_id in included_cells]
        for cell_id in included_cells
    }
    leftmost_rank_cache: dict[int, int] = {}

    def leftmost_rank(cell_id: int) -> int:
        cached = leftmost_rank_cache.get(cell_id)
        if cached is not None:
            return cached
        children = filtered_children.get(cell_id, [])
        if not children:
            cfg.require(cell_id in leaf_rank, f"Included leaf cell {cell_id} must be one of the sampled terminals.")
            leftmost_rank_cache[cell_id] = leaf_rank[cell_id]
            return leaf_rank[cell_id]
        rank = min(leftmost_rank(child_id) for child_id in children)
        leftmost_rank_cache[cell_id] = rank
        return rank

    y_positions: dict[int, float] = {}

    def assign_y(cell_id: int) -> float:
        if cell_id in y_positions:
            return y_positions[cell_id]
        children = filtered_children.get(cell_id, [])
        if not children:
            y_positions[cell_id] = float(leaf_rank[cell_id])
            return y_positions[cell_id]
        ordered_children = sorted(children, key=leftmost_rank)
        child_positions = [assign_y(child_id) for child_id in ordered_children]
        y_positions[cell_id] = float(np.mean(child_positions))
        return y_positions[cell_id]

    roots = [cell_id for cell_id in included_cells if parent_by_cell.get(cell_id) is None or parent_by_cell.get(cell_id) not in included_cells]
    for root in sorted(roots, key=leftmost_rank):
        assign_y(root)

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.8 * len(selected_terminal_ids) + 2.0)))
    state_cmap = dict(zip(range(cfg.N_STATES), STATE_COLORS))

    for cell_id in included_cells:
        points = state_points[cell_id]
        sorted_times = sorted(points.keys())
        y = y_positions[cell_id]
        birth_time = birth_times.get(cell_id, min(sorted_times))
        end_time = death_times.get(cell_id, division_times.get(cell_id, latest_time))

        if birth_time in points:
            start_state = points[birth_time]
        else:
            earlier_times = [time for time in sorted_times if time <= birth_time]
            start_state = points[max(earlier_times)] if earlier_times else points[sorted_times[0]]

        ax.hlines(y, birth_time, end_time, color=state_cmap[start_state], linewidth=2.5, alpha=0.45)

        for time in sorted_times:
            ax.scatter(time, y, s=26, color=state_cmap[points[time]], edgecolors="none", zorder=3)

        parent_id = parent_by_cell.get(cell_id)
        if parent_id is not None and parent_id in included_cells:
            ax.plot(
                [birth_time, birth_time],
                [y_positions[parent_id], y],
                color="#9ca3af",
                linewidth=1.1,
                alpha=0.8,
            )

        if cell_id in death_times:
            ax.scatter(death_times[cell_id], y, marker="x", s=42, color="#111827", linewidths=1.3, zorder=4)

    ax.set_yticks([y_positions[cell_id] for cell_id in selected_terminal_ids])
    ax.set_yticklabels([f"Cell {cell_id}" for cell_id in selected_terminal_ids])
    ax.set_xlabel("Time")
    ax.set_title(title)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=STATE_COLORS[idx], markersize=7, label=state_name)
        for idx, state_name in enumerate(cfg.STATE_NAMES)
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="x", color="#111827", linestyle="none", markersize=7, label="Death"))
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    return _save(fig, save_path)
