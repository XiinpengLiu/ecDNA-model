"""
Minimal plotting utilities for the ecDNA model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from simulation import SimulationResult


STATE_COLORS = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")


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
