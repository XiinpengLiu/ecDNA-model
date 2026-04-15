"""
Minimal plotting utilities for the ecDNA v4 model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import v4_config as cfg
from v4_simulation import SimulationResult

STATE_COLORS = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_results(result: SimulationResult, title: str = "ecDNA v4 simulation", save_path: str | Path | None = None) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    state_fractions = np.asarray(result.state_fractions, dtype=float)
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

    axes[1, 1].plot(times, result.mean_stress, label="Stress", linewidth=2)
    axes[1, 1].plot(times, result.mean_survival, label="Survival reserve", linewidth=2)
    axes[1, 1].set_title("Latent stress and survival")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_observation_proxies(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    times = np.asarray(result.times, dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)
    counts = np.asarray(result.population_sizes, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    state_fractions = np.asarray(result.state_fractions, dtype=float)
    axes[0].stackplot(times, state_fractions.T, labels=cfg.STATE_NAMES)
    axes[0].set_title("Flow/staining proxy")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fraction")

    for idx, species in enumerate(cfg.SPECIES):
        axes[1].plot(times, bulk[:, idx], linewidth=2, label=species)
    axes[1].set_title("qPCR proxy")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mean copies")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(times, counts, color="#7c3aed", linewidth=2)
    axes[2].set_title("Cell-count proxy")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Predicted count")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_event_summary(result: SimulationResult, save_path: str | Path | None = None) -> plt.Figure:
    event_counts: dict[str, int] = {}
    for _, event_type, _, _ in result.events:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(event_counts.keys())
    values = [event_counts[key] for key in labels]
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
    for raw_index in range(len(terminal_ids)):
        if len(sampled_indices) >= n_terminal_cells:
            break
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

    for cell_id in included_cells:
        cfg.require(bool(state_points[cell_id]), f"Missing observed state path for cell {cell_id}.")

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

    root_cells = sorted(
        [cell_id for cell_id in included_cells if parent_by_cell.get(cell_id) not in included_cells],
        key=leftmost_rank,
    )
    cfg.require(bool(root_cells), "Lineage state plot requires at least one root cell in the sampled subgraph.")
    for root_cell in root_cells:
        assign_y(root_cell)

    time_padding = max(0.5, 0.03 * max(latest_time - float(result.times[0]), 1.0))
    fig_height = max(4.0, 1.1 * len(selected_terminal_ids) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for cell_id in sorted(included_cells, key=leftmost_rank):
        ordered_points = sorted(state_points[cell_id].items())
        y_position = y_positions[cell_id]
        for (start_time, state_idx), (end_time, _) in zip(ordered_points[:-1], ordered_points[1:]):
            ax.plot(
                [start_time, end_time],
                [y_position, y_position],
                color=STATE_COLORS[state_idx],
                linewidth=3.0,
                solid_capstyle="round",
                zorder=2,
            )
        if cell_id in death_times:
            ax.scatter(death_times[cell_id], y_position, color="#111827", marker="x", s=36, linewidths=1.5, zorder=4)

    for parent_id, child_ids in filtered_children.items():
        if len(child_ids) < 2:
            continue
        division_time = division_times[parent_id]
        child_y_positions = [y_positions[child_id] for child_id in sorted(child_ids, key=leftmost_rank)]
        ax.plot(
            [division_time, division_time],
            [min(child_y_positions), max(child_y_positions)],
            color="#6b7280",
            linewidth=1.5,
            zorder=1,
        )

    label_x = latest_time + time_padding
    for terminal_id in selected_terminal_ids:
        ax.text(label_x, y_positions[terminal_id], f"cell {terminal_id}", va="center", fontsize=9)

    legend_handles = [Line2D([0], [0], color=STATE_COLORS[idx], lw=3, label=state_name) for idx, state_name in enumerate(cfg.STATE_NAMES)]
    legend_handles.append(
        Line2D([0], [0], color="#111827", marker="x", linestyle="None", markersize=6, label="Death")
    )
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", ncol=3)
    ax.set_title(f"{title} (dominant state, n={len(selected_terminal_ids)} terminal cells)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sampled lineage")
    ax.set_xlim(float(result.times[0]), latest_time + 4.0 * time_padding)
    ax.set_ylim(-0.8, max(y_positions.values()) + 0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    return _save(fig, save_path)
