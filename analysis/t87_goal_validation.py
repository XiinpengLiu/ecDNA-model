"""
Run the T87 goal validation against the current config defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.simulation import run_simulation


CONDITIONS = ("ctrl", "P10", "P50", "P250", "R20", "R100", "R500")
SPECIES = ("MYC", "CDK4", "PDGFRA")


def _load_targets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ddpcr = pd.read_csv(raw_dir / "ddpcr.csv")
    cell_count = pd.read_csv(raw_dir / "cell_count.csv")
    return ddpcr, cell_count


def _target_cell_count(cell_count: pd.DataFrame, condition: str, week: int) -> float:
    rows = cell_count[(cell_count["condition"] == condition) & (cell_count["week"].astype(int) == int(week))]
    if rows.empty:
        return float("nan")
    return float(rows["total_cell_count"].median())


def _target_copies(ddpcr: pd.DataFrame, condition: str, week: int) -> dict[str, float]:
    rows = ddpcr[(ddpcr["condition"] == condition) & (ddpcr["week"].astype(int) == int(week))]
    values = rows.groupby("species")["ddpcr_copy_number"].median().to_dict()
    return {species: float(values.get(species, np.nan)) for species in SPECIES}


def _condition_label(condition: str) -> str:
    drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    if condition == "ctrl":
        return "ctrl"
    return f"{drug} {dose:g} nM"


def _simulation_params(args: argparse.Namespace) -> cfg.ModelParameters:
    base = cfg.DEFAULT_MODEL_PARAMETERS
    simulation = replace(
        base.simulation,
        n_init=int(args.n_init),
        target_population_size=int(args.target_population_size),
        max_pop_size=int(args.max_pop_size),
        random_seed=int(args.seed),
        record_events=bool(args.record_events),
    )
    return replace(base, simulation=simulation)


def _run_condition(
    condition: str,
    *,
    params: cfg.ModelParameters,
    raw_dir: Path,
    seed: int,
    rows_per_state: int,
) -> tuple[pd.DataFrame, dict]:
    initialization = cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=raw_dir / "ddpcr.csv",
        seed=seed,
        rows_per_state=rows_per_state,
    )
    initial_copy_means = np.zeros(cfg.N_SPECIES, dtype=float)
    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        matrix = np.asarray(initialization.empirical_sorted_copy_distributions[state_name], dtype=float)
        initial_copy_means += float(initialization.empirical_flow_fractions[state_idx]) * np.mean(matrix, axis=0)

    result = run_simulation(
        params=params,
        initialization=initialization,
        input_schedules=cfg.t87_input_schedules_for_condition(condition),
        seed=seed,
        verbose=False,
    )

    rows = [
        {
            "condition": condition,
            "condition_label": _condition_label(condition),
            "sim_time": 0.0,
            "week": 1,
            "sim_representative_cells": int(params.simulation.n_init),
            "sim_MYC": float(initial_copy_means[cfg.MYC]),
            "sim_CDK4": float(initial_copy_means[cfg.CDK4]),
            "sim_PDGFRA": float(initial_copy_means[cfg.PDGFRA]),
            "stop_reason": result.stop_reason,
            "stop_time": float(result.stop_time) if result.stop_time is not None else np.nan,
        }
    ]
    for time, population_size, copy_means in zip(result.times, result.population_sizes, result.bulk_copy_means):
        raw_week = int(round(float(time))) + 1
        rows.append(
            {
                "condition": condition,
                "condition_label": _condition_label(condition),
                "sim_time": float(time),
                "week": raw_week,
                "sim_representative_cells": int(population_size),
                "sim_MYC": float(copy_means[cfg.MYC]),
                "sim_CDK4": float(copy_means[cfg.CDK4]),
                "sim_PDGFRA": float(copy_means[cfg.PDGFRA]),
                "stop_reason": result.stop_reason,
                "stop_time": float(result.stop_time) if result.stop_time is not None else np.nan,
            }
        )
    stop = {
        "condition": condition,
        "stop_reason": result.stop_reason,
        "stop_time": float(result.stop_time) if result.stop_time is not None else np.nan,
        "final_representative_cells": int(result.population_sizes[-1]) if result.population_sizes else 0,
    }
    return pd.DataFrame(rows), stop


def _add_targets_and_errors(
    timeline: pd.DataFrame,
    ddpcr: pd.DataFrame,
    cell_count: pd.DataFrame,
    n_init: int,
) -> pd.DataFrame:
    enriched = timeline.copy()
    week1_count = {
        condition: _target_cell_count(cell_count, condition, 1)
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        mask = enriched["condition"] == condition
        scale = float(week1_count[condition]) / float(n_init)
        enriched.loc[mask, "sim_weighted_cell_count"] = enriched.loc[mask, "sim_representative_cells"] * scale

    enriched["exp_cell_count"] = [
        _target_cell_count(cell_count, row.condition, int(row.week))
        for row in enriched.itertuples(index=False)
    ]
    for species in SPECIES:
        enriched[f"exp_{species}"] = [
            _target_copies(ddpcr, row.condition, int(row.week))[species]
            for row in enriched.itertuples(index=False)
        ]
        enriched[f"copy_error_{species}"] = enriched[f"sim_{species}"] - enriched[f"exp_{species}"]
        enriched[f"copy_percent_error_{species}"] = (
            100.0 * enriched[f"copy_error_{species}"] / enriched[f"exp_{species}"].replace(0.0, np.nan)
        )

    enriched["sim_log10_cell_count"] = np.log10(enriched["sim_weighted_cell_count"].clip(lower=1.0))
    enriched["exp_log10_cell_count"] = np.log10(enriched["exp_cell_count"].clip(lower=1.0))
    enriched["log10_cell_error"] = enriched["sim_log10_cell_count"] - enriched["exp_log10_cell_count"]
    return enriched


def _day56_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    week6_parts = []
    ordered = timeline.sort_values(["condition", "week"])
    for condition, group in ordered.groupby("condition", sort=False):
        week6 = group[group["week"].astype(int) == 6]
        week6_parts.append((week6 if not week6.empty else group.tail(1)).copy())
    week6 = pd.concat(week6_parts, ignore_index=True)
    ctrl = week6[week6["condition"] == "ctrl"].iloc[0]
    rows = []
    for row in week6.sort_values("condition").itertuples(index=False):
        out = {
            "condition": row.condition,
            "condition_label": row.condition_label,
            "week": int(row.week),
            "stop_reason": row.stop_reason,
            "stop_time": row.stop_time,
            "sim_representative_cells": row.sim_representative_cells,
            "sim_log10_cell_count": row.sim_log10_cell_count,
            "exp_log10_cell_count": row.exp_log10_cell_count,
            "log10_cell_error": row.log10_cell_error,
            "sim_cell_ratio_vs_ctrl": row.sim_weighted_cell_count / ctrl.sim_weighted_cell_count,
            "exp_cell_ratio_vs_ctrl": row.exp_cell_count / ctrl.exp_cell_count,
        }
        for species in SPECIES:
            sim_value = getattr(row, f"sim_{species}")
            exp_value = getattr(row, f"exp_{species}")
            ctrl_sim = getattr(ctrl, f"sim_{species}")
            ctrl_exp = getattr(ctrl, f"exp_{species}")
            out[f"sim_{species}"] = sim_value
            out[f"exp_{species}"] = exp_value
            out[f"sim_{species}_ratio_vs_ctrl"] = sim_value / ctrl_sim
            out[f"exp_{species}_ratio_vs_ctrl"] = exp_value / ctrl_exp
            out[f"{species}_percent_error"] = 100.0 * (sim_value - exp_value) / exp_value
        rows.append(out)
    return pd.DataFrame(rows)


def _initial_state_summary(raw_dir: Path, rows_per_state: int, seed: int) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        initialization = cfg.build_t87_initialization_parameters(
            condition,
            ddpcr_path=raw_dir / "ddpcr.csv",
            seed=seed,
            rows_per_state=rows_per_state,
        )
        for state_name in cfg.STATE_NAMES:
            matrix = np.asarray(initialization.empirical_sorted_copy_distributions[state_name], dtype=float)
            means = np.mean(matrix, axis=0)
            rows.append(
                {
                    "condition": condition,
                    "state": state_name,
                    "state_fraction": float(initialization.empirical_flow_fractions[cfg.STATE_INDEX[state_name]]),
                    "cycle_Q": float(initialization.cycle_probabilities[cfg.Q]),
                    "cycle_G1": float(initialization.cycle_probabilities[cfg.G1]),
                    "cycle_S": float(initialization.cycle_probabilities[cfg.S]),
                    "cycle_G2M": float(initialization.cycle_probabilities[cfg.G2M]),
                    "init_MYC_mean": float(means[cfg.MYC]),
                    "init_CDK4_mean": float(means[cfg.CDK4]),
                    "init_PDGFRA_mean": float(means[cfg.PDGFRA]),
                }
            )
    return pd.DataFrame(rows)


def _write_markdown_report(output_dir: Path, day56: pd.DataFrame, stops: pd.DataFrame) -> None:
    special = {
        "P10_CDK4_gt_ctrl": bool(
            day56.loc[day56["condition"] == "P10", "sim_CDK4_ratio_vs_ctrl"].iloc[0] > 1.0
        ),
        "P50_CDK4_gt_ctrl": bool(
            day56.loc[day56["condition"] == "P50", "sim_CDK4_ratio_vs_ctrl"].iloc[0] > 1.0
        ),
        "P250_CDK4_not_gt_ctrl": bool(
            day56.loc[day56["condition"] == "P250", "sim_CDK4_ratio_vs_ctrl"].iloc[0] <= 1.0
        ),
        "R500_PDGFRA_low": bool(
            day56.loc[day56["condition"] == "R500", "sim_PDGFRA_ratio_vs_ctrl"].iloc[0] < 0.75
        ),
    }
    lines = [
        "# T87 Goal Validation",
        "",
        "Mechanism priority: selection, drug effective signal, and condition-specific initial heterogeneity. Turnover parameters are not used as the primary fitting lever in this validation.",
        "",
        "## Stop Reasons",
        "",
        stops.to_markdown(index=False),
        "",
        "## Day56 Summary",
        "",
        day56.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Special Checks",
        "",
    ]
    lines.extend(f"- `{name}`: {value}" for name, value in special.items())
    lines.append("")
    output_dir.joinpath("validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _simulation_params(args)
    cfg.validate_model_parameters(params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    ddpcr, cell_count = _load_targets(raw_dir)
    timeline_parts = []
    stop_rows = []
    for condition in CONDITIONS:
        timeline, stop = _run_condition(
            condition,
            params=params,
            raw_dir=raw_dir,
            seed=int(args.seed),
            rows_per_state=int(args.rows_per_state),
        )
        timeline_parts.append(timeline)
        stop_rows.append(stop)

    timeline = pd.concat(timeline_parts, ignore_index=True)
    timeline = _add_targets_and_errors(timeline, ddpcr, cell_count, n_init=int(args.n_init))
    day56 = _day56_summary(timeline)
    stops = pd.DataFrame(stop_rows)
    initial = _initial_state_summary(raw_dir, rows_per_state=int(args.rows_per_state), seed=int(args.seed))

    timeline.to_csv(output_dir / "timeline_comparison.csv", index=False)
    day56.to_csv(output_dir / "day56_summary.csv", index=False)
    stops.to_csv(output_dir / "stop_reasons.csv", index=False)
    initial.to_csv(output_dir / "initial_state_summary.csv", index=False)
    _write_markdown_report(output_dir, day56, stops)

    print(f"Wrote validation outputs to {output_dir}")
    print(day56[["condition", "sim_CDK4_ratio_vs_ctrl", "exp_CDK4_ratio_vs_ctrl", "sim_PDGFRA_ratio_vs_ctrl", "exp_PDGFRA_ratio_vs_ctrl"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw") / "t87_drug_bulkfit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "t87_goal_validation")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--rows-per-state", type=int, default=512)
    parser.add_argument("--n-init", type=int, default=1200)
    parser.add_argument("--target-population-size", type=int, default=10000)
    parser.add_argument("--max-pop-size", type=int, default=10000)
    parser.add_argument("--record-events", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
