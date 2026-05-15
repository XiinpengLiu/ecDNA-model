"""
Run the T87 goal validation against the current config defaults.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.simulation import run_simulation


CONDITIONS = ("ctrl", "P10", "P50", "P250", "R20", "R100", "R500")
SPECIES = ("MYC", "CDK4", "PDGFRA")
FILTERED_DDPCR_SOURCE = Path("data") / "2026-05-04-ddPCR-T87-drug-treatment-days-28-35-42-filtered.csv"
DDPCR_TARGET_TO_SPECIES = {"ecMyc": "MYC", "ecCDK4": "CDK4", "ecPDGFRA": "PDGFRA"}
DAY_START = 14
DAY_END = 56
T_DAY56 = 12.0
DAYS_PER_SIM_TIME = (DAY_END - DAY_START) / T_DAY56


def _day_to_sim_time(day: int | float) -> float:
    return (float(day) - DAY_START) / DAYS_PER_SIM_TIME


def _sim_time_to_day(time: int | float) -> int:
    return int(round(DAY_START + DAYS_PER_SIM_TIME * float(time)))


def _filtered_ddpcr_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "day",
                "sim_time",
                "condition",
                "replicate",
                "species",
                "ddpcr_copy_number",
                "ddpcr_sd_or_ci",
                "batch_id",
            ]
        )

    raw = pd.read_csv(path)
    parsed = raw["Sample"].astype(str).str.extract(r"^d(?P<day>\d+)\s+(?P<condition>\S+)$")
    rows = raw.join(parsed)
    rows = rows[rows["condition"].isin(CONDITIONS) & rows["Target"].isin(DDPCR_TARGET_TO_SPECIES)].copy()
    rows["day"] = rows["day"].astype(int)
    rows = rows[rows["day"].between(DAY_START, DAY_END)].copy()
    rows["sim_time"] = rows["day"].map(_day_to_sim_time)
    rows["replicate"] = "filtered"
    rows["species"] = rows["Target"].map(DDPCR_TARGET_TO_SPECIES)
    rows["ddpcr_copy_number"] = rows["CNV"].astype(float)
    rows["ddpcr_sd_or_ci"] = (
        rows["PoissonCNVMax"].astype(float) - rows["PoissonCNVMin"].astype(float)
    ) / 2.0
    rows["batch_id"] = "2026-05-04-filtered-d" + rows["day"].astype(str)
    columns = ["day", "sim_time", "condition", "replicate", "species", "ddpcr_copy_number", "ddpcr_sd_or_ci", "batch_id"]
    return rows[columns].sort_values(["day", "condition", "species"]).reset_index(drop=True)


def _raw_week1_anchors(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "ddpcr.csv")


def _load_targets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_week1 = _raw_week1_anchors(raw_dir)
    ddpcr = _filtered_ddpcr_targets(FILTERED_DDPCR_SOURCE)
    if ddpcr.empty:
        anchor = raw_week1.copy()
        anchor["day"] = DAY_START
        anchor["sim_time"] = 0.0
        ddpcr = anchor[
            ["day", "sim_time", "condition", "replicate", "species", "ddpcr_copy_number", "ddpcr_sd_or_ci", "batch_id"]
        ]
    cell_count = pd.read_csv(raw_dir / "cell_count.csv")
    return raw_week1, ddpcr, cell_count


def _target_cell_count(cell_count: pd.DataFrame, condition: str, week: int) -> float:
    rows = cell_count[(cell_count["condition"] == condition) & (cell_count["week"].astype(int) == int(week))]
    if rows.empty:
        return float("nan")
    return float(rows["total_cell_count"].median())


def _cell_count_week_for_day(day: int) -> int | None:
    week = int(round((int(day) - DAY_START) / 7.0))
    return week if week >= 1 else None


def _target_cell_count_for_day(cell_count: pd.DataFrame, condition: str, day: int) -> float:
    week = _cell_count_week_for_day(day)
    if week is None:
        return float("nan")
    return _target_cell_count(cell_count, condition, week)


def _copy_target_rows(ddpcr: pd.DataFrame, condition: str, day: int) -> pd.DataFrame:
    condition_rows = ddpcr[ddpcr["condition"] == condition].copy()
    if condition_rows.empty:
        return condition_rows

    condition_rows["day"] = condition_rows["day"].astype(int)
    return condition_rows[condition_rows["day"] == int(day)]


def _target_copies(ddpcr: pd.DataFrame, condition: str, day: int) -> dict[str, float]:
    rows = _copy_target_rows(ddpcr, condition, day)
    values = rows.groupby("species")["ddpcr_copy_number"].median().to_dict()
    return {species: float(values.get(species, np.nan)) for species in SPECIES}


def _target_copy_day(ddpcr: pd.DataFrame, condition: str, day: int) -> float:
    rows = _copy_target_rows(ddpcr, condition, day)
    if rows.empty:
        return float("nan")
    return float(rows["day"].astype(int).max())


def _condition_label(condition: str) -> str:
    drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    if condition == "ctrl":
        return "ctrl"
    return f"{drug} {dose:g} nM"


def _simulation_params(args: argparse.Namespace) -> cfg.ModelParameters:
    base = cfg.DEFAULT_MODEL_PARAMETERS
    target_population_size = int(args.target_population_size)
    simulation = replace(
        base.simulation,
        n_init=int(args.n_init),
        target_population_size=None if target_population_size <= 0 else target_population_size,
        max_pop_size=int(args.max_pop_size),
        random_seed=int(args.seed),
        record_events=bool(args.record_events),
    )
    return replace(base, simulation=simulation)


def _initial_copy_means(initialization: cfg.InitializationParameters) -> np.ndarray:
    initial_copy_means = np.zeros(cfg.N_SPECIES, dtype=float)
    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        matrix = np.asarray(initialization.empirical_sorted_copy_distributions[state_name], dtype=float)
        initial_copy_means += float(initialization.empirical_flow_fractions[state_idx]) * np.mean(matrix, axis=0)
    return initial_copy_means


def _run_condition(
    condition: str,
    *,
    params: cfg.ModelParameters,
    raw_dir: Path,
    seed: int,
    rows_per_state: int,
) -> tuple[pd.DataFrame, dict, cfg.InitializationParameters]:
    initialization = cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=raw_dir / "ddpcr.csv",
        seed=seed,
        rows_per_state=rows_per_state,
    )
    initial_copy_means = _initial_copy_means(initialization)

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
            "day": DAY_START,
            "sim_representative_cells": int(params.simulation.n_init),
            "sim_MYC": float(initial_copy_means[cfg.MYC]),
            "sim_CDK4": float(initial_copy_means[cfg.CDK4]),
            "sim_PDGFRA": float(initial_copy_means[cfg.PDGFRA]),
            "stop_reason": result.stop_reason,
            "stop_time": float(result.stop_time) if result.stop_time is not None else np.nan,
        }
    ]
    for time, population_size, copy_means in zip(result.times, result.population_sizes, result.bulk_copy_means):
        day = _sim_time_to_day(float(time))
        rows.append(
            {
                "condition": condition,
                "condition_label": _condition_label(condition),
                "sim_time": float(time),
                "day": day,
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
    if result.truth_snapshots:
        final_truth = result.truth_snapshots[-1]
        dominant = np.asarray(final_truth["dominant_state_fractions"], dtype=float)
        soft = np.asarray(final_truth["soft_state_fractions"], dtype=float)
        for idx, state_name in enumerate(cfg.STATE_NAMES):
            token = state_name.replace("-like", "").replace("-", "_")
            stop[f"final_hard_fraction_{token}"] = float(dominant[idx])
            stop[f"final_soft_fraction_{token}"] = float(soft[idx])
        stop["final_hard_fraction_NPC_OPC"] = float(dominant[cfg.NPC] + dominant[cfg.OPC])
        stop["final_hard_fraction_AC_MES"] = float(dominant[cfg.AC] + dominant[cfg.MES])
    return pd.DataFrame(rows), stop, initialization


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
        _target_cell_count_for_day(cell_count, row.condition, int(row.day))
        for row in enriched.itertuples(index=False)
    ]
    enriched["exp_copy_target_day"] = [
        _target_copy_day(ddpcr, row.condition, int(row.day))
        for row in enriched.itertuples(index=False)
    ]
    for species in SPECIES:
        enriched[f"exp_{species}"] = [
            _target_copies(ddpcr, row.condition, int(row.day))[species]
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


def _growth_curve_metrics(timeline: pd.DataFrame) -> pd.DataFrame:
    metrics = timeline[timeline["day"].astype(int).between(21, DAY_END)].copy()
    metrics["day"] = metrics["day"].astype(int)
    metrics["sim_time"] = metrics["sim_time"].astype(float)

    first_by_condition = metrics.sort_values(["condition", "sim_time"]).groupby("condition", sort=False)[
        ["sim_log10_cell_count", "exp_log10_cell_count"]
    ].transform("first")
    metrics["sim_log10_fold_growth"] = metrics["sim_log10_cell_count"] - first_by_condition["sim_log10_cell_count"]
    metrics["exp_log10_fold_growth"] = metrics["exp_log10_cell_count"] - first_by_condition["exp_log10_cell_count"]
    metrics["log10_fold_growth_error"] = metrics["sim_log10_fold_growth"] - metrics["exp_log10_fold_growth"]

    ctrl = metrics[metrics["condition"] == "ctrl"][
        ["day", "sim_log10_cell_count", "exp_log10_cell_count", "sim_log10_fold_growth", "exp_log10_fold_growth"]
    ].rename(
        columns={
            "sim_log10_cell_count": "ctrl_sim_log10_cell_count",
            "exp_log10_cell_count": "ctrl_exp_log10_cell_count",
            "sim_log10_fold_growth": "ctrl_sim_log10_fold_growth",
            "exp_log10_fold_growth": "ctrl_exp_log10_fold_growth",
        }
    )
    metrics = metrics.merge(ctrl, on="day", how="left")
    metrics["sim_log10_ratio_vs_ctrl"] = metrics["sim_log10_cell_count"] - metrics["ctrl_sim_log10_cell_count"]
    metrics["exp_log10_ratio_vs_ctrl"] = metrics["exp_log10_cell_count"] - metrics["ctrl_exp_log10_cell_count"]
    metrics["log10_ratio_vs_ctrl_error"] = metrics["sim_log10_ratio_vs_ctrl"] - metrics["exp_log10_ratio_vs_ctrl"]
    metrics["sim_log10_fold_growth_vs_ctrl"] = (
        metrics["sim_log10_fold_growth"] - metrics["ctrl_sim_log10_fold_growth"]
    )
    metrics["exp_log10_fold_growth_vs_ctrl"] = (
        metrics["exp_log10_fold_growth"] - metrics["ctrl_exp_log10_fold_growth"]
    )
    metrics["log10_fold_growth_vs_ctrl_error"] = (
        metrics["sim_log10_fold_growth_vs_ctrl"] - metrics["exp_log10_fold_growth_vs_ctrl"]
    )

    condition_order = pd.Categorical(metrics["condition"], categories=CONDITIONS, ordered=True)
    metrics = metrics.assign(condition_order=condition_order).sort_values(["condition_order", "sim_time"])
    columns = [
        "condition",
        "condition_label",
        "day",
        "sim_time",
        "sim_representative_cells",
        "sim_weighted_cell_count",
        "exp_cell_count",
        "sim_log10_cell_count",
        "exp_log10_cell_count",
        "sim_log10_fold_growth",
        "exp_log10_fold_growth",
        "log10_fold_growth_error",
        "sim_log10_ratio_vs_ctrl",
        "exp_log10_ratio_vs_ctrl",
        "log10_ratio_vs_ctrl_error",
        "sim_log10_fold_growth_vs_ctrl",
        "exp_log10_fold_growth_vs_ctrl",
        "log10_fold_growth_vs_ctrl_error",
    ]
    return metrics[columns].reset_index(drop=True)


def _day56_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    week6_parts = []
    ordered = timeline.sort_values(["condition", "sim_time"])
    for condition, group in ordered.groupby("condition", sort=False):
        day56 = group[group["day"].astype(int) == DAY_END]
        week6_parts.append((day56 if not day56.empty else group.tail(1)).copy())
    week6 = pd.concat(week6_parts, ignore_index=True)
    ctrl = week6[week6["condition"] == "ctrl"].iloc[0]
    rows = []
    for row in week6.sort_values("condition").itertuples(index=False):
        out = {
            "condition": row.condition,
            "condition_label": row.condition_label,
            "day": int(row.day),
            "sim_time": float(row.sim_time),
            "stop_reason": row.stop_reason,
            "stop_time": row.stop_time,
            "sim_representative_cells": row.sim_representative_cells,
            "sim_log10_cell_count": row.sim_log10_cell_count,
            "exp_log10_cell_count": row.exp_log10_cell_count,
            "log10_cell_error": row.log10_cell_error,
            "exp_copy_target_day": row.exp_copy_target_day,
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


def _copy_timeline_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_days = set(range(DAY_START, DAY_END + 1, 7))
    for row in timeline[timeline["day"].astype(int).isin(target_days)].sort_values(["condition", "day"]).itertuples(index=False):
        for species in SPECIES:
            observed = float(getattr(row, f"exp_{species}"))
            simulated = float(getattr(row, f"sim_{species}"))
            rows.append(
                {
                    "condition": row.condition,
                    "condition_label": row.condition_label,
                    "day": int(row.day),
                    "sim_time": float(row.sim_time),
                    "species": species,
                    "observed_copy_number": observed,
                    "simulated_copy_number": simulated,
                    "absolute_error": simulated - observed,
                    "percent_error": 100.0 * (simulated - observed) / observed if observed > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _direction_label(percent_change: float) -> str:
    if percent_change > 5.0:
        return "increase"
    if percent_change < -5.0:
        return "decrease"
    return "flat"


def _copy_direction_checks(copy_timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, species), group in copy_timeline.groupby(["condition", "species"], sort=False):
        ordered = group.sort_values("day")
        start = ordered[ordered["day"].astype(int) == DAY_START]
        end = ordered[ordered["day"].astype(int) == DAY_END]
        if start.empty or end.empty:
            continue
        observed_start = float(start["observed_copy_number"].iloc[0])
        observed_end = float(end["observed_copy_number"].iloc[0])
        simulated_start = float(start["simulated_copy_number"].iloc[0])
        simulated_end = float(end["simulated_copy_number"].iloc[0])
        observed_change = 100.0 * (observed_end - observed_start) / observed_start
        simulated_change = 100.0 * (simulated_end - simulated_start) / simulated_start
        observed_direction = _direction_label(observed_change)
        simulated_direction = _direction_label(simulated_change)
        rows.append(
            {
                "condition": condition,
                "species": species,
                "observed_day14_to_day56_percent": observed_change,
                "simulated_day14_to_day56_percent": simulated_change,
                "observed_direction": observed_direction,
                "simulated_direction": simulated_direction,
                "direction_pass": observed_direction == simulated_direction,
            }
        )
    return pd.DataFrame(rows)


def _special_checks(day56: pd.DataFrame) -> pd.DataFrame:
    d = day56.set_index("condition")
    rows = [
        {
            "check": "P10 day56 CDK4 ratio > 1",
            "pass": bool(d.loc["P10", "sim_CDK4_ratio_vs_ctrl"] > 1.0),
            "value": float(d.loc["P10", "sim_CDK4_ratio_vs_ctrl"]),
        },
        {
            "check": "P50 day56 CDK4 ratio > 1",
            "pass": bool(d.loc["P50", "sim_CDK4_ratio_vs_ctrl"] > 1.0),
            "value": float(d.loc["P50", "sim_CDK4_ratio_vs_ctrl"]),
        },
        {
            "check": "P250 MYC/CDK4/PDGFRA all below ctrl",
            "pass": bool(
                (d.loc["P250", ["sim_MYC_ratio_vs_ctrl", "sim_CDK4_ratio_vs_ctrl", "sim_PDGFRA_ratio_vs_ctrl"]] < 1.0).all()
            ),
            "value": float(d.loc["P250", ["sim_MYC_ratio_vs_ctrl", "sim_CDK4_ratio_vs_ctrl", "sim_PDGFRA_ratio_vs_ctrl"]].max()),
        },
        {
            "check": "P250 CDK4 is strongest P250 drop",
            "pass": bool(
                d.loc["P250", "sim_CDK4_ratio_vs_ctrl"]
                <= d.loc["P250", ["sim_MYC_ratio_vs_ctrl", "sim_PDGFRA_ratio_vs_ctrl"]].min()
            ),
            "value": float(d.loc["P250", "sim_CDK4_ratio_vs_ctrl"]),
        },
        {
            "check": "R20 CDK4/PDGFRA near ctrl and MYC slightly low",
            "pass": bool(
                abs(float(d.loc["R20", "sim_CDK4_ratio_vs_ctrl"]) - 1.0) <= 0.10
                and abs(float(d.loc["R20", "sim_PDGFRA_ratio_vs_ctrl"]) - 1.0) <= 0.10
                and float(d.loc["R20", "sim_MYC_ratio_vs_ctrl"]) < 1.0
            ),
            "value": float(d.loc["R20", "sim_MYC_ratio_vs_ctrl"]),
        },
        {
            "check": "R100 MYC and PDGFRA below ctrl",
            "pass": bool(
                float(d.loc["R100", "sim_MYC_ratio_vs_ctrl"]) < 1.0
                and float(d.loc["R100", "sim_PDGFRA_ratio_vs_ctrl"]) < 1.0
            ),
            "value": float(max(d.loc["R100", "sim_MYC_ratio_vs_ctrl"], d.loc["R100", "sim_PDGFRA_ratio_vs_ctrl"])),
        },
        {
            "check": "R500 MYC and PDGFRA strongest R-group drops, CDK4 more retained",
            "pass": bool(
                float(d.loc["R500", "sim_MYC_ratio_vs_ctrl"]) < float(d.loc["R100", "sim_MYC_ratio_vs_ctrl"])
                and float(d.loc["R500", "sim_PDGFRA_ratio_vs_ctrl"]) < float(d.loc["R100", "sim_PDGFRA_ratio_vs_ctrl"])
                and float(d.loc["R500", "sim_CDK4_ratio_vs_ctrl"]) > float(d.loc["R500", "sim_PDGFRA_ratio_vs_ctrl"])
            ),
            "value": float(d.loc["R500", "sim_PDGFRA_ratio_vs_ctrl"]),
        },
    ]
    return pd.DataFrame(rows)


def _growth_summary(growth_curve: pd.DataFrame) -> pd.DataFrame:
    window = growth_curve.copy()
    rows = []
    for condition, group in window.sort_values(["condition", "sim_time"]).groupby("condition", sort=False):
        group = group.sort_values("sim_time")
        first = group.iloc[0]
        last = group.iloc[-1]
        week_span = float(last["sim_time"] - first["sim_time"])
        sim_slope = (float(last["sim_log10_fold_growth"]) - float(first["sim_log10_fold_growth"])) / week_span
        exp_slope = (float(last["exp_log10_fold_growth"]) - float(first["exp_log10_fold_growth"])) / week_span
        rows.append(
            {
                "condition": condition,
                "condition_label": str(last["condition_label"]),
                "day_start": int(first["day"]),
                "day_end": int(last["day"]),
                "sim_time_start": float(first["sim_time"]),
                "sim_time_end": float(last["sim_time"]),
                "sim_day56_representative_cells": int(last["sim_representative_cells"]),
                "exp_day56_cell_count": float(last["exp_cell_count"]),
                "sim_log10_fold_growth_day56": float(last["sim_log10_fold_growth"]),
                "exp_log10_fold_growth_day56": float(last["exp_log10_fold_growth"]),
                "log10_fold_growth_error_day56": float(last["log10_fold_growth_error"]),
                "sim_log10_slope": sim_slope,
                "exp_log10_slope": exp_slope,
                "sim_day56_log10_ratio_vs_ctrl": float(last["sim_log10_ratio_vs_ctrl"]),
                "exp_day56_log10_ratio_vs_ctrl": float(last["exp_log10_ratio_vs_ctrl"]),
                "day56_log10_ratio_vs_ctrl_error": float(last["log10_ratio_vs_ctrl_error"]),
                "sim_log10_ratio_vs_ctrl_delta": float(
                    last["sim_log10_ratio_vs_ctrl"] - first["sim_log10_ratio_vs_ctrl"]
                ),
                "exp_log10_ratio_vs_ctrl_delta": float(
                    last["exp_log10_ratio_vs_ctrl"] - first["exp_log10_ratio_vs_ctrl"]
                ),
                "sim_monotonic_growth": bool(
                    (group["sim_representative_cells"].astype(float).diff().dropna() > 0.0).all()
                ),
                "exp_monotonic_growth": bool((group["exp_cell_count"].astype(float).diff().dropna() > 0.0).all()),
            }
        )
    summary = pd.DataFrame(rows)
    ctrl = summary[summary["condition"] == "ctrl"].iloc[0]
    summary["sim_slope_ratio_vs_ctrl"] = summary["sim_log10_slope"] / float(ctrl["sim_log10_slope"])
    summary["exp_slope_ratio_vs_ctrl"] = summary["exp_log10_slope"] / float(ctrl["exp_log10_slope"])
    summary["slope_ratio_error"] = summary["sim_slope_ratio_vs_ctrl"] - summary["exp_slope_ratio_vs_ctrl"]
    summary["sim_fold_growth_ratio_vs_ctrl"] = summary["sim_log10_fold_growth_day56"] / float(
        ctrl["sim_log10_fold_growth_day56"]
    )
    summary["exp_fold_growth_ratio_vs_ctrl"] = summary["exp_log10_fold_growth_day56"] / float(
        ctrl["exp_log10_fold_growth_day56"]
    )
    summary["fold_growth_ratio_error"] = (
        summary["sim_fold_growth_ratio_vs_ctrl"] - summary["exp_fold_growth_ratio_vs_ctrl"]
    )
    return summary


def _initial_state_summary(initializations: dict[str, cfg.InitializationParameters]) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        initialization = initializations[condition]
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


def _initial_anchor_summary(
    raw_dir: Path,
    initializations: dict[str, cfg.InitializationParameters],
) -> pd.DataFrame:
    raw = _raw_week1_anchors(raw_dir)
    rows = []
    for condition in CONDITIONS:
        initialization = initializations[condition]
        realized = _initial_copy_means(initialization)

        for species in SPECIES:
            raw_value = float(
                raw[
                    (raw["condition"].astype(str) == condition)
                    & (raw["week"].astype(int) == 1)
                    & (raw["species"].astype(str) == species)
                ]["ddpcr_copy_number"].median()
            )
            species_idx = cfg.SPECIES_INDEX[species]
            expected_value = float(initialization.parametric_copy_number_mean[species_idx])
            realized_value = float(realized[species_idx])
            rows.append(
                {
                    "condition": condition,
                    "species": species,
                    "raw_week1_ddpcr_copy_number": raw_value,
                    "initialized_bulk_mean": expected_value,
                    "difference": expected_value - raw_value,
                    "realized_distribution_bulk_mean": realized_value,
                    "realized_sampling_difference": realized_value - raw_value,
                    "uses_initial_copy_scale": "no",
                }
            )
    return pd.DataFrame(rows)


def _write_markdown_report(
    output_dir: Path,
    day56: pd.DataFrame,
    growth: pd.DataFrame,
    stops: pd.DataFrame,
    special: pd.DataFrame,
    directions: pd.DataFrame,
    anchors: pd.DataFrame,
) -> None:
    lines = [
        "# T87 Goal Validation",
        "",
        f"Time mapping: day14 is `t=0`, day56 is `t={T_DAY56:g}`. Validation uses `t_max={cfg.DEFAULT_MODEL_PARAMETERS.simulation.t_max:g}` and `record_times={cfg.DEFAULT_MODEL_PARAMETERS.simulation.record_times}`.",
        "Mechanism priority: selection, drug effective signal, and state-specific heterogeneity with raw week1 bulk means locked. Turnover parameters are not used as the primary fitting lever in this validation.",
        "Copy-number targets use exact measured ddPCR days from the filtered day14-day56 table; no day77 cell-count target is used for copy-number fitting.",
        "",
        "## Stop Reasons",
        "",
        stops.to_markdown(index=False),
        "",
        "## Final State Fractions",
        "",
        stops[
            [
                "condition",
                "final_hard_fraction_NPC_OPC",
                "final_hard_fraction_AC_MES",
                "final_hard_fraction_NPC",
                "final_hard_fraction_OPC",
                "final_hard_fraction_AC",
                "final_hard_fraction_MES",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Day56 Summary",
        "",
        day56.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Growth Summary",
        "",
        "`sim_log10_slope` and fold-growth ratios are computed from day21-to-day56 normalized `log10(N_t / N_start)`. Per-timepoint normalized growth and relative-to-ctrl gaps are written to `growth_curve_metrics.csv`.",
        "",
        growth.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Initial Anchor Check",
        "",
        anchors.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Copy Direction Checks",
        "",
        directions.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Special Checks",
        "",
    ]
    lines.append(special.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")
    output_dir.joinpath("validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _simulation_params(args)
    cfg.validate_model_parameters(params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    _, ddpcr, cell_count = _load_targets(raw_dir)
    max_workers = min(max(1, int(args.workers)), len(CONDITIONS))
    timeline_by_condition: dict[str, pd.DataFrame] = {}
    stop_by_condition: dict[str, dict] = {}
    initializations: dict[str, cfg.InitializationParameters] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_condition,
                condition,
                params=params,
                raw_dir=raw_dir,
                seed=int(args.seed),
                rows_per_state=int(args.rows_per_state),
            ): condition
            for condition in CONDITIONS
        }
        for future in as_completed(futures):
            condition = futures[future]
            timeline, stop, initialization = future.result()
            timeline_by_condition[condition] = timeline
            stop_by_condition[condition] = stop
            initializations[condition] = initialization
            print(
                f"{condition}: stop={stop['stop_reason']} "
                f"at t={float(stop['stop_time']):.2f}, "
                f"final_cells={stop['final_representative_cells']}"
            )

    timeline_parts = [timeline_by_condition[condition] for condition in CONDITIONS]
    stop_rows = [stop_by_condition[condition] for condition in CONDITIONS]

    timeline = pd.concat(timeline_parts, ignore_index=True)
    timeline = _add_targets_and_errors(timeline, ddpcr, cell_count, n_init=int(args.n_init))
    day56 = _day56_summary(timeline)
    copy_timeline = _copy_timeline_summary(timeline)
    directions = _copy_direction_checks(copy_timeline)
    special = _special_checks(day56)
    growth_curve = _growth_curve_metrics(timeline)
    growth = _growth_summary(growth_curve)
    stops = pd.DataFrame(stop_rows)
    initial = _initial_state_summary(initializations)
    anchors = _initial_anchor_summary(raw_dir, initializations)

    timeline.to_csv(output_dir / "timeline_comparison.csv", index=False)
    copy_timeline.to_csv(output_dir / "copy_timeline_comparison.csv", index=False)
    day56.to_csv(output_dir / "day56_summary.csv", index=False)
    directions.to_csv(output_dir / "copy_direction_checks.csv", index=False)
    special.to_csv(output_dir / "special_checks.csv", index=False)
    growth_curve.to_csv(output_dir / "growth_curve_metrics.csv", index=False)
    growth.to_csv(output_dir / "growth_summary.csv", index=False)
    stops.to_csv(output_dir / "stop_reasons.csv", index=False)
    initial.to_csv(output_dir / "initial_state_summary.csv", index=False)
    anchors.to_csv(output_dir / "initial_anchor_check.csv", index=False)
    _write_markdown_report(output_dir, day56, growth, stops, special, directions, anchors)

    print(f"Wrote validation outputs to {output_dir}")
    print(day56[["condition", "sim_CDK4_ratio_vs_ctrl", "exp_CDK4_ratio_vs_ctrl", "sim_PDGFRA_ratio_vs_ctrl", "exp_PDGFRA_ratio_vs_ctrl"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw") / "t87_drug_bulkfit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "t87_goal_validation")
    defaults = cfg.DEFAULT_MODEL_PARAMETERS.simulation
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--rows-per-state", type=int, default=512)
    parser.add_argument("--n-init", type=int, default=defaults.n_init)
    parser.add_argument("--target-population-size", type=int, default=0, help="Use 0 to stop at t_max.")
    parser.add_argument("--max-pop-size", type=int, default=cfg.DEFAULT_MODEL_PARAMETERS.simulation.max_pop_size)
    parser.add_argument(
        "--workers",
        type=int,
        default=len(CONDITIONS),
        help="Condition worker processes. Default runs one process per condition.",
    )
    parser.add_argument("--record-events", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
