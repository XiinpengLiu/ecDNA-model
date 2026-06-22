"""Write and validate the nine local ABC-SMC output files (fit_method.md section 13)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

from .io_utils import ensure_dir, write_json, write_table, write_yaml
from .parameters import PARAMETER_SPECS


REQUIRED_OUTPUTS = (
    "fit_config.yaml",
    "baseline_config_prediction.csv",
    "baseline_config_rmse.json",
    "candidates_all.parquet",
    "accepted_particles.parquet",
    "generation_summary.csv",
    "ddpcr_predictions_long.parquet",
    "final_posterior_summary.csv",
    "ppc_summary.csv",
)

ACCEPTED_COLUMNS = [
    "generation",
    "particle_id",
    "candidate_id",
    "global_id",
    "rank",
    "proposal_type",
    "parent_generation",
    "parent_candidate_id",
    "seed",
    "rmse_ddpcr",
    "epsilon_generation",
    "distance_to_config",
    *[spec.theta_column for spec in PARAMETER_SPECS],
    *[spec.log2_fold_column for spec in PARAMETER_SPECS],
    *[spec.phi_column for spec in PARAMETER_SPECS],
]

GENERATION_SUMMARY_COLUMNS = [
    "generation",
    "N_candidates",
    "K_accepted",
    "epsilon",
    "rmse_min",
    "rmse_median_all",
    "rmse_median_accepted",
    "rmse_config",
    "distance_to_config_median_accepted",
    "accepted_anchor",
    "small_accepted_count",
    "large_accepted_count",
]

POSTERIOR_SUMMARY_COLUMNS = [
    "parameter",
    "theta_config",
    "median",
    "q05",
    "q10",
    "q90",
    "q95",
    "mean",
    "sd",
    "min",
    "max",
    "local_interval_width",
    "distance_from_config_median",
    "median_log2_fold_vs_config",
    "q05_log2_fold_vs_config",
    "q95_log2_fold_vs_config",
]

PPC_SUMMARY_COLUMNS = [
    "condition",
    "week",
    "species",
    "ddpcr_obs",
    "pred_median",
    "pred_q05",
    "pred_q95",
    "pred_min",
    "pred_max",
    "rmse_component",
    "covered_by_q05_q95",
]


def accepted_particles_frame(ranked: pd.DataFrame) -> pd.DataFrame:
    """Reshape the accepted (rank <= K) rows into the accepted_particles schema."""
    accepted = ranked[ranked["accepted"]].copy()
    accepted.insert(1, "particle_id", accepted["global_id"].astype(int))
    return accepted.loc[:, ACCEPTED_COLUMNS].reset_index(drop=True)


def generation_summary_row(ranked: pd.DataFrame) -> dict[str, Any]:
    accepted = ranked[ranked["accepted"]].copy()
    anchor = ranked[ranked["proposal_type"] == "anchor"].sort_values("rmse_ddpcr")
    return {
        "generation": int(ranked["generation"].iloc[0]),
        "N_candidates": int(len(ranked)),
        "K_accepted": int(len(accepted)),
        "epsilon": float(accepted["rmse_ddpcr"].max()),
        "rmse_min": float(ranked["rmse_ddpcr"].min()),
        "rmse_median_all": float(ranked["rmse_ddpcr"].median()),
        "rmse_median_accepted": float(accepted["rmse_ddpcr"].median()),
        "rmse_config": float(anchor["rmse_ddpcr"].iloc[0]) if not anchor.empty else float("nan"),
        "distance_to_config_median_accepted": float(accepted["distance_to_config"].median()),
        "accepted_anchor": bool((accepted["proposal_type"] == "anchor").any()),
        "small_accepted_count": int((accepted["proposal_type"] == "small").sum()),
        "large_accepted_count": int((accepted["proposal_type"] == "large").sum()),
    }


def posterior_summary(final_accepted: pd.DataFrame, phi0: np.ndarray) -> pd.DataFrame:
    """Per-parameter quantiles over the final accepted particles (K=20)."""
    rows = []
    for idx, spec in enumerate(PARAMETER_SPECS):
        theta = final_accepted[spec.theta_column].astype(float)
        folds = final_accepted[spec.log2_fold_column].astype(float)
        rows.append(
            {
                "parameter": spec.config_path,
                "theta_config": float(math.exp(phi0[idx])),
                "median": float(theta.median()),
                "q05": float(theta.quantile(0.05)),
                "q10": float(theta.quantile(0.10)),
                "q90": float(theta.quantile(0.90)),
                "q95": float(theta.quantile(0.95)),
                "mean": float(theta.mean()),
                "sd": float(theta.std(ddof=1)) if len(theta) > 1 else 0.0,
                "min": float(theta.min()),
                "max": float(theta.max()),
                "local_interval_width": float(theta.quantile(0.95) - theta.quantile(0.05)),
                "distance_from_config_median": float(theta.median() - math.exp(phi0[idx])),
                "median_log2_fold_vs_config": float(folds.median()),
                "q05_log2_fold_vs_config": float(folds.quantile(0.05)),
                "q95_log2_fold_vs_config": float(folds.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)[POSTERIOR_SUMMARY_COLUMNS]


def ppc_summary(final_predictions: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, week, species) predictive quantiles from the final particles."""
    rows = []
    for keys, group in final_predictions.groupby(["condition", "week", "species"], sort=True):
        condition, week, species = keys
        obs = float(group["ddpcr_obs"].iloc[0])
        sims = group["ddpcr_sim"].astype(float)
        pred_median = float(sims.median())
        pred_q05 = float(sims.quantile(0.05))
        pred_q95 = float(sims.quantile(0.95))
        rows.append(
            {
                "condition": str(condition),
                "week": int(week),
                "species": str(species),
                "ddpcr_obs": obs,
                "pred_median": pred_median,
                "pred_q05": pred_q05,
                "pred_q95": pred_q95,
                "pred_min": float(sims.min()),
                "pred_max": float(sims.max()),
                "rmse_component": float((np.log2(pred_median + 1.0) - np.log2(obs + 1.0)) ** 2),
                "covered_by_q05_q95": bool(pred_q05 <= obs <= pred_q95),
            }
        )
    return pd.DataFrame(rows)[PPC_SUMMARY_COLUMNS]


def write_fit_config(
    output_dir: Path,
    *,
    conditions: tuple[str, ...],
    generations: int,
    n_per_generation: int,
    accepted_count: int,
    seed: int,
    n_init: int,
    rows_per_state: int,
    target_population_size: int | None,
    max_pop_size: int,
    targets: pd.DataFrame,
) -> Path:
    """Record the full replay configuration (fit_method.md section 13.1)."""
    from .proposal import PROPOSAL_SCHEDULE, proposal_counts

    schedules = {}
    for generation in range(int(generations)):
        anchor, small, large = proposal_counts(int(n_per_generation), generation)
        schedule = dict(PROPOSAL_SCHEDULE[generation]) if generation in PROPOSAL_SCHEDULE else {}
        schedule.update({"anchor": anchor, "small": small, "large": large})
        schedules[f"generation_{generation}"] = schedule

    payload = {
        "analysis_name": "config_centered_local_abc_smc_fit",
        "purpose": "reconstruct_final_stage_local_abc_path_around_config",
        "N_total": int(generations) * int(n_per_generation),
        "generations": int(generations),
        "N_per_generation": int(n_per_generation),
        "K_accepted": min(int(accepted_count), int(n_per_generation)),
        "conditions": list(conditions),
        "ddpcr_target_rows": int(len(targets)),
        "distance": "log2_ddpcr_rmse",
        "posterior": "equal_weight_topK_final_generation",
        "parameters": [
            {"name": spec.name, "config_path": spec.config_path, "transform": "log"}
            for spec in PARAMETER_SPECS
        ],
        "proposal_schedule": schedules,
        "simulation": {
            "n_init": int(n_init),
            "rows_per_state": int(rows_per_state),
            "target_population_size": target_population_size,
            "max_pop_size": int(max_pop_size),
            "seed": int(seed),
        },
    }
    return write_yaml(payload, output_dir / "fit_config.yaml")


def write_baseline_prediction(output_dir: Path, predictions: pd.DataFrame, rmse: float) -> None:
    """Step 0 anchor outputs: baseline ddPCR prediction and its RMSE."""
    baseline = predictions.drop(columns=["generation", "candidate_id", "global_id", "accepted"], errors="ignore")
    baseline.to_csv(output_dir / "baseline_config_prediction.csv", index=False)
    write_json(
        {"rmse_ddpcr": float(rmse), "distance": "log2_ddpcr_rmse", "n_points": int(len(baseline))},
        output_dir / "baseline_config_rmse.json",
    )


def write_all_outputs(
    output_dir: Path,
    *,
    candidates_all: pd.DataFrame,
    accepted_all: pd.DataFrame,
    predictions_all: pd.DataFrame,
    generation_summary: pd.DataFrame,
    final_accepted: pd.DataFrame,
    final_predictions: pd.DataFrame,
    phi0: np.ndarray,
) -> None:
    """Write the six generation/posterior artifacts (config + baseline written separately)."""
    ensure_dir(output_dir)
    write_table(candidates_all, output_dir / "candidates_all.parquet")
    write_table(accepted_all, output_dir / "accepted_particles.parquet")
    write_table(predictions_all, output_dir / "ddpcr_predictions_long.parquet")
    generation_summary.to_csv(output_dir / "generation_summary.csv", index=False)
    posterior_summary(final_accepted, phi0).to_csv(output_dir / "final_posterior_summary.csv", index=False)
    ppc_summary(final_predictions).to_csv(output_dir / "ppc_summary.csv", index=False)


def validate_outputs(output_dir: Path) -> None:
    """Check all nine files exist with the required columns and no NaNs."""
    output_dir = Path(output_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    cfg.require(not missing, f"Missing fit output files: {missing}")

    candidates = pd.read_parquet(output_dir / "candidates_all.parquet")
    accepted = pd.read_parquet(output_dir / "accepted_particles.parquet")
    predictions = pd.read_parquet(output_dir / "ddpcr_predictions_long.parquet")
    ppc = pd.read_csv(output_dir / "ppc_summary.csv")
    posterior = pd.read_csv(output_dir / "final_posterior_summary.csv")
    summary = pd.read_csv(output_dir / "generation_summary.csv")

    required_candidates = {
        "generation", "candidate_id", "global_id", "proposal_type", "parent_generation",
        "parent_candidate_id", "seed", "accepted", "rank", "rmse_ddpcr", "epsilon_generation",
        "distance_to_config",
    }
    required_accepted = {
        "generation", "particle_id", "rank", "proposal_type", "rmse_ddpcr",
        "epsilon_generation", "distance_to_config",
    }
    required_predictions = {
        "generation", "candidate_id", "accepted", "condition", "week", "species",
        "ddpcr_obs", "ddpcr_sim", "log2_obs", "log2_sim", "residual",
    }
    required_ppc = {
        "condition", "week", "species", "ddpcr_obs", "pred_median", "pred_q05", "pred_q95",
        "pred_min", "pred_max", "rmse_component", "covered_by_q05_q95",
    }
    for frame, columns, name in (
        (candidates, required_candidates, "candidates_all.parquet"),
        (accepted, required_accepted, "accepted_particles.parquet"),
        (predictions, required_predictions, "ddpcr_predictions_long.parquet"),
        (ppc, required_ppc, "ppc_summary.csv"),
    ):
        missing_columns = columns - set(frame.columns)
        cfg.require(not missing_columns, f"{name} is missing columns: {sorted(missing_columns)}")

    complete = [
        "rmse_ddpcr", "epsilon_generation", "distance_to_config",
        *[spec.theta_column for spec in PARAMETER_SPECS],
        *[spec.log2_fold_column for spec in PARAMETER_SPECS],
    ]
    cfg.require(candidates[complete].notna().all().all(), "candidates_all has missing parameter or score data.")
    cfg.require(
        predictions[["ddpcr_obs", "ddpcr_sim", "log2_obs", "log2_sim", "residual"]].notna().all().all(),
        "ddpcr_predictions_long has missing data.",
    )
    cfg.require(not posterior.empty and posterior.notna().all().all(), "final_posterior_summary is incomplete.")
    cfg.require(not summary.empty and summary.notna().all().all(), "generation_summary is incomplete.")
