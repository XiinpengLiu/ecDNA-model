"""Final report layer for the bulk-only fit method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_table, write_text_pdf


def build_final_report_layer(
    observation_dir: str | Path,
    lite_dir: str | Path,
    full_dir: str | Path,
    output_dir: str | Path,
    validation_dir: str | Path | None = None,
) -> dict[str, Path]:
    obs_dir = Path(observation_dir)
    lite = Path(lite_dir)
    full = Path(full_dir)
    validation = Path(validation_dir) if validation_dir is not None else full.parent / "09_validation"
    out = ensure_dir(output_dir)
    data_constrained = _data_constrained(lite)
    latent = _latent_results()
    interp = _interpretability(validation, full)
    scenarios = _scenario_classes(data_constrained, interp)
    write_table(data_constrained, out / "FINAL_data_constrained_results.csv")
    write_table(latent, out / "FINAL_latent_model_dependent_results.csv")
    write_table(interp, out / "FINAL_parameter_interpretability_table.csv")
    write_table(_hidden_4state_summary(lite), out / "FULL_hidden_4state_summary.parquet")
    write_table(_hidden_copy_distribution_summary(lite), out / "FULL_hidden_copy_distribution_summary.parquet")
    write_table(_event_summary(full), out / "FULL_event_summary.parquet")
    write_table(scenarios, out / "FULL_scenario_classes.parquet")
    _copy_zarr_metadata(full / "FULL_replay_histories.zarr", out / "FULL_latent_history_samples.zarr")
    write_text_pdf(
        out / "FINAL_bulkfit_main_report.pdf",
        "Final Bulk Fit Main Report",
        [
            "Primary results are restricted to bulk ddPCR trajectories, cell counts, drug/dose bulk effects, and flow3 steady projection.",
            "Latent histories are model-dependent and current data do not directly identify them.",
            "qPCDR, ecTAG, flow4, and state-specific copy likelihoods were closed throughout the pipeline.",
        ],
    )
    write_text_pdf(
        out / "FULL_scenario_summary.pdf",
        "Full Scenario Summary",
        [f"{row.scenario}: {row.interpretation}" for row in scenarios.itertuples(index=False)],
    )
    write_text_pdf(out / "FINAL_scenario_summary.pdf", "Final Scenario Summary", [f"{row.scenario}: {row.posterior_weight:.3f}" for row in scenarios.itertuples(index=False)])
    write_json(
        out / "FINAL_method_manifest.json",
        {
            "method_source": "markdown/fit_method.md",
            "workflow": ["00_manifest_bulk", "01_clean_qc_bulk", "02_observation_bulk", "03_v4_lite_bulk", "04_parameter_registry", "05_prior_predictive_gate", "06_moment_prescreen", "07_full_initialization", "08_full_partialobs_smc", "09_validation_identifiability", "10_final_report"],
            "fit_mask": read_json(obs_dir / "obs_params_for_full.json")["fit_mask"],
            "outputs": list(schemas.FINAL_OUTPUTS),
        },
    )
    failures = _fit_incompatible(validation, full)
    if failures:
        (out / "FULL_bulkfit_incompatible_under_biological_priors.md").write_text(
            "# FULL Bulk Fit Incompatible Under Biological Priors\n\n"
            "fit requires boundary parameter values or failed validation gates.\n\n"
            + "\n".join(f"- {reason}" for reason in failures)
            + "\n",
            encoding="utf-8",
        )
    return {name: out / name for name in schemas.FINAL_OUTPUTS}


def materialize_method_layout(output_root: str | Path) -> dict[str, Path]:
    # The pipeline already writes the method layout directly.
    layout = schemas.ResultLayout(Path(output_root))
    return {
        "manifest": layout.manifest,
        "clean_data": layout.clean_data,
        "observation": layout.observation,
        "lite": layout.lite,
        "parameter_registry": layout.parameter_registry,
        "prior_predictive": layout.prior_predictive,
        "moment_prescreen": layout.moment_prescreen,
        "full_init": layout.full_init,
        "full_smc": layout.full_smc,
        "validation": layout.validation,
        "final_report": layout.final_report,
    }


def validate_final_artifacts(final_dir: str | Path) -> None:
    base = Path(final_dir)
    missing = [name for name in schemas.FINAL_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing final report artifacts: {', '.join(missing)}")
    interp = pd.read_csv(base / "FINAL_parameter_interpretability_table.csv")
    required = {"parameter", "role", "posterior_contraction", "prior_shift_z", "boundary_mass", "ridge_partner", "interpretation_status"}
    schemas.validate_required_columns(set(interp.columns), tuple(required), "FINAL_parameter_interpretability_table")


def _data_constrained(lite: Path) -> pd.DataFrame:
    dd = pd.read_parquet(lite / "BULK_LITE_ddpcr_trajectories.parquet")
    cc = pd.read_parquet(lite / "BULK_LITE_cell_count_trajectories.parquet")
    flow = read_json(lite / "BULK_LITE_flow3_steady.json")
    rows = []
    for row in dd.itertuples(index=False):
        rows.append({"result_type": "bulk ddPCR trajectory", "week": row.week, "condition": row.condition, "replicate": row.replicate, "species": row.species, "value": row.bulk_mean})
    for row in cc.itertuples(index=False):
        rows.append({"result_type": "cell count trajectory", "week": row.week, "condition": row.condition, "replicate": row.replicate, "species": "", "value": row.total_cell_count})
    for group, value in flow["fractions"].items():
        rows.append({"result_type": "flow3 steady projection", "week": "", "condition": "all", "replicate": "", "species": group, "value": value})
    rows.extend(_drug_effect_rows(dd, cc))
    return pd.DataFrame(rows)


def _drug_effect_rows(dd: pd.DataFrame, cc: pd.DataFrame) -> list[dict]:
    rows = []
    if "ctrl" not in set(dd["condition"]):
        return rows
    final_week = int(dd["week"].max())
    ctrl_dd = dd[(dd["condition"] == "ctrl") & (dd["week"] == final_week)].set_index("species")
    for condition, group in dd[(dd["condition"] != "ctrl") & (dd["week"] == final_week)].groupby("condition"):
        for row in group.itertuples(index=False):
            if row.species in ctrl_dd.index:
                effect = float(np.log(row.bulk_mean + 1e-9) - np.log(float(ctrl_dd.loc[row.species, "bulk_mean"]) + 1e-9))
                rows.append({"result_type": "drug/dose effect on bulk ecDNA mean", "week": final_week, "condition": condition, "replicate": row.replicate, "species": row.species, "value": effect})
    ctrl_cc = cc[(cc["condition"] == "ctrl") & (cc["week"] == final_week)]
    if not ctrl_cc.empty:
        ctrl_value = float(ctrl_cc["total_cell_count"].median())
        for row in cc[(cc["condition"] != "ctrl") & (cc["week"] == final_week)].itertuples(index=False):
            effect = float(np.log(row.total_cell_count + 1.0) - np.log(ctrl_value + 1.0))
            rows.append({"result_type": "drug/dose effect on growth", "week": final_week, "condition": row.condition, "replicate": row.replicate, "species": "", "value": effect})
    return rows


def _latent_results() -> pd.DataFrame:
    quantities = ["single-cell copy distribution", "NPC/OPC split", "gain/loss events", "division/death split", "state-specific copy burden", "co-segregation", "state transition histories"]
    return pd.DataFrame({"quantity": quantities, "interpretation": ["model-dependent; current data do not directly identify this quantity"] * len(quantities)})


def _interpretability(validation: Path, full: Path) -> pd.DataFrame:
    path = validation / "FULL_identifiability_report.csv"
    if path.exists():
        return pd.read_csv(path)
    params = pd.read_parquet(full / "FULL_particle_parameters.parquet")
    return pd.DataFrame({"parameter": params.columns, "role": "unknown", "posterior_contraction": 0.0, "prior_shift_z": 0.0, "boundary_mass": 0.0, "ridge_partner": "", "interpretation_status": "prior-driven"})


def _hidden_4state_summary(lite: Path) -> pd.DataFrame:
    init = pd.read_parquet(lite / "BULK_LITE_hidden_4state_initializer.parquet")
    init["metadata"] = "current data do not directly identify this quantity"
    return init


def _hidden_copy_distribution_summary(lite: Path) -> pd.DataFrame:
    dd = pd.read_parquet(lite / "BULK_LITE_ddpcr_trajectories.parquet")
    rows = []
    for row in dd.itertuples(index=False):
        rows.append({"week": row.week, "condition": row.condition, "replicate": row.replicate, "species": row.species, "bulk_mean_anchor": row.bulk_mean, "distribution": "mean-matched ZINB prior", "metadata": "current data do not directly identify this quantity"})
    return pd.DataFrame(rows)


def _event_summary(full: Path) -> pd.DataFrame:
    sidecar = full / "FULL_replay_histories_event_summary.parquet"
    if sidecar.exists():
        events = pd.read_parquet(sidecar)
        if not events.empty and "event_type" in events:
            result = events.groupby("event_type", as_index=False).size().rename(columns={"size": "weighted_count_mean"})
            result["metadata"] = "current data do not directly identify this quantity"
            return result
    params = pd.read_parquet(full / "FULL_particle_parameters.parquet")
    rows = []
    for name in ("division_death_turnover", "ecDNA_gain_loss_turnover"):
        if name in params:
            rows.append({"event_type": name, "weighted_count_mean": float(params[name].median()), "metadata": "current data do not directly identify this quantity"})
    return pd.DataFrame(rows)


def _scenario_classes(data: pd.DataFrame, interp: pd.DataFrame) -> pd.DataFrame:
    status = "boundary-forced" if (interp["interpretation_status"] == "boundary-forced").any() else "mixed latent histories"
    return pd.DataFrame({"scenario": [status], "posterior_weight": [1.0], "interpretation": ["scenario summarizes compatible bulk histories, not unique microscopic rates"]})


def _copy_zarr_metadata(src: Path, dst: Path) -> None:
    import shutil

    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _fit_incompatible(validation: Path, full: Path) -> list[str]:
    failures: list[str] = []
    boundary_path = validation / "FULL_boundary_forcing_report.csv"
    if boundary_path.exists():
        boundary = pd.read_csv(boundary_path)
        active = boundary[boundary["role"] == "active_effective_control"]
        if bool((active["boundary_mass"].astype(float) > 0.3).any()):
            failures.append("active effective control boundary mass exceeds 0.3")
    ddpcr_path = validation / "FULL_ddpcr_ppc.parquet"
    if ddpcr_path.exists():
        coverage = float(pd.read_parquet(ddpcr_path)["covered"].mean())
        if coverage < 0.85:
            failures.append(f"ddPCR PPC coverage below 0.85 ({coverage:.3f})")
    count_path = validation / "FULL_cellcount_ppc.parquet"
    if count_path.exists():
        coverage = float(pd.read_parquet(count_path)["covered"].mean())
        if coverage < 0.85:
            failures.append(f"cell count PPC coverage below 0.85 ({coverage:.3f})")
    flow_path = validation / "FULL_flow3steady_ppc.parquet"
    if flow_path.exists():
        error = float(pd.read_parquet(flow_path)["abs_error"].mean())
        if error >= 0.07:
            failures.append(f"flow3 steady mean absolute error is not <0.07 ({error:.3f})")
    weights_path = full / "FULL_particle_weights.parquet"
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
        accepted = weights[weights["accepted"]].copy()
        if accepted.empty:
            failures.append("no accepted final particles")
        else:
            w = accepted["weight"].astype(float).to_numpy()
            total = float(w.sum())
            if total > 0.0:
                w = w / total
                ess = float(1.0 / np.sum(w * w))
                if ess / max(1, len(accepted)) < 0.20:
                    failures.append(f"final ESS below 20% of retained particles ({ess:.3f}/{len(accepted)})")
    scores_path = full / "FULL_particle_scores.parquet"
    if scores_path.exists() and weights_path.exists():
        scores = pd.read_parquet(scores_path)
        weights = pd.read_parquet(weights_path)
        accepted_ids = set(weights.loc[weights["accepted"], "particle_id"].astype(int))
        accepted_scores = scores[scores["particle_id"].astype(int).isin(accepted_ids)]
        if "D_biology" in accepted_scores and bool((accepted_scores["D_biology"].astype(float) > 0.0).any()):
            failures.append("accepted particles violate biological hard boundary")
        if "D_prior" in accepted_scores and "D_prior" in scores and not accepted_scores.empty:
            prior_region_path = full.parent / "05_prior_predictive" / "PRIOR_predictive_accepted_region.parquet"
            if prior_region_path.exists():
                prior_region = pd.read_parquet(prior_region_path)
                prior_limit = float(prior_region["D_prior"].astype(float).quantile(0.99))
            else:
                prior_limit = float(scores["D_prior"].astype(float).quantile(0.99))
            top_half = accepted_scores.nsmallest(max(1, len(accepted_scores) // 2), "score")
            if float(top_half["D_prior"].astype(float).max()) > prior_limit:
                failures.append("top 50% accepted particles exceed prior predictive 99% D_prior threshold")
    holdout_path = validation / "FULL_holdout_validation.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path)
        for channel in ("ddpcr", "cell_count"):
            subset = holdout[holdout["channel"] == channel]
            if not subset.empty and float(subset["covered"].mean()) < 0.80:
                failures.append(f"held-out {channel} coverage below 0.80 ({float(subset['covered'].mean()):.3f})")
    return failures
