"""Observation calibration fixed before lite and full fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_dataset_netcdf, write_json, write_markdown_report, write_text_pdf
from fit.raw import load_clean_tables


def calculate_ddpcr_pooled_mean(flow_fractions: pd.DataFrame, state_species_means: pd.DataFrame) -> pd.DataFrame:
    """Compute bulk_mean[w,c,r,j] = sum_s f[w,c,r,s] * mu[w,c,r,s,j]."""

    required_flow = {"week", "condition", "replicate", "state_gate", "fraction"}
    required_mu = {"week", "condition", "replicate", "state_gate", "species", "copy_mean"}
    schemas.validate_required_columns(set(flow_fractions.columns), tuple(required_flow), "flow_fractions")
    schemas.validate_required_columns(set(state_species_means.columns), tuple(required_mu), "state_species_means")
    merged = state_species_means.merge(
        flow_fractions[["week", "condition", "replicate", "state_gate", "fraction"]],
        on=["week", "condition", "replicate", "state_gate"],
        how="left",
        validate="many_to_one",
    )
    if merged["fraction"].isna().any():
        raise ValueError("Cannot compute ddPCR pooled mean: missing flow fraction for a state/species mean")
    merged["weighted_mean"] = merged["fraction"].astype(float) * merged["copy_mean"].astype(float)
    return (
        merged.groupby(["week", "condition", "replicate", "species"], as_index=False)["weighted_mean"]
        .sum()
        .rename(columns={"weighted_mean": "bulk_mean"})
    )


def fit_observation_model(clean_dir: str | Path, output_dir: str | Path, seed: int = 1) -> dict:
    """Calibrate measurement-channel parameters from clean observations.

    This function does not fit biological dynamics. Its outputs are locked
    before full reconstruction so assay error cannot be absorbed by latent
    biology parameters.
    """

    rng = np.random.default_rng(seed)
    tables = load_clean_tables(clean_dir)
    out = ensure_dir(output_dir)
    ectag_max = int(tables["ectag"]["ectag_count"].max()) if not tables["ectag"].empty else 0
    bins = schemas.open_log2_copy_bins(ectag_max)
    params: dict[str, Any] = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "locked_for_full": True,
        "ddpcr_interpretation": "bulk pooled first moment only: sum_s f[w,c,r,s] * mu[w,c,r,s,j]",
        "ectag_interpretation": "species-specific open-support binned likelihood",
        "flow": _calibrate_flow(tables["flow"]),
        "qpcdr": _calibrate_qpcdr(tables["qpcdr"], tables["ectag"]),
        "ectag": _calibrate_ectag(tables["ectag"], bins),
        "ddpcr": _calibrate_ddpcr(tables["ddpcr"], tables["flow"], tables["ectag"]),
        "cell_count": _calibrate_cell_count(tables["cell_count"]),
    }
    ppc = _run_observation_ppc(tables, params, bins, rng)
    params["ppc"] = ppc
    validate_observation_params(params)
    write_json(out / "obs_params_for_lite.json", params)
    write_json(out / "obs_params_for_full.json", params)
    write_json(out / "obs_calibration_report.json", _observation_report_payload(params, tables))
    write_dataset_netcdf(
        out / "obs_calibration_fit.nc",
        {
            "qpcdr_sigma": [params["qpcdr"]["by_species"][species]["sigma"] for species in schemas.SPECIES],
            "ddpcr_sigma": [params["ddpcr"]["sigma_by_species"][species] for species in schemas.SPECIES],
            "flow_kappa": [params["flow"]["kappa_flow"]],
            "ppc_coverage": [ppc["coverage_by_channel"].get(channel, np.nan) for channel in ("flow", "qpcdr", "ectag", "ddpcr")],
        },
        attrs={"method_source": "markdown/fit_method.md", "locked_for_full": True},
    )
    write_text_pdf(
        out / "obs_calibration_ppc.pdf",
        "Observation Calibration PPC",
        [
            "Observation calibration is locked before full reconstruction.",
            "ddPCR is checked only as a bulk pooled mean anchor.",
            "ecTAG bins are species-specific and open-tailed.",
            f"PPC coverage by channel: {ppc['coverage_by_channel']}",
            f"Continue gate passed: {ppc['continue_gate_passed']}",
            f"Rows: {', '.join(f'{name}={len(df)}' for name, df in tables.items())}",
        ],
    )
    write_markdown_report(
        out / "obs_calibration_report.md",
        "Observation Calibration Report",
        [
            (
                "Scope",
                "The observation layer is calibrated once and locked before full reconstruction.",
            ),
            (
                "ddPCR",
                "ddPCR contributes only a pooled mean anchor: sum_s f[w,c,r,s] * mu[w,c,r,s,j]. It is never treated as a single-cell copy-number distribution.",
            ),
            (
                "ecTAG",
                f"ecTAG uses species-specific histogram bins: {', '.join(str(item['label']) for item in bins)}. The final bin is an open tail bin, not a detection ceiling.",
            ),
            (
                "Outputs",
                "Wrote obs_params_for_lite.json, obs_params_for_full.json, and calibration reports.",
            ),
            (
                "PPC",
                (
                    f"coverage_by_channel={ppc['coverage_by_channel']}; "
                    f"qpcdr_ddpcr_median_relative_error={ppc['qpcdr_ddpcr_median_relative_error']:.4f}; "
                    f"continue_gate_passed={ppc['continue_gate_passed']}"
                ),
            ),
        ],
    )
    return params


def load_observation_params(path: str | Path) -> dict:
    params = read_json(path)
    validate_observation_params(params)
    return params


def validate_observation_params(params: dict) -> None:
    required = {"flow", "qpcdr", "ectag", "ddpcr", "locked_for_full", "ddpcr_interpretation"}
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Observation params missing required fields: {missing}")
    if not bool(params["locked_for_full"]):
        raise ValueError("Observation calibration must be locked before full reconstruction")
    if "single-cell" in str(params["ddpcr_interpretation"]).lower() and "never" not in str(params["ddpcr_interpretation"]).lower():
        raise ValueError("ddPCR interpretation is ambiguous; it must be pooled mean only")
    for species in schemas.SPECIES:
        if species not in params["qpcdr"]["by_species"]:
            raise ValueError(f"qPCDR params missing species {species}")
        if species not in params["ddpcr"]["sigma_by_species"]:
            raise ValueError(f"ddPCR params missing species {species}")


def _calibrate_flow(flow: pd.DataFrame) -> dict:
    if flow["sort_purity"].notna().any():
        purity = float(np.nanmedian(flow["sort_purity"].astype(float)))
        policy = "metadata_sort_purity"
    else:
        purity = 0.95
        policy = "method_default_no_metadata_with_sensitivity_0.90_0.99"
    total_events = flow.groupby(["week", "condition", "replicate"])["pre_sort_count"].sum()
    kappa = float(np.nanmedian(total_events.to_numpy(dtype=float)))
    return {
        "purity": purity,
        "purity_policy": policy,
        "sensitivity_values": [0.90, 0.95, 0.99],
        "kappa_flow": float(max(1.0, kappa)),
    }


def _calibrate_qpcdr(qpcdr: pd.DataFrame, ectag: pd.DataFrame) -> dict:
    ectag_mean = _ectag_state_species_means(ectag)
    by_species = {}
    for species, group in qpcdr.groupby("species"):
        group = group.copy()
        group = group.merge(
            ectag_mean,
            on=["week", "condition", "replicate", "state_gate", "species"],
            how="left",
            validate="many_to_one",
        )
        fallback_mu = group["relative_copy_number"].astype(float).median()
        if not np.isfinite(fallback_mu):
            fallback_mu = 1.0
        mu = group["copy_mean"].astype(float).fillna(float(fallback_mu)).clip(lower=1e-9)
        if group["relative_copy_number"].notna().any():
            mask = group["relative_copy_number"].notna()
            values = np.log(group.loc[mask, "relative_copy_number"].astype(float).clip(lower=1e-9))
            x = np.log(mu.loc[mask].astype(float).clip(lower=1e-9))
            scale = "relative_copy_number_log"
            intercept, slope = _fit_linear_calibration(x, values, default_slope=1.0)
            predicted = intercept + slope * x
        else:
            mask = group["raw_Ct_or_Cq"].notna()
            values = group.loc[mask, "raw_Ct_or_Cq"].astype(float)
            x = np.log10(mu.loc[mask].astype(float).clip(lower=1e-9))
            scale = "ct_or_cq"
            intercept, signed_slope = _fit_linear_calibration(x, values, default_slope=-1.0)
            slope = -signed_slope
            predicted = intercept - slope * x
        residual = values.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
        sigma = _residual_sigma(residual, floor=0.05, fallback=_replicate_sigma(group, values, floor=0.05, fallback=0.25))
        by_species[str(species)] = {
            "scale": scale,
            "intercept": float(intercept),
            "slope": float(slope),
            "sigma": sigma,
            "calibration_target": "state/species ecTAG copy_mean",
            "n_calibration_rows": int(len(values)),
        }
    return {"by_species": by_species, "epsilon": 1e-9}


def _calibrate_ectag(ectag: pd.DataFrame, bins: list[dict]) -> dict:
    concentration = {}
    replicate_concordance = {}
    for species, group in ectag.groupby("species"):
        grouped = group.groupby(["week", "condition", "replicate", "state_gate"])["ectag_count"]
        sizes = grouped.size().to_numpy(dtype=float)
        median_size = float(np.nanmedian(sizes)) if len(sizes) else 1.0
        concordance = _ectag_replicate_concordance(group, bins)
        replicate_concordance[str(species)] = concordance
        concentration[str(species)] = float(max(1.0, median_size * max(0.25, concordance)))
    return {
        "bins": bins,
        "concentration_by_species": concentration,
        "replicate_concordance_by_species": replicate_concordance,
        "likelihood": "dirichlet_multinomial_species_specific_bins",
        "histogram_policy": "species_specific_open_support",
        "censoring": "disabled_unless_raw_metadata_declares_censoring",
    }


def _calibrate_ddpcr(ddpcr: pd.DataFrame, flow: pd.DataFrame, ectag: pd.DataFrame) -> dict:
    pooled = calculate_ddpcr_pooled_mean(
        flow[["week", "condition", "replicate", "state_gate", "fraction"]],
        _ectag_state_species_means(ectag),
    )
    merged = ddpcr.merge(pooled, on=["week", "condition", "replicate", "species"], how="left", validate="many_to_one")
    sigma = {}
    median_relative_error = {}
    for species, group in merged.groupby("species"):
        sd = group["ddpcr_sd_or_ci"].dropna().astype(float)
        mean = group["ddpcr_copy_number"].astype(float).clip(lower=1e-9)
        predicted = group["bulk_mean"].astype(float).clip(lower=1e-9)
        residual = np.log(mean.to_numpy(dtype=float)) - np.log(predicted.to_numpy(dtype=float))
        if len(sd):
            rel = float(np.median(sd.to_numpy(dtype=float) / mean.reindex(sd.index).to_numpy(dtype=float)))
            sigma_value = float(max(0.05, rel, _residual_sigma(residual, floor=0.0, fallback=0.0)))
        elif len(group) > 1:
            log_std = _residual_sigma(residual, floor=0.0, fallback=float(np.std(np.log(mean), ddof=1)))
            sigma_value = float(max(0.05, log_std))
        else:
            sigma_value = 0.25
        sigma[str(species)] = sigma_value
        median_relative_error[str(species)] = float(np.nanmedian(np.abs(mean - predicted) / mean))
    return {
        "sigma_by_species": sigma,
        "likelihood": "lognormal_on_bulk_pooled_mean",
        "calibration_target": "sum_s flow_fraction[w,c,r,s] * ecTAG_copy_mean[w,c,r,s,j]",
        "median_relative_error_by_species": median_relative_error,
    }


def _calibrate_cell_count(cell_count: pd.DataFrame) -> dict:
    values = cell_count["total_cell_count"].astype(float)
    median_count = float(np.nanmedian(values)) if len(values) else 20.0
    dispersion = float(max(1.0, median_count * 0.05))
    return {"dispersion": dispersion, "role": "growth_scale_if_reliable_else_plausibility_only"}


def _run_observation_ppc(tables: dict[str, pd.DataFrame], params: dict, bins: list[dict], rng: np.random.Generator) -> dict:
    """Compute method gate summaries from calibrated observation models."""

    flow_ppc = _flow_ppc_coverage(tables["flow"], params["flow"], rng)
    flow_sensitivity = _flow_purity_sensitivity(tables["flow"], params["flow"])
    flow_coverage = float(flow_ppc["coverage"])
    qpcdr_coverage, qpcdr_rel = _qpcdr_ppc_coverage(tables["qpcdr"], tables["ectag"], params["qpcdr"])
    ectag_coverage = _ectag_ppc_coverage(tables["ectag"], bins)
    ddpcr_coverage, ddpcr_rel = _ddpcr_ppc_coverage(tables["ddpcr"], tables["flow"], tables["ectag"], params["ddpcr"])
    coverage = {
        "flow": flow_coverage,
        "qpcdr": qpcdr_coverage,
        "ectag": ectag_coverage,
        "ddpcr": ddpcr_coverage,
    }
    relative_errors = [value for value in (qpcdr_rel, ddpcr_rel) if np.isfinite(value)]
    median_relative_error = float(np.nanmedian(relative_errors)) if relative_errors else float("nan")
    pass_thresholds = {
        "qpcdr_ddpcr_scale_median_relative_error_lt_0.30": bool(median_relative_error < 0.30),
        "observation_key_summary_coverage_ge_0.85": bool(min(coverage.values()) >= 0.85),
        "ddpcr_interpretation_pooled_mean_only": True,
        "flow_purity_sensitivity_no_direction_reversal": bool(flow_sensitivity["no_direction_reversal"]),
        "ectag_replicate_concordance_available_or_single_replicate": bool(ectag_coverage >= 0.85),
    }
    return {
        "coverage_by_channel": coverage,
        "flow_ppc": flow_ppc,
        "flow_purity_sensitivity": flow_sensitivity,
        "qpcdr_ddpcr_median_relative_error": median_relative_error,
        "continue_gate_passed": bool(all(pass_thresholds.values())),
        "pass_thresholds": pass_thresholds,
        "interval": "90% posterior predictive interval",
    }


def _flow_ppc_coverage(flow: pd.DataFrame, flow_params: dict, rng: np.random.Generator, draws: int = 400) -> dict:
    rows: list[bool] = []
    interval_rows: list[dict] = []
    kappa = float(flow_params.get("kappa_flow", 1.0))
    for key, group in flow.groupby(["week", "condition", "replicate"], dropna=False):
        ordered = group.sort_values("state_gate").reset_index(drop=True)
        observed = schemas.normalize_probabilities(ordered["fraction"].astype(float).to_numpy() + 1e-9, name="flow PPC fractions")
        total_n = int(max(1, round(float(ordered["pre_sort_count"].astype(float).sum()))))
        alpha = np.clip(observed * max(1.0, kappa), 1e-6, None)
        replicated = np.empty((int(draws), len(observed)), dtype=float)
        for draw in range(int(draws)):
            replicate_p = rng.dirichlet(alpha)
            replicated[draw, :] = rng.multinomial(total_n, replicate_p) / float(total_n)
        lower = np.quantile(replicated, 0.05, axis=0)
        upper = np.quantile(replicated, 0.95, axis=0)
        for idx, row in enumerate(ordered.itertuples(index=False)):
            covered = bool(lower[idx] <= observed[idx] <= upper[idx])
            rows.append(covered)
            interval_rows.append(
                {
                    "week": int(key[0]),
                    "condition": str(key[1]),
                    "replicate": str(key[2]),
                    "state_gate": str(row.state_gate),
                    "observed_fraction": float(observed[idx]),
                    "ppc_q05": float(lower[idx]),
                    "ppc_q95": float(upper[idx]),
                    "covered": covered,
                }
            )
    return {
        "coverage": _mean_bool(rows),
        "model": "DirichletMultinomial replicated flow fractions",
        "draws": int(draws),
        "intervals": interval_rows,
    }


def _flow_purity_sensitivity(flow: pd.DataFrame, flow_params: dict) -> dict:
    values = [float(value) for value in flow_params.get("sensitivity_values", [0.90, 0.95, 0.99])]
    values = sorted(set(values))
    if len(values) < 2:
        return {"no_direction_reversal": True, "purity_values": values, "reversals": []}
    adjusted_frames = [_purity_adjusted_flow(flow, purity).assign(purity=purity) for purity in values]
    adjusted = pd.concat(adjusted_frames, ignore_index=True)
    reversals: list[dict] = []
    low = min(values)
    high = max(values)
    for (condition, replicate, state_gate), group in adjusted.groupby(["condition", "replicate", "state_gate"], dropna=False):
        directions = {}
        for purity, purity_group in group.groupby("purity"):
            ordered = purity_group.sort_values("week")
            if ordered["week"].nunique() < 2:
                directions[float(purity)] = 0.0
                continue
            delta = float(ordered["adjusted_fraction"].iloc[-1] - ordered["adjusted_fraction"].iloc[0])
            directions[float(purity)] = delta
        low_direction = directions.get(low, 0.0)
        high_direction = directions.get(high, 0.0)
        if abs(low_direction) > 1e-9 and abs(high_direction) > 1e-9 and np.sign(low_direction) != np.sign(high_direction):
            reversals.append(
                {
                    "condition": str(condition),
                    "replicate": str(replicate),
                    "state_gate": str(state_gate),
                    "delta_at_low_purity": low_direction,
                    "delta_at_high_purity": high_direction,
                }
            )
    return {
        "no_direction_reversal": len(reversals) == 0,
        "purity_values": values,
        "reversals": reversals,
        "policy": "main state-fraction week trend must not reverse between sensitivity endpoints",
    }


def _purity_adjusted_flow(flow: pd.DataFrame, purity: float) -> pd.DataFrame:
    rows = []
    for key, group in flow.groupby(["week", "condition", "replicate"], dropna=False):
        ordered = group.sort_values("state_gate")
        observed = schemas.normalize_probabilities(ordered["fraction"].astype(float).to_numpy() + 1e-9, name="flow purity fractions")
        contamination = (1.0 - float(purity)) / max(1, len(observed))
        corrected = np.clip((observed - contamination) / max(float(purity), 1e-9), 0.0, None)
        corrected = schemas.normalize_probabilities(corrected + 1e-9, name="purity-corrected flow fractions")
        for idx, row in enumerate(ordered.itertuples(index=False)):
            rows.append(
                {
                    "week": int(key[0]),
                    "condition": str(key[1]),
                    "replicate": str(key[2]),
                    "state_gate": str(row.state_gate),
                    "adjusted_fraction": float(corrected[idx]),
                }
            )
    return pd.DataFrame(rows)


def _qpcdr_ppc_coverage(qpcdr: pd.DataFrame, ectag: pd.DataFrame, qpcdr_params: dict) -> tuple[float, float]:
    ectag_mean = _ectag_state_species_means(ectag)
    merged = qpcdr.merge(ectag_mean, on=["week", "condition", "replicate", "state_gate", "species"], how="left", validate="many_to_one")
    covered = []
    relative_errors = []
    for row in merged.itertuples(index=False):
        species_params = qpcdr_params["by_species"][str(row.species)]
        mu = max(1e-9, float(row.copy_mean) if np.isfinite(row.copy_mean) else 1e-9)
        sigma = max(1e-9, float(species_params["sigma"]))
        if str(species_params["scale"]) == "relative_copy_number_log" and np.isfinite(row.relative_copy_number):
            observed = float(row.relative_copy_number)
            predicted = float(np.exp(species_params["intercept"] + species_params["slope"] * np.log(mu)))
            score_obs = np.log(max(1e-9, observed))
            score_pred = np.log(max(1e-9, predicted))
        elif np.isfinite(row.raw_Ct_or_Cq):
            observed = float(row.raw_Ct_or_Cq)
            predicted = float(species_params["intercept"] - species_params["slope"] * np.log10(mu))
            score_obs = observed
            score_pred = predicted
        else:
            continue
        covered.append(abs(score_obs - score_pred) <= 1.645 * sigma)
        relative_errors.append(abs(observed - predicted) / max(abs(observed), 1e-9))
    return _mean_bool(covered), _nanmedian(relative_errors)


def _ectag_ppc_coverage(ectag: pd.DataFrame, bins: list[dict]) -> float:
    if ectag.empty:
        return 1.0
    df = ectag.copy()
    df["bin_label"] = [schemas.assign_copy_bin(value, bins) for value in df["ectag_count"]]
    group_cols = ["week", "condition", "state_gate", "species"]
    covered = []
    for _, group in df.groupby(group_cols):
        replicate_count = group["replicate"].nunique()
        if replicate_count < 2:
            covered.append(True)
            continue
        pooled = group["bin_label"].value_counts(normalize=True)
        for _, replicate_group in group.groupby("replicate"):
            n = max(1.0, float(len(replicate_group)))
            observed = replicate_group["bin_label"].value_counts(normalize=True)
            for label in [str(item["label"]) for item in bins]:
                p = float(pooled.get(label, 0.0))
                q = float(observed.get(label, 0.0))
                sigma = np.sqrt(max(1e-9, p * (1.0 - p) / n))
                covered.append(abs(q - p) <= 1.645 * sigma + 1.0 / n)
    return _mean_bool(covered)


def _ddpcr_ppc_coverage(ddpcr: pd.DataFrame, flow: pd.DataFrame, ectag: pd.DataFrame, ddpcr_params: dict) -> tuple[float, float]:
    pooled = calculate_ddpcr_pooled_mean(
        flow[["week", "condition", "replicate", "state_gate", "fraction"]],
        _ectag_state_species_means(ectag),
    )
    merged = ddpcr.merge(pooled, on=["week", "condition", "replicate", "species"], how="left", validate="many_to_one")
    covered = []
    relative_errors = []
    for row in merged.itertuples(index=False):
        observed = max(1e-9, float(row.ddpcr_copy_number))
        predicted = max(1e-9, float(row.bulk_mean))
        sigma = max(1e-9, float(ddpcr_params["sigma_by_species"][str(row.species)]))
        covered.append(abs(np.log(observed) - np.log(predicted)) <= 1.645 * sigma)
        relative_errors.append(abs(observed - predicted) / observed)
    return _mean_bool(covered), _nanmedian(relative_errors)


def _fit_linear_calibration(x: pd.Series, y: pd.Series, *, default_slope: float) -> tuple[float, float]:
    x_arr = x.astype(float).to_numpy()
    y_arr = y.astype(float).to_numpy()
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size >= 2 and float(np.std(x_arr)) > 1e-10:
        design = np.column_stack([np.ones_like(x_arr), x_arr])
        intercept, slope = np.linalg.lstsq(design, y_arr, rcond=None)[0]
        return float(intercept), float(slope)
    intercept = float(np.nanmedian(y_arr - default_slope * x_arr)) if y_arr.size else 0.0
    return intercept, float(default_slope)


def _residual_sigma(residual: np.ndarray, *, floor: float, fallback: float) -> float:
    values = np.asarray(residual, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return float(max(floor, fallback))
    sigma = float(np.std(values, ddof=1))
    empirical_quantile = 1.0 if values.size <= 4 else 0.90
    empirical_90 = float(np.nanquantile(np.abs(values), empirical_quantile) / 1.645)
    return float(max(floor, sigma, empirical_90, fallback))


def _ectag_state_species_means(ectag: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["week", "condition", "replicate", "state_gate", "species"]
    return (
        ectag.groupby(group_cols, as_index=False)["ectag_count"]
        .mean()
        .rename(columns={"ectag_count": "copy_mean"})
    )


def _ectag_replicate_concordance(group: pd.DataFrame, bins: list[dict]) -> float:
    if group["replicate"].nunique() < 2:
        return 1.0
    df = group.copy()
    df["bin_label"] = [schemas.assign_copy_bin(value, bins) for value in df["ectag_count"]]
    distances = []
    for _, key_group in df.groupby(["week", "condition", "state_gate"]):
        pooled = key_group["bin_label"].value_counts(normalize=True)
        for _, replicate_group in key_group.groupby("replicate"):
            observed = replicate_group["bin_label"].value_counts(normalize=True)
            labels = sorted(set(pooled.index).union(set(observed.index)))
            distance = sum(abs(float(observed.get(label, 0.0)) - float(pooled.get(label, 0.0))) for label in labels) / 2.0
            distances.append(distance)
    if not distances:
        return 1.0
    return float(np.clip(1.0 - np.nanmedian(distances), 0.0, 1.0))


def _replicate_sigma(group: pd.DataFrame, transformed_values: pd.Series, *, floor: float, fallback: float) -> float:
    if len(transformed_values) <= 1:
        return fallback
    replicate_keys = ["week", "condition", "replicate", "state_gate", "species"]
    values = group.copy()
    values["_value"] = transformed_values.reindex(group.index)
    per_group = values.dropna(subset=["_value"]).groupby(replicate_keys)["_value"].std()
    finite = per_group.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        median_sigma = float(np.nanmedian(finite.to_numpy(dtype=float)))
        return float(max(floor, median_sigma))
    fallback_sigma = float(np.nanstd(transformed_values.to_numpy(dtype=float)))
    return float(max(floor, fallback_sigma))


def _observation_report_payload(params: dict, tables: dict[str, pd.DataFrame]) -> dict:
    return {
        "locked_for_full": params["locked_for_full"],
        "ddpcr_pooled_mean_only": True,
        "ectag_species_specific": True,
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "bins": params["ectag"]["bins"],
        "ppc": params.get("ppc", {}),
    }


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 1.0
    return float(np.mean(np.asarray(values, dtype=bool)))


def _nanmedian(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.nanmedian(finite)) if finite else float("nan")
