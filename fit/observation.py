"""Observation calibration fixed before lite and full fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_markdown_report
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

    del seed
    tables = load_clean_tables(clean_dir)
    out = ensure_dir(output_dir)
    ectag_max = int(tables["ectag"]["ectag_count"].max()) if not tables["ectag"].empty else 0
    bins = schemas.open_log2_copy_bins(ectag_max)
    params = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "locked_for_full": True,
        "ddpcr_interpretation": "bulk pooled first moment only: sum_s f[w,c,r,s] * mu[w,c,r,s,j]",
        "ectag_interpretation": "species-specific open-support binned likelihood",
        "flow": _calibrate_flow(tables["flow"]),
        "qpcdr": _calibrate_qpcdr(tables["qpcdr"]),
        "ectag": _calibrate_ectag(tables["ectag"], bins),
        "ddpcr": _calibrate_ddpcr(tables["ddpcr"]),
        "cell_count": _calibrate_cell_count(tables["cell_count"]),
    }
    validate_observation_params(params)
    write_json(out / "obs_params_for_lite.json", params)
    write_json(out / "obs_params_for_full.json", params)
    write_json(out / "obs_calibration_report.json", _observation_report_payload(params, tables))
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


def _calibrate_qpcdr(qpcdr: pd.DataFrame) -> dict:
    by_species = {}
    for species, group in qpcdr.groupby("species"):
        if group["relative_copy_number"].notna().any():
            values = np.log(group["relative_copy_number"].dropna().astype(float).clip(lower=1e-9))
            scale = "relative_copy_number_log"
            intercept = 0.0
            slope = 1.0
        else:
            values = group["raw_Ct_or_Cq"].dropna().astype(float)
            scale = "ct_or_cq"
            intercept = float(np.nanmedian(values)) if len(values) else 0.0
            slope = -1.0
        sigma = _replicate_sigma(group, values, floor=0.05, fallback=0.25)
        by_species[str(species)] = {
            "scale": scale,
            "intercept": intercept,
            "slope": slope,
            "sigma": sigma,
        }
    return {"by_species": by_species, "epsilon": 1e-9}


def _calibrate_ectag(ectag: pd.DataFrame, bins: list[dict]) -> dict:
    concentration = {}
    for species, group in ectag.groupby("species"):
        grouped = group.groupby(["week", "condition", "replicate", "state_gate"])["ectag_count"]
        sizes = grouped.size().to_numpy(dtype=float)
        median_size = float(np.nanmedian(sizes)) if len(sizes) else 1.0
        concentration[str(species)] = float(max(1.0, median_size))
    return {
        "bins": bins,
        "concentration_by_species": concentration,
        "histogram_policy": "species_specific_open_support",
        "censoring": "disabled_unless_raw_metadata_declares_censoring",
    }


def _calibrate_ddpcr(ddpcr: pd.DataFrame) -> dict:
    sigma = {}
    for species, group in ddpcr.groupby("species"):
        sd = group["ddpcr_sd_or_ci"].dropna().astype(float)
        mean = group["ddpcr_copy_number"].astype(float).clip(lower=1e-9)
        if len(sd):
            rel = float(np.median(sd.to_numpy(dtype=float) / mean.reindex(sd.index).to_numpy(dtype=float)))
            sigma_value = float(max(0.05, rel))
        elif len(group) > 1:
            log_std = float(np.std(np.log(mean), ddof=1))
            sigma_value = float(max(0.05, log_std))
        else:
            sigma_value = 0.25
        sigma[str(species)] = sigma_value
    return {"sigma_by_species": sigma, "likelihood": "lognormal_on_bulk_pooled_mean"}


def _calibrate_cell_count(cell_count: pd.DataFrame) -> dict:
    values = cell_count["total_cell_count"].astype(float)
    median_count = float(np.nanmedian(values)) if len(values) else 20.0
    dispersion = float(max(1.0, median_count * 0.05))
    return {"dispersion": dispersion, "role": "growth_scale_if_reliable_else_plausibility_only"}


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
    }
