"""Bulk observation model fixed before full fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_markdown_report
from fit.raw import load_clean_tables


def calculate_ddpcr_pooled_mean(flow_fractions: pd.DataFrame, state_species_means: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper: pooled bulk mean from state fractions and state means."""

    required_flow = {"week", "condition", "replicate", "state_gate", "fraction"}
    required_mu = {"week", "condition", "replicate", "state_gate", "species", "copy_mean"}
    schemas.validate_required_columns(set(flow_fractions.columns), tuple(required_flow), "flow_fractions")
    schemas.validate_required_columns(set(state_species_means.columns), tuple(required_mu), "state_species_means")
    merged = state_species_means.merge(flow_fractions, on=["week", "condition", "replicate", "state_gate"], how="left", validate="many_to_one")
    if merged["fraction"].isna().any():
        raise ValueError("Cannot compute ddPCR pooled mean: missing flow fraction")
    merged["weighted_mean"] = merged["fraction"].astype(float) * merged["copy_mean"].astype(float)
    return merged.groupby(["week", "condition", "replicate", "species"], as_index=False)["weighted_mean"].sum().rename(columns={"weighted_mean": "bulk_mean"})


def fit_observation_model(clean_dir: str | Path, output_dir: str | Path, seed: int = 1) -> dict:
    """Write locked observation parameters for ddPCR, cell count, and flow3."""

    del seed
    tables = load_clean_tables(clean_dir)
    out = ensure_dir(output_dir)
    flow_target = _flow3_steady_target(tables["flow3"])
    projection = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    params: dict[str, Any] = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "locked_for_full": True,
        "fit_mask": dict(schemas.FIT_MASK),
        "ddpcr": {
            "likelihood": "lognormal_on_bulk_mean",
            "default_log_sd": 0.20,
            "log_sd_by_species": _ddpcr_log_sd(tables["ddpcr"]),
            "interpretation": "bulk pooled mean only; never a single-cell copy-number distribution",
        },
        "cell_count": {"likelihood": "student_t_log_count", "nu": 4, "default_log_sd": 0.25, "log_sd": 0.25},
        "flow3": {"likelihood": "steady_three_group_projection", "absolute_fraction_sd": 0.05, "target": flow_target},
        "disabled_modalities": {"qpcdr": 0, "ectag": 0, "flow4": 0, "state_specific_copy": 0},
    }
    validate_observation_params(params)
    write_json(out / "obs_params_for_bulk_lite.json", params)
    write_json(out / "obs_params_for_full.json", params)
    np.save(out / "flow3_projection_matrix.npy", projection)
    write_json(out / "flow3_steady_target.json", flow_target)
    write_markdown_report(
        out / "observation_qc_report.md",
        "Bulk Observation QC Report",
        [
            ("ddPCR", "Lognormal likelihood on bulk mean only. ddPCR is not interpreted as a single-cell distribution."),
            ("Cell Count", "Student-t log-count likelihood with nu=4 and default log sd 0.25."),
            ("Flow3", f"Projection matrix shape={projection.shape}; target={flow_target['fractions']}."),
            ("Closed Modalities", "qPCDR, ecTAG, flow4, and state-specific copy likelihood weights are fixed at zero."),
        ],
    )
    return params


def load_observation_params(path: str | Path) -> dict:
    params = read_json(path)
    validate_observation_params(params)
    return params


def validate_observation_params(params: dict) -> None:
    required = {"ddpcr", "cell_count", "flow3", "locked_for_full", "fit_mask", "disabled_modalities"}
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Observation params missing required fields: {missing}")
    if not bool(params["locked_for_full"]):
        raise ValueError("Observation calibration must be locked before full fitting")
    for key in ("use_qpcdr", "use_ectag", "use_flow4", "use_state_specific_copy", "use_lite_summary_in_final_score"):
        if bool(params["fit_mask"].get(key)):
            raise ValueError(f"{key} must be false for the bulk-only fit")
    if "single-cell" in str(params["ddpcr"].get("interpretation", "")).lower() and "never" not in str(params["ddpcr"].get("interpretation", "")).lower():
        raise ValueError("ddPCR interpretation must explicitly forbid single-cell use")


def _ddpcr_log_sd(ddpcr: pd.DataFrame) -> dict[str, float]:
    result = {}
    for species, group in ddpcr.groupby("species"):
        mean = group["ddpcr_copy_number"].astype(float).clip(lower=1e-9)
        sd = group["ddpcr_sd_or_ci"].astype(float)
        rel = sd / mean
        finite = rel.replace([np.inf, -np.inf], np.nan).dropna()
        result[str(species)] = float(max(0.05, np.nanmedian(finite) if len(finite) else 0.20))
    for species in schemas.SPECIES:
        result.setdefault(species, 0.20)
    return result


def _flow3_steady_target(flow3: pd.DataFrame) -> dict:
    grouped = flow3.groupby("group", as_index=False).agg(fraction=("fraction", "mean"), total_events=("total_events", "max"))
    values = grouped.set_index("group")["fraction"].reindex(schemas.FLOW3_GROUPS).fillna(0.0).to_numpy(dtype=float)
    fractions = schemas.normalize_probabilities(values + 1e-12, name="flow3 steady target")
    n_eff_values = flow3["total_events"].dropna().astype(float)
    n_eff = float(np.nanmedian(n_eff_values)) if len(n_eff_values) else 300.0
    if not np.isfinite(n_eff) or n_eff <= 0:
        n_eff = 300.0
    return {
        "groups": list(schemas.FLOW3_GROUPS),
        "fractions": dict(zip(schemas.FLOW3_GROUPS, fractions.tolist())),
        "n_eff": n_eff,
        "policy": "steady three-group projection; no four-state flow likelihood",
    }
