"""Shared schemas and deterministic helpers for the fit pipeline.

The constants in this module describe data contracts, not experimental facts.
Any biological assumption used by the fit stages must be traceable to
``markdown/fit_method.md`` or an explicit input artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config as cfg

STATE_NAMES: tuple[str, ...] = tuple(cfg.STATE_NAMES)
SPECIES: tuple[str, ...] = tuple(cfg.SPECIES)

RAW_TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "flow": (
        "week",
        "condition",
        "replicate",
        "state_gate",
        "pre_sort_count",
        "post_sort_count",
        "fraction",
        "sort_purity",
        "marker_panel",
        "batch_id",
    ),
    "qpcdr": (
        "week",
        "condition",
        "replicate",
        "state_gate",
        "species",
        "technical_rep",
        "raw_Ct_or_Cq",
        "relative_copy_number",
        "plate_id",
        "batch_id",
    ),
    "ectag": (
        "week",
        "condition",
        "replicate",
        "state_gate",
        "cell_id",
        "species",
        "ectag_count",
        "image_qc_pass",
        "batch_id",
    ),
    "ddpcr": (
        "week",
        "condition",
        "replicate",
        "species",
        "ddpcr_copy_number",
        "ddpcr_sd_or_ci",
        "batch_id",
    ),
    "cell_count": (
        "week",
        "condition",
        "replicate",
        "total_cell_count",
        "viability",
        "passage_info",
        "batch_id",
    ),
}

OBSERVATION_OUTPUTS: tuple[str, ...] = (
    "obs_params_for_lite.json",
    "obs_params_for_full.json",
    "obs_calibration_report.md",
)

LITE_OUTPUTS: tuple[str, ...] = (
    "LITE_final_fit.json",
    "LITE_snapshot_posterior.parquet",
    "LITE_summary_target_vector.parquet",
    "LITE_summary_covariance.npz",
    "LITE_distance_weights.json",
    "LITE_initial_population_sampler.json",
    "LITE_to_FULL_prior_scales.json",
    "LITE_final_report.md",
)

FULL_OUTPUTS: tuple[str, ...] = (
    "accepted_histories.jsonl",
    "particle_parameters.parquet",
    "particle_weights.parquet",
    "full_snapshot_summaries.parquet",
    "event_summaries.parquet",
    "scenario_classes.parquet",
    "full_ppc_report.md",
)


@dataclass(frozen=True)
class ResultLayout:
    """Default result paths used by CLI and workflow systems."""

    root: Path

    @property
    def clean_data(self) -> Path:
        return self.root / "01_clean_data"

    @property
    def observation(self) -> Path:
        return self.root / "02_observation_model"

    @property
    def empirical(self) -> Path:
        return self.root / "03_empirical_summary"

    @property
    def lite(self) -> Path:
        return self.root / "03_v4_lite"

    @property
    def full_init(self) -> Path:
        return self.root / "04_full_initialization"

    @property
    def full_smc(self) -> Path:
        return self.root / "05_full_smc"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stable_feature_id(channel: str, **parts: Any) -> str:
    ordered = "|".join(f"{key}={parts[key]}" for key in sorted(parts))
    return f"{channel}|{ordered}" if ordered else channel


def validate_required_columns(columns: set[str], required: tuple[str, ...], table_name: str) -> None:
    missing = [column for column in required if column not in columns]
    require(not missing, f"{table_name} is missing required columns: {', '.join(missing)}")


def validate_states(values: Any, table_name: str) -> None:
    invalid = sorted(set(str(value) for value in values if str(value) not in STATE_NAMES))
    require(not invalid, f"{table_name} contains invalid state_gate values: {invalid}")


def validate_species(values: Any, table_name: str) -> None:
    invalid = sorted(set(str(value) for value in values if str(value) not in SPECIES))
    require(not invalid, f"{table_name} contains invalid species values: {invalid}")


def validate_nonnegative(values: Any, field: str, table_name: str) -> None:
    arr = np.asarray(values, dtype=float)
    bad = arr[np.isfinite(arr) & (arr < 0)]
    require(bad.size == 0, f"{table_name}.{field} contains negative values")


def validate_weeks(values: Any, table_name: str) -> None:
    arr = np.asarray(values, dtype=float)
    require(bool(np.all(np.isfinite(arr))), f"{table_name}.week contains non-finite values")
    require(bool(np.all(arr >= 0)), f"{table_name}.week contains negative values")


def open_log2_copy_bins(max_observed: int) -> list[dict[str, int | str | None]]:
    """Return method-specified open-support log2-like bins.

    The final bin is a histogram tail bin, not a detection ceiling. It extends
    to infinity and is sampled with an explicit tail distribution in the full
    reconstruction stage.
    """

    max_value = max(0, int(max_observed))
    bins: list[dict[str, int | str | None]] = [
        {"label": "0", "low": 0, "high": 0},
        {"label": "1", "low": 1, "high": 1},
    ]
    low = 2
    while low <= max_value:
        high = 2 * low - 1
        bins.append({"label": f"{low}-{high}", "low": low, "high": high})
        low *= 2
    if max_value < 2:
        low = 2
    tail_low = low
    bins.append({"label": f"{tail_low}+", "low": tail_low, "high": None})
    return bins


def assign_copy_bin(value: int | float, bins: list[dict[str, int | str | None]]) -> str:
    copy_value = int(value)
    require(copy_value >= 0, "copy number values must be non-negative")
    for item in bins:
        low = _required_int(item["low"], "bin low")
        high = item["high"]
        if high is None:
            if copy_value >= low:
                return str(item["label"])
        elif low <= copy_value <= _required_int(high, "bin high"):
            return str(item["label"])
    raise ValueError(f"copy number {copy_value} did not match any open-support bin")


def bin_center(item: dict[str, int | str | None]) -> float:
    low = _required_int(item["low"], "bin low")
    high = item["high"]
    if high is None:
        return float(low * 1.5)
    return 0.5 * (low + _required_int(high, "bin high"))


def normalize_probabilities(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    require(arr.ndim == 1 and arr.size > 0, f"{name} must be a non-empty vector")
    require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    require(total > 0.0, f"{name} must have positive total mass")
    return arr / total


def softplus(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def _required_int(value: int | str | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} cannot be None")
    return int(value)
