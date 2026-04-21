"""
Stage-aware parameter registry for the fitting shell.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping

import numpy as np

import config as cfg
from fit_data import CanonicalFitDataset


CATEGORY_FIXED = "fixed"
CATEGORY_STAGE1 = "free-in-stage-1"
CATEGORY_STAGE2 = "free-in-stage-2"
CATEGORY_DERIVED = "derived"

TRANSFORM_IDENTITY = "identity"
TRANSFORM_LOG = "log"
TRANSFORM_LOGIT = "logit"
TRANSFORM_ZERO_SUM = "zero_sum"
TRANSFORM_SIMPLEX = "simplex"
TRANSFORM_COLUMN_SIMPLEX = "column_simplex"


class ParameterBoundsError(ValueError):
    def __init__(self, spec_name: str, message: str):
        super().__init__(message)
        self.spec_name = spec_name


@dataclass(frozen=True)
class ClassificationRow:
    path: str
    category: str
    default_text: str
    source: str
    sweep_registered: str
    rationale: str


@dataclass
class ParameterBundle:
    model: cfg.ModelParameters
    observation: cfg.ObservationParameters

    def deep_copy(self) -> "ParameterBundle":
        return ParameterBundle(model=copy.deepcopy(self.model), observation=copy.deepcopy(self.observation))


@dataclass(frozen=True)
class FitParameterSpec:
    name: str
    report_paths: tuple[str, ...]
    category: str
    block: str
    transform: str
    raw_size: int
    unconstrained_size: int
    lower: np.ndarray
    upper: np.ndarray
    prior_center: np.ndarray
    prior_scale: np.ndarray
    shrinkage: bool
    getter: Callable[[ParameterBundle], np.ndarray]
    setter: Callable[[ParameterBundle, np.ndarray], None]
    rationale: str

    def raw_values(self, bundle: ParameterBundle) -> np.ndarray:
        values = np.asarray(self.getter(bundle), dtype=float).reshape(-1)
        cfg.require(values.size == self.raw_size, f"{self.name} getter returned {values.size} values, expected {self.raw_size}.")
        return values

    def validate_hard_bounds(self, values: np.ndarray, *, tolerance: float = 1e-10) -> None:
        flat = np.asarray(values, dtype=float).reshape(-1)
        cfg.require(flat.size == self.raw_size, f"{self.name} bound check received {flat.size} values, expected {self.raw_size}.")
        lower_violation = flat < (self.lower - tolerance)
        upper_violation = flat > (self.upper + tolerance)
        if not np.any(lower_violation) and not np.any(upper_violation):
            return
        problems: list[str] = []
        for index in np.where(lower_violation)[0].tolist():
            problems.append(f"index {index}: {flat[index]:.6g} < lower {self.lower[index]:.6g}")
        for index in np.where(upper_violation)[0].tolist():
            problems.append(f"index {index}: {flat[index]:.6g} > upper {self.upper[index]:.6g}")
        raise ParameterBoundsError(self.name, f"{self.name} violated hard biological bounds: {'; '.join(problems)}")

    def apply(self, bundle: ParameterBundle, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=float).reshape(-1)
        cfg.require(flat.size == self.raw_size, f"{self.name} setter received {flat.size} values, expected {self.raw_size}.")
        self.validate_hard_bounds(flat)
        projected = np.clip(flat, self.lower, self.upper)
        self.setter(bundle, projected)


def _parse_markdown_tables(markdown_path: str | Path) -> dict[str, list[ClassificationRow]]:
    text = Path(markdown_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_header: str | None = None
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_header is not None:
                sections[current_header] = current_lines
            current_header = line.strip()
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)
    if current_header is not None:
        sections[current_header] = current_lines

    header_map = {
        "## 2. Fixed 参数": CATEGORY_FIXED,
        "## 3. Free-in-stage-1 参数": CATEGORY_STAGE1,
        "## 4. Free-in-stage-2 参数": CATEGORY_STAGE2,
        "## 5. Derived / report-only quantities": CATEGORY_DERIVED,
    }
    parsed: dict[str, list[ClassificationRow]] = {category: [] for category in header_map.values()}
    for header, category in header_map.items():
        section_lines = sections.get(header)
        cfg.require(section_lines is not None, f"Missing section {header} in parameter classification markdown.")
        table_lines = [line for line in section_lines if line.startswith("|")]
        cfg.require(len(table_lines) >= 2, f"Section {header} does not contain a Markdown table.")
        header_cells = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
        expected = ["path", "default", "source", "sweep_registered", "rationale"]
        cfg.require(header_cells == expected, f"Unexpected table columns in {header}: {header_cells}.")
        for row_line in table_lines[2:]:
            cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
            if len(cells) != len(expected):
                continue
            parsed[category].append(
                ClassificationRow(
                    path=cells[0],
                    category=category,
                    default_text=cells[1],
                    source=cells[2],
                    sweep_registered=cells[3],
                    rationale=cells[4],
                )
            )
    return parsed


def _identity_bounds(center: np.ndarray, minimum_width: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    width = np.maximum(minimum_width, 2.0 * np.abs(center) + 0.5)
    return center - width, center + width


def _log_bounds(center: np.ndarray, lower_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(lower_floor, 0.25 * center)
    upper = np.maximum(center + 1.0, 4.0 * center)
    return lower, upper


def _logit_bounds(size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(size, 1e-6, dtype=float), np.full(size, 1.0 - 1e-6, dtype=float)


def _simplex_bounds(size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(size, 1e-6, dtype=float), np.full(size, 1.0, dtype=float)


def _nonnegative_identity_bounds(center: np.ndarray, upper_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.maximum(0.0, np.asarray(center, dtype=float).reshape(-1))
    upper = np.maximum(upper_floor, 4.0 * clipped + upper_floor)
    return np.zeros_like(clipped), upper


def _prior_scale(center: np.ndarray, transform: str, *, shrinkage: bool) -> np.ndarray:
    if transform == TRANSFORM_LOG:
        scale = np.maximum(0.10, 0.50 * center)
    elif transform == TRANSFORM_LOGIT:
        scale = np.full(center.size, 0.15, dtype=float)
    elif transform in (TRANSFORM_SIMPLEX, TRANSFORM_COLUMN_SIMPLEX):
        scale = np.full(center.size, 0.20, dtype=float)
    else:
        scale = np.maximum(0.35, 0.75 * np.maximum(np.abs(center), 0.25))
    if shrinkage:
        scale = 0.5 * scale
    return scale.astype(float)


def _normalize_simplex(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    cfg.require(np.all(flat >= 0.0), "Simplex values must be non-negative.")
    total = float(np.sum(flat))
    cfg.require(total > 0.0, "Simplex values must sum to a positive number.")
    return flat / total


def _center_zero_sum(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return flat - np.mean(flat)


def _normalize_column_simplex(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float).reshape(cfg.N_STATES, cfg.N_STATES)
    cfg.require(np.all(matrix >= 0.0), "sort_purity_matrix entries must be non-negative.")
    column_sums = np.sum(matrix, axis=0)
    cfg.require(np.all(column_sums > 0.0), "Every sort_purity_matrix column must sum to a positive number.")
    return matrix / column_sums


def _set_scalar(bundle: ParameterBundle, container_name: str, field_name: str, value: float) -> None:
    container = getattr(bundle.model, container_name)
    object.__setattr__(container, field_name, float(value))


def _set_observation_scalar(bundle: ParameterBundle, field_name: str, value: float) -> None:
    object.__setattr__(bundle.observation, field_name, float(value))


def _set_model_array(bundle: ParameterBundle, container_name: str, field_name: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=float)
    target = getattr(getattr(bundle.model, container_name), field_name)
    target[...] = array.reshape(target.shape)


def _set_turnover_scalar(bundle: ParameterBundle, species_name: str, field_name: str, value: float) -> None:
    species_params = bundle.model.turnover[species_name]
    object.__setattr__(species_params, field_name, float(value))


def _set_observation_array(bundle: ParameterBundle, field_name: str, values: np.ndarray) -> None:
    target = getattr(bundle.observation, field_name)
    target[...] = np.asarray(values, dtype=float).reshape(target.shape)


class ParameterRegistry:
    def __init__(
        self,
        classification_rows: Mapping[str, ClassificationRow],
        specs: Iterable[FitParameterSpec],
    ):
        self.classification_rows = dict(classification_rows)
        self.specs = tuple(specs)
        self.specs_by_name = {spec.name: spec for spec in self.specs}
        cfg.require(
            len(self.specs_by_name) == len(self.specs),
            "Duplicate parameter spec names detected in the registry.",
        )

    @classmethod
    def from_markdown(
        cls,
        markdown_path: str | Path,
        *,
        dataset: CanonicalFitDataset | None = None,
        base_bundle: ParameterBundle | None = None,
    ) -> "ParameterRegistry":
        parsed_tables = _parse_markdown_tables(markdown_path)
        classification_rows = {row.path: row for rows in parsed_tables.values() for row in rows}
        bundle = (
            ParameterBundle(model=copy.deepcopy(cfg.DEFAULT_MODEL_PARAMETERS), observation=copy.deepcopy(cfg.DEFAULT_OBSERVATION_PARAMETERS))
            if base_bundle is None
            else base_bundle.deep_copy()
        )
        specs = _build_supported_specs(classification_rows, bundle)
        registry = cls(classification_rows=classification_rows, specs=specs)
        if dataset is not None:
            registry.validate_dataset_compatibility(dataset)
        return registry

    def default_bundle(
        self,
        *,
        model: cfg.ModelParameters | None = None,
        observation: cfg.ObservationParameters | None = None,
    ) -> ParameterBundle:
        bundle = ParameterBundle(
            model=copy.deepcopy(cfg.DEFAULT_MODEL_PARAMETERS if model is None else model),
            observation=copy.deepcopy(cfg.DEFAULT_OBSERVATION_PARAMETERS if observation is None else observation),
        )
        self.project_bundle(bundle)
        return bundle

    def project_bundle(self, bundle: ParameterBundle) -> None:
        for spec in self.specs:
            spec.apply(bundle, spec.raw_values(bundle))
        cfg.validate_model_parameters(bundle.model)
        cfg.validate_observation_parameters(bundle.observation)

    def validate_dataset_compatibility(self, dataset: CanonicalFitDataset) -> None:
        cfg.require(
            len({record.batch for record in dataset.qpcdr}) <= 1,
            "The current fitting shell only supports a single qPCDR batch because ObservationParameters is batchless.",
        )

    def supported_specs(
        self,
        *,
        categories: Iterable[str] | None = None,
        blocks: Iterable[str] | None = None,
    ) -> tuple[FitParameterSpec, ...]:
        category_filter = None if categories is None else set(categories)
        block_filter = None if blocks is None else set(blocks)
        result: list[FitParameterSpec] = []
        for spec in self.specs:
            if category_filter is not None and spec.category not in category_filter:
                continue
            if block_filter is not None and spec.block not in block_filter:
                continue
            result.append(spec)
        return tuple(result)

    def unsupported_free_rows(self) -> tuple[ClassificationRow, ...]:
        covered_paths = {path for spec in self.specs for path in spec.report_paths}
        unsupported = [
            row
            for row in self.classification_rows.values()
            if row.category in {CATEGORY_STAGE1, CATEGORY_STAGE2} and row.path not in covered_paths
        ]
        return tuple(sorted(unsupported, key=lambda row: row.path))

    def prior_penalty(self, bundle: ParameterBundle, specs: Iterable[FitParameterSpec]) -> tuple[float, float]:
        prior_total = 0.0
        boundary_total = 0.0
        for spec in specs:
            raw = spec.raw_values(bundle)
            z = (raw - spec.prior_center) / spec.prior_scale
            multiplier = 2.0 if spec.shrinkage else 1.0
            prior_total += 0.5 * multiplier * float(np.dot(z, z)) / max(1, raw.size)
            span = spec.upper - spec.lower
            finite_mask = np.isfinite(span) & (span > 0.0)
            if np.any(finite_mask):
                lower_margin = raw[finite_mask] - spec.lower[finite_mask]
                upper_margin = spec.upper[finite_mask] - raw[finite_mask]
                normalized_margin = np.minimum(lower_margin, upper_margin) / span[finite_mask]
                clipped = np.clip(0.05 - normalized_margin, 0.0, None)
                boundary_total += float(np.dot(clipped / 0.05, clipped / 0.05))
        return prior_total, boundary_total

    def boundary_report(self, bundle: ParameterBundle, specs: Iterable[FitParameterSpec]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in specs:
            raw = spec.raw_values(bundle)
            span = spec.upper - spec.lower
            finite_mask = np.isfinite(span) & (span > 0.0)
            if np.any(finite_mask):
                lower_margin = raw[finite_mask] - spec.lower[finite_mask]
                upper_margin = spec.upper[finite_mask] - raw[finite_mask]
                normalized_margin = np.minimum(lower_margin, upper_margin) / span[finite_mask]
                min_margin = float(np.min(normalized_margin))
            else:
                min_margin = float("inf")
            rows.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "block": spec.block,
                    "min_normalized_margin": min_margin,
                    "touching_boundary": bool(np.isfinite(min_margin) and min_margin <= 0.02),
                }
            )
        return rows


def _build_supported_specs(
    classification_rows: Mapping[str, ClassificationRow],
    bundle: ParameterBundle,
) -> tuple[FitParameterSpec, ...]:
    specs: list[FitParameterSpec] = []

    def add_spec(
        *,
        name: str,
        report_paths: tuple[str, ...],
        category: str,
        block: str,
        transform: str,
        getter: Callable[[ParameterBundle], np.ndarray],
        setter: Callable[[ParameterBundle, np.ndarray], None],
        lower: np.ndarray,
        upper: np.ndarray,
        shrinkage: bool = False,
    ) -> None:
        rationale_parts = [classification_rows[path].rationale for path in report_paths if path in classification_rows]
        raw = np.asarray(getter(bundle), dtype=float).reshape(-1)
        spec = FitParameterSpec(
            name=name,
            report_paths=report_paths,
            category=category,
            block=block,
            transform=transform,
            raw_size=raw.size,
            unconstrained_size=_unconstrained_size(transform, raw.size),
            lower=np.asarray(lower, dtype=float).reshape(-1),
            upper=np.asarray(upper, dtype=float).reshape(-1),
            prior_center=raw.copy(),
            prior_scale=_prior_scale(raw, transform, shrinkage=shrinkage),
            shrinkage=shrinkage,
            getter=getter,
            setter=setter,
            rationale=" ".join(rationale_parts),
        )
        cfg.require(spec.lower.size == spec.raw_size, f"{name} lower bound size mismatch.")
        cfg.require(spec.upper.size == spec.raw_size, f"{name} upper bound size mismatch.")
        specs.append(spec)

    def state_paths(field_name: str) -> tuple[str, ...]:
        return tuple(f"landscape.{field_name}[{state_name}]" for state_name in cfg.STATE_NAMES)

    def division_species_paths(field_name: str) -> tuple[str, ...]:
        return tuple(f"division.{field_name}[{species_name}]" for species_name in cfg.SPECIES)

    def exposure_species_paths(field_name: str) -> tuple[str, ...]:
        return tuple(f"exposure.{field_name}[{species_name}]" for species_name in cfg.SPECIES)

    def turnover_paths(field_name: str) -> tuple[str, ...]:
        return tuple(f"turnover.{species_name}.{field_name}" for species_name in cfg.SPECIES)

    add_spec(
        name="obs.qpcdr_intercept",
        report_paths=("obs.qpcdr_intercept[j,batch]",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_IDENTITY,
        getter=lambda current: current.observation.qpcdr_intercept.astype(float).copy(),
        setter=lambda current, values: _set_observation_array(current, "qpcdr_intercept", values),
        lower=np.array([-10.0, -10.0, -10.0], dtype=float),
        upper=np.array([10.0, 10.0, 10.0], dtype=float),
    )
    add_spec(
        name="obs.qpcdr_slope",
        report_paths=("obs.qpcdr_slope[j,batch]",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_LOG,
        getter=lambda current: current.observation.qpcdr_slope.astype(float).copy(),
        setter=lambda current, values: _set_observation_array(current, "qpcdr_slope", values),
        lower=np.full(cfg.N_SPECIES, 1e-6, dtype=float),
        upper=np.full(cfg.N_SPECIES, 10.0, dtype=float),
    )
    add_spec(
        name="obs.qpcdr_sigma",
        report_paths=("obs.qpcdr_sigma[j,batch]",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_LOG,
        getter=lambda current: current.observation.qpcdr_sigma.astype(float).copy(),
        setter=lambda current, values: _set_observation_array(current, "qpcdr_sigma", values),
        lower=np.full(cfg.N_SPECIES, 1e-6, dtype=float),
        upper=np.full(cfg.N_SPECIES, 5.0, dtype=float),
    )
    add_spec(
        name="obs.ecTAG_emission",
        report_paths=("obs.ecTAG_emission[j]",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_LOG,
        getter=lambda current: current.observation.ecTAG_detection_efficiency.astype(float).copy(),
        setter=lambda current, values: _set_observation_array(current, "ecTAG_detection_efficiency", values),
        lower=np.full(cfg.N_SPECIES, 1e-6, dtype=float),
        upper=np.full(cfg.N_SPECIES, 10.0, dtype=float),
    )
    add_spec(
        name="obs.ecTAG_overdispersion",
        report_paths=("obs.ecTAG_overdispersion[j]",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_LOG,
        getter=lambda current: current.observation.ecTAG_overdispersion.astype(float).copy(),
        setter=lambda current, values: _set_observation_array(current, "ecTAG_overdispersion", values),
        lower=np.full(cfg.N_SPECIES, 1e-6, dtype=float),
        upper=np.full(cfg.N_SPECIES, 5.0, dtype=float),
    )
    add_spec(
        name="obs.flow_overdispersion",
        report_paths=("obs.flow_overdispersion",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_IDENTITY,
        getter=lambda current: np.array([current.observation.flow_overdispersion], dtype=float),
        setter=lambda current, values: _set_observation_scalar(current, "flow_overdispersion", float(values[0])),
        lower=np.array([0.0], dtype=float),
        upper=np.array([5.0], dtype=float),
    )
    add_spec(
        name="obs.count_overdispersion",
        report_paths=("obs.count_overdispersion",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_IDENTITY,
        getter=lambda current: np.array([current.observation.count_overdispersion], dtype=float),
        setter=lambda current, values: _set_observation_scalar(current, "count_overdispersion", float(values[0])),
        lower=np.array([0.0], dtype=float),
        upper=np.array([5.0], dtype=float),
    )
    add_spec(
        name="obs.sort_purity_matrix",
        report_paths=("obs.sort_purity_matrix",),
        category=CATEGORY_STAGE1,
        block="observation_core",
        transform=TRANSFORM_COLUMN_SIMPLEX,
        getter=lambda current: _normalize_column_simplex(current.observation.sort_purity_matrix).reshape(-1),
        setter=lambda current, values: _set_observation_array(
            current,
            "sort_purity_matrix",
            _normalize_column_simplex(values).reshape(cfg.N_STATES, cfg.N_STATES),
        ),
        lower=np.zeros(cfg.N_STATES * cfg.N_STATES, dtype=float),
        upper=np.ones(cfg.N_STATES * cfg.N_STATES, dtype=float),
    )

    for field_name, block in (("nu_C", "drug_core"), ("nu_P", "drug_core")):
        center = getattr(bundle.model.exposure, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"exposure.{field_name}",
            report_paths=(f"exposure.{field_name}",),
            category=CATEGORY_STAGE1,
            block=block,
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.exposure, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "exposure", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )

    state_block_map = {
        "alpha": "baseline_state_core",
        "gamma_C": "baseline_state_core",
        "gamma_P": "baseline_state_core",
        "eta_a": "cue_core",
        "eta_m": "cue_core",
        "gamma_M": "stage2_myc_plasticity",
        "xi_B": "stage2_myc_plasticity",
    }
    stage_category = {
        "alpha": CATEGORY_STAGE1,
        "gamma_C": CATEGORY_STAGE1,
        "gamma_P": CATEGORY_STAGE1,
        "eta_a": CATEGORY_STAGE1,
        "eta_m": CATEGORY_STAGE1,
        "gamma_M": CATEGORY_STAGE2,
        "xi_B": CATEGORY_STAGE2,
    }
    shrinkage_groups = {"gamma_M", "xi_B"}
    for field_name, block_name in state_block_map.items():
        raw = _center_zero_sum(getattr(bundle.model.landscape, field_name))
        lower, upper = _identity_bounds(raw)
        add_spec(
            name=f"landscape.{field_name}",
            report_paths=state_paths(field_name),
            category=stage_category[field_name],
            block=block_name,
            transform=TRANSFORM_ZERO_SUM,
            getter=lambda current, field_name=field_name: _center_zero_sum(getattr(current.model.landscape, field_name)),
            setter=lambda current, values, field_name=field_name: _set_model_array(current, "landscape", field_name, _center_zero_sum(values)),
            lower=lower,
            upper=upper,
            shrinkage=field_name in shrinkage_groups,
        )

    b_u_diag = np.diag(bundle.model.landscape.B_U).astype(float)
    lower, upper = _log_bounds(b_u_diag)
    add_spec(
        name="landscape.B_U_diag",
        report_paths=tuple(f"landscape.B_U[{index},{index}]" for index in range(cfg.LATENT_DIM)),
        category=CATEGORY_STAGE2,
        block="stage2_myc_plasticity",
        transform=TRANSFORM_LOG,
        getter=lambda current: np.diag(current.model.landscape.B_U).astype(float),
        setter=lambda current, values: _set_model_array(
            current,
            "landscape",
            "B_U",
            np.diag(np.asarray(values, dtype=float)),
        ),
        lower=lower,
        upper=upper,
    )
    for field_name in ("sigma_0", "sigma_M"):
        center = getattr(bundle.model.landscape, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"landscape.{field_name}",
            report_paths=(f"landscape.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_myc_plasticity",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.landscape, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "landscape", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )

    burden_raw = _normalize_simplex(bundle.model.exposure.burden_weights)
    lower, upper = _simplex_bounds(cfg.N_SPECIES)
    add_spec(
        name="exposure.burden_weights",
        report_paths=exposure_species_paths("burden_weights"),
        category=CATEGORY_STAGE2,
        block="stage2_weights",
        transform=TRANSFORM_SIMPLEX,
        getter=lambda current: _normalize_simplex(current.model.exposure.burden_weights),
        setter=lambda current, values: _set_model_array(current, "exposure", "burden_weights", _normalize_simplex(values)),
        lower=lower,
        upper=upper,
    )
    prolif_raw = _normalize_simplex(bundle.model.exposure.proliferative_weights)
    lower, upper = _simplex_bounds(2)
    add_spec(
        name="exposure.proliferative_weights",
        report_paths=("exposure.proliferative_weights[MYC]", "exposure.proliferative_weights[CDK4]"),
        category=CATEGORY_STAGE2,
        block="stage2_weights",
        transform=TRANSFORM_SIMPLEX,
        getter=lambda current: _normalize_simplex(current.model.exposure.proliferative_weights),
        setter=lambda current, values: _set_model_array(
            current,
            "exposure",
            "proliferative_weights",
            _normalize_simplex(values),
        ),
        lower=lower,
        upper=upper,
    )

    for field_name in (
        "alpha_R",
        "r_B",
        "r_S",
        "r_C",
        "r_P",
        "r_m",
        "alpha_V",
        "v_M",
        "v_A",
        "v_Q",
        "v_R",
        "v_C",
        "v_P",
        "v_a",
    ):
        center = getattr(bundle.model.stress_survival, field_name)
        lower, upper = _identity_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"stress_survival.{field_name}",
            report_paths=(f"stress_survival.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_stress_survival",
            transform=TRANSFORM_IDENTITY,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.stress_survival, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "stress_survival", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("b_R", "sigma_R", "b_V", "sigma_V"):
        center = getattr(bundle.model.stress_survival, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"stress_survival.{field_name}",
            report_paths=(f"stress_survival.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_stress_survival",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.stress_survival, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "stress_survival", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )

    for field_name in ("qbar_G1S", "qbar_G1Q", "qbar_QG1", "qbar_SG2M"):
        center = getattr(bundle.model.cycle, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"cycle.{field_name}",
            report_paths=(f"cycle.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_cycle",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.cycle, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "cycle", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in (
        "beta_0",
        "beta_P",
        "beta_NO",
        "beta_R",
        "beta_V",
        "beta_C",
        "beta_Pg",
        "gamma_0",
        "gamma_M",
        "gamma_R",
        "gamma_m",
        "gamma_V",
        "delta_0",
        "delta_P",
        "delta_V",
        "delta_NO",
        "delta_R",
        "delta_m",
        "kappa_0",
        "kappa_R",
        "kappa_V",
    ):
        center = getattr(bundle.model.cycle, field_name)
        lower, upper = _identity_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"cycle.{field_name}",
            report_paths=(f"cycle.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_cycle",
            transform=TRANSFORM_IDENTITY,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.cycle, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "cycle", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )

    turnover_stage1_fields = ("gain_ceiling", "loss_ceiling", "a0", "b0", "a_C", "a_P", "b_C", "b_P")
    turnover_stage2_fields = ("a_R", "a_prol", "b_R", "b_V")
    for species_name in cfg.SPECIES:
        for field_name in turnover_stage1_fields + turnover_stage2_fields:
            center = float(getattr(bundle.model.turnover[species_name], field_name))
            if field_name in {"gain_ceiling", "loss_ceiling"}:
                transform = TRANSFORM_LOG
                lower, upper = _log_bounds(np.array([center], dtype=float))
                block_name = "baseline_turnover_core"
                category = CATEGORY_STAGE1
            elif field_name in {"a_C", "a_P", "b_C", "b_P"}:
                transform = TRANSFORM_IDENTITY
                lower, upper = _identity_bounds(np.array([center], dtype=float))
                block_name = "drug_core"
                category = CATEGORY_STAGE1
            elif field_name in turnover_stage2_fields:
                transform = TRANSFORM_IDENTITY
                lower, upper = _identity_bounds(np.array([center], dtype=float))
                block_name = "stage2_detailed_turnover"
                category = CATEGORY_STAGE2
            else:
                transform = TRANSFORM_IDENTITY
                lower, upper = _identity_bounds(np.array([center], dtype=float))
                block_name = "baseline_turnover_core"
                category = CATEGORY_STAGE1
            add_spec(
                name=f"turnover.{species_name}.{field_name}",
                report_paths=(f"turnover.{species_name}.{field_name}",),
                category=category,
                block=block_name,
                transform=transform,
                getter=lambda current, species_name=species_name, field_name=field_name: np.array(
                    [getattr(current.model.turnover[species_name], field_name)],
                    dtype=float,
                ),
                setter=lambda current, values, species_name=species_name, field_name=field_name: _set_turnover_scalar(
                    current,
                    species_name,
                    field_name,
                    float(values[0]),
                ),
                lower=lower,
                upper=upper,
            )

    for field_name in ("lambda_div_ceiling", "lambda_death_ceiling"):
        center = getattr(bundle.model.hazard, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"hazard.{field_name}",
            report_paths=(f"hazard.{field_name}",),
            category=CATEGORY_STAGE1,
            block="baseline_hazard_core",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "hazard", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("theta_0", "phi_0"):
        center = getattr(bundle.model.hazard, field_name)
        lower, upper = _identity_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"hazard.{field_name}",
            report_paths=(f"hazard.{field_name}",),
            category=CATEGORY_STAGE1,
            block="baseline_hazard_core",
            transform=TRANSFORM_IDENTITY,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "hazard", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("chi_C", "chi_P"):
        center = getattr(bundle.model.hazard, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"hazard.{field_name}",
            report_paths=(f"hazard.{field_name}",),
            category=CATEGORY_STAGE1,
            block="drug_core",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "hazard", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("theta_P", "theta_NO", "theta_R", "theta_V", "phi_R", "phi_V", "phi_M", "phi_B"):
        center = getattr(bundle.model.hazard, field_name)
        lower, upper = _identity_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"hazard.{field_name}",
            report_paths=(f"hazard.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_detailed_hazard",
            transform=TRANSFORM_IDENTITY,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "hazard", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("B_star", "chi_B"):
        center = getattr(bundle.model.hazard, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"hazard.{field_name}",
            report_paths=(f"hazard.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_detailed_hazard",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "hazard", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    lower, upper = _logit_bounds(1)
    add_spec(
        name="hazard.omega_O_given_C",
        report_paths=("hazard.omega_O_given_C",),
        category=CATEGORY_STAGE2,
        block="stage2_detailed_hazard",
        transform=TRANSFORM_LOGIT,
        getter=lambda current: np.array([current.model.hazard.omega_O_given_C], dtype=float),
        setter=lambda current, values: _set_scalar(current, "hazard", "omega_O_given_C", float(values[0])),
        lower=lower,
        upper=upper,
    )

    lower, upper = _log_bounds(bundle.model.division.delta.astype(float))
    add_spec(
        name="division.delta",
        report_paths=division_species_paths("delta"),
        category=CATEGORY_STAGE1,
        block="division_core",
        transform=TRANSFORM_LOG,
        getter=lambda current: current.model.division.delta.astype(float).copy(),
        setter=lambda current, values: _set_model_array(current, "division", "delta", np.asarray(values, dtype=float)),
        lower=lower,
        upper=upper,
    )
    for field_name in ("lambda_amp_ceiling", "c0", "cR", "cC", "cP"):
        raw = getattr(bundle.model.division, field_name).astype(float)
        transform = TRANSFORM_IDENTITY
        if field_name == "lambda_amp_ceiling":
            lower, upper = _nonnegative_identity_bounds(raw)
        else:
            lower, upper = _identity_bounds(raw)
        add_spec(
            name=f"division.{field_name}",
            report_paths=division_species_paths(field_name),
            category=CATEGORY_STAGE2,
            block="stage2_division_daughter",
            transform=transform,
            getter=lambda current, field_name=field_name: getattr(current.model.division, field_name).astype(float).copy(),
            setter=lambda current, values, field_name=field_name: _set_model_array(
                current,
                "division",
                field_name,
                np.asarray(values, dtype=float),
            ),
            lower=lower,
            upper=upper,
            shrinkage=(field_name == "lambda_amp_ceiling"),
        )
    for field_name in ("rho_U", "rho_R", "rho_V"):
        lower, upper = _logit_bounds(1)
        add_spec(
            name=f"division.{field_name}",
            report_paths=(f"division.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_division_daughter",
            transform=TRANSFORM_LOGIT,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.division, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "division", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    omega_raw = np.diag(bundle.model.division.Omega_U).astype(float)
    lower, upper = _log_bounds(np.maximum(omega_raw, 1e-6))
    add_spec(
        name="division.Omega_U_diag",
        report_paths=tuple(f"division.Omega_U[{index},{index}]" for index in range(cfg.LATENT_DIM)),
        category=CATEGORY_STAGE2,
        block="stage2_division_daughter",
        transform=TRANSFORM_LOG,
        getter=lambda current: np.diag(current.model.division.Omega_U).astype(float),
        setter=lambda current, values: _set_model_array(
            current,
            "division",
            "Omega_U",
            np.diag(np.asarray(values, dtype=float)),
        ),
        lower=lower,
        upper=upper,
    )
    for field_name in ("sigma_R0", "sigma_V0"):
        center = getattr(bundle.model.division, field_name)
        lower, upper = _log_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"division.{field_name}",
            report_paths=(f"division.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_division_daughter",
            transform=TRANSFORM_LOG,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.division, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "division", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )
    for field_name in ("zeta_0", "zeta_R", "zeta_M", "zeta_a", "zeta_m"):
        center = getattr(bundle.model.division, field_name)
        lower, upper = _identity_bounds(np.array([center], dtype=float))
        add_spec(
            name=f"division.{field_name}",
            report_paths=(f"division.{field_name}",),
            category=CATEGORY_STAGE2,
            block="stage2_division_daughter",
            transform=TRANSFORM_IDENTITY,
            getter=lambda current, field_name=field_name: np.array([getattr(current.model.division, field_name)], dtype=float),
            setter=lambda current, values, field_name=field_name: _set_scalar(current, "division", field_name, float(values[0])),
            lower=lower,
            upper=upper,
        )

    return tuple(specs)


def _unconstrained_size(transform: str, raw_size: int) -> int:
    if transform == TRANSFORM_ZERO_SUM:
        cfg.require(raw_size == cfg.N_STATES, f"zero_sum transform currently expects {cfg.N_STATES} raw values.")
        return cfg.LATENT_DIM
    if transform == TRANSFORM_SIMPLEX:
        cfg.require(raw_size >= 2, "simplex transform requires at least 2 raw values.")
        return raw_size - 1
    if transform == TRANSFORM_COLUMN_SIMPLEX:
        return cfg.N_STATES * (cfg.N_STATES - 1)
    return raw_size
