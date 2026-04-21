"""
External untreated sweep engine for the ecDNA v4 model.

This module does not modify the on-disk model source files. It applies
parameter overrides and initialization overrides only inside a runtime
context while calling the existing simulator.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, field, replace
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import config as cfg
from simulation import SimulationResult, run_simulation
from treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts


UNTREATED_SCHEDULES = {"u_C": lambda _t: 0.0, "u_P": lambda _t: 0.0, "a": lambda _t: 0.0, "m": lambda _t: 0.0}
# Hard-coded candidate/parameter-parallel worker count. Set to 1 to disable parallelism.
PARAMETER_EVAL_WORKERS = 6
# Detailed parallel diagnostics. Set to False to silence worker-level logs.
PARALLEL_DEBUG_PRINT = True


def _progress_print(message: str) -> None:
    print(f"[sweep] {message}", flush=True)


def _debug_print(message: str) -> None:
    if PARALLEL_DEBUG_PRINT:
        _progress_print(message)


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    group: str
    container_steps: tuple[str, ...]
    field_name: str
    index: tuple[int, ...] | None
    default_value: float
    lower_bound: float
    upper_bound: float
    mode: str
    family: str | None = None

    def to_manifest_row(self) -> dict[str, object]:
        return {
            "path": self.path,
            "group": self.group,
            "default_value": self.default_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "mode": self.mode,
            "family": self.family,
        }


@dataclass(frozen=True)
class InitializationOverride:
    copy_number_mean: np.ndarray
    state_dirichlet_alpha: np.ndarray
    cycle_probabilities: np.ndarray
    age_scale: float


@dataclass(frozen=True)
class ScreeningExecutionPlan:
    protocol_name: str = "untreated"
    t_max: float = 10.0
    record_interval: float = 1.0
    n_init: int = 80
    target_population_size: int | None = 500
    max_pop_size: int = 2000
    baseline_seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106)
    oat_points_per_parameter: int = 5
    ranking_top_parameters_per_category: int = 4
    two_param_grid_size: int = 5
    top_pairs_per_group: int = 2
    representative_top_parameters: int = 4

    @classmethod
    def compact_test_plan(cls) -> "ScreeningExecutionPlan":
        return cls(
            baseline_seeds=(11,),
            t_max=2.0,
            record_interval=1.0,
            n_init=20,
            target_population_size=60,
            max_pop_size=120,
            oat_points_per_parameter=3,
            ranking_top_parameters_per_category=1,
            two_param_grid_size=3,
            top_pairs_per_group=1,
            representative_top_parameters=1,
        )


@dataclass(frozen=True)
class ScreeningEvaluationConfig:
    protocol_name: str
    t_max: float
    record_interval: float
    n_init: int
    target_population_size: int | None
    max_pop_size: int
    seeds: tuple[int, ...]


@dataclass
class ScreeningSeedMetric:
    candidate_id: str
    phase: str
    seed: int
    stop_reason: str
    stop_time: float
    terminal_population: int
    growth_rate: float
    terminal_stress: float
    terminal_survival: float
    division_count: int
    death_count: int
    division_death_ratio: float
    auc_population: float
    auc_stress: float
    auc_survival: float
    terminal_state_NPC: float
    terminal_state_OPC: float
    terminal_state_AC: float
    terminal_state_MES: float
    terminal_bulk_MYC: float
    terminal_bulk_CDK4: float
    terminal_bulk_PDGFRA: float
    bulk_trend_MYC: float
    bulk_trend_CDK4: float
    bulk_trend_PDGFRA: float


@dataclass(frozen=True)
class TrajectorySeries:
    times: np.ndarray
    population_sizes: np.ndarray
    mean_stress: np.ndarray
    mean_survival: np.ndarray
    npc_fractions: np.ndarray


SCREENING_METRIC_KEYS = (
    "growth_rate",
    "terminal_population",
    "terminal_stress",
    "terminal_survival",
    "division_count",
    "death_count",
    "division_death_ratio",
    "auc_population",
    "auc_stress",
    "auc_survival",
    "terminal_state_NPC",
    "terminal_state_OPC",
    "terminal_state_AC",
    "terminal_state_MES",
    "terminal_bulk_MYC",
    "terminal_bulk_CDK4",
    "terminal_bulk_PDGFRA",
    "bulk_trend_MYC",
    "bulk_trend_CDK4",
    "bulk_trend_PDGFRA",
)


SCREENING_METRIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "growth": ("growth_rate", "terminal_population", "auc_population"),
    "state": ("terminal_state_NPC", "terminal_state_OPC", "terminal_state_AC", "terminal_state_MES"),
    "ecdna": ("terminal_bulk_MYC", "terminal_bulk_CDK4", "terminal_bulk_PDGFRA", "bulk_trend_MYC", "bulk_trend_CDK4", "bulk_trend_PDGFRA"),
    "stress_survival": ("terminal_stress", "terminal_survival", "auc_stress", "auc_survival"),
    "events": ("division_count", "death_count", "division_death_ratio"),
}


def _git_commit_or_unknown(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        return value if value else "unknown"
    except Exception:
        return "unknown"


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count == 1:
            continue
        members = np.where(inverse == idx)[0]
        ranks[members] = np.mean(ranks[members])
    return ranks


def _spearman_absolute(values: np.ndarray, scores: np.ndarray) -> float:
    if len(values) < 3 or np.allclose(values, values[0]):
        return 0.0
    ranked_x = _rankdata(values)
    ranked_y = _rankdata(scores)
    corr = np.corrcoef(ranked_x, ranked_y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _cycle_family_paths() -> tuple[str, ...]:
    return tuple(f"init_cycle_probs[{name}]" for name in cfg.CYCLE_NAMES)


def _resolve_container(root: object, steps: Sequence[str]) -> object:
    current = root
    for step in steps:
        if isinstance(current, dict):
            current = current[step]
        else:
            current = getattr(current, step)
    return current


def _set_spec_value(root: object, spec: ParameterSpec, value: float) -> None:
    container = _resolve_container(root, spec.container_steps)
    if spec.index is None:
        object.__setattr__(container, spec.field_name, float(value))
        return
    array = getattr(container, spec.field_name)
    array[spec.index] = float(value)


def _make_spec(
    *,
    path: str,
    group: str,
    container_steps: Sequence[str],
    field_name: str,
    default_value: float,
    mode: str,
    index: tuple[int, ...] | None = None,
    family: str | None = None,
) -> ParameterSpec:
    cfg.require(np.isfinite(default_value), f"Default value for {path} must be finite.")
    if mode == "log":
        cfg.require(default_value > 0.0, f"Log-scaled parameter {path} must have a positive default value.")
        lower_bound = 0.5 * default_value
        upper_bound = 2.0 * default_value
    elif mode == "signed_ratio":
        if default_value == 0.0:
            lower_bound = -0.25
            upper_bound = 0.25
            mode = "additive"
        else:
            values = np.array([0.5 * default_value, 1.5 * default_value], dtype=float)
            lower_bound = float(np.min(values))
            upper_bound = float(np.max(values))
    elif mode == "additive":
        lower_bound = float(default_value - 0.25)
        upper_bound = float(default_value + 0.25)
    elif mode == "probability":
        lower_bound = float(max(0.05, default_value - 0.10))
        upper_bound = float(min(0.95, default_value + 0.10))
    else:
        raise ValueError(f"Unsupported parameter mode: {mode}")
    return ParameterSpec(
        path=path,
        group=group,
        container_steps=tuple(container_steps),
        field_name=field_name,
        index=index,
        default_value=float(default_value),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mode=mode,
        family=family,
    )


def build_parameter_specs() -> dict[str, ParameterSpec]:
    specs: dict[str, ParameterSpec] = {}
    base_params = cfg.DEFAULT_MODEL_PARAMETERS

    def register(spec: ParameterSpec) -> None:
        cfg.require(spec.path not in specs, f"Duplicate parameter path in whitelist: {spec.path}")
        specs[spec.path] = spec

    register(
        _make_spec(
            path="simulation.dt",
            group="validation_only",
            container_steps=("simulation",),
            field_name="dt",
            default_value=float(base_params.simulation.dt),
            mode="log",
        )
    )

    for index, name in enumerate(cfg.SPECIES):
        register(_make_spec(path=f"init_copy_mean[{name}]", group="init", container_steps=("initialization",), field_name="copy_number_mean", index=(index,), default_value=float([5.5, 6.5, 6.0][index]), mode="log"))
    for index, name in enumerate(cfg.STATE_NAMES):
        register(_make_spec(path=f"init_state_alpha[{name}]", group="init", container_steps=("initialization",), field_name="state_dirichlet_alpha", index=(index,), default_value=float([3.0, 2.8, 1.6, 1.4][index]), mode="log"))
    cycle_defaults = [0.15, 0.55, 0.20, 0.10]
    for index, name in enumerate(cfg.CYCLE_NAMES):
        register(_make_spec(path=f"init_cycle_probs[{name}]", group="init", container_steps=("initialization",), field_name="cycle_probabilities", index=(index,), default_value=float(cycle_defaults[index]), mode="probability", family="init_cycle_probs"))
    register(_make_spec(path="init_age_scale", group="init", container_steps=("initialization",), field_name="age_scale", default_value=2.0, mode="log"))

    for index, name in enumerate(cfg.SPECIES):
        register(_make_spec(path=f"exposure.burden_weights[{name}]", group="landscape_exposure", container_steps=("exposure",), field_name="burden_weights", index=(index,), default_value=float(base_params.exposure.burden_weights[index]), mode="log"))
    for index, label in enumerate(("MYC", "CDK4")):
        register(_make_spec(path=f"exposure.proliferative_weights[{label}]", group="landscape_exposure", container_steps=("exposure",), field_name="proliferative_weights", index=(index,), default_value=float(base_params.exposure.proliferative_weights[index]), mode="log"))

    for field_name, mode in (("alpha", "additive"), ("gamma_M", "signed_ratio"), ("gamma_C", "signed_ratio"), ("gamma_P", "signed_ratio"), ("xi_B", "signed_ratio")):
        values = getattr(base_params.landscape, field_name)
        for index, state_name in enumerate(cfg.STATE_NAMES):
            register(_make_spec(path=f"landscape.{field_name}[{state_name}]", group="landscape_exposure", container_steps=("landscape",), field_name=field_name, index=(index,), default_value=float(values[index]), mode=mode))
    for index in range(cfg.LATENT_DIM):
        register(_make_spec(path=f"landscape.B_U[{index},{index}]", group="landscape_exposure", container_steps=("landscape",), field_name="B_U", index=(index, index), default_value=float(base_params.landscape.B_U[index, index]), mode="log"))
    register(_make_spec(path="landscape.sigma_0", group="landscape_exposure", container_steps=("landscape",), field_name="sigma_0", default_value=float(base_params.landscape.sigma_0), mode="log"))
    register(_make_spec(path="landscape.sigma_M", group="landscape_exposure", container_steps=("landscape",), field_name="sigma_M", default_value=float(base_params.landscape.sigma_M), mode="log"))

    for field_name, mode in (("alpha_R", "additive"), ("r_B", "signed_ratio"), ("r_S", "signed_ratio"), ("r_m", "signed_ratio"), ("b_R", "log"), ("sigma_R", "log"), ("alpha_V", "additive"), ("v_M", "signed_ratio"), ("v_A", "signed_ratio"), ("v_Q", "signed_ratio"), ("v_R", "signed_ratio"), ("b_V", "log"), ("sigma_V", "log")):
        register(_make_spec(path=f"stress_survival.{field_name}", group="stress_survival", container_steps=("stress_survival",), field_name=field_name, default_value=float(getattr(base_params.stress_survival, field_name)), mode=mode))

    for field_name in ("qbar_G1S", "qbar_G1Q", "qbar_QG1", "qbar_SG2M"):
        register(_make_spec(path=f"cycle.{field_name}", group="cycle_hazard", container_steps=("cycle",), field_name=field_name, default_value=float(getattr(base_params.cycle, field_name)), mode="log"))
    for field_name in ("beta_0", "gamma_0", "delta_0", "kappa_0"):
        register(_make_spec(path=f"cycle.{field_name}", group="cycle_hazard", container_steps=("cycle",), field_name=field_name, default_value=float(getattr(base_params.cycle, field_name)), mode="additive"))
    for field_name in ("beta_P", "beta_NO", "beta_R", "beta_V", "gamma_M", "gamma_R", "gamma_V", "delta_P", "delta_V", "delta_NO", "delta_R", "kappa_R", "kappa_V"):
        register(_make_spec(path=f"cycle.{field_name}", group="cycle_hazard", container_steps=("cycle",), field_name=field_name, default_value=float(getattr(base_params.cycle, field_name)), mode="signed_ratio"))

    for field_name, mode in (("lambda_div_ceiling", "log"), ("lambda_death_ceiling", "log"), ("theta_0", "additive"), ("theta_P", "signed_ratio"), ("theta_NO", "signed_ratio"), ("theta_R", "signed_ratio"), ("theta_V", "signed_ratio"), ("B_star", "log"), ("chi_B", "log"), ("phi_0", "additive"), ("phi_R", "signed_ratio"), ("phi_V", "signed_ratio"), ("phi_M", "signed_ratio"), ("phi_B", "signed_ratio")):
        register(_make_spec(path=f"hazard.{field_name}", group="cycle_hazard", container_steps=("hazard",), field_name=field_name, default_value=float(getattr(base_params.hazard, field_name)), mode=mode))

    for field_name, mode in (("eta_1", "log"), ("eta_2", "log"), ("r_L", "log"), ("r_U", "log")):
        register(_make_spec(path=f"turnover_window.{field_name}", group="turnover_division", container_steps=("turnover_window",), field_name=field_name, default_value=float(getattr(base_params.turnover_window, field_name)), mode=mode))

    for species_name in cfg.SPECIES:
        species_params = base_params.turnover[species_name]
        for field_name, mode in (("gain_ceiling", "log"), ("loss_ceiling", "log"), ("a0", "additive"), ("a_R", "signed_ratio"), ("a_prol", "signed_ratio"), ("b0", "additive"), ("b_R", "signed_ratio"), ("b_V", "signed_ratio")):
            register(_make_spec(path=f"turnover.{species_name}.{field_name}", group="turnover_division", container_steps=("turnover", species_name), field_name=field_name, default_value=float(getattr(species_params, field_name)), mode=mode))

    register(_make_spec(path="division.tau", group="turnover_division", container_steps=("division",), field_name="tau", default_value=float(base_params.division.tau), mode="log"))
    for index, species_name in enumerate(cfg.SPECIES):
        register(_make_spec(path=f"division.delta[{species_name}]", group="turnover_division", container_steps=("division",), field_name="delta", index=(index,), default_value=float(base_params.division.delta[index]), mode="signed_ratio"))
    for field_name in ("rho_U", "rho_R", "rho_V"):
        register(_make_spec(path=f"division.{field_name}", group="turnover_division", container_steps=("division",), field_name=field_name, default_value=float(getattr(base_params.division, field_name)), mode="probability"))
    for index in range(cfg.LATENT_DIM):
        register(_make_spec(path=f"division.Omega_U[{index},{index}]", group="turnover_division", container_steps=("division",), field_name="Omega_U", index=(index, index), default_value=float(base_params.division.Omega_U[index, index]), mode="log"))
    for field_name, mode in (("sigma_R0", "log"), ("sigma_V0", "log"), ("zeta_0", "additive"), ("zeta_R", "signed_ratio"), ("zeta_M", "signed_ratio")):
        register(_make_spec(path=f"division.{field_name}", group="turnover_division", container_steps=("division",), field_name=field_name, default_value=float(getattr(base_params.division, field_name)), mode=mode))
    return specs


def default_parameter_values(specs: dict[str, ParameterSpec], groups: Iterable[str] | None = None) -> dict[str, float]:
    active_groups = None if groups is None else set(groups)
    result: dict[str, float] = {}
    for path, spec in specs.items():
        if spec.group == "validation_only":
            continue
        if active_groups is not None and spec.group not in active_groups:
            continue
        result[path] = spec.default_value
    return result


def _normalize_cycle_probabilities(values: dict[str, float]) -> dict[str, float]:
    family_paths = _cycle_family_paths()
    family_values = np.array([values[path] for path in family_paths], dtype=float)
    cfg.require(np.all(np.isfinite(family_values)), "Initial cycle probabilities must be finite.")
    cfg.require(np.all(family_values > 0.0), "Initial cycle probabilities must be strictly positive.")
    normalized = family_values / np.sum(family_values)
    return {path: float(normalized[idx]) for idx, path in enumerate(family_paths)}


def apply_parameter_overrides(
    specs: dict[str, ParameterSpec],
    overrides: dict[str, float],
    *,
    base_params: cfg.ModelParameters | None = None,
) -> tuple[cfg.ModelParameters, InitializationOverride]:
    params_copy = copy.deepcopy(cfg.DEFAULT_MODEL_PARAMETERS if base_params is None else base_params)
    init_defaults_source = cfg.DEFAULT_INITIALIZATION_PARAMETERS
    init_defaults = {
        "init_copy_mean[MYC]": float(init_defaults_source.parametric_copy_number_mean[cfg.MYC]),
        "init_copy_mean[CDK4]": float(init_defaults_source.parametric_copy_number_mean[cfg.CDK4]),
        "init_copy_mean[PDGFRA]": float(init_defaults_source.parametric_copy_number_mean[cfg.PDGFRA]),
        "init_state_alpha[NPC-like]": float(init_defaults_source.parametric_state_dirichlet_alpha[cfg.NPC]),
        "init_state_alpha[OPC-like]": float(init_defaults_source.parametric_state_dirichlet_alpha[cfg.OPC]),
        "init_state_alpha[AC-like]": float(init_defaults_source.parametric_state_dirichlet_alpha[cfg.AC]),
        "init_state_alpha[MES-like]": float(init_defaults_source.parametric_state_dirichlet_alpha[cfg.MES]),
        "init_cycle_probs[Q]": float(init_defaults_source.cycle_probabilities[cfg.Q]),
        "init_cycle_probs[G1]": float(init_defaults_source.cycle_probabilities[cfg.G1]),
        "init_cycle_probs[S]": float(init_defaults_source.cycle_probabilities[cfg.S]),
        "init_cycle_probs[G2M]": float(init_defaults_source.cycle_probabilities[cfg.G2M]),
        "init_age_scale": float(init_defaults_source.age_scale),
    }
    init_values = dict(init_defaults)
    for path, value in overrides.items():
        cfg.require(path in specs, f"Parameter path is not in the whitelist: {path}")
        spec = specs[path]
        value = float(value)
        cfg.require(np.isfinite(value), f"Override for {path} must be finite.")
        if spec.family != "init_cycle_probs":
            cfg.require(spec.lower_bound <= value <= spec.upper_bound, f"Override for {path}={value} is outside [{spec.lower_bound}, {spec.upper_bound}].")
        if path.startswith("init_"):
            init_values[path] = value
            continue
        _set_spec_value(params_copy, spec, value)
    init_values.update(_normalize_cycle_probabilities({path: init_values[path] for path in _cycle_family_paths()}))
    initialization = InitializationOverride(
        copy_number_mean=np.array([init_values[f"init_copy_mean[{name}]"] for name in cfg.SPECIES], dtype=float),
        state_dirichlet_alpha=np.array([init_values[f"init_state_alpha[{name}]"] for name in cfg.STATE_NAMES], dtype=float),
        cycle_probabilities=np.array([init_values[f"init_cycle_probs[{name}]"] for name in cfg.CYCLE_NAMES], dtype=float),
        age_scale=float(init_values["init_age_scale"]),
    )
    cfg.require(np.all(initialization.copy_number_mean > 0.0), "Initial copy-number means must be strictly positive.")
    cfg.require(np.all(initialization.state_dirichlet_alpha > 0.0), "Initial state alpha values must be strictly positive.")
    cfg.require(initialization.age_scale > 0.0, "Initial age scale must be strictly positive.")
    return params_copy, initialization


def _to_initialization_parameters(initialization: InitializationOverride) -> cfg.InitializationParameters:
    params = cfg.InitializationParameters(
        mode=cfg.PARAMETRIC,
        parametric_copy_number_mean=initialization.copy_number_mean.copy(),
        parametric_state_dirichlet_alpha=initialization.state_dirichlet_alpha.copy(),
        cycle_probabilities=initialization.cycle_probabilities.copy(),
        age_scale=float(initialization.age_scale),
    )
    cfg.validate_initialization_parameters(params)
    return params


def _event_counts(result: SimulationResult) -> tuple[int, int]:
    division_count = 0
    death_count = 0
    for _, event_type, _, _ in result.events:
        if event_type == "division":
            division_count += 1
        elif event_type == "death":
            death_count += 1
    return division_count, death_count


def _safe_path_token(value: str) -> str:
    token = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
    cfg.require(bool(token), "Path token must not be empty.")
    return token


def _write_trajectory_summary_csv(path: Path, result: SimulationResult) -> None:
    cfg.require(bool(result.times), "Trajectory summary requires recorded simulation times.")
    cfg.require(
        len(result.times)
        == len(result.population_sizes)
        == len(result.mean_stress_scores)
        == len(result.mean_survival_scores)
        == len(result.soft_state_fractions),
        "Trajectory summary arrays must have identical lengths.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("time", "population_size", "mean_stress", "mean_survival", "npc_fraction"),
        )
        writer.writeheader()
        for index, time_value in enumerate(result.times):
            state_fraction = np.asarray(result.soft_state_fractions[index], dtype=float)
            cfg.require(state_fraction.shape == (cfg.N_STATES,), "Trajectory state fraction row has invalid shape.")
            writer.writerow(
                {
                    "time": float(time_value),
                    "population_size": int(result.population_sizes[index]),
                    "mean_stress": float(result.mean_stress_scores[index]),
                    "mean_survival": float(result.mean_survival_scores[index]),
                    "npc_fraction": float(state_fraction[cfg.NPC]),
                }
            )


def _read_trajectory_summary_csv(path: Path) -> TrajectorySeries:
    cfg.require(path.exists(), f"Trajectory summary file does not exist: {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = ("time", "population_size", "mean_stress", "mean_survival", "npc_fraction")
        cfg.require(reader.fieldnames is not None, f"Trajectory summary file is missing a header: {path}")
        cfg.require(all(field in reader.fieldnames for field in required_fields), f"Trajectory summary file has invalid columns: {path}")
        rows = list(reader)
    cfg.require(bool(rows), f"Trajectory summary file contains no data rows: {path}")
    return TrajectorySeries(
        times=np.array([float(row["time"]) for row in rows], dtype=float),
        population_sizes=np.array([int(row["population_size"]) for row in rows], dtype=int),
        mean_stress=np.array([float(row["mean_stress"]) for row in rows], dtype=float),
        mean_survival=np.array([float(row["mean_survival"]) for row in rows], dtype=float),
        npc_fractions=np.array([float(row["npc_fraction"]) for row in rows], dtype=float),
    )


def _resolve_parameter_worker_count(task_count: int) -> int:
    cfg.require(task_count > 0, "Parallel execution requires a positive task count.")
    cfg.require(PARAMETER_EVAL_WORKERS >= 1, f"PARAMETER_EVAL_WORKERS must be >= 1, got {PARAMETER_EVAL_WORKERS}.")
    cpu_count = os.cpu_count() or 1
    return int(min(task_count, cpu_count, PARAMETER_EVAL_WORKERS))


def extract_screening_metrics(result: SimulationResult) -> dict[str, float]:
    cfg.require(bool(result.times), "Screening metrics require recorded simulation times.")
    cfg.require(bool(result.population_sizes), "Screening metrics require recorded population sizes.")
    cfg.require(bool(result.soft_state_fractions), "Screening metrics require recorded state fractions.")
    cfg.require(bool(result.bulk_copy_means), "Screening metrics require recorded bulk ecDNA means.")
    cfg.require(bool(result.mean_stress_scores), "Screening metrics require recorded stress values.")
    cfg.require(bool(result.mean_survival_scores), "Screening metrics require recorded survival values.")
    times = np.asarray(result.times, dtype=float)
    populations = np.asarray(result.population_sizes, dtype=float)
    stress = np.asarray(result.mean_stress_scores, dtype=float)
    survival = np.asarray(result.mean_survival_scores, dtype=float)
    state = np.asarray(result.soft_state_fractions, dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)
    cfg.require(times.ndim == 1 and len(times) >= 1, "Screening metrics require a one-dimensional time axis.")
    cfg.require(np.all(np.isfinite(times)), "Screening times must be finite.")
    cfg.require(np.all(np.isfinite(populations)), "Screening populations must be finite.")
    cfg.require(np.all(np.isfinite(stress)), "Screening stress values must be finite.")
    cfg.require(np.all(np.isfinite(survival)), "Screening survival values must be finite.")
    cfg.require(np.all(np.isfinite(state)), "Screening state values must be finite.")
    cfg.require(np.all(np.isfinite(bulk)), "Screening ecDNA values must be finite.")
    terminal_state = state[-1]
    terminal_bulk = bulk[-1]
    division_count, death_count = _event_counts(result)
    division_death_ratio = float(division_count / max(death_count, 1))
    bulk_trends = compute_bulk_copy_trends(result)
    return {
        "growth_rate": compute_growth_rate(result),
        "terminal_population": float(populations[-1]),
        "terminal_stress": float(stress[-1]),
        "terminal_survival": float(survival[-1]),
        "division_count": float(division_count),
        "death_count": float(death_count),
        "division_death_ratio": division_death_ratio,
        "auc_population": float(np.trapz(populations, times)),
        "auc_stress": float(np.trapz(stress, times)),
        "auc_survival": float(np.trapz(survival, times)),
        "terminal_state_NPC": float(terminal_state[cfg.NPC]),
        "terminal_state_OPC": float(terminal_state[cfg.OPC]),
        "terminal_state_AC": float(terminal_state[cfg.AC]),
        "terminal_state_MES": float(terminal_state[cfg.MES]),
        "terminal_bulk_MYC": float(terminal_bulk[cfg.MYC]),
        "terminal_bulk_CDK4": float(terminal_bulk[cfg.CDK4]),
        "terminal_bulk_PDGFRA": float(terminal_bulk[cfg.PDGFRA]),
        "bulk_trend_MYC": float(bulk_trends["MYC"]),
        "bulk_trend_CDK4": float(bulk_trends["CDK4"]),
        "bulk_trend_PDGFRA": float(bulk_trends["PDGFRA"]),
    }


def evaluate_screening_candidate(
    *,
    candidate_id: str,
    phase: str,
    parameter_values: dict[str, float],
    specs: dict[str, ParameterSpec],
    screening_config: ScreeningEvaluationConfig,
    trajectory_output_dir: Path | None = None,
) -> tuple[list[ScreeningSeedMetric], list[Path]]:
    eval_start = time.perf_counter()
    params, initialization = apply_parameter_overrides(specs, parameter_values)
    protocol_schedules = {"untreated": UNTREATED_SCHEDULES}
    cfg.require(
        screening_config.protocol_name in protocol_schedules,
        f"Unsupported screening protocol: {screening_config.protocol_name}.",
    )
    seed_values = tuple(int(seed) for seed in screening_config.seeds)
    cfg.require(bool(seed_values), "Screening seed list must contain at least one seed.")
    _debug_print(
        f"dispatch mode=screening phase={phase} candidate={candidate_id} parent_pid={os.getpid()} "
        f"seeds={list(seed_values)} seed_execution_mode=sequential"
    )
    seed_metrics: list[ScreeningSeedMetric] = []
    trajectory_paths: list[Path] = []
    for seed in seed_values:
        trajectory_path = None
        if trajectory_output_dir is not None:
            trajectory_path = trajectory_output_dir / f"{_safe_path_token(candidate_id)}_seed_{seed}_summary.csv"
            trajectory_paths.append(trajectory_path)
        seed_metrics.append(
            _evaluate_screening_seed(
                seed,
                candidate_id,
                phase,
                params,
                initialization,
                screening_config,
                trajectory_output_path=trajectory_path,
            )
        )
    _debug_print(
        f"dispatch-done mode=screening phase={phase} candidate={candidate_id} "
        f"total_seeds={len(seed_metrics)} duration_s={time.perf_counter() - eval_start:.3f}"
    )
    return seed_metrics, trajectory_paths


def _evaluate_screening_candidate_task(
    candidate_id: str,
    phase: str,
    parameter_values: dict[str, float],
    specs: dict[str, ParameterSpec],
    screening_config: ScreeningEvaluationConfig,
) -> tuple[str, dict[str, float], list[ScreeningSeedMetric], list[Path]]:
    seed_metrics, trajectory_paths = evaluate_screening_candidate(
        candidate_id=candidate_id,
        phase=phase,
        parameter_values=parameter_values,
        specs=specs,
        screening_config=screening_config,
    )
    return candidate_id, parameter_values, seed_metrics, trajectory_paths


def _evaluate_screening_seed(
    seed: int,
    candidate_id: str,
    phase: str,
    params: cfg.ModelParameters,
    initialization: InitializationOverride,
    screening_config: ScreeningEvaluationConfig,
    trajectory_output_path: Path | None = None,
) -> ScreeningSeedMetric:
    task_start = time.perf_counter()
    _debug_print(f"worker-start mode=screening pid={os.getpid()} phase={phase} candidate={candidate_id} seed={seed}")
    protocol_schedules = {"untreated": UNTREATED_SCHEDULES}
    cfg.require(
        screening_config.protocol_name in protocol_schedules,
        f"Unsupported screening protocol: {screening_config.protocol_name}.",
    )
    screening_params = replace(
        params,
        simulation=replace(
            params.simulation,
            time_unit="week",
            t_max=float(screening_config.t_max),
            n_init=int(screening_config.n_init),
            target_population_size=screening_config.target_population_size,
            max_pop_size=int(screening_config.max_pop_size),
            random_seed=int(seed),
            fitting_mode=False,
            record_full_snapshots=False,
            record_events=True,
        ),
    )
    result = run_simulation(
        params=screening_params,
        initialization=_to_initialization_parameters(initialization),
        seed=seed,
        input_schedules=protocol_schedules[screening_config.protocol_name],
        record_interval=screening_config.record_interval,
        verbose=False,
    )
    metrics = extract_screening_metrics(result)
    seed_metric = ScreeningSeedMetric(
        candidate_id=candidate_id,
        phase=phase,
        seed=int(seed),
        stop_reason=result.stop_reason,
        stop_time=float(result.stop_time or screening_config.t_max),
        **metrics,
    )
    if trajectory_output_path is not None:
        _write_trajectory_summary_csv(trajectory_output_path, result)
    _debug_print(
        f"worker-end mode=screening pid={os.getpid()} phase={phase} candidate={candidate_id} seed={seed} "
        f"status=ok duration_s={time.perf_counter() - task_start:.3f} stop_reason={result.stop_reason}"
    )
    return seed_metric


def _format_seed_scope(seeds: Sequence[int]) -> str:
    return ",".join(str(seed) for seed in seeds)


def _sample_value(spec: ParameterSpec, rng: np.random.Generator, center: float | None = None, trust_radius: float | None = None) -> float:
    center_value = spec.default_value if center is None else float(center)
    if spec.mode == "log":
        low = spec.lower_bound if trust_radius is None else max(spec.lower_bound, center_value * (1.0 - trust_radius))
        high = spec.upper_bound if trust_radius is None else min(spec.upper_bound, center_value * (1.0 + trust_radius))
        cfg.require(low > 0.0 and high > 0.0 and high >= low, f"Invalid log sampling bounds for {spec.path}.")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if spec.mode == "signed_ratio":
        if trust_radius is None:
            low = spec.lower_bound
            high = spec.upper_bound
        else:
            values = np.array([center_value * (1.0 - trust_radius), center_value * (1.0 + trust_radius)], dtype=float)
            low = max(spec.lower_bound, float(np.min(values)))
            high = min(spec.upper_bound, float(np.max(values)))
        return float(rng.uniform(low, high))
    if spec.mode == "additive":
        span = 0.10 if trust_radius is not None else 0.25
        return float(rng.uniform(max(spec.lower_bound, center_value - span), min(spec.upper_bound, center_value + span)))
    if spec.mode == "probability":
        span = 0.05 if trust_radius is not None else 0.10
        return float(rng.uniform(max(spec.lower_bound, center_value - span), min(spec.upper_bound, center_value + span)))
    raise ValueError(f"Unsupported sampling mode: {spec.mode}")


def _expand_family_paths(paths: Sequence[str]) -> list[str]:
    expanded = list(paths)
    if any(path.startswith("init_cycle_probs[") for path in paths):
        for path in _cycle_family_paths():
            if path not in expanded:
                expanded.append(path)
    return expanded


def sample_candidate(
    *,
    rng: np.random.Generator,
    specs: dict[str, ParameterSpec],
    active_paths: Sequence[str],
    base_values: dict[str, float],
    selected_paths: Sequence[str],
    trust_radius: float | None = None,
) -> dict[str, float]:
    candidate = dict(base_values)
    sampled_paths = _expand_family_paths(selected_paths)
    for path in sampled_paths:
        candidate[path] = _sample_value(specs[path], rng, center=base_values[path], trust_radius=trust_radius)
    if any(path.startswith("init_cycle_probs[") for path in sampled_paths):
        candidate.update(_normalize_cycle_probabilities({path: candidate[path] for path in _cycle_family_paths()}))
    return {path: candidate[path] for path in active_paths}


def _oat_perturbation_values(spec: ParameterSpec, n_points: int) -> list[float]:
    cfg.require(n_points in (3, 5), f"OAT perturbation grid only supports 3 or 5 points, got {n_points}.")
    if spec.mode == "log":
        multipliers = [0.5, 1.0, 2.0] if n_points == 3 else [0.5, 0.75, 1.0, 1.5, 2.0]
        values = [spec.default_value * multiplier for multiplier in multipliers]
    elif spec.mode == "probability":
        offsets = [-0.10, 0.0, 0.10] if n_points == 3 else [-0.10, -0.05, 0.0, 0.05, 0.10]
        values = [spec.default_value + offset for offset in offsets]
    else:
        offsets = [-0.25, 0.0, 0.25] if n_points == 3 else [-0.25, -0.125, 0.0, 0.125, 0.25]
        values = [spec.default_value + offset for offset in offsets]
    clipped = [float(min(spec.upper_bound, max(spec.lower_bound, value))) for value in values]
    ordered = []
    for value in clipped:
        if not ordered or abs(ordered[-1] - value) > 1e-12:
            ordered.append(value)
    return ordered


def _metric_scale(metric_name: str, baseline_value: float) -> float:
    if metric_name.startswith("terminal_state_"):
        return 1.0
    if metric_name.startswith("bulk_trend_"):
        return max(abs(baseline_value), 1e-3)
    return max(abs(baseline_value), 1e-6)


def _normalize_parameter_delta(spec: ParameterSpec, value: float) -> float:
    if spec.mode == "log":
        return abs(math.log(value / spec.default_value))
    if spec.mode == "probability":
        return abs(value - spec.default_value) / 0.10
    return abs(value - spec.default_value) / 0.25


def _aggregate_screening_rows(rows: Sequence[dict[str, object]], group_fields: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, object]] = []
    for key, members in grouped.items():
        base = {field: key[index] for index, field in enumerate(group_fields)}
        for metric_name in SCREENING_METRIC_KEYS:
            values = np.array([float(member[metric_name]) for member in members], dtype=float)
            base[f"{metric_name}_mean"] = float(np.mean(values))
            base[f"{metric_name}_std"] = float(np.std(values))
        aggregated.append(base)
    return aggregated


def rank_parameter_sensitivity(
    summary_rows: Sequence[dict[str, object]],
    specs: dict[str, ParameterSpec],
) -> list[dict[str, object]]:
    baseline_candidates = [row for row in summary_rows if bool(row["is_baseline"])]
    cfg.require(bool(baseline_candidates), "Sensitivity ranking requires a baseline summary row.")
    baseline_row = baseline_candidates[0]
    ranked_rows: list[dict[str, object]] = []
    for parameter_path, spec in specs.items():
        if spec.group == "validation_only":
            continue
        parameter_rows = [row for row in summary_rows if str(row["parameter_path"]) == parameter_path and not bool(row["is_baseline"])]
        if not parameter_rows:
            continue
        category_scores = {category: 0.0 for category in SCREENING_METRIC_CATEGORIES}
        metric_scores = {metric_name: 0.0 for metric_name in SCREENING_METRIC_KEYS}
        monotonicity_values = []
        sorted_rows = sorted(parameter_rows, key=lambda row: float(row["parameter_value"]))
        if len(sorted_rows) >= 3:
            deltas = np.diff(np.array([float(row["growth_rate_mean"]) for row in sorted_rows], dtype=float))
            signs = np.sign(deltas[np.abs(deltas) > 1e-10])
            monotonicity = 1.0 if len(signs) <= 1 else float(np.mean(signs == signs[0]))
        else:
            monotonicity = 1.0
        monotonicity_values.append(monotonicity)
        for row in parameter_rows:
            parameter_delta = _normalize_parameter_delta(spec, float(row["parameter_value"]))
            if parameter_delta <= 1e-12:
                continue
            for metric_name in SCREENING_METRIC_KEYS:
                baseline_metric = float(baseline_row[f"{metric_name}_mean"])
                perturbed_metric = float(row[f"{metric_name}_mean"])
                metric_delta = abs(perturbed_metric - baseline_metric) / _metric_scale(metric_name, baseline_metric)
                normalized_sensitivity = metric_delta / parameter_delta
                metric_scores[metric_name] = max(metric_scores[metric_name], float(normalized_sensitivity))
        for category_name, metric_names in SCREENING_METRIC_CATEGORIES.items():
            category_scores[category_name] = float(np.mean([metric_scores[metric_name] for metric_name in metric_names]))
        ranked_rows.append(
            {
                "parameter_path": parameter_path,
                "group": spec.group,
                "overall_score": float(np.mean(list(category_scores.values()))),
                "monotonicity_score": float(np.mean(monotonicity_values)),
                **{f"{category_name}_score": score for category_name, score in category_scores.items()},
            }
        )
    ranked_rows.sort(key=lambda row: float(row["overall_score"]), reverse=True)
    return ranked_rows


def build_two_parameter_grid(
    *,
    parameter_a: str,
    parameter_b: str,
    specs: dict[str, ParameterSpec],
    base_values: dict[str, float],
    grid_size: int,
) -> list[dict[str, float]]:
    spec_a = specs[parameter_a]
    spec_b = specs[parameter_b]
    values_a = _oat_perturbation_values(spec_a, 3 if grid_size <= 3 else 5)
    values_b = _oat_perturbation_values(spec_b, 3 if grid_size <= 3 else 5)
    if grid_size < len(values_a):
        values_a = [values_a[0], values_a[len(values_a) // 2], values_a[-1]]
        values_b = [values_b[0], values_b[len(values_b) // 2], values_b[-1]]
    candidates: list[dict[str, float]] = []
    for value_a, value_b in itertools.product(values_a, values_b):
        candidate = dict(base_values)
        candidate[parameter_a] = float(value_a)
        candidate[parameter_b] = float(value_b)
        if parameter_a.startswith("init_cycle_probs[") or parameter_b.startswith("init_cycle_probs["):
            candidate.update(_normalize_cycle_probabilities({path: candidate[path] for path in _cycle_family_paths()}))
        candidates.append(candidate)
    return candidates


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _screening_rows_from_seed_metrics(
    *,
    phase: str,
    parameter_path: str,
    parameter_group: str,
    parameter_value: float,
    is_baseline: bool,
    seed_metrics: Sequence[ScreeningSeedMetric],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in seed_metrics:
        row = {
            "phase": phase,
            "candidate_id": metric.candidate_id,
            "parameter_path": parameter_path,
            "parameter_group": parameter_group,
            "parameter_value": parameter_value,
            "is_baseline": is_baseline,
            **asdict(metric),
        }
        rows.append(row)
    return rows


def _screening_manifest(
    *,
    cwd: Path,
    phase: str,
    seeds: Sequence[int],
    parameter_paths: Sequence[str],
    plan: ScreeningExecutionPlan,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "phase": phase,
        "seed_scope": _format_seed_scope(seeds),
        "protocol_name": plan.protocol_name,
        "t_max": plan.t_max,
        "record_interval": plan.record_interval,
        "n_init": plan.n_init,
        "target_population_size": plan.target_population_size,
        "max_pop_size": plan.max_pop_size,
        "parameter_paths": list(parameter_paths),
        "code_version": _git_commit_or_unknown(cwd),
    }
    if extra:
        manifest.update(extra)
    return manifest


def _plot_screening_rank_bars(ranked_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * min(len(ranked_rows), 20))))
    top_rows = list(ranked_rows[:20])
    labels = [str(row["parameter_path"]) for row in top_rows][::-1]
    values = [float(row["overall_score"]) for row in top_rows][::-1]
    if values:
        ax.barh(labels, values, color="#1d4ed8")
    ax.set_title("Parameter sensitivity ranking")
    ax.set_xlabel("Overall sensitivity score")
    _save_figure(fig, output_path)


def _plot_screening_metric_correlation(summary_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    baseline_summary = [row for row in summary_rows if bool(row["is_baseline"])]
    fig, ax = plt.subplots(figsize=(10, 8))
    if not baseline_summary:
        ax.text(0.5, 0.5, "No summary rows", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, output_path)
        return
    matrix = np.array([[float(row[f"{metric}_mean"]) for metric in SCREENING_METRIC_KEYS] for row in summary_rows], dtype=float)
    corr = np.corrcoef(matrix, rowvar=False)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(SCREENING_METRIC_KEYS)))
    ax.set_xticklabels(SCREENING_METRIC_KEYS, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(SCREENING_METRIC_KEYS)))
    ax.set_yticklabels(SCREENING_METRIC_KEYS, fontsize=7)
    ax.set_title("Screening metric correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, output_path)


def _plot_oat_response_curves(
    summary_rows: Sequence[dict[str, object]],
    ranked_rows: Sequence[dict[str, object]],
    output_dir: Path,
    top_n: int,
) -> None:
    baseline_candidates = [row for row in summary_rows if bool(row["is_baseline"])]
    if not baseline_candidates:
        return
    baseline_row = baseline_candidates[0]
    for row in ranked_rows[:top_n]:
        parameter_path = str(row["parameter_path"])
        parameter_rows = sorted(
            [item for item in summary_rows if str(item["parameter_path"]) == parameter_path],
            key=lambda item: float(item["parameter_value"]),
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        x_values = [float(item["parameter_value"]) for item in parameter_rows]
        plotting_metrics = ("growth_rate", "terminal_population", "terminal_stress", "terminal_bulk_MYC")
        titles = ("Growth rate", "Terminal population", "Terminal stress", "Terminal MYC")
        for axis, metric_name, title in zip(axes.flatten(), plotting_metrics, titles):
            mean_values = [float(item[f"{metric_name}_mean"]) for item in parameter_rows]
            std_values = [float(item[f"{metric_name}_std"]) for item in parameter_rows]
            axis.plot(x_values, mean_values, linewidth=2)
            axis.fill_between(x_values, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), alpha=0.2)
            axis.axhline(float(baseline_row[f"{metric_name}_mean"]), color="#6b7280", linestyle="--", linewidth=1.0)
            axis.set_title(title)
            axis.set_xlabel(parameter_path)
        fig.suptitle(f"OAT response: {parameter_path}")
        _save_figure(fig, output_dir / f"{parameter_path.replace('.', '_').replace('[', '_').replace(']', '')}_oat_response.png")


def _plot_two_parameter_heatmap(
    rows: Sequence[dict[str, object]],
    parameter_a: str,
    parameter_b: str,
    metric_name: str,
    output_path: Path,
) -> None:
    values_a = sorted({float(row["parameter_a_value"]) for row in rows})
    values_b = sorted({float(row["parameter_b_value"]) for row in rows})
    matrix = np.full((len(values_b), len(values_a)), np.nan, dtype=float)
    for row in rows:
        index_a = values_a.index(float(row["parameter_a_value"]))
        index_b = values_b.index(float(row["parameter_b_value"]))
        matrix[index_b, index_a] = float(row[f"{metric_name}_mean"])
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(values_a)))
    ax.set_xticklabels([f"{value:.3g}" for value in values_a], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(values_b)))
    ax.set_yticklabels([f"{value:.3g}" for value in values_b], fontsize=8)
    ax.set_xlabel(parameter_a)
    ax.set_ylabel(parameter_b)
    ax.set_title(f"{metric_name} heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, output_path)


def _plot_representative_trajectory(
    baseline_summary_path: Path,
    low_summary_path: Path,
    high_summary_path: Path,
    parameter_path: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    runs = (
        ("baseline", _read_trajectory_summary_csv(baseline_summary_path), "#1d4ed8"),
        ("low", _read_trajectory_summary_csv(low_summary_path), "#dc2626"),
        ("high", _read_trajectory_summary_csv(high_summary_path), "#16a34a"),
    )
    for label, trajectory, color in runs:
        axes[0, 0].plot(trajectory.times, trajectory.population_sizes, label=label, color=color)
        axes[0, 1].plot(trajectory.times, trajectory.mean_stress, label=label, color=color)
        axes[1, 0].plot(trajectory.times, trajectory.mean_survival, label=label, color=color)
        axes[1, 1].plot(trajectory.times, trajectory.npc_fractions, label=label, color=color)
    axes[0, 0].set_title("Population")
    axes[0, 1].set_title("Stress")
    axes[1, 0].set_title("Survival")
    axes[1, 1].set_title("NPC fraction")
    for axis in axes.flatten():
        axis.set_xlabel("Time")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Representative trajectory: {parameter_path}")
    _save_figure(fig, output_path)


class ScreeningEngine:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        plan: ScreeningExecutionPlan | None = None,
        specs: dict[str, ParameterSpec] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plan = plan or ScreeningExecutionPlan()
        self.specs = build_parameter_specs() if specs is None else specs
        self.defaults = default_parameter_values(self.specs)
        self.workspace = Path.cwd()

    def _screening_config(self) -> ScreeningEvaluationConfig:
        return ScreeningEvaluationConfig(
            protocol_name=self.plan.protocol_name,
            t_max=self.plan.t_max,
            record_interval=self.plan.record_interval,
            n_init=self.plan.n_init,
            target_population_size=self.plan.target_population_size,
            max_pop_size=self.plan.max_pop_size,
            seeds=self.plan.baseline_seeds,
        )

    def _evaluate_screening_candidates(
        self,
        *,
        phase: str,
        candidates: list[tuple[str, dict[str, float]]],
    ) -> list[tuple[str, dict[str, float], list[ScreeningSeedMetric], list[Path]]]:
        cfg.require(bool(candidates), f"Screening phase {phase} must provide at least one candidate.")
        screening_config = self._screening_config()
        worker_count = _resolve_parameter_worker_count(len(candidates))
        execution_mode = "parallel" if worker_count > 1 else "sequential"
        candidate_ids = [candidate_id for candidate_id, _candidate in candidates]
        candidate_values = [dict(candidate) for _candidate_id, candidate in candidates]
        _debug_print(
            f"phase-dispatch mode=screening phase={phase} parent_pid={os.getpid()} candidates={len(candidates)} "
            f"seeds={list(screening_config.seeds)} configured_workers={PARAMETER_EVAL_WORKERS} "
            f"resolved_workers={worker_count} execution_mode={execution_mode}"
        )
        phase_start = time.perf_counter()
        if worker_count == 1:
            results = [
                _evaluate_screening_candidate_task(candidate_id, phase, candidate, self.specs, screening_config)
                for candidate_id, candidate in zip(candidate_ids, candidate_values)
            ]
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        _evaluate_screening_candidate_task,
                        candidate_ids,
                        itertools.repeat(phase),
                        candidate_values,
                        itertools.repeat(self.specs),
                        itertools.repeat(screening_config),
                    )
                )
        _debug_print(
            f"phase-dispatch-done mode=screening phase={phase} candidates={len(results)} "
            f"duration_s={time.perf_counter() - phase_start:.3f}"
        )
        return results

    def _baseline_results(self) -> tuple[list[dict[str, object]], list[Path]]:
        _progress_print(
            f"phase_01_oat_local baseline: protocol={self.plan.protocol_name}, seeds={_format_seed_scope(self.plan.baseline_seeds)}, "
            f"t_max={self.plan.t_max}, n_init={self.plan.n_init}, target_population_size={self.plan.target_population_size}, max_pop_size={self.plan.max_pop_size}"
        )
        seed_metrics, trajectory_paths = evaluate_screening_candidate(
            candidate_id="baseline",
            phase="phase_01_oat_local",
            parameter_values=dict(self.defaults),
            specs=self.specs,
            screening_config=self._screening_config(),
            trajectory_output_dir=self.output_dir / "trajectory_cache" / "baseline",
        )
        rows = _screening_rows_from_seed_metrics(
            phase="phase_01_oat_local",
            parameter_path="baseline",
            parameter_group="baseline",
            parameter_value=1.0,
            is_baseline=True,
            seed_metrics=seed_metrics,
        )
        cfg.require(bool(trajectory_paths), "Baseline trajectory summaries must be written for final plots.")
        return rows, trajectory_paths

    def _run_oat_local(self) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path]]:
        phase_dir = self.output_dir / "phase_01_oat_local"
        phase_dir.mkdir(parents=True, exist_ok=True)
        per_seed_rows, baseline_trajectory_paths = self._baseline_results()
        parameter_paths = [path for path, spec in self.specs.items() if spec.group != "validation_only"]
        _progress_print(
            f"phase_01_oat_local start: parameters={len(parameter_paths)}, oat_points={self.plan.oat_points_per_parameter}, output={phase_dir}"
        )
        for parameter_index, parameter_path in enumerate(parameter_paths, start=1):
            spec = self.specs[parameter_path]
            perturb_values = _oat_perturbation_values(spec, self.plan.oat_points_per_parameter)
            _progress_print(
                f"phase_01_oat_local parameter {parameter_index}/{len(parameter_paths)}: {parameter_path} "
                f"(group={spec.group}, points={len(perturb_values)})"
            )
            candidate_batch: list[tuple[str, dict[str, float]]] = []
            for value in perturb_values:
                candidate = dict(self.defaults)
                candidate[parameter_path] = value
                if parameter_path.startswith("init_cycle_probs["):
                    candidate.update(_normalize_cycle_probabilities({path: candidate[path] for path in _cycle_family_paths()}))
                candidate_batch.append((f"{parameter_path}:{value:.6g}", candidate))
            results = self._evaluate_screening_candidates(phase="phase_01_oat_local", candidates=candidate_batch)
            for _candidate_id, candidate, seed_metrics, _trajectory_paths in results:
                per_seed_rows.extend(
                    _screening_rows_from_seed_metrics(
                        phase="phase_01_oat_local",
                        parameter_path=parameter_path,
                        parameter_group=spec.group,
                        parameter_value=float(candidate[parameter_path]),
                        is_baseline=abs(float(candidate[parameter_path]) - spec.default_value) <= 1e-12,
                        seed_metrics=seed_metrics,
                    )
                )
        summary_rows = _aggregate_screening_rows(
            per_seed_rows,
            ("phase", "parameter_path", "parameter_group", "parameter_value", "is_baseline"),
        )
        _write_csv(phase_dir / "screening_metrics_per_seed.csv", per_seed_rows)
        _write_csv(phase_dir / "screening_metrics_summary.csv", summary_rows)
        with open(phase_dir / "phase_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                _screening_manifest(
                    cwd=self.workspace,
                    phase="phase_01_oat_local",
                    seeds=self.plan.baseline_seeds,
                    parameter_paths=parameter_paths,
                    plan=self.plan,
                    extra={"oat_points_per_parameter": self.plan.oat_points_per_parameter},
                ),
                handle,
                indent=2,
            )
        _progress_print(
            f"phase_01_oat_local done: per_seed_rows={len(per_seed_rows)}, summary_rows={len(summary_rows)}, manifest={phase_dir / 'phase_manifest.json'}"
        )
        return per_seed_rows, summary_rows, baseline_trajectory_paths

    def _run_rank_and_select(self, summary_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        phase_dir = self.output_dir / "phase_02_rank_and_select"
        phase_dir.mkdir(parents=True, exist_ok=True)
        _progress_print(f"phase_02_rank_and_select start: summary_rows={len(summary_rows)}, output={phase_dir}")
        ranked_rows = rank_parameter_sensitivity(summary_rows, self.specs)
        _write_csv(phase_dir / "parameter_sensitivity_rank.csv", ranked_rows)
        _plot_screening_rank_bars(ranked_rows, phase_dir / "parameter_sensitivity_rank.png")
        _plot_screening_metric_correlation(summary_rows, phase_dir / "metric_correlation_heatmap.png")
        _plot_oat_response_curves(summary_rows, ranked_rows, phase_dir, top_n=min(10, len(ranked_rows)))
        with open(phase_dir / "phase_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                _screening_manifest(
                    cwd=self.workspace,
                    phase="phase_02_rank_and_select",
                    seeds=self.plan.baseline_seeds,
                    parameter_paths=[str(row["parameter_path"]) for row in ranked_rows[:20]],
                    plan=self.plan,
                ),
                handle,
                indent=2,
            )
        if ranked_rows:
            top_row = ranked_rows[0]
            _progress_print(
                f"phase_02_rank_and_select done: ranked={len(ranked_rows)}, top={top_row['parameter_path']}, overall_score={float(top_row['overall_score']):.4f}"
            )
        else:
            _progress_print("phase_02_rank_and_select done: ranked=0")
        return ranked_rows

    def _select_two_parameter_pairs(self, ranked_rows: Sequence[dict[str, object]]) -> list[tuple[str, str]]:
        selected_by_group: dict[str, list[str]] = {}
        for row in ranked_rows:
            group = self.specs[str(row["parameter_path"])].group
            selected_by_group.setdefault(group, [])
            if len(selected_by_group[group]) < self.plan.ranking_top_parameters_per_category:
                selected_by_group[group].append(str(row["parameter_path"]))
        pairs: list[tuple[str, str]] = []
        for group, parameter_paths in selected_by_group.items():
            if len(parameter_paths) < 2:
                continue
            for pair in itertools.islice(itertools.combinations(parameter_paths, 2), self.plan.top_pairs_per_group):
                pairs.append(pair)
        init_parameters = selected_by_group.get("init", [])
        top_growth = [str(row["parameter_path"]) for row in ranked_rows if self.specs[str(row["parameter_path"])].group != "init"][: self.plan.top_pairs_per_group]
        for parameter_a in init_parameters[: self.plan.top_pairs_per_group]:
            for parameter_b in top_growth:
                pairs.append((parameter_a, parameter_b))
        unique_pairs: list[tuple[str, str]] = []
        for parameter_a, parameter_b in pairs:
            if parameter_a == parameter_b:
                continue
            normalized = tuple(sorted((parameter_a, parameter_b)))
            if normalized not in unique_pairs:
                unique_pairs.append(normalized)
        return unique_pairs

    def _run_two_param_maps(self, ranked_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        phase_dir = self.output_dir / "phase_03_two_param_maps"
        phase_dir.mkdir(parents=True, exist_ok=True)
        pairs = self._select_two_parameter_pairs(ranked_rows)
        _progress_print(
            f"phase_03_two_param_maps start: pairs={len(pairs)}, grid_size={self.plan.two_param_grid_size}, output={phase_dir}"
        )
        per_seed_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        for pair_index, (parameter_a, parameter_b) in enumerate(pairs, start=1):
            candidates = build_two_parameter_grid(
                parameter_a=parameter_a,
                parameter_b=parameter_b,
                specs=self.specs,
                base_values=self.defaults,
                grid_size=self.plan.two_param_grid_size,
            )
            _progress_print(
                f"phase_03_two_param_maps pair {pair_index}/{len(pairs)}: {parameter_a} x {parameter_b}, candidates={len(candidates)}"
            )
            pair_seed_rows: list[dict[str, object]] = []
            candidate_batch = [(f"grid_{index:04d}", candidate) for index, candidate in enumerate(candidates)]
            results = self._evaluate_screening_candidates(phase="phase_03_two_param_maps", candidates=candidate_batch)
            for _candidate_id, candidate, seed_metrics, _trajectory_paths in results:
                for metric in seed_metrics:
                    row = {
                        "phase": "phase_03_two_param_maps",
                        "candidate_id": metric.candidate_id,
                        "parameter_a": parameter_a,
                        "parameter_b": parameter_b,
                        "parameter_a_value": float(candidate[parameter_a]),
                        "parameter_b_value": float(candidate[parameter_b]),
                        **asdict(metric),
                    }
                    pair_seed_rows.append(row)
            per_seed_rows.extend(pair_seed_rows)
            pair_summary_rows = _aggregate_screening_rows(
                pair_seed_rows,
                ("phase", "parameter_a", "parameter_b", "parameter_a_value", "parameter_b_value"),
            )
            summary_rows.extend(pair_summary_rows)
            _plot_two_parameter_heatmap(
                pair_summary_rows,
                parameter_a=parameter_a,
                parameter_b=parameter_b,
                metric_name="growth_rate",
                output_path=phase_dir / f"{parameter_a.replace('.', '_')}_{parameter_b.replace('.', '_')}_growth_heatmap.png",
            )
        _write_csv(phase_dir / "two_parameter_grid_per_seed.csv", per_seed_rows)
        _write_csv(phase_dir / "two_parameter_grid_summary.csv", summary_rows)
        with open(phase_dir / "phase_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                _screening_manifest(
                    cwd=self.workspace,
                    phase="phase_03_two_param_maps",
                    seeds=self.plan.baseline_seeds,
                    parameter_paths=[f"{a}|{b}" for a, b in pairs],
                    plan=self.plan,
                    extra={"pair_count": len(pairs), "grid_size": self.plan.two_param_grid_size},
                ),
                handle,
                indent=2,
            )
        _progress_print(
            f"phase_03_two_param_maps done: per_seed_rows={len(per_seed_rows)}, summary_rows={len(summary_rows)}, manifest={phase_dir / 'phase_manifest.json'}"
        )
        return summary_rows

    def _write_final_outputs(
        self,
        ranked_rows: Sequence[dict[str, object]],
        baseline_trajectory_paths: Sequence[Path],
        summary_rows: Sequence[dict[str, object]],
    ) -> None:
        final_dir = self.output_dir / "final_outputs"
        final_dir.mkdir(parents=True, exist_ok=True)
        top_parameters = [str(row["parameter_path"]) for row in ranked_rows[: self.plan.representative_top_parameters]]
        cfg.require(bool(baseline_trajectory_paths), "Final trajectory plots require a baseline trajectory summary.")
        baseline_summary_path = baseline_trajectory_paths[0]
        trajectory_dir = self.output_dir / "trajectory_cache" / "final_outputs"
        _progress_print(
            f"final_outputs start: representative_parameters={len(top_parameters)}, output={final_dir}"
        )
        for parameter_path in top_parameters:
            spec = self.specs[parameter_path]
            perturb_values = _oat_perturbation_values(spec, self.plan.oat_points_per_parameter)
            low_value = perturb_values[0]
            high_value = perturb_values[-1]
            _progress_print(
                f"final_outputs trajectory: {parameter_path}, low={low_value:.6g}, high={high_value:.6g}"
            )
            low_candidate = dict(self.defaults)
            low_candidate[parameter_path] = low_value
            high_candidate = dict(self.defaults)
            high_candidate[parameter_path] = high_value
            if parameter_path.startswith("init_cycle_probs["):
                low_candidate.update(_normalize_cycle_probabilities({path: low_candidate[path] for path in _cycle_family_paths()}))
                high_candidate.update(_normalize_cycle_probabilities({path: high_candidate[path] for path in _cycle_family_paths()}))
            _, low_trajectory_paths = evaluate_screening_candidate(
                candidate_id=f"{parameter_path}:low",
                phase="final_outputs",
                parameter_values=low_candidate,
                specs=self.specs,
                screening_config=self._screening_config(),
                trajectory_output_dir=trajectory_dir,
            )
            _, high_trajectory_paths = evaluate_screening_candidate(
                candidate_id=f"{parameter_path}:high",
                phase="final_outputs",
                parameter_values=high_candidate,
                specs=self.specs,
                screening_config=self._screening_config(),
                trajectory_output_dir=trajectory_dir,
            )
            cfg.require(bool(low_trajectory_paths) and bool(high_trajectory_paths), "Final trajectory summaries must be written before plotting.")
            _plot_representative_trajectory(
                baseline_summary_path=baseline_summary_path,
                low_summary_path=low_trajectory_paths[0],
                high_summary_path=high_trajectory_paths[0],
                parameter_path=parameter_path,
                output_path=final_dir / f"{parameter_path.replace('.', '_').replace('[', '_').replace(']', '')}_trajectory.png",
            )
        report_lines = [
            "# Screening Summary Report",
            "",
            f"- Protocol: `{self.plan.protocol_name}`",
            f"- Seeds: `{_format_seed_scope(self.plan.baseline_seeds)}`",
            "",
            "## Top Sensitive Parameters",
        ]
        for row in ranked_rows[:10]:
            report_lines.append(
                f"- {row['parameter_path']}: overall={float(row['overall_score']):.4f}, growth={float(row['growth_score']):.4f}, state={float(row['state_score']):.4f}, ecdna={float(row['ecdna_score']):.4f}, stress_survival={float(row['stress_survival_score']):.4f}, events={float(row['events_score']):.4f}, monotonicity={float(row['monotonicity_score']):.4f}"
            )
        report_lines.extend(["", "## Interpretation"])
        for category_name in ("growth", "state", "ecdna", "stress_survival", "events"):
            category_rows = sorted(ranked_rows, key=lambda row: float(row[f"{category_name}_score"]), reverse=True)[:3]
            summary = ", ".join(f"{row['parameter_path']}={float(row[f'{category_name}_score']):.3f}" for row in category_rows)
            report_lines.append(f"- {category_name}: {summary}")
        report_lines.extend(["", "## Output Notes", "- `phase_01_oat_local` contains one-at-a-time perturbation metrics.", "- `phase_02_rank_and_select` contains sensitivity rankings and response curves.", "- `phase_03_two_param_maps` contains pairwise response heatmaps."])
        with open(final_dir / "summary_report.md", "w", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines) + "\n")
        with open(final_dir / "phase_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                _screening_manifest(
                    cwd=self.workspace,
                    phase="final_outputs",
                    seeds=self.plan.baseline_seeds,
                    parameter_paths=top_parameters,
                    plan=self.plan,
                    extra={"summary_row_count": len(summary_rows)},
                ),
                handle,
                indent=2,
            )
        _progress_print(f"final_outputs done: summary_report={final_dir / 'summary_report.md'}")

    def run(self) -> dict[str, object]:
        _progress_print(
            f"screening run start: protocol={self.plan.protocol_name}, parameters={len([path for path, spec in self.specs.items() if spec.group != 'validation_only'])}, "
            f"seeds={_format_seed_scope(self.plan.baseline_seeds)}, output={self.output_dir}"
        )
        per_seed_rows, summary_rows, baseline_trajectory_paths = self._run_oat_local()
        ranked_rows = self._run_rank_and_select(summary_rows)
        two_param_summary = self._run_two_param_maps(ranked_rows)
        self._write_final_outputs(ranked_rows, baseline_trajectory_paths, summary_rows)
        _progress_print("screening run done")
        return {
            "phase_01_oat_local": {"per_seed_rows": per_seed_rows, "summary_rows": summary_rows},
            "phase_02_rank_and_select": {"ranked_rows": ranked_rows},
            "phase_03_two_param_maps": {"summary_rows": two_param_summary},
        }


def run_screening(
    *,
    output_dir: str | Path,
    plan: ScreeningExecutionPlan | None = None,
    specs: dict[str, ParameterSpec] | None = None,
) -> dict[str, object]:
    engine = ScreeningEngine(output_dir=output_dir, plan=plan, specs=specs)
    return engine.run()
