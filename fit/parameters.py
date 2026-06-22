"""The ten fitted parameters and the log transform phi = log(theta).

Implements fit_method.md equations (1) and (2): all ten positive
parameters are proposed jointly in log-transformed space.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402


@dataclass(frozen=True)
class ParameterSpec:
    """One fitted parameter: its location inside ``ModelParameters``."""

    name: str          # markdown symbol group, e.g. "nu_C"
    config_path: str   # dotted config path, e.g. "exposure.nu_C"
    object_path: tuple[str, ...]  # (block, field) walked via getattr/replace

    @property
    def column_token(self) -> str:
        return self.config_path.replace(".", "_")

    @property
    def theta_column(self) -> str:
        return f"theta_{self.column_token}"

    @property
    def phi_column(self) -> str:
        return f"phi_{self.column_token}"

    @property
    def log2_fold_column(self) -> str:
        return f"log2_fold_{self.column_token}_vs_config"


# The ten positive parameters perturbed in log space (fit_method.md equation 1):
#   nu_C, nu_P, beta_C, beta_Pg, chi_C, chi_P, theta_P, phi_B, chi_B, B_star
PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("nu_C", "exposure.nu_C", ("exposure", "nu_C")),
    ParameterSpec("nu_P", "exposure.nu_P", ("exposure", "nu_P")),
    ParameterSpec("beta_C", "cycle.beta_C", ("cycle", "beta_C")),
    ParameterSpec("beta_Pg", "cycle.beta_Pg", ("cycle", "beta_Pg")),
    ParameterSpec("chi_C", "hazard.chi_C", ("hazard", "chi_C")),
    ParameterSpec("chi_P", "hazard.chi_P", ("hazard", "chi_P")),
    ParameterSpec("theta_P", "hazard.theta_P", ("hazard", "theta_P")),
    ParameterSpec("phi_B", "hazard.phi_B", ("hazard", "phi_B")),
    ParameterSpec("chi_B", "hazard.chi_B", ("hazard", "chi_B")),
    ParameterSpec("B_star", "hazard.B_star", ("hazard", "B_star")),
)

N_PARAMETERS = len(PARAMETER_SPECS)


def _get_parameter_value(params: cfg.ModelParameters, spec: ParameterSpec) -> float:
    current: Any = params
    for part in spec.object_path:
        current = getattr(current, part)
    return float(current)


def _set_parameter_value(params: cfg.ModelParameters, spec: ParameterSpec, value: float) -> cfg.ModelParameters:
    block_name, field_name = spec.object_path
    new_block = replace(getattr(params, block_name), **{field_name: float(value)})
    return replace(params, **{block_name: new_block})


def reference_params() -> cfg.ModelParameters:
    """The reference parameterization phi_0 is taken from ``config.py``."""
    return cfg.DEFAULT_MODEL_PARAMETERS


def reference_phi() -> np.ndarray:
    """phi_0 = log(theta_0) for the reference parameterization."""
    return phi_from_params(reference_params())


def phi_from_params(params: cfg.ModelParameters) -> np.ndarray:
    """phi_m = log(theta_m); requires all fitted parameters to be positive."""
    values = []
    for spec in PARAMETER_SPECS:
        value = _get_parameter_value(params, spec)
        cfg.require(value > 0.0, f"{spec.config_path} must be positive for log perturbation, got {value}.")
        values.append(math.log(value))
    return np.asarray(values, dtype=float)


def params_from_phi(base: cfg.ModelParameters, phi: np.ndarray) -> cfg.ModelParameters:
    """theta_m = exp(phi_m); validate the resulting ModelParameters."""
    params = base
    for value, spec in zip(np.asarray(phi, dtype=float), PARAMETER_SPECS):
        theta = math.exp(float(value))
        params = _set_parameter_value(params, spec, theta)
    cfg.validate_model_parameters(params)
    return params


def phi_to_theta(phi: np.ndarray) -> np.ndarray:
    """Elementwise theta = exp(phi)."""
    return np.exp(np.asarray(phi, dtype=float))


def distance_to_config(phi: np.ndarray, phi0: np.ndarray) -> float:
    """sqrt(mean((phi - phi0)^2)) in log-parameter space (fit_method.md ranking criterion)."""
    return float(np.sqrt(np.mean((np.asarray(phi, dtype=float) - phi0) ** 2)))


def log2_fold_vs_config(phi: np.ndarray, phi0: np.ndarray) -> np.ndarray:
    """Per-parameter log2 fold change relative to the reference (phi - phi0)/log(2)."""
    return (np.asarray(phi, dtype=float) - phi0) / math.log(2.0)
