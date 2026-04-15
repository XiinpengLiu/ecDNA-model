"""
Configuration and shared math utilities for the ecDNA v4 model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np


SPECIES = ("MYC", "CDK4", "PDGFRA")
STATE_NAMES = ("NPC-like", "OPC-like", "AC-like", "MES-like")
CYCLE_NAMES = ("Q", "G1", "S", "G2M")

SPECIES_INDEX = {name: idx for idx, name in enumerate(SPECIES)}
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_NAMES)}
CYCLE_INDEX = {name: idx for idx, name in enumerate(CYCLE_NAMES)}

N_SPECIES = len(SPECIES)
N_STATES = len(STATE_NAMES)
LATENT_DIM = N_STATES - 1
N_CYCLE = len(CYCLE_NAMES)

MYC = SPECIES_INDEX["MYC"]
CDK4 = SPECIES_INDEX["CDK4"]
PDGFRA = SPECIES_INDEX["PDGFRA"]

NPC = STATE_INDEX["NPC-like"]
OPC = STATE_INDEX["OPC-like"]
AC = STATE_INDEX["AC-like"]
MES = STATE_INDEX["MES-like"]

Q = CYCLE_INDEX["Q"]
G1 = CYCLE_INDEX["G1"]
S = CYCLE_INDEX["S"]
G2M = CYCLE_INDEX["G2M"]

HELMERT_SUBMATRIX = np.array(
    [
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
        [0.0, -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
        [0.0, 0.0, -3.0 / np.sqrt(12.0)],
    ],
    dtype=float,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x_array = np.asarray(x, dtype=float)
    x_clamped = np.clip(x_array, -500.0, 500.0)
    result = 1.0 / (1.0 + np.exp(-x_clamped))
    if np.isscalar(x):
        return float(result)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    require(logits.shape == (N_STATES,), f"Expected {N_STATES} logits, got {logits.shape}.")
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    total = np.sum(weights)
    require(np.isfinite(total) and total > 0.0, "Softmax denominator must be positive.")
    return weights / total


def closure(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    require(values.shape == (N_STATES,), f"Expected {N_STATES} composition entries, got {values.shape}.")
    require(np.all(np.isfinite(values)), "Composition values must be finite.")
    require(np.all(values > 0.0), "Composition values must be strictly positive.")
    total = float(np.sum(values))
    require(np.isfinite(total) and total > 0.0, "Composition sum must be finite and positive.")
    return values / total


def ilr(composition: np.ndarray) -> np.ndarray:
    composition = closure(composition)
    latent = HELMERT_SUBMATRIX.T @ np.log(composition)
    require(latent.shape == (LATENT_DIM,), f"Expected latent dimension {LATENT_DIM}, got {latent.shape}.")
    return latent


def inverse_ilr(latent: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=float)
    require(latent.shape == (LATENT_DIM,), f"Expected latent dimension {LATENT_DIM}, got {latent.shape}.")
    return closure(np.exp(HELMERT_SUBMATRIX @ latent))


def validate_copy_vector(copy_vector: np.ndarray) -> None:
    copy_vector = np.asarray(copy_vector)
    require(copy_vector.shape == (N_SPECIES,), f"Expected {N_SPECIES} ecDNA species, got {copy_vector.shape}.")
    require(np.issubdtype(copy_vector.dtype, np.integer), "ecDNA copy numbers must be integer-valued.")
    require(np.all(copy_vector >= 0), "ecDNA copy numbers must be non-negative.")


def validate_simplex(composition: np.ndarray) -> None:
    composition = np.asarray(composition, dtype=float)
    require(composition.shape == (N_STATES,), f"Expected {N_STATES}-state composition, got {composition.shape}.")
    require(np.all(np.isfinite(composition)), "Soft state composition must be finite.")
    require(np.all(composition > 0.0), "Soft state composition must be strictly positive.")
    total = float(np.sum(composition))
    require(abs(total - 1.0) <= 1e-8, f"Soft state composition must sum to 1, got {total}.")


def validate_cycle_state(cycle_state: int) -> None:
    require(cycle_state in range(N_CYCLE), f"Invalid cycle state index: {cycle_state}.")


@dataclass(frozen=True)
class ExposureParameters:
    k_C: float = 0.25
    k_P: float = 0.25
    eta_C: float = 1.0
    eta_P: float = 1.0
    D_C0: float = 0.0
    D_P0: float = 0.0
    nu_C: float = 0.85
    nu_P: float = 0.85
    burden_weights: np.ndarray = field(default_factory=lambda: np.array([0.65, 0.70, 0.70], dtype=float))
    proliferative_weights: np.ndarray = field(default_factory=lambda: np.array([0.95, 0.75], dtype=float))


@dataclass(frozen=True)
class StateLandscapeParameters:
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.15, -0.05, -0.25], dtype=float))
    gamma_M: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.05, 0.00, 0.05], dtype=float))
    gamma_C: np.ndarray = field(default_factory=lambda: np.array([1.10, 0.15, -0.20, -0.10], dtype=float))
    gamma_P: np.ndarray = field(default_factory=lambda: np.array([0.05, 1.05, -0.15, -0.10], dtype=float))
    eta_a: np.ndarray = field(default_factory=lambda: np.array([-0.15, -0.10, 1.15, -0.10], dtype=float))
    eta_m: np.ndarray = field(default_factory=lambda: np.array([-0.20, -0.15, -0.10, 1.10], dtype=float))
    xi_B: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.10, 0.06, 0.12], dtype=float))
    B_U: np.ndarray = field(default_factory=lambda: np.diag(np.array([0.85, 0.80, 0.75], dtype=float)))
    sigma_0: float = 0.12
    sigma_M: float = 0.05


@dataclass(frozen=True)
class StressSurvivalParameters:
    alpha_R: float = 0.05
    r_B: float = 0.18
    r_S: float = 0.45
    r_C: float = 0.35
    r_P: float = 0.35
    r_m: float = 0.20
    b_R: float = 0.90
    sigma_R: float = 0.12
    alpha_V: float = 0.40
    v_M: float = 0.45
    v_A: float = 0.32
    v_Q: float = 0.22
    v_R: float = 0.35
    v_C: float = 0.18
    v_P: float = 0.18
    v_a: float = 0.24
    b_V: float = 0.85
    sigma_V: float = 0.10


@dataclass(frozen=True)
class CycleTransitionParameters:
    qbar_G1S: float = 0.70
    qbar_G1Q: float = 0.14
    qbar_QG1: float = 0.42
    qbar_SG2M: float = 0.65
    beta_0: float = -0.90
    beta_P: float = 1.05
    beta_NO: float = 0.65
    beta_R: float = 0.45
    beta_V: float = 0.55
    beta_C: float = 0.75
    beta_Pg: float = 0.35
    gamma_0: float = -1.70
    gamma_M: float = 0.65
    gamma_R: float = 0.65
    gamma_m: float = 0.35
    gamma_V: float = 0.40
    delta_0: float = -0.95
    delta_P: float = 0.95
    delta_V: float = 0.50
    delta_NO: float = 0.55
    delta_R: float = 0.40
    delta_m: float = 0.25
    kappa_0: float = -0.35
    kappa_R: float = 0.45
    kappa_V: float = 0.40


@dataclass(frozen=True)
class TurnoverWindowParameters:
    eta_1: float = 2.4
    eta_2: float = 2.2
    r_L: float = 0.35
    r_U: float = 1.10


@dataclass(frozen=True)
class TurnoverSpeciesParameters:
    gain_ceiling: float
    loss_ceiling: float
    a0: float
    a_R: float
    a_prol: float
    a_C: float
    a_P: float
    b0: float
    b_R: float
    b_V: float
    b_C: float
    b_P: float


@dataclass(frozen=True)
class HazardParameters:
    lambda_div_ceiling: float = 0.62
    lambda_death_ceiling: float = 0.24
    theta_0: float = -0.15
    theta_P: float = 1.00
    theta_NO: float = 0.70
    theta_R: float = 0.50
    theta_V: float = 0.65
    B_star: float = 3.9
    chi_B: float = 0.10
    phi_0: float = -2.10
    phi_R: float = 0.75
    phi_V: float = 0.95
    phi_M: float = 0.15
    phi_B: float = 0.10
    chi_C: float = 0.55
    chi_P: float = 0.55
    omega_O_given_C: float = 0.45


@dataclass(frozen=True)
class DivisionParameters:
    lambda_amp_ceiling: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    c0: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cR: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cC: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cP: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    tau: float = 0.85
    delta: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9, 0.9], dtype=float))
    rho_U: float = 0.60
    rho_R: float = 0.55
    rho_V: float = 0.55
    Omega_U: np.ndarray = field(default_factory=lambda: np.diag(np.array([0.05, 0.05, 0.05], dtype=float)))
    sigma_R0: float = 0.10
    sigma_V0: float = 0.10
    zeta_0: float = -1.50
    zeta_R: float = 0.55
    zeta_M: float = 0.55
    zeta_a: float = 0.35
    zeta_m: float = 0.25


@dataclass(frozen=True)
class TransitionGeneratorParameters:
    base_edges: Dict[tuple[int, int], float] = field(
        default_factory=lambda: {
            (NPC, OPC): 0.25,
            (OPC, NPC): 0.25,
            (OPC, AC): 0.20,
            (AC, OPC): 0.20,
            (AC, MES): 0.22,
            (MES, AC): 0.22,
            (NPC, AC): 0.08,
            (AC, NPC): 0.08,
        }
    )
    gamma_edges: Dict[tuple[int, int], float] = field(
        default_factory=lambda: {
            (NPC, OPC): 1.0,
            (OPC, NPC): 1.0,
            (OPC, AC): 1.0,
            (AC, OPC): 1.0,
            (AC, MES): 1.0,
            (MES, AC): 1.0,
            (NPC, AC): 0.65,
            (AC, NPC): 0.65,
        }
    )


@dataclass(frozen=True)
class SimulationParameters:
    dt: float = 0.1
    t_max: float = 72.0
    record_interval: float = 1.0
    n_init: int = 80
    target_population_size: int | None = None
    max_pop_size: int = 5000
    random_seed: int = 42


@dataclass(frozen=True)
class ModelParameters:
    exposure: ExposureParameters = field(default_factory=ExposureParameters)
    landscape: StateLandscapeParameters = field(default_factory=StateLandscapeParameters)
    stress_survival: StressSurvivalParameters = field(default_factory=StressSurvivalParameters)
    cycle: CycleTransitionParameters = field(default_factory=CycleTransitionParameters)
    turnover_window: TurnoverWindowParameters = field(default_factory=TurnoverWindowParameters)
    turnover: Dict[str, TurnoverSpeciesParameters] = field(
        default_factory=lambda: {
            "MYC": TurnoverSpeciesParameters(0.26, 0.12, -1.10, 0.90, 0.65, 0.10, 0.10, -1.25, 0.45, 0.55, 0.18, 0.18),
            "CDK4": TurnoverSpeciesParameters(0.23, 0.12, -1.15, 0.85, 0.60, 0.16, 0.08, -1.25, 0.45, 0.52, 0.22, 0.14),
            "PDGFRA": TurnoverSpeciesParameters(0.23, 0.12, -1.15, 0.85, 0.60, 0.08, 0.16, -1.25, 0.45, 0.52, 0.14, 0.22),
        }
    )
    hazard: HazardParameters = field(default_factory=HazardParameters)
    division: DivisionParameters = field(default_factory=DivisionParameters)
    generator: TransitionGeneratorParameters = field(default_factory=TransitionGeneratorParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)


PARAMS = ModelParameters()


DEFAULT_INPUT_SCHEDULES: Dict[str, Callable[[float], float]] = {
    "u_C": lambda _t: 0.0,
    "u_P": lambda _t: 0.0,
    "a": lambda _t: 0.0,
    "m": lambda _t: 0.0,
}


def sample_initial_cycle_state(rng: np.random.Generator) -> int:
    return int(rng.choice([Q, G1, S, G2M], p=[0.15, 0.55, 0.20, 0.10]))


def sample_initial_copy_numbers(rng: np.random.Generator) -> np.ndarray:
    mean = np.array([5.5, 6.5, 6.0], dtype=float)
    copies = rng.poisson(mean).astype(int)
    validate_copy_vector(copies)
    return copies


def sample_initial_soft_state(rng: np.random.Generator) -> np.ndarray:
    composition = rng.dirichlet(np.array([3.0, 2.8, 1.6, 1.4], dtype=float))
    validate_simplex(composition)
    return composition


def sample_initial_age(rng: np.random.Generator) -> float:
    return float(rng.exponential(scale=2.0))
