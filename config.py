"""
Configuration and shared math utilities for the ecDNA model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
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

EMPIRICAL_WEEK1 = "empirical_week1"
PARAMETRIC = "parametric"

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


def validate_probability_vector(values: np.ndarray, *, name: str, expected_shape: tuple[int, ...]) -> None:
    values = np.asarray(values, dtype=float)
    require(values.shape == expected_shape, f"{name} must have shape {expected_shape}, got {values.shape}.")
    require(np.all(np.isfinite(values)), f"{name} must be finite.")
    require(np.all(values >= 0.0), f"{name} must be non-negative.")
    total = float(np.sum(values))
    require(abs(total - 1.0) <= 1e-8, f"{name} must sum to 1, got {total}.")


def _weekly_record_times() -> tuple[float, ...]:
    return tuple(float(week) for week in range(1, 6))


@dataclass(frozen=True)
class ExposureParameters:
    k_C: float = 0.35
    k_P: float = 0.35
    eta_C: float = 0.0025
    eta_P: float = 0.007
    D_C0: float = 0.0
    D_P0: float = 0.0
    nu_C: float = 0.22
    nu_P: float = 0.65
    burden_weights: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.50, 0.25], dtype=float)
    )
    proliferative_weights: np.ndarray = field(
        default_factory=lambda: np.array([0.42, 0.58], dtype=float)
    )


@dataclass(frozen=True)
class StateLandscapeParameters:
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.22, 0.08, 0.05], dtype=float))
    gamma_M: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.00, 0.00, 0.01], dtype=float))
    gamma_C: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.06, -0.06, -0.06], dtype=float))
    gamma_P: np.ndarray = field(default_factory=lambda: np.array([0.00, 0.12, -0.03, -0.03], dtype=float))
    eta_a: np.ndarray = field(default_factory=lambda: np.array([-0.15, -0.10, 1.15, -0.10], dtype=float))
    eta_m: np.ndarray = field(default_factory=lambda: np.array([-0.20, -0.15, -0.10, 1.10], dtype=float))
    xi_B: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.04, 0.05], dtype=float))
    B_U: np.ndarray = field(default_factory=lambda: np.diag(np.array([0.85, 0.80, 0.75], dtype=float)))
    sigma_0: float = 0.08
    sigma_M: float = 0.025


@dataclass(frozen=True)
class StressSurvivalParameters:
    alpha_R: float = 0.05
    r_B: float = 0.22
    r_S: float = 0.45
    r_C: float = 1.25
    r_P: float = 0.60
    r_m: float = 0.20
    b_R: float = 0.90
    sigma_R: float = 0.12
    alpha_V: float = 0.40
    v_M: float = 0.50
    v_A: float = 0.32
    v_Q: float = 0.22
    v_R: float = 0.35
    v_C: float = 0.35
    v_P: float = 0.23
    v_a: float = 0.24
    b_V: float = 0.85
    sigma_V: float = 0.10


@dataclass(frozen=True)
class CycleTransitionParameters:
    qbar_G1S: float = 1.60
    qbar_G1Q: float = 0.14
    qbar_QG1: float = 1.00
    qbar_SG2M: float = 4.00
    beta_0: float = -0.90
    beta_P: float = 2.00
    beta_NO: float = 0.85
    beta_R: float = 0.45
    beta_V: float = 0.55
    beta_C: float = 5.50
    beta_Pg: float = 1.70
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
    lambda_div_ceiling: float = 9.00
    lambda_death_ceiling: float = 1.40
    theta_0: float = -0.15
    theta_P: float = 2.30
    theta_NO: float = 1.30
    theta_R: float = 0.55
    theta_V: float = 1.00
    B_star: float = 3.60
    chi_B: float = 0.30
    phi_0: float = -5.15
    phi_R: float = 1.20
    phi_V: float = 1.10
    phi_M: float = 0.15
    phi_B: float = 0.10
    chi_C: float = 0.85
    chi_P: float = 0.55
    omega_O_given_C: float = 0.12
    min_division_age: float = 0.25
    age_gate_slope: float = 6.0


@dataclass(frozen=True)
class DivisionParameters:
    lambda_amp_ceiling: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    c0: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cR: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cC: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    cP: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    tau: float = 1.05
    delta: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))
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
    dt: float = 0.20
    time_unit: str = "week"
    record_times: tuple[float, ...] = field(default_factory=_weekly_record_times)
    t_max: float = 5.0
    n_init: int = 1200
    target_population_size: int | None = 10000
    max_pop_size: int = 10000
    random_seed: int = 20260504
    fitting_mode: bool = False
    record_full_snapshots: bool = False
    record_events: bool = True
    record_histograms: bool = True
    max_cells_saved_per_snapshot: int = 1000


@dataclass(frozen=True)
class ModelParameters:
    exposure: ExposureParameters = field(default_factory=ExposureParameters)
    landscape: StateLandscapeParameters = field(default_factory=StateLandscapeParameters)
    stress_survival: StressSurvivalParameters = field(default_factory=StressSurvivalParameters)
    cycle: CycleTransitionParameters = field(default_factory=CycleTransitionParameters)
    turnover_window: TurnoverWindowParameters = field(default_factory=TurnoverWindowParameters)
    turnover: Dict[str, TurnoverSpeciesParameters] = field(
        default_factory=lambda: {
            "MYC": TurnoverSpeciesParameters(0.12, 1.20, -1.10, 0.90, 0.65, 0.00, -0.02, -3.20, 0.45, 0.55, 0.05, 0.35),
            "CDK4": TurnoverSpeciesParameters(0.18, 1.10, -1.15, 0.85, 0.60, -0.05, -0.02, -3.25, 0.45, 0.52, 1.55, 0.15),
            "PDGFRA": TurnoverSpeciesParameters(0.18, 1.35, -1.15, 0.85, 0.60, 0.00, -0.05, -3.25, 0.45, 0.52, 0.08, 0.90),
        }
    )
    hazard: HazardParameters = field(default_factory=HazardParameters)
    division: DivisionParameters = field(default_factory=DivisionParameters)
    generator: TransitionGeneratorParameters = field(default_factory=TransitionGeneratorParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)


@dataclass(frozen=True)
class ObservationParameters:
    qpcdr_intercept: np.ndarray = field(default_factory=lambda: np.zeros(N_SPECIES, dtype=float))
    qpcdr_slope: np.ndarray = field(default_factory=lambda: np.ones(N_SPECIES, dtype=float))
    qpcdr_sigma: np.ndarray = field(default_factory=lambda: np.full(N_SPECIES, 0.25, dtype=float))
    ecTAG_detection_efficiency: np.ndarray = field(default_factory=lambda: np.ones(N_SPECIES, dtype=float))
    ecTAG_background: np.ndarray = field(default_factory=lambda: np.full(N_SPECIES, 0.10, dtype=float))
    ecTAG_overdispersion: np.ndarray = field(default_factory=lambda: np.full(N_SPECIES, 0.15, dtype=float))
    ecTAG_max_observed: int = 30
    flow_overdispersion: float = 0.0
    sort_purity_matrix: np.ndarray = field(default_factory=lambda: np.eye(N_STATES, dtype=float))
    count_overdispersion: float = 0.0


@dataclass(frozen=True)
class InitializationParameters:
    mode: str = PARAMETRIC
    parametric_copy_number_mean: np.ndarray = field(default_factory=lambda: np.array([114.0, 101.7, 107.1], dtype=float))
    parametric_state_dirichlet_alpha: np.ndarray = field(default_factory=lambda: np.array([33.0, 37.0, 14.0, 16.0], dtype=float))
    cycle_probabilities: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.58, 0.22, 0.08], dtype=float))
    age_scale: float = 1.0
    empirical_flow_fractions: np.ndarray | None = None
    empirical_sorted_copy_distributions: dict[str, np.ndarray] | None = None
    empirical_soft_state_concentration: float = 25.0


DEFAULT_MODEL_PARAMETERS = ModelParameters()
DEFAULT_OBSERVATION_PARAMETERS = ObservationParameters()
DEFAULT_INITIALIZATION_PARAMETERS = InitializationParameters()


DEFAULT_INPUT_SCHEDULES: Dict[str, Callable[[float], float]] = {
    "u_C": lambda _t: 0.0,
    "u_P": lambda _t: 0.0,
    "a": lambda _t: 0.0,
    "m": lambda _t: 0.0,
}


T87_CONDITION_TREATMENTS: dict[str, tuple[str, float]] = {
    "ctrl": ("vehicle", 0.0),
    "P10": ("Palbociclib", 10.0),
    "P50": ("Palbociclib", 50.0),
    "P250": ("Palbociclib", 250.0),
    "R20": ("Ripretinib", 20.0),
    "R100": ("Ripretinib", 100.0),
    "R500": ("Ripretinib", 500.0),
}

T87_INITIAL_STATE_FRACTIONS = np.array([0.33, 0.37, 0.14, 0.16], dtype=float)
T87_INITIAL_CYCLE_PROBABILITIES = np.array([0.12, 0.58, 0.22, 0.08], dtype=float)

T87_BASE_STATE_COPY_MULTIPLIERS = np.array(
    [
        [1.05, 1.15, 0.90],
        [1.00, 1.05, 1.16],
        [0.95, 0.80, 0.90],
        [0.95, 0.75, 0.90],
    ],
    dtype=float,
)

T87_CDK4I_STATE_COPY_MULTIPLIERS = np.array(
    [
        [1.05, 1.32, 0.90],
        [1.00, 1.08, 1.16],
        [0.95, 0.62, 0.90],
        [0.95, 0.58, 0.90],
    ],
    dtype=float,
)

T87_CONDITION_COPY_SCALERS: dict[str, np.ndarray] = {
    "ctrl": np.array([1.00, 1.00, 1.00], dtype=float),
    "P10": np.array([1.00, 1.24, 1.00], dtype=float),
    "P50": np.array([1.00, 1.18, 1.00], dtype=float),
    "P250": np.array([1.00, 1.30, 1.00], dtype=float),
    "R20": np.array([1.00, 1.00, 1.00], dtype=float),
    "R100": np.array([1.00, 1.00, 1.00], dtype=float),
    "R500": np.array([0.85, 1.00, 0.80], dtype=float),
}


def t87_input_schedules_for_condition(condition: str) -> dict[str, Callable[[float], float]]:
    """Return continuous CDK4i/PDGFRAi schedules for a T87 condition."""

    if condition not in T87_CONDITION_TREATMENTS:
        raise ValueError(f"Unsupported T87 condition: {condition}")
    drug, dose = T87_CONDITION_TREATMENTS[condition]
    return {
        "u_C": lambda _t, drug=drug, dose=dose: dose if drug == "Palbociclib" else 0.0,
        "u_P": lambda _t, drug=drug, dose=dose: dose if drug == "Ripretinib" else 0.0,
        "a": lambda _t: 0.0,
        "m": lambda _t: 0.0,
    }


def _read_t87_week1_ddpcr_means(ddpcr_path: str | Path, condition: str) -> np.ndarray:
    means_by_species: dict[str, float] = {}
    with Path(ddpcr_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["week"])) != 1:
                continue
            if str(row["condition"]) != condition:
                continue
            species = str(row["species"])
            if species in SPECIES:
                means_by_species[species] = float(row["ddpcr_copy_number"])
    missing = [species for species in SPECIES if species not in means_by_species]
    require(not missing, f"Missing week-1 ddPCR anchors for {condition}: {missing}.")
    return np.array([means_by_species[species] for species in SPECIES], dtype=float)


def build_t87_initialization_parameters(
    condition: str,
    *,
    ddpcr_path: str | Path = Path("raw") / "t87_drug_bulkfit" / "ddpcr.csv",
    seed: int = 20260504,
    rows_per_state: int = 8192,
    overdispersion_phi: float = 3.5,
) -> InitializationParameters:
    """Build the condition-specific T87 initial population.

    The initializer anchors each condition to week-1 ddPCR bulk means, then
    creates overdispersed state-specific copy pools. CDK4i low/intermediate
    conditions include a CDK4-high reservoir so the day56 CDK4i enrichment can
    arise from selection without making turnover the primary explanation.
    """

    if condition not in T87_CONDITION_TREATMENTS:
        raise ValueError(f"Unsupported T87 condition: {condition}")
    require(rows_per_state > 0, "rows_per_state must be strictly positive.")
    require(overdispersion_phi > 0.0, "overdispersion_phi must be strictly positive.")

    mean_by_species = _read_t87_week1_ddpcr_means(ddpcr_path, condition)
    mean_by_species = mean_by_species * T87_CONDITION_COPY_SCALERS[condition]

    multipliers = (
        T87_CDK4I_STATE_COPY_MULTIPLIERS
        if condition in {"P10", "P50"}
        else T87_BASE_STATE_COPY_MULTIPLIERS
    ).copy()
    weighted_means = T87_INITIAL_STATE_FRACTIONS @ multipliers
    multipliers = multipliers / weighted_means

    rng = np.random.default_rng(int(seed))
    distributions: dict[str, np.ndarray] = {}
    for state_idx, state_name in enumerate(STATE_NAMES):
        state_means = mean_by_species * multipliers[state_idx]
        matrix = np.zeros((int(rows_per_state), N_SPECIES), dtype=int)
        for species_idx, mean in enumerate(state_means):
            gamma_rates = rng.gamma(
                shape=float(overdispersion_phi),
                scale=float(mean) / float(overdispersion_phi),
                size=int(rows_per_state),
            )
            matrix[:, species_idx] = rng.poisson(gamma_rates).astype(int)
        distributions[state_name] = matrix

    initialization = InitializationParameters(
        mode=EMPIRICAL_WEEK1,
        parametric_copy_number_mean=mean_by_species.copy(),
        parametric_state_dirichlet_alpha=T87_INITIAL_STATE_FRACTIONS * 100.0,
        cycle_probabilities=T87_INITIAL_CYCLE_PROBABILITIES.copy(),
        age_scale=1.0,
        empirical_flow_fractions=T87_INITIAL_STATE_FRACTIONS.copy(),
        empirical_sorted_copy_distributions=distributions,
        empirical_soft_state_concentration=25.0,
    )
    validate_initialization_parameters(initialization)
    return initialization


def _validate_finite_vector(values: np.ndarray, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    require(values.shape == shape, f"{name} must have shape {shape}, got {values.shape}.")
    require(np.all(np.isfinite(values)), f"{name} must be finite.")
    return values


def validate_simulation_parameters(params: SimulationParameters) -> None:
    require(bool(params.time_unit), "Simulation time_unit must be non-empty.")
    require(params.dt > 0.0, "Simulation dt must be strictly positive.")
    require(params.t_max > 0.0, "Simulation t_max must be strictly positive.")
    require(params.n_init > 0, "Simulation n_init must be strictly positive.")
    require(params.max_pop_size > 0, "Simulation max_pop_size must be strictly positive.")
    record_times = np.asarray(params.record_times, dtype=float)
    require(record_times.ndim == 1 and record_times.size > 0, "Simulation record_times must be a non-empty 1D sequence.")
    require(np.all(np.isfinite(record_times)), "Simulation record_times must be finite.")
    require(np.all(record_times >= 0.0), "Simulation record_times must be non-negative.")
    require(np.all(np.diff(record_times) > 0.0), "Simulation record_times must be strictly increasing.")
    require(abs(float(record_times[-1]) - params.t_max) <= 1e-8, "Simulation t_max must equal the final record time.")
    if params.target_population_size is not None:
        require(params.target_population_size > 0, "Simulation target_population_size must be strictly positive.")
        require(
            params.target_population_size <= params.max_pop_size,
            "Simulation target_population_size cannot exceed max_pop_size.",
        )
    if params.fitting_mode:
        require(params.target_population_size is None, "target_population_size is forbidden when fitting_mode=True.")
    require(params.max_cells_saved_per_snapshot > 0, "Simulation max_cells_saved_per_snapshot must be strictly positive.")


def validate_model_parameters(params: ModelParameters) -> None:
    exposure = params.exposure
    _validate_finite_vector(exposure.burden_weights, shape=(N_SPECIES,), name="exposure.burden_weights")
    _validate_finite_vector(exposure.proliferative_weights, shape=(2,), name="exposure.proliferative_weights")
    require(np.all(exposure.burden_weights >= 0.0), "Exposure burden_weights must be non-negative.")
    require(np.all(exposure.proliferative_weights >= 0.0), "Exposure proliferative_weights must be non-negative.")
    validate_probability_vector(exposure.burden_weights, name="exposure.burden_weights", expected_shape=(N_SPECIES,))
    validate_probability_vector(
        exposure.proliferative_weights,
        name="exposure.proliferative_weights",
        expected_shape=(2,),
    )
    for value_name in ("k_C", "k_P", "eta_C", "eta_P", "D_C0", "D_P0", "nu_C", "nu_P"):
        value = float(getattr(exposure, value_name))
        require(np.isfinite(value) and value >= 0.0, f"Exposure parameter {value_name} must be finite and non-negative.")

    landscape = params.landscape
    for field_name in ("alpha", "gamma_M", "gamma_C", "gamma_P", "eta_a", "eta_m", "xi_B"):
        _validate_finite_vector(getattr(landscape, field_name), shape=(N_STATES,), name=f"landscape.{field_name}")
    B_U = _validate_finite_vector(landscape.B_U, shape=(LATENT_DIM, LATENT_DIM), name="landscape.B_U")
    eigenvalues = np.linalg.eigvalsh(B_U)
    require(np.all(eigenvalues > 0.0), "landscape.B_U must be positive definite.")
    require(landscape.sigma_0 >= 0.0, "landscape.sigma_0 must be non-negative.")
    require(landscape.sigma_M >= 0.0, "landscape.sigma_M must be non-negative.")

    stress_survival = params.stress_survival
    for field_name in (
        "alpha_R",
        "r_B",
        "r_S",
        "r_C",
        "r_P",
        "r_m",
        "b_R",
        "sigma_R",
        "alpha_V",
        "v_M",
        "v_A",
        "v_Q",
        "v_R",
        "v_C",
        "v_P",
        "v_a",
        "b_V",
        "sigma_V",
    ):
        value = float(getattr(stress_survival, field_name))
        require(np.isfinite(value), f"stress_survival.{field_name} must be finite.")
    require(stress_survival.b_R >= 0.0, "stress_survival.b_R must be non-negative.")
    require(stress_survival.sigma_R >= 0.0, "stress_survival.sigma_R must be non-negative.")
    require(stress_survival.b_V >= 0.0, "stress_survival.b_V must be non-negative.")
    require(stress_survival.sigma_V >= 0.0, "stress_survival.sigma_V must be non-negative.")

    cycle = params.cycle
    for field_name in (
        "qbar_G1S",
        "qbar_G1Q",
        "qbar_QG1",
        "qbar_SG2M",
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
        value = float(getattr(cycle, field_name))
        require(np.isfinite(value), f"cycle.{field_name} must be finite.")
    for field_name in ("qbar_G1S", "qbar_G1Q", "qbar_QG1", "qbar_SG2M"):
        require(float(getattr(cycle, field_name)) >= 0.0, f"cycle.{field_name} must be non-negative.")

    turnover_window = params.turnover_window
    require(turnover_window.eta_1 > 0.0, "turnover_window.eta_1 must be strictly positive.")
    require(turnover_window.eta_2 > 0.0, "turnover_window.eta_2 must be strictly positive.")
    require(turnover_window.r_L < turnover_window.r_U, "turnover_window.r_L must be smaller than r_U.")

    require(set(params.turnover.keys()) == set(SPECIES), "turnover parameters must be present for every species.")
    for species_name in SPECIES:
        species_params = params.turnover[species_name]
        for field_name in (
            "gain_ceiling",
            "loss_ceiling",
            "a0",
            "a_R",
            "a_prol",
            "a_C",
            "a_P",
            "b0",
            "b_R",
            "b_V",
            "b_C",
            "b_P",
        ):
            value = float(getattr(species_params, field_name))
            require(np.isfinite(value), f"turnover.{species_name}.{field_name} must be finite.")
        require(species_params.gain_ceiling >= 0.0, f"turnover.{species_name}.gain_ceiling must be non-negative.")
        require(species_params.loss_ceiling >= 0.0, f"turnover.{species_name}.loss_ceiling must be non-negative.")

    hazard = params.hazard
    for field_name in (
        "lambda_div_ceiling",
        "lambda_death_ceiling",
        "theta_0",
        "theta_P",
        "theta_NO",
        "theta_R",
        "theta_V",
        "B_star",
        "chi_B",
        "phi_0",
        "phi_R",
        "phi_V",
        "phi_M",
        "phi_B",
        "chi_C",
        "chi_P",
        "omega_O_given_C",
        "min_division_age",
        "age_gate_slope",
    ):
        value = float(getattr(hazard, field_name))
        require(np.isfinite(value), f"hazard.{field_name} must be finite.")
    require(hazard.lambda_div_ceiling >= 0.0, "hazard.lambda_div_ceiling must be non-negative.")
    require(hazard.lambda_death_ceiling >= 0.0, "hazard.lambda_death_ceiling must be non-negative.")
    require(hazard.chi_B >= 0.0, "hazard.chi_B must be non-negative.")
    require(hazard.B_star >= 0.0, "hazard.B_star must be non-negative.")
    require(hazard.min_division_age >= 0.0, "hazard.min_division_age must be non-negative.")
    require(hazard.age_gate_slope > 0.0, "hazard.age_gate_slope must be strictly positive.")
    require(0.0 <= hazard.omega_O_given_C <= 1.0, "hazard.omega_O_given_C must lie in [0, 1].")

    division = params.division
    for field_name in ("lambda_amp_ceiling", "c0", "cR", "cC", "cP", "delta"):
        _validate_finite_vector(getattr(division, field_name), shape=(N_SPECIES,), name=f"division.{field_name}")
    Omega_U = _validate_finite_vector(division.Omega_U, shape=(LATENT_DIM, LATENT_DIM), name="division.Omega_U")
    omega_eigs = np.linalg.eigvalsh(Omega_U)
    require(np.all(omega_eigs >= -1e-10), "division.Omega_U must be positive semidefinite.")
    require(np.all(division.lambda_amp_ceiling >= 0.0), "division.lambda_amp_ceiling must be non-negative.")
    require(division.tau >= 0.0, "division.tau must be non-negative.")
    require(division.sigma_R0 >= 0.0, "division.sigma_R0 must be non-negative.")
    require(division.sigma_V0 >= 0.0, "division.sigma_V0 must be non-negative.")
    for field_name in ("rho_U", "rho_R", "rho_V"):
        value = float(getattr(division, field_name))
        require(0.0 <= value <= 1.0, f"division.{field_name} must lie in [0, 1].")
    for field_name in ("zeta_0", "zeta_R", "zeta_M", "zeta_a", "zeta_m"):
        value = float(getattr(division, field_name))
        require(np.isfinite(value), f"division.{field_name} must be finite.")

    validate_simulation_parameters(params.simulation)


def validate_observation_parameters(params: ObservationParameters) -> None:
    for field_name in (
        "qpcdr_intercept",
        "qpcdr_slope",
        "qpcdr_sigma",
        "ecTAG_detection_efficiency",
        "ecTAG_background",
        "ecTAG_overdispersion",
    ):
        _validate_finite_vector(getattr(params, field_name), shape=(N_SPECIES,), name=f"observation.{field_name}")
    purity = _validate_finite_vector(params.sort_purity_matrix, shape=(N_STATES, N_STATES), name="observation.sort_purity_matrix")
    require(np.all(purity >= 0.0), "observation.sort_purity_matrix must be non-negative.")
    column_sums = np.sum(purity, axis=0)
    require(np.allclose(column_sums, 1.0, atol=1e-8), "Each column of sort_purity_matrix must sum to 1.")
    require(np.all(params.qpcdr_sigma >= 0.0), "observation.qpcdr_sigma must be non-negative.")
    require(np.all(params.ecTAG_detection_efficiency >= 0.0), "observation.ecTAG_detection_efficiency must be non-negative.")
    require(np.all(params.ecTAG_background >= 0.0), "observation.ecTAG_background must be non-negative.")
    require(np.all(params.ecTAG_overdispersion >= 0.0), "observation.ecTAG_overdispersion must be non-negative.")
    require(params.ecTAG_max_observed > 0, "observation.ecTAG_max_observed must be strictly positive.")
    require(params.flow_overdispersion >= 0.0, "observation.flow_overdispersion must be non-negative.")
    require(params.count_overdispersion >= 0.0, "observation.count_overdispersion must be non-negative.")


def validate_initialization_parameters(params: InitializationParameters) -> None:
    require(params.mode in (PARAMETRIC, EMPIRICAL_WEEK1), f"Unsupported initialization mode: {params.mode}.")
    _validate_finite_vector(
        params.parametric_copy_number_mean,
        shape=(N_SPECIES,),
        name="initialization.parametric_copy_number_mean",
    )
    _validate_finite_vector(
        params.parametric_state_dirichlet_alpha,
        shape=(N_STATES,),
        name="initialization.parametric_state_dirichlet_alpha",
    )
    validate_probability_vector(
        params.cycle_probabilities,
        name="initialization.cycle_probabilities",
        expected_shape=(N_CYCLE,),
    )
    require(np.all(params.parametric_copy_number_mean > 0.0), "parametric_copy_number_mean must be strictly positive.")
    require(
        np.all(params.parametric_state_dirichlet_alpha > 0.0),
        "parametric_state_dirichlet_alpha must be strictly positive.",
    )
    require(params.age_scale > 0.0, "initialization.age_scale must be strictly positive.")
    require(
        params.empirical_soft_state_concentration > 0.0,
        "initialization.empirical_soft_state_concentration must be strictly positive.",
    )

    if params.mode != EMPIRICAL_WEEK1:
        return

    require(params.empirical_flow_fractions is not None, "empirical_flow_fractions is required in empirical_week1 mode.")
    validate_probability_vector(
        np.asarray(params.empirical_flow_fractions, dtype=float),
        name="initialization.empirical_flow_fractions",
        expected_shape=(N_STATES,),
    )
    require(
        params.empirical_sorted_copy_distributions is not None,
        "empirical_sorted_copy_distributions is required in empirical_week1 mode.",
    )
    require(
        set(params.empirical_sorted_copy_distributions.keys()) == set(STATE_NAMES),
        "empirical_sorted_copy_distributions must contain every state gate.",
    )
    for state_name, values in params.empirical_sorted_copy_distributions.items():
        matrix = np.asarray(values)
        require(matrix.ndim == 2 and matrix.shape[1] == N_SPECIES, f"{state_name} empirical copy matrix must have shape (n, {N_SPECIES}).")
        require(matrix.shape[0] > 0, f"{state_name} empirical copy matrix must be non-empty.")
        require(np.issubdtype(matrix.dtype, np.integer), f"{state_name} empirical copy matrix must be integer-valued.")
        require(np.all(matrix >= 0), f"{state_name} empirical copy matrix must be non-negative.")


def sample_initial_cycle_state(rng: np.random.Generator, initialization: InitializationParameters) -> int:
    return int(rng.choice(np.arange(N_CYCLE), p=np.asarray(initialization.cycle_probabilities, dtype=float)))


def sample_initial_copy_numbers(rng: np.random.Generator, initialization: InitializationParameters, gate_index: int | None = None) -> np.ndarray:
    if initialization.mode == EMPIRICAL_WEEK1:
        require(gate_index is not None, "empirical_week1 initialization requires a sampled gate index.")
        require(initialization.empirical_sorted_copy_distributions is not None, "Missing empirical copy distributions.")
        gate_name = STATE_NAMES[gate_index]
        copy_pool = np.asarray(initialization.empirical_sorted_copy_distributions[gate_name], dtype=int)
        row_index = int(rng.integers(copy_pool.shape[0]))
        copies = copy_pool[row_index].astype(int, copy=True)
        validate_copy_vector(copies)
        return copies
    copies = rng.poisson(np.asarray(initialization.parametric_copy_number_mean, dtype=float)).astype(int)
    validate_copy_vector(copies)
    return copies


def sample_initial_soft_state(rng: np.random.Generator, initialization: InitializationParameters, gate_index: int | None = None) -> np.ndarray:
    if initialization.mode == EMPIRICAL_WEEK1:
        require(gate_index is not None, "empirical_week1 initialization requires a sampled gate index.")
        alpha = np.ones(N_STATES, dtype=float)
        alpha[gate_index] = float(initialization.empirical_soft_state_concentration)
        composition = rng.dirichlet(alpha)
    else:
        composition = rng.dirichlet(np.asarray(initialization.parametric_state_dirichlet_alpha, dtype=float))
    validate_simplex(composition)
    return composition


def sample_initial_age(rng: np.random.Generator, initialization: InitializationParameters) -> float:
    return float(rng.exponential(scale=initialization.age_scale))


def sample_initial_gate(rng: np.random.Generator, initialization: InitializationParameters) -> int | None:
    if initialization.mode != EMPIRICAL_WEEK1:
        return None
    require(initialization.empirical_flow_fractions is not None, "Missing empirical flow fractions.")
    return int(rng.choice(np.arange(N_STATES), p=np.asarray(initialization.empirical_flow_fractions, dtype=float)))
