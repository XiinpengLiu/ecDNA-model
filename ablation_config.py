"""
Ablation-specific configuration for T87 simulations.

Each ablation starts from ``config.DEFAULT_MODEL_PARAMETERS`` and changes only
the parameters needed to close the requested mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable

import numpy as np

import config as cfg


DEFAULT_ABLATION_SEEDS = (24, 1212, 20020823, 20020611)
DEFAULT_ABLATION_ENSEMBLE_ID = "ablation"
DEFAULT_ABLATION_CONDITION = "ctrl"

# Validation requires B_U to remain positive definite; the simulator only
# multiplies by B_U in the state update, so this is a config-level freeze.
NEAR_ZERO_STATE_DRIFT = 1e-12
STRICT_INITIAL_RESERVOIR_OVERDISPERSION_PHI = 1e6
STRICT_INITIAL_STATE_COPY_MULTIPLIERS = np.ones((cfg.N_STATES, cfg.N_SPECIES), dtype=float)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    disabled_mechanisms: tuple[str, ...]
    build_parameters: Callable[[cfg.ModelParameters], cfg.ModelParameters]
    implementation_notes: tuple[str, ...] = ()


def _zero_turnover(params: cfg.ModelParameters) -> dict[str, cfg.TurnoverSpeciesParameters]:
    return {
        species: replace(species_params, gain_ceiling=0.0, loss_ceiling=0.0)
        for species, species_params in params.turnover.items()
    }


def _growth_only(base: cfg.ModelParameters) -> cfg.ModelParameters:
    exposure = replace(base.exposure, nu_C=0.0, nu_P=0.0)
    hazard = replace(
        base.hazard,
        theta_P=0.0,
        theta_NO=0.0,
        theta_R=0.0,
        theta_V=0.0,
        chi_B=0.0,
        phi_R=0.0,
        phi_V=0.0,
        phi_M=0.0,
        phi_B=0.0,
        chi_C=0.0,
        chi_P=0.0,
    )
    cycle = replace(
        base.cycle,
        beta_P=0.0,
        beta_NO=0.0,
        beta_R=0.0,
        beta_V=0.0,
        beta_C=0.0,
        beta_Pg=0.0,
        gamma_M=0.0,
        gamma_R=0.0,
        gamma_m=0.0,
        gamma_V=0.0,
        delta_P=0.0,
        delta_V=0.0,
        delta_NO=0.0,
        delta_R=0.0,
        delta_m=0.0,
        kappa_R=0.0,
        kappa_V=0.0,
    )
    stress_survival = replace(
        base.stress_survival,
        alpha_R=0.0,
        r_B=0.0,
        r_S=0.0,
        r_C=0.0,
        r_P=0.0,
        r_m=0.0,
        b_R=0.0,
        sigma_R=0.0,
        alpha_V=0.0,
        v_M=0.0,
        v_A=0.0,
        v_Q=0.0,
        v_R=0.0,
        v_C=0.0,
        v_P=0.0,
        v_a=0.0,
        b_V=0.0,
        sigma_V=0.0,
    )
    landscape = replace(
        base.landscape,
        alpha=np.zeros(cfg.N_STATES, dtype=float),
        gamma_M=np.zeros(cfg.N_STATES, dtype=float),
        gamma_C=np.zeros(cfg.N_STATES, dtype=float),
        gamma_P=np.zeros(cfg.N_STATES, dtype=float),
        eta_a=np.zeros(cfg.N_STATES, dtype=float),
        eta_m=np.zeros(cfg.N_STATES, dtype=float),
        xi_B=np.zeros(cfg.N_STATES, dtype=float),
        B_U=np.eye(cfg.LATENT_DIM, dtype=float) * NEAR_ZERO_STATE_DRIFT,
        sigma_0=0.0,
        sigma_M=0.0,
    )
    division = replace(
        base.division,
        lambda_amp_ceiling=np.zeros(cfg.N_SPECIES, dtype=float),
        tau=0.0,
        delta=np.zeros(cfg.N_SPECIES, dtype=float),
        rho_U=1.0,
        rho_R=1.0,
        rho_V=1.0,
        Omega_U=np.zeros((cfg.LATENT_DIM, cfg.LATENT_DIM), dtype=float),
        sigma_R0=0.0,
        sigma_V0=0.0,
    )
    return replace(
        base,
        exposure=exposure,
        hazard=hazard,
        cycle=cycle,
        stress_survival=stress_survival,
        landscape=landscape,
        turnover=_zero_turnover(base),
        division=division,
    )


def _no_copy_selection(base: cfg.ModelParameters) -> cfg.ModelParameters:
    hazard = replace(
        base.hazard,
        theta_P=0.0,
        theta_NO=0.0,
        chi_B=0.0,
        phi_B=0.0,
        phi_M=0.0,
        chi_C=0.0,
        chi_P=0.0,
    )
    cycle = replace(
        base.cycle,
        beta_P=0.0,
        beta_NO=0.0,
        gamma_M=0.0,
        delta_P=0.0,
        delta_NO=0.0,
    )
    stress_survival = replace(base.stress_survival, r_B=0.0, v_M=0.0, v_A=0.0)
    division = replace(base.division, zeta_M=0.0)
    return replace(base, hazard=hazard, cycle=cycle, stress_survival=stress_survival, division=division)


def _no_target_action(base: cfg.ModelParameters) -> cfg.ModelParameters:
    exposure = replace(base.exposure, nu_C=0.0, nu_P=0.0)
    hazard = replace(base.hazard, chi_C=0.0, chi_P=0.0)
    turnover = {
        species: replace(species_params, a_C=0.0, a_P=0.0, b_C=0.0, b_P=0.0)
        for species, species_params in base.turnover.items()
    }
    return replace(base, exposure=exposure, hazard=hazard, turnover=turnover)


def _no_turnover(base: cfg.ModelParameters) -> cfg.ModelParameters:
    return replace(base, turnover=_zero_turnover(base))


def _no_stochastic_inheritance(base: cfg.ModelParameters) -> cfg.ModelParameters:
    division = replace(
        base.division,
        lambda_amp_ceiling=np.zeros(cfg.N_SPECIES, dtype=float),
        tau=0.0,
        delta=np.ones(cfg.N_SPECIES, dtype=float),
    )
    return replace(base, division=division)


def _no_state_dynamics(base: cfg.ModelParameters) -> cfg.ModelParameters:
    landscape = replace(
        base.landscape,
        B_U=np.eye(cfg.LATENT_DIM, dtype=float) * NEAR_ZERO_STATE_DRIFT,
        sigma_0=0.0,
        sigma_M=0.0,
    )
    division = replace(base.division, rho_U=1.0, Omega_U=np.zeros((cfg.LATENT_DIM, cfg.LATENT_DIM), dtype=float))
    return replace(base, landscape=landscape, division=division)


def _no_initial_reservoir(base: cfg.ModelParameters) -> cfg.ModelParameters:
    return base


ABLATION_CONFIGS: dict[str, AblationConfig] = {
    "GROWTH_ONLY": AblationConfig(
        name="GROWTH_ONLY",
        disabled_mechanisms=(
            "copy-number fitness effects",
            "copy-number state-landscape effects",
            "state-dependent growth and death effects",
            "latent state drift and diffusion",
            "stress/survival dynamics",
            "drug exposure effects on growth and stress",
            "non-mitotic ecDNA turnover",
            "division-coupled amplification",
            "correlated co-segregation; binomial split remains in the simulator kernel",
        ),
        build_parameters=_growth_only,
    ),
    "NO_COPY_SELECTION": AblationConfig(
        name="NO_COPY_SELECTION",
        disabled_mechanisms=(
            "proliferative-signal effects on division and cycle entry",
            "burden fitness costs",
            "target-copy death interactions",
            "burden-driven stress survival",
            "state-mediated fitness relay downstream of copy-number state shifts",
        ),
        build_parameters=_no_copy_selection,
    ),
    "NO_TARGET_ACTION": AblationConfig(
        name="NO_TARGET_ACTION",
        disabled_mechanisms=(
            "CDK4 target-effective signaling attenuation",
            "PDGFRA target-effective signaling attenuation",
            "target-copy-dependent drug death interactions",
            "drug-specific ecDNA turnover remodeling",
        ),
        build_parameters=_no_target_action,
    ),
    "NO_TURNOVER": AblationConfig(
        name="NO_TURNOVER",
        disabled_mechanisms=("non-mitotic ecDNA gain/loss",),
        build_parameters=_no_turnover,
    ),
    "NO_STOCHASTIC_INHERITANCE": AblationConfig(
        name="NO_STOCHASTIC_INHERITANCE",
        disabled_mechanisms=(
            "division-coupled amplification",
            "correlated co-segregation",
        ),
        build_parameters=_no_stochastic_inheritance,
    ),
    "NO_STATE_DYNAMICS": AblationConfig(
        name="NO_STATE_DYNAMICS",
        disabled_mechanisms=(
            "latent state drift, approximated by near-zero positive-definite B_U",
            "MYC-dependent phenotypic volatility",
            "daughter latent-state reset noise",
        ),
        build_parameters=_no_state_dynamics,
        implementation_notes=(
            "Config-only near-frozen latent-state approximation; this does not implement an exact X_i(t)=X_i(0) simulator lock.",
        ),
    ),
    "NO_INITIAL_RESERVOIR": AblationConfig(
        name="NO_INITIAL_RESERVOIR",
        disabled_mechanisms=(
            "condition-specific initial state-copy multipliers",
            "state-specific initial copy enrichment",
            "overdispersed initial high-copy tail",
        ),
        build_parameters=_no_initial_reservoir,
        implementation_notes=(
            "Strict initialization ablation: uniform state-copy multipliers plus near-Poisson gamma-Poisson sampling; model parameters are unchanged.",
        ),
    ),
}

ABLATION_NAMES = tuple(ABLATION_CONFIGS)


def parse_ablation_names(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return ABLATION_NAMES
    names = tuple(token.strip().upper() for token in raw.split(",") if token.strip())
    cfg.require(bool(names), "At least one ablation name is required.")
    unknown = [name for name in names if name not in ABLATION_CONFIGS]
    cfg.require(not unknown, f"Unsupported ablation(s): {unknown}.")
    return tuple(dict.fromkeys(names))


def build_model_parameters(ablation_name: str, base: cfg.ModelParameters) -> cfg.ModelParameters:
    config = ABLATION_CONFIGS[ablation_name]
    params = config.build_parameters(base)
    cfg.validate_model_parameters(params)
    return params


def build_initialization_parameters(
    ablation_name: str,
    condition: str,
    *,
    ddpcr_path: str | Path,
    seed: int,
    rows_per_state: int,
) -> cfg.InitializationParameters:
    if ablation_name == "NO_INITIAL_RESERVOIR":
        return _build_t87_initialization_with_multipliers(
            condition,
            STRICT_INITIAL_STATE_COPY_MULTIPLIERS,
            ddpcr_path=ddpcr_path,
            seed=seed,
            rows_per_state=rows_per_state,
            overdispersion_phi=STRICT_INITIAL_RESERVOIR_OVERDISPERSION_PHI,
        )
    return cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=ddpcr_path,
        seed=seed,
        rows_per_state=rows_per_state,
    )


def metadata_for_ablation(ablation_name: str) -> dict[str, object]:
    config = ABLATION_CONFIGS[ablation_name]
    metadata: dict[str, object] = {
        "ablation_name": config.name,
        "disabled_mechanisms": list(config.disabled_mechanisms),
        "config_file": "ablation_config.py",
    }
    if config.implementation_notes:
        metadata["implementation_notes"] = list(config.implementation_notes)
    return metadata


def _build_t87_initialization_with_multipliers(
    condition: str,
    state_copy_multipliers: np.ndarray,
    *,
    ddpcr_path: str | Path,
    seed: int,
    rows_per_state: int,
    overdispersion_phi: float = 3.5,
) -> cfg.InitializationParameters:
    if condition not in cfg.T87_CONDITION_TREATMENTS:
        raise ValueError(f"Unsupported T87 condition: {condition}")
    cfg.require(rows_per_state > 0, "rows_per_state must be strictly positive.")
    cfg.require(overdispersion_phi > 0.0, "overdispersion_phi must be strictly positive.")

    mean_by_species = cfg._read_t87_week1_ddpcr_means(ddpcr_path, condition)
    multipliers = np.asarray(state_copy_multipliers, dtype=float).copy()
    weighted_means = cfg.T87_INITIAL_STATE_FRACTIONS @ multipliers
    multipliers = multipliers / weighted_means

    rng = np.random.default_rng(int(seed))
    distributions: dict[str, np.ndarray] = {}
    for state_idx, state_name in enumerate(cfg.STATE_NAMES):
        state_means = mean_by_species * multipliers[state_idx]
        matrix = np.zeros((int(rows_per_state), cfg.N_SPECIES), dtype=int)
        for species_idx, mean in enumerate(state_means):
            gamma_rates = rng.gamma(
                shape=float(overdispersion_phi),
                scale=float(mean) / float(overdispersion_phi),
                size=int(rows_per_state),
            )
            matrix[:, species_idx] = rng.poisson(gamma_rates).astype(int)
        distributions[state_name] = matrix

    initialization_kwargs = {
        "mode": cfg.EMPIRICAL_WEEK1,
        "parametric_copy_number_mean": mean_by_species.copy(),
        "parametric_state_dirichlet_alpha": cfg.T87_INITIAL_STATE_FRACTIONS * 100.0,
        "cycle_probabilities": cfg.T87_INITIAL_CYCLE_PROBABILITIES.copy(),
        "age_scale": 1.0,
        "empirical_flow_fractions": cfg.T87_INITIAL_STATE_FRACTIONS.copy(),
        "empirical_sorted_copy_distributions": distributions,
        "empirical_soft_state_concentration": 25.0,
    }
    initialization_fields = {field.name for field in fields(cfg.InitializationParameters)}
    if "exact_bulk_copy_number_mean" in initialization_fields:
        initialization_kwargs["exact_bulk_copy_number_mean"] = mean_by_species.copy()
    initialization = cfg.InitializationParameters(**initialization_kwargs)
    cfg.validate_initialization_parameters(initialization)
    return initialization
