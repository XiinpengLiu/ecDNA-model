"""
Derived quantities, continuous dynamics, and event-rate calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass(frozen=True)
class ReplicateContext:
    time: float
    u_C: float
    u_P: float
    D_C: float
    D_P: float
    astrocytic_cue: float
    mesenchymal_cue: float


@dataclass(frozen=True)
class DerivedQuantities:
    log_copies: np.ndarray
    effective_signaling: np.ndarray
    burden: float
    proliferative_signal: float
    logits: np.ndarray
    target_composition: np.ndarray
    target_latent: np.ndarray
    diffusion_scale: float


def compute_log_copies(copy_numbers: np.ndarray) -> np.ndarray:
    return np.log1p(copy_numbers.astype(float))


def compute_effective_signaling(
    log_copies: np.ndarray,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> np.ndarray:
    exposure = params.exposure
    signaling = np.array(
        [
            log_copies[cfg.MYC],
            log_copies[cfg.CDK4] / (1.0 + exposure.nu_C * context.D_C),
            log_copies[cfg.PDGFRA] / (1.0 + exposure.nu_P * context.D_P),
        ],
        dtype=float,
    )
    cfg.require(np.all(np.isfinite(signaling)), "Effective signaling must be finite.")
    return signaling


def compute_total_burden(log_copies: np.ndarray, params: cfg.ModelParameters) -> float:
    burden = float(np.dot(params.exposure.burden_weights, log_copies))
    cfg.require(np.isfinite(burden), "Total ecDNA burden must be finite.")
    return burden


def compute_proliferative_signal(effective_signaling: np.ndarray, params: cfg.ModelParameters) -> float:
    proliferative = float(
        params.exposure.proliferative_weights[0] * effective_signaling[cfg.MYC]
        + params.exposure.proliferative_weights[1] * effective_signaling[cfg.CDK4]
    )
    cfg.require(np.isfinite(proliferative), "Proliferative signal must be finite.")
    return proliferative


def compute_state_logits(
    effective_signaling: np.ndarray,
    burden: float,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> np.ndarray:
    landscape = params.landscape
    logits = (
        landscape.alpha
        + landscape.gamma_M * effective_signaling[cfg.MYC]
        + landscape.gamma_C * effective_signaling[cfg.CDK4]
        + landscape.gamma_P * effective_signaling[cfg.PDGFRA]
        + landscape.eta_a * context.astrocytic_cue
        + landscape.eta_m * context.mesenchymal_cue
        - landscape.xi_B * burden
    )
    return np.asarray(logits, dtype=float)


def compute_target_composition(logits: np.ndarray) -> np.ndarray:
    return cfg.softmax(logits)


def compute_target_latent(target_composition: np.ndarray) -> np.ndarray:
    return cfg.ilr(target_composition)


def compute_diffusion_scale(log_copies: np.ndarray, params: cfg.ModelParameters) -> float:
    scale = params.landscape.sigma_0 + params.landscape.sigma_M * log_copies[cfg.MYC]
    cfg.require(scale >= 0.0, "Latent diffusion scale must be non-negative.")
    return float(scale)


def compute_derived_quantities(
    cell: "Cell",
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> DerivedQuantities:
    cached = cell.get_cached_derived_quantities(context, params)
    if cached is not None:
        return cached
    log_copies = compute_log_copies(cell.copy_numbers)
    signaling = compute_effective_signaling(log_copies, context, params)
    burden = compute_total_burden(log_copies, params)
    proliferative = compute_proliferative_signal(signaling, params)
    logits = compute_state_logits(signaling, burden, context, params)
    target_composition = compute_target_composition(logits)
    target_latent = compute_target_latent(target_composition)
    diffusion_scale = compute_diffusion_scale(log_copies, params)
    derived = DerivedQuantities(
        log_copies=log_copies,
        effective_signaling=signaling,
        burden=burden,
        proliferative_signal=proliferative,
        logits=logits,
        target_composition=target_composition,
        target_latent=target_latent,
        diffusion_scale=diffusion_scale,
    )
    cell.cache_derived_quantities(context, params, derived)
    return derived


def compute_stress_attractor(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
    cycle_state: int | None = None,
) -> float:
    stress_params = params.stress_survival
    current_cycle = cell.cycle_state if cycle_state is None else cycle_state
    in_replication = 1.0 if current_cycle in (cfg.S, cfg.G2M) else 0.0
    return (
        stress_params.alpha_R
        + stress_params.r_B * derived.burden
        + stress_params.r_S * in_replication
        + stress_params.r_C * context.D_C
        + stress_params.r_P * context.D_P
        + stress_params.r_m * context.mesenchymal_cue
    )


def compute_survival_attractor(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
    cycle_state: int | None = None,
) -> float:
    survival_params = params.stress_survival
    current_cycle = cell.cycle_state if cycle_state is None else cycle_state
    is_quiescent = 1.0 if current_cycle == cfg.Q else 0.0
    return (
        survival_params.alpha_V
        + survival_params.v_M * cell.soft_state[cfg.MES]
        + survival_params.v_A * cell.soft_state[cfg.AC]
        + survival_params.v_Q * is_quiescent
        - survival_params.v_R * cell.stress_score
        - survival_params.v_C * context.D_C
        - survival_params.v_P * context.D_P
        + survival_params.v_a * context.astrocytic_cue
    )


def update_continuous_state(
    cell: "Cell",
    context: ReplicateContext,
    duration: float,
    rng: np.random.Generator,
    params: cfg.ModelParameters,
) -> DerivedQuantities:
    cfg.require(duration >= 0.0, "Flow duration must be non-negative.")
    if duration == 0.0:
        return compute_derived_quantities(cell, context, params)

    derived_before = compute_derived_quantities(cell, context, params)
    latent_noise = rng.normal(size=cfg.LATENT_DIM)
    cell.latent_state = (
        cell.latent_state
        - params.landscape.B_U @ (cell.latent_state - derived_before.target_latent) * duration
        + derived_before.diffusion_scale * np.sqrt(duration) * latent_noise
    )
    cell.soft_state = cfg.inverse_ilr(cell.latent_state)
    cell.invalidate_derived_cache()

    derived_after_u = compute_derived_quantities(cell, context, params)
    stress_mean = compute_stress_attractor(cell, derived_after_u, context, params)
    cell.stress_score = (
        cell.stress_score
        - params.stress_survival.b_R * (cell.stress_score - stress_mean) * duration
        + params.stress_survival.sigma_R * np.sqrt(duration) * rng.normal()
    )

    survival_mean = compute_survival_attractor(cell, derived_after_u, context, params)
    cell.survival_score = (
        cell.survival_score
        - params.stress_survival.b_V * (cell.survival_score - survival_mean) * duration
        + params.stress_survival.sigma_V * np.sqrt(duration) * rng.normal()
    )
    cell.age += duration
    cfg.require(np.isfinite(cell.stress_score), "Stress score must remain finite after flow update.")
    cfg.require(np.isfinite(cell.survival_score), "Survival score must remain finite after flow update.")
    cfg.require(cell.age >= 0.0, "Cell age must remain non-negative after flow update.")
    return derived_after_u


def stress_window(stress_score: float, params: cfg.ModelParameters) -> float:
    turnover_window = params.turnover_window
    return float(
        cfg.sigmoid(turnover_window.eta_1 * (stress_score - turnover_window.r_L))
        - cfg.sigmoid(turnover_window.eta_2 * (stress_score - turnover_window.r_U))
    )


def compute_cycle_transition_rates(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> dict[str, float]:
    cycle = params.cycle
    rates: dict[str, float] = {}
    x_no = cell.soft_state[cfg.NPC] + cell.soft_state[cfg.OPC]
    if cell.cycle_state == cfg.G1:
        eta_g1s = (
            cycle.beta_0
            + cycle.beta_P * derived.proliferative_signal
            + cycle.beta_NO * x_no
            - cycle.beta_R * cell.stress_score
            + cycle.beta_V * cell.survival_score
            - cycle.beta_C * context.D_C
            - cycle.beta_Pg * context.D_P
        )
        eta_g1q = (
            cycle.gamma_0
            + cycle.gamma_M * cell.soft_state[cfg.MES]
            + cycle.gamma_R * cell.stress_score
            + cycle.gamma_m * context.mesenchymal_cue
            - cycle.gamma_V * cell.survival_score
        )
        rates["G1_to_S"] = cycle.qbar_G1S * cfg.sigmoid(eta_g1s)
        rates["G1_to_Q"] = cycle.qbar_G1Q * cfg.sigmoid(eta_g1q)
    elif cell.cycle_state == cfg.Q:
        eta_qg1 = (
            cycle.delta_0
            + cycle.delta_P * derived.proliferative_signal
            + cycle.delta_V * cell.survival_score
            + cycle.delta_NO * x_no
            - cycle.delta_R * cell.stress_score
            - cycle.delta_m * context.mesenchymal_cue
        )
        rates["Q_to_G1"] = cycle.qbar_QG1 * cfg.sigmoid(eta_qg1)
    elif cell.cycle_state == cfg.S:
        eta_sg2m = cycle.kappa_0 - cycle.kappa_R * cell.stress_score + cycle.kappa_V * cell.survival_score
        rates["S_to_G2M"] = cycle.qbar_SG2M * cfg.sigmoid(eta_sg2m)
    return rates


def compute_turnover_rates(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    window_value = stress_window(cell.stress_score, params)
    for species_name in cfg.SPECIES:
        species_idx = cfg.SPECIES_INDEX[species_name]
        species_params = params.turnover[species_name]
        if cell.cycle_state in (cfg.S, cfg.G2M):
            gain_eta = (
                species_params.a0
                + species_params.a_R * window_value
                + species_params.a_prol * derived.proliferative_signal
                + species_params.a_C * context.D_C
                + species_params.a_P * context.D_P
            )
            rates[f"gain_{species_name}"] = species_params.gain_ceiling * cfg.sigmoid(gain_eta)

        if cell.copy_numbers[species_idx] > 0:
            loss_eta = (
                species_params.b0
                + species_params.b_R * cell.stress_score
                - species_params.b_V * cell.survival_score
                + species_params.b_C * context.D_C
                + species_params.b_P * context.D_P
            )
            rates[f"loss_{species_name}"] = species_params.loss_ceiling * cfg.sigmoid(loss_eta)
    return rates


def burden_penalty(burden: float, params: cfg.ModelParameters) -> float:
    return params.hazard.chi_B * (burden - params.hazard.B_star) ** 2


def compute_division_hazard(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> float:
    if cell.cycle_state != cfg.G2M:
        return 0.0
    hazard = params.hazard
    x_no = cell.soft_state[cfg.NPC] + cell.soft_state[cfg.OPC]
    eta = (
        hazard.theta_0
        + hazard.theta_P * derived.proliferative_signal
        + hazard.theta_NO * x_no
        - hazard.theta_R * cell.stress_score
        - burden_penalty(derived.burden, params)
        + hazard.theta_V * cell.survival_score
    )
    age_gate = cfg.sigmoid(hazard.age_gate_slope * (cell.age - hazard.min_division_age))
    return float(hazard.lambda_div_ceiling * cfg.sigmoid(eta) * age_gate)


def compute_death_hazard(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> float:
    hazard = params.hazard
    W_C = cell.soft_state[cfg.NPC] + hazard.omega_O_given_C * cell.soft_state[cfg.OPC]
    W_P = cell.soft_state[cfg.OPC]
    eta = (
        hazard.phi_0
        + hazard.phi_R * cell.stress_score
        - hazard.phi_V * cell.survival_score
        + hazard.phi_M * cell.soft_state[cfg.MES]
        + hazard.phi_B * derived.burden
        + hazard.chi_C * context.D_C * derived.log_copies[cfg.CDK4] * W_C
        + hazard.chi_P * context.D_P * derived.log_copies[cfg.PDGFRA] * W_P
    )
    return float(hazard.lambda_death_ceiling * cfg.sigmoid(eta))


def compute_all_event_rates(
    cell: "Cell",
    derived: DerivedQuantities,
    context: ReplicateContext,
    params: cfg.ModelParameters,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    rates.update(compute_cycle_transition_rates(cell, derived, context, params))
    rates.update(compute_turnover_rates(cell, derived, context, params))
    rates["division"] = compute_division_hazard(cell, derived, context, params)
    rates["death"] = compute_death_hazard(cell, derived, context, params)
    return {name: float(rate) for name, rate in rates.items() if rate > 0.0}


def apply_nonterminal_event(cell: "Cell", event_name: str) -> None:
    if event_name == "G1_to_S":
        cell.cycle_state = cfg.S
    elif event_name == "G1_to_Q":
        cell.cycle_state = cfg.Q
    elif event_name == "Q_to_G1":
        cell.cycle_state = cfg.G1
    elif event_name == "S_to_G2M":
        cell.cycle_state = cfg.G2M
    elif event_name.startswith("gain_"):
        species = event_name.split("_", 1)[1]
        idx = cfg.SPECIES_INDEX[species]
        cell.copy_numbers[idx] += 1
    elif event_name.startswith("loss_"):
        species = event_name.split("_", 1)[1]
        idx = cfg.SPECIES_INDEX[species]
        cfg.require(cell.copy_numbers[idx] > 0, f"Cannot lose ecDNA from zero-copy species {species}.")
        cell.copy_numbers[idx] -= 1
    else:
        raise ValueError(f"Unsupported nonterminal event: {event_name}")
    cell.invalidate_derived_cache()
    cell.validate()


def compute_local_transition_generator(logits: np.ndarray, params: cfg.ModelParameters) -> np.ndarray:
    generator = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    for (source, target), base_rate in params.generator.base_edges.items():
        gamma = params.generator.gamma_edges[(source, target)]
        generator[source, target] = base_rate * np.exp(gamma * (logits[target] - logits[source]))
    for idx in range(cfg.N_STATES):
        generator[idx, idx] = -np.sum(generator[idx, :])
    return generator
