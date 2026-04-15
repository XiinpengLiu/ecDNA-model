"""
Derived quantities, continuous dynamics, and event-rate calculations for the ecDNA v4 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import v4_config as cfg


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


def update_exposure(current_exposure: float, dose: float, decay: float, conversion: float, dt: float) -> float:
    next_value = current_exposure + dt * (-decay * current_exposure + conversion * dose)
    cfg.require(next_value >= -1e-10, "Internal drug exposure must remain non-negative.")
    return max(0.0, next_value)


def compute_log_copies(copy_numbers: np.ndarray) -> np.ndarray:
    return np.log1p(copy_numbers.astype(float))


def compute_effective_signaling(log_copies: np.ndarray, context: ReplicateContext) -> np.ndarray:
    params = cfg.PARAMS.exposure
    signaling = np.array(
        [
            log_copies[cfg.MYC],
            log_copies[cfg.CDK4] / (1.0 + params.nu_C * context.D_C),
            log_copies[cfg.PDGFRA] / (1.0 + params.nu_P * context.D_P),
        ],
        dtype=float,
    )
    cfg.require(np.all(np.isfinite(signaling)), "Effective signaling must be finite.")
    return signaling


def compute_total_burden(log_copies: np.ndarray) -> float:
    burden = float(np.dot(cfg.PARAMS.exposure.burden_weights, log_copies))
    cfg.require(np.isfinite(burden), "Total ecDNA burden must be finite.")
    return burden


def compute_proliferative_signal(effective_signaling: np.ndarray) -> float:
    proliferative = float(
        cfg.PARAMS.exposure.proliferative_weights[0] * effective_signaling[cfg.MYC]
        + cfg.PARAMS.exposure.proliferative_weights[1] * effective_signaling[cfg.CDK4]
    )
    cfg.require(np.isfinite(proliferative), "Proliferative signal must be finite.")
    return proliferative


def compute_state_logits(effective_signaling: np.ndarray, burden: float, context: ReplicateContext) -> np.ndarray:
    landscape = cfg.PARAMS.landscape
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
    composition = cfg.softmax(logits)
    return composition


def compute_target_latent(target_composition: np.ndarray) -> np.ndarray:
    return cfg.ilr(target_composition)


def compute_diffusion_scale(log_copies: np.ndarray) -> float:
    scale = cfg.PARAMS.landscape.sigma_0 + cfg.PARAMS.landscape.sigma_M * log_copies[cfg.MYC]
    cfg.require(scale >= 0.0, "Latent diffusion scale must be non-negative.")
    return float(scale)


def compute_derived_quantities(cell: "Cell", context: ReplicateContext) -> DerivedQuantities:
    cached = cell.get_cached_derived_quantities(context)
    if cached is not None:
        return cached
    log_copies = compute_log_copies(cell.copy_numbers)
    signaling = compute_effective_signaling(log_copies, context)
    burden = compute_total_burden(log_copies)
    proliferative = compute_proliferative_signal(signaling)
    logits = compute_state_logits(signaling, burden, context)
    target_composition = compute_target_composition(logits)
    target_latent = compute_target_latent(target_composition)
    diffusion_scale = compute_diffusion_scale(log_copies)
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
    cell.cache_derived_quantities(context, derived)
    return derived


def compute_stress_attractor(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext, cycle_state: int | None = None) -> float:
    params = cfg.PARAMS.stress_survival
    current_cycle = cell.cycle_state if cycle_state is None else cycle_state
    in_replication = 1.0 if current_cycle in (cfg.S, cfg.G2M) else 0.0
    return (
        params.alpha_R
        + params.r_B * derived.burden
        + params.r_S * in_replication
        + params.r_C * context.D_C
        + params.r_P * context.D_P
        + params.r_m * context.mesenchymal_cue
    )


def compute_survival_attractor(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext, cycle_state: int | None = None) -> float:
    params = cfg.PARAMS.stress_survival
    current_cycle = cell.cycle_state if cycle_state is None else cycle_state
    is_quiescent = 1.0 if current_cycle == cfg.Q else 0.0
    return (
        params.alpha_V
        + params.v_M * cell.soft_state[cfg.MES]
        + params.v_A * cell.soft_state[cfg.AC]
        + params.v_Q * is_quiescent
        - params.v_R * cell.stress
        - params.v_C * context.D_C
        - params.v_P * context.D_P
        + params.v_a * context.astrocytic_cue
    )


def update_continuous_state(cell: "Cell", context: ReplicateContext, duration: float, rng: np.random.Generator) -> DerivedQuantities:
    cfg.require(duration >= 0.0, "Flow duration must be non-negative.")
    if duration == 0.0:
        return compute_derived_quantities(cell, context)

    derived_before = compute_derived_quantities(cell, context)
    landscape = cfg.PARAMS.landscape
    latent_noise = rng.normal(size=cfg.LATENT_DIM)
    cell.latent_state = (
        cell.latent_state
        - landscape.B_U @ (cell.latent_state - derived_before.target_latent) * duration
        + derived_before.diffusion_scale * np.sqrt(duration) * latent_noise
    )
    cell.soft_state = cfg.inverse_ilr(cell.latent_state)
    cell.invalidate_derived_cache()

    derived_after_u = compute_derived_quantities(cell, context)
    params = cfg.PARAMS.stress_survival
    stress_mean = compute_stress_attractor(cell, derived_after_u, context)
    cell.stress = cell.stress - params.b_R * (cell.stress - stress_mean) * duration + params.sigma_R * np.sqrt(duration) * rng.normal()

    survival_mean = compute_survival_attractor(cell, derived_after_u, context)
    cell.survival = cell.survival - params.b_V * (cell.survival - survival_mean) * duration + params.sigma_V * np.sqrt(duration) * rng.normal()
    cell.age += duration
    cfg.require(np.isfinite(cell.stress), "Stress must remain finite after flow update.")
    cfg.require(np.isfinite(cell.survival), "Survival reserve must remain finite after flow update.")
    cfg.require(cell.age >= 0.0, "Cell age must remain non-negative after flow update.")
    return derived_after_u


def stress_window(stress_value: float) -> float:
    params = cfg.PARAMS.turnover_window
    return float(
        cfg.sigmoid(params.eta_1 * (stress_value - params.r_L))
        - cfg.sigmoid(params.eta_2 * (stress_value - params.r_U))
    )


def compute_cycle_transition_rates(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext) -> dict[str, float]:
    params = cfg.PARAMS.cycle
    rates: dict[str, float] = {}
    x_no = cell.soft_state[cfg.NPC] + cell.soft_state[cfg.OPC]
    if cell.cycle_state == cfg.G1:
        eta_g1s = (
            params.beta_0
            + params.beta_P * derived.proliferative_signal
            + params.beta_NO * x_no
            - params.beta_R * cell.stress
            + params.beta_V * cell.survival
            - params.beta_C * context.D_C
            - params.beta_Pg * context.D_P
        )
        eta_g1q = (
            params.gamma_0
            + params.gamma_M * cell.soft_state[cfg.MES]
            + params.gamma_R * cell.stress
            + params.gamma_m * context.mesenchymal_cue
            - params.gamma_V * cell.survival
        )
        rates["G1_to_S"] = params.qbar_G1S * cfg.sigmoid(eta_g1s)
        rates["G1_to_Q"] = params.qbar_G1Q * cfg.sigmoid(eta_g1q)
    elif cell.cycle_state == cfg.Q:
        eta_qg1 = (
            params.delta_0
            + params.delta_P * derived.proliferative_signal
            + params.delta_V * cell.survival
            + params.delta_NO * x_no
            - params.delta_R * cell.stress
            - params.delta_m * context.mesenchymal_cue
        )
        rates["Q_to_G1"] = params.qbar_QG1 * cfg.sigmoid(eta_qg1)
    elif cell.cycle_state == cfg.S:
        eta_sg2m = params.kappa_0 - params.kappa_R * cell.stress + params.kappa_V * cell.survival
        rates["S_to_G2M"] = params.qbar_SG2M * cfg.sigmoid(eta_sg2m)
    return rates


def compute_turnover_rates(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext) -> dict[str, float]:
    rates: dict[str, float] = {}
    window_value = stress_window(cell.stress)
    for species_name in cfg.SPECIES:
        species_idx = cfg.SPECIES_INDEX[species_name]
        species_params = cfg.PARAMS.turnover[species_name]
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
                + species_params.b_R * cell.stress
                - species_params.b_V * cell.survival
                + species_params.b_C * context.D_C
                + species_params.b_P * context.D_P
            )
            rates[f"loss_{species_name}"] = species_params.loss_ceiling * cfg.sigmoid(loss_eta)
    return rates


def burden_penalty(burden: float) -> float:
    params = cfg.PARAMS.hazard
    return params.chi_B * (burden - params.B_star) ** 2


def compute_division_hazard(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext) -> float:
    if cell.cycle_state != cfg.G2M:
        return 0.0
    params = cfg.PARAMS.hazard
    x_no = cell.soft_state[cfg.NPC] + cell.soft_state[cfg.OPC]
    eta = (
        params.theta_0
        + params.theta_P * derived.proliferative_signal
        + params.theta_NO * x_no
        - params.theta_R * cell.stress
        - burden_penalty(derived.burden)
        + params.theta_V * cell.survival
    )
    return params.lambda_div_ceiling * cfg.sigmoid(eta)


def compute_death_hazard(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext) -> float:
    params = cfg.PARAMS.hazard
    W_C = cell.soft_state[cfg.NPC] + params.omega_O_given_C * cell.soft_state[cfg.OPC]
    W_P = cell.soft_state[cfg.OPC]
    eta = (
        params.phi_0
        + params.phi_R * cell.stress
        - params.phi_V * cell.survival
        + params.phi_M * cell.soft_state[cfg.MES]
        + params.phi_B * derived.burden
        + params.chi_C * context.D_C * derived.log_copies[cfg.CDK4] * W_C
        + params.chi_P * context.D_P * derived.log_copies[cfg.PDGFRA] * W_P
    )
    return params.lambda_death_ceiling * cfg.sigmoid(eta)


def compute_all_event_rates(cell: "Cell", derived: DerivedQuantities, context: ReplicateContext) -> dict[str, float]:
    rates = {}
    rates.update(compute_cycle_transition_rates(cell, derived, context))
    rates.update(compute_turnover_rates(cell, derived, context))
    rates["division"] = compute_division_hazard(cell, derived, context)
    rates["death"] = compute_death_hazard(cell, derived, context)
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


def compute_local_transition_generator(logits: np.ndarray) -> np.ndarray:
    params = cfg.PARAMS.generator
    generator = np.zeros((cfg.N_STATES, cfg.N_STATES), dtype=float)
    for (source, target), base_rate in params.base_edges.items():
        gamma = params.gamma_edges[(source, target)]
        generator[source, target] = base_rate * np.exp(gamma * (logits[target] - logits[source]))
    for idx in range(cfg.N_STATES):
        generator[idx, idx] = -np.sum(generator[idx, :])
    return generator
