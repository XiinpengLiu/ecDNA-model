"""
Division kernel and daughter initialization for the ecDNA v4 model.
"""

from __future__ import annotations

import numpy as np

import config as cfg
import dynamics as dyn
from cell import Cell


class DivisionKernel:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def amplification_rate(self, parent: Cell, species_idx: int, context: dyn.ReplicateContext) -> float:
        params = cfg.PARAMS.division
        if parent.cycle_state not in (cfg.S, cfg.G2M):
            return 0.0
        ceiling = params.lambda_amp_ceiling[species_idx]
        if ceiling <= 0.0:
            return 0.0
        rate_logit = (
            params.c0[species_idx]
            + params.cR[species_idx] * dyn.stress_window(parent.stress)
            + params.cC[species_idx] * context.D_C
            + params.cP[species_idx] * context.D_P
        )
        return ceiling * cfg.sigmoid(rate_logit)

    def replicated_copies(self, parent: Cell, context: dyn.ReplicateContext) -> np.ndarray:
        amplified = np.zeros(cfg.N_SPECIES, dtype=int)
        for idx in range(cfg.N_SPECIES):
            amp_rate = self.amplification_rate(parent, idx, context)
            amplified[idx] = self.rng.poisson(amp_rate) if amp_rate > 0.0 else 0
        replicated = 2 * parent.copy_numbers + amplified
        cfg.validate_copy_vector(replicated)
        return replicated

    def segregate(self, replicated_copies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = cfg.PARAMS.division
        mitotic_shock = self.rng.normal(loc=0.0, scale=params.tau)
        daughter_one = np.zeros(cfg.N_SPECIES, dtype=int)
        daughter_two = np.zeros(cfg.N_SPECIES, dtype=int)
        for idx in range(cfg.N_SPECIES):
            segregation_probability = cfg.sigmoid(params.delta[idx] * mitotic_shock)
            daughter_one[idx] = self.rng.binomial(int(replicated_copies[idx]), float(segregation_probability))
            daughter_two[idx] = int(replicated_copies[idx] - daughter_one[idx])
        cfg.validate_copy_vector(daughter_one)
        cfg.validate_copy_vector(daughter_two)
        return daughter_one, daughter_two

    def initialize_daughter(self, parent: Cell, daughter_copies: np.ndarray, context: dyn.ReplicateContext) -> Cell:
        params = cfg.PARAMS.division
        draft = Cell(
            cycle_state=cfg.G1,
            copy_numbers=daughter_copies.copy(),
            latent_state=np.zeros(cfg.LATENT_DIM, dtype=float),
            soft_state=np.full(cfg.N_STATES, 1.0 / cfg.N_STATES, dtype=float),
            stress=0.0,
            survival=0.0,
            age=0.0,
            parent_id=parent.cell_id,
        )
        derived = dyn.compute_derived_quantities(draft, context)
        latent_noise = self.rng.multivariate_normal(mean=np.zeros(cfg.LATENT_DIM), cov=params.Omega_U)
        draft.latent_state = params.rho_U * parent.latent_state + (1.0 - params.rho_U) * derived.target_latent + latent_noise
        draft.soft_state = cfg.inverse_ilr(draft.latent_state)

        updated_derived = dyn.compute_derived_quantities(draft, context)
        stress_mean = dyn.compute_stress_attractor(draft, updated_derived, context, cycle_state=cfg.G1)
        draft.stress = params.rho_R * parent.stress + (1.0 - params.rho_R) * stress_mean + params.sigma_R0 * self.rng.normal()

        survival_mean = dyn.compute_survival_attractor(draft, updated_derived, context, cycle_state=cfg.G1)
        draft.survival = params.rho_V * parent.survival + (1.0 - params.rho_V) * survival_mean + params.sigma_V0 * self.rng.normal()

        p_quiescent = cfg.sigmoid(
            params.zeta_0
            + params.zeta_R * parent.stress
            + params.zeta_M * parent.soft_state[cfg.MES]
            + params.zeta_a * context.astrocytic_cue
            + params.zeta_m * context.mesenchymal_cue
        )
        draft.cycle_state = cfg.Q if self.rng.random() < p_quiescent else cfg.G1
        draft.age = 0.0
        draft.validate()
        return draft

    def divide(self, parent: Cell, context: dyn.ReplicateContext) -> tuple[Cell, Cell]:
        replicated = self.replicated_copies(parent, context)
        daughter_one_copies, daughter_two_copies = self.segregate(replicated)
        daughter_one = self.initialize_daughter(parent, daughter_one_copies, context)
        daughter_two = self.initialize_daughter(parent, daughter_two_copies, context)
        return daughter_one, daughter_two
