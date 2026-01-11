"""
ecDNA Copy-Number Kinetics Model - PDMP Dynamics
Implements the flow (φ), jump intensities (λ), and transition kernel (Q).
"""

import numpy as np
from numba import jit
from typing import Tuple, Dict, List
from cell import Cell
import config as cfg



# Flow Between Events


def flow_age(a: float, delta: float) -> float:
    """
    Age evolution: dA/dt = 1
    A(t + Δ) = a + Δ
    """
    return a + delta


def flow_phenotype(y: np.ndarray, e: int, c: int, s: int, x: int,
                   k_total: int, delta: float, rng: np.random.Generator = None) -> np.ndarray:
    """
    Phenomic dynamics (OU with diffusion):
    dY = -B_{e,m}(Y - mu_{e,m}) dt + Sigma_{e,m} dW
    """
    if delta <= 0:
        return y

    if rng is None:
        rng = np.random.default_rng()

    mu = cfg.get_mu(e, c, s, x, k_total)
    B = cfg.get_B(e, c, s, x)
    Sigma = cfg.get_Sigma(e, c, s, x)

    # Extract per-dimension relaxation rates.
    if np.isscalar(B):
        beta = np.full(cfg.P_DIM, float(B))
    elif isinstance(B, np.ndarray) and B.ndim == 2 and B.shape[0] == B.shape[1]:
        b0 = B[0, 0]
        if np.allclose(B, np.eye(B.shape[0]) * b0):
            beta = np.full(B.shape[0], float(b0))
        else:
            beta = np.diagonal(B).copy()
    else:
        beta = np.full(cfg.P_DIM, cfg.B_RELAX_RATE)

    # Extract per-dimension diffusion.
    if np.isscalar(Sigma):
        sigma = np.full(cfg.P_DIM, float(Sigma))
    elif isinstance(Sigma, np.ndarray) and Sigma.ndim == 1:
        sigma = Sigma.copy()
    elif isinstance(Sigma, np.ndarray) and Sigma.ndim == 2:
        sigma = np.diagonal(Sigma).copy()
    else:
        sigma = np.zeros(cfg.P_DIM)

    beta_safe = np.maximum(beta, 1e-12)
    exp_term = np.exp(-beta_safe * delta)
    mean = mu + exp_term * (y - mu)
    var = (sigma ** 2) / (2.0 * beta_safe) * (1.0 - np.exp(-2.0 * beta_safe * delta))
    var = np.maximum(var, 0.0)
    noise = rng.normal(0.0, 1.0, size=mu.shape) * np.sqrt(var)
    return mean + noise


def apply_flow(cell: Cell, delta: float, rng: np.random.Generator = None) -> None:
    """
    Apply PDMP flow φ(z, Δ) to cell in-place.
    Updates age and phenotype; (E, M, K) remain constant.
    """
    cell.a = flow_age(cell.a, delta)
    k_total = cell.total_ecdna()
    cell.y = flow_phenotype(cell.y, cell.e, cell.c, cell.s, cell.x, k_total, delta, rng=rng)


def lazy_apply_flow(cell: Cell, target_time: float, rng: np.random.Generator = None) -> None:
    """
    Lazily apply PDMP flow to cell in-place.
    Only applies flow if target_time > cell.last_update_time.
    Updates cell.last_update_time after applying flow.
    
    Args:
        cell: Cell to update
        target_time: Target time to advance to
    """
    delta = target_time - cell.last_update_time
    if delta > 1e-12:  # Only apply if meaningful time has passed
        apply_flow(cell, delta, rng=rng)
        cell.last_update_time = target_time


def batch_lazy_apply_flow(cells: list, target_time: float, rng: np.random.Generator = None) -> None:
    """
    Lazily apply flow to all cells, advancing each to target_time.
    Used at record times when all cell states must be synchronized.
    
    Args:
        cells: List of Cell objects
        target_time: Target time to advance all cells to
    """
    for cell in cells:
        lazy_apply_flow(cell, target_time, rng=rng)



# Jump Channel Intensities


@jit(nopython=True, cache=True)
def sigmoid(x: float) -> float:
    """Sigmoid function σ(x) = 1/(1+exp(-x))."""
    # Manual clamp to keep numba nopython-friendly on scalars.
    if x < -500.0:
        x = -500.0
    elif x > 500.0:
        x = 500.0
    return 1.0 / (1.0 + np.exp(-x))


def ddr_hump(y_ddr: float) -> float:
    """Bounded hump for DDR-driven gain."""
    width = max(cfg.DDR_HUMP_WIDTH, 1e-6)
    return sigmoid((y_ddr - cfg.DDR_HUMP_THETA1) / width) - sigmoid((y_ddr - cfg.DDR_HUMP_THETA2) / width)


def drug_effect(u: float, drug: cfg.DrugParams, effect_type: str) -> float:
    """
    Compute drug modulator using Hill/Emax form.
    
    Inhibition: 1 - Emax * u^n / (EC50^n + u^n)
    Activation: 1 + Emax * u^n / (EC50^n + u^n)
    """
    if u <= 0:
        return 1.0
    
    hill = (u ** drug.hill_n) / (drug.ec50 ** drug.hill_n + u ** drug.hill_n)
    
    if effect_type == "inhibit":
        return 1.0 - drug.emax * hill
    else:  # activate
        return 1.0 + drug.emax * hill


class JumpIntensities:
    """
    Computes all jump channel intensities for a cell state.
    """
    
    def __init__(self, drug_schedule: Dict = None):
        self.drug_schedule = drug_schedule or cfg.DRUG_SCHEDULE
    
    def get_drug_conc(self, t: float) -> Dict[str, float]:
        """Get current drug concentrations."""
        return {name: schedule(t) for name, schedule in self.drug_schedule.items()}
    

    # CTMC Switching Rates

    def cycle_switch_rate(self, cell: Cell, c_new: int, t: float,
                          drug_conc: Dict[str, float] = None) -> float:
        """Cell-cycle switching rate q^cycle_{c→c'}."""
        if (cell.c, c_new) not in cfg.CYCLE_RATES:
            return 0.0
        
        base_rate = cfg.CYCLE_RATES[(cell.c, c_new)]
        
        # Drug modulation (e.g., CDK inhibitor blocks G1→S)
        drug_mod = 1.0
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("cell_cycle_inhibitor", 0.0)
        if u > 0 and cell.c == 1 and c_new == 2:  # G1→S
            drug_mod = drug_effect(u, cfg.DRUGS["cell_cycle_inhibitor"], "inhibit")
        
        return cfg.Q_MAX_CYCLE * sigmoid(np.log(base_rate / (1 - base_rate + 1e-10))) * drug_mod
    
    def sen_switch_rate(self, cell: Cell, s_new: int, t: float,
                        k_total: int = None) -> float:
        """Senescence switching rate q^sen_{s→s'}."""
        if (cell.s, s_new) not in cfg.SEN_RATES:
            return 0.0
        
        base_rate = cfg.SEN_RATES[(cell.s, s_new)]
        
        # ecDNA can accelerate senescence
        if k_total is None:
            k_total = cell.total_ecdna()
        k_effect = 1.0 + 0.01 * k_total
        y_ddr = cell.y[cfg.Y_DDR_IDX]
        eta = np.log(base_rate * k_effect + 1e-10) + cfg.SEN_DDR_EFFECT * y_ddr
        return cfg.Q_MAX_SEN * sigmoid(eta) 
    
    def expr_switch_rate(self, cell: Cell, x_new: int, t: float) -> float:
        """Expression program switching rate q^expr_{x→x'}."""
        if (cell.x, x_new) not in cfg.EXPR_RATES:
            return 0.0
        
        base_rate = cfg.EXPR_RATES[(cell.x, x_new)]
        return cfg.Q_MAX_EXPR * sigmoid(np.log(base_rate / (1 - base_rate + 1e-10)))
    
    def all_switch_rates(self, cell: Cell, t: float,
                         drug_conc: Dict[str, float] = None,
                         k_total: int = None) -> List[Tuple[str, dict, float]]:
        """Get all non-zero switching rates."""
        rates = []
        
        # Cycle switches
        for c_new in cfg.CYCLE_STATES:
            if c_new != cell.c:
                r = self.cycle_switch_rate(cell, c_new, t, drug_conc=drug_conc)
                if r > 0:
                    rates.append(("cycle", {"c_new": c_new}, r))
        
        # Senescence switches
        for s_new in cfg.SEN_STATES:
            if s_new != cell.s:
                r = self.sen_switch_rate(cell, s_new, t, k_total=k_total)
                if r > 0:
                    rates.append(("sen", {"s_new": s_new}, r))
        
        # Expression switches
        for x_new in cfg.EXPR_STATES:
            if x_new != cell.x:
                r = self.expr_switch_rate(cell, x_new, t)
                if r > 0:
                    rates.append(("expr", {"x_new": x_new}, r))
        
        return rates
    
    
    # Inter-division ecDNA Gain/Loss
    
    
    def ecdna_gain_rate(self, cell: Cell, j: int, t: float,
                        drug_conc: Dict[str, float] = None) -> float:
        """ecDNA gain rate μ^gain_{e,m,j}(k,y,a;u)."""
        if not cfg.ENABLE_INTERDIV_ECDNA:
            return 0.0
        
        y_ddr = cell.y[cfg.Y_DDR_IDX]
        k_j = cell.k[j]
        myc_flag = 1.0 if cell.x in cfg.MYC_STATES else 0.0
        eta = (
            cfg.GAIN_ETA_BASE
            + cfg.GAIN_ETA_K * np.log1p(k_j)
            + cfg.GAIN_ETA_MYC * myc_flag
            + cfg.GAIN_ETA_DDR_HUMP * ddr_hump(y_ddr)
        )
        
        # Drug modulation (logit shift, bounded by MU_GAIN_MAX)
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("ecdna_destabilizer", 0.0)
        if u > 0:
            mod = drug_effect(u, cfg.DRUGS["ecdna_destabilizer"], "inhibit")
            eta += np.log(max(mod, 1e-12))
        
        return cfg.MU_GAIN_MAX * sigmoid(eta)
    
    def ecdna_loss_rate(self, cell: Cell, j: int, t: float,
                        drug_conc: Dict[str, float] = None) -> float:
        """ecDNA loss rate μ^loss_{e,m,j}(k,y,a;u) * k_j (per-copy loss)."""
        if not cfg.ENABLE_INTERDIV_ECDNA:
            return 0.0
        if cell.k[j] <= 1: # Prevent losing the last copy
            return 0.0
        
        y_ddr = cell.y[cfg.Y_DDR_IDX]
        y_surv = cell.y[cfg.Y_SURV_IDX]
        k_j = cell.k[j]
        eta = (
            cfg.LOSS_ETA_BASE
            + cfg.LOSS_ETA_K * np.log1p(k_j)
            + cfg.LOSS_ETA_DDR * y_ddr
            - cfg.LOSS_ETA_SURV * y_surv
        )
        
        # Drug modulation (logit shift, bounded by MU_LOSS_MAX)
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("ecdna_destabilizer", 0.0)
        if u > 0:
            mod = drug_effect(u, cfg.DRUGS["ecdna_destabilizer"], "activate")
            eta += np.log(max(mod, 1e-12))
        
        return cfg.MU_LOSS_MAX * sigmoid(eta)
    
    def all_ecdna_rates(self, cell: Cell, t: float,
                        drug_conc: Dict[str, float] = None) -> List[Tuple[str, dict, float]]:
        """Get all ecDNA gain/loss rates."""
        rates = []
        for j in range(cfg.J_ECDNA):
            # Gain
            r_gain = self.ecdna_gain_rate(cell, j, t, drug_conc=drug_conc)
            if r_gain > 0 and cell.k[j] < cfg.K_MAX[j]:
                rates.append(("ecdna_gain", {"j": j}, r_gain))
            
            # Loss
            r_loss = self.ecdna_loss_rate(cell, j, t, drug_conc=drug_conc)
            if r_loss > 0:
                rates.append(("ecdna_loss", {"j": j}, r_loss))
        
        return rates
    
    
    # Division and Death Hazards
    
    
    def division_hazard(self, cell: Cell, t: float,
                        drug_conc: Dict[str, float] = None,
                        k_total: int = None) -> float:
        """
        Division hazard λ^div_i(a, y; u).
        Only G2M phase can divide.
        
        ecDNA effect: Inverted-U (Gaussian) relationship
        - Optimal ecDNA copy number maximizes division rate
        - Too few or too many copies reduce division fitness
        """
        # Cycle-phase gating
        base_rate = cfg.DIV_HAZARD_BY_CYCLE.get(cell.c, 0.0)
        if base_rate == 0:
            return 0.0
        
        # Senescent cells don't divide
        if cell.s == 2:
            return 0.0
        
        # Age dependence (maturation)
        if cfg.USE_AGE_DEPENDENT_HAZARD:
            age_factor = 1.0 - np.exp(-cfg.AGE_HAZARD_SCALE * cell.a)
        else:
            age_factor = 1.0
        
        # ecDNA fitness effect: Inverted-U (Gaussian) function
        # k_effect = baseline + (peak - baseline) * exp(-((k - k_opt) / sigma)^2)
        if k_total is None:
            k_total = cell.total_ecdna()
        deviation = (k_total - cfg.ECDNA_OPTIMAL_COPIES) / cfg.ECDNA_FITNESS_WIDTH
        k_effect = cfg.ECDNA_FITNESS_BASELINE + \
                   (cfg.ECDNA_FITNESS_PEAK - cfg.ECDNA_FITNESS_BASELINE) * np.exp(-deviation ** 2)
        
        # Drug modulation
        drug_mod = 1.0
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("cell_cycle_inhibitor", 0.0)
        if u > 0:
            drug_mod = drug_effect(u, cfg.DRUGS["cell_cycle_inhibitor"], "inhibit")
        
        eta = np.log(base_rate * age_factor * k_effect * drug_mod + 1e-10)
        return cfg.LAMBDA_DIV_MAX * sigmoid(eta)
    
    def death_hazard(self, cell: Cell, t: float,
                     drug_conc: Dict[str, float] = None,
                     k_total: int = None) -> float:
        """
        Death hazard λ^death_i(a, y; u).
        
        ecDNA effect: Linear relationship (higher ecDNA -> higher death risk)
        This reflects genomic instability cost of carrying many ecDNA copies.
        """
        # Base death rate
        base_rate = cfg.DEATH_HAZARD_BASE
        
        # Senescence increases death risk
        sen_mult = cfg.DEATH_HAZARD_SEN_MULT.get(cell.s, 1.0)
        
        # ecDNA increases death risk (linear relationship)
        if k_total is None:
            k_total = cell.total_ecdna()
        k_effect = 1.0 + cfg.ECDNA_DEATH_EFFECT * k_total
        
        # Drug modulation (senolytic targets senescent cells)
        drug_mod = 1.0
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("senolytic", 0.0)
        if u > 0 and cell.s >= 1:  # senolytic targets pre-senescent and senescent
            drug_mod = drug_effect(u, cfg.DRUGS["senolytic"], "activate")
        
        y_ddr = cell.y[cfg.Y_DDR_IDX]
        y_surv = cell.y[cfg.Y_SURV_IDX]
        beta_ddr = cfg.DEATH_BETA_DDR_BY_EXPR[cell.x]
        eta = (
            np.log(base_rate * sen_mult * k_effect * drug_mod + 1e-10)
            + beta_ddr * y_ddr
            - cfg.DEATH_BETA_SURV * y_surv
        )
        return cfg.LAMBDA_DEATH_MAX * sigmoid(eta)
    
    
    # Total Intensity
    
    
    def compute_all_rates(self, cell: Cell, t: float) -> Tuple[List[Tuple], float]:
        """
        Compute all channel rates and total intensity.
        
        Returns:
            channels: List of (channel_type, params, rate)
            total: Total intensity λ(z; t)
        """
        drug_conc = self.get_drug_conc(t)
        k_total = cell.total_ecdna()
        channels = []
        
        # Switching rates
        channels.extend(self.all_switch_rates(cell, t, drug_conc=drug_conc, k_total=k_total))
        
        # ecDNA rates
        channels.extend(self.all_ecdna_rates(cell, t, drug_conc=drug_conc))
        
        # Terminal events
        div_rate = self.division_hazard(cell, t, drug_conc=drug_conc, k_total=k_total)
        if div_rate > 0:
            channels.append(("division", {}, div_rate))
        
        death_rate = self.death_hazard(cell, t, drug_conc=drug_conc, k_total=k_total)
        if death_rate > 0:
            channels.append(("death", {}, death_rate))
        
        total = sum(r for _, _, r in channels)
        return channels, total
    
    def compute_dominating_bound(self) -> float:
        """
        Compute global upper bound r̄ for Ogata thinning.
        Sum of all channel maxima.
        """
        bound = 0.0
        
        # Switching channels
        bound += cfg.Q_MAX_CYCLE * len(cfg.CYCLE_RATES)
        bound += cfg.Q_MAX_SEN * len(cfg.SEN_RATES)
        bound += cfg.Q_MAX_EXPR * len(cfg.EXPR_RATES)
        
        # ecDNA channels
        if cfg.ENABLE_INTERDIV_ECDNA:
            for j in range(cfg.J_ECDNA):
                bound += cfg.MU_GAIN_MAX  # gain
                bound += cfg.MU_LOSS_MAX  # loss
        
        # Terminal events
        bound += cfg.LAMBDA_DIV_MAX
        bound += cfg.LAMBDA_DEATH_MAX
        
        return bound



# Transition Kernel (non-branching channels)


def apply_transition(cell: Cell, channel_type: str, params: dict) -> None:
    """
    Apply state transition for non-branching channels.
    Modifies cell in-place.
    """
    if channel_type == "cycle":
        cell.c = params["c_new"]
    elif channel_type == "sen":
        cell.s = params["s_new"]
    elif channel_type == "expr":
        cell.x = params["x_new"]
    elif channel_type == "ecdna_gain":
        j = params["j"]
        cell.k[j] = min(cell.k[j] + 1, cfg.K_MAX[j])
    elif channel_type == "ecdna_loss":
        j = params["j"]
        cell.k[j] = max(cell.k[j] - 1, 0)
    # division and death handled separately
