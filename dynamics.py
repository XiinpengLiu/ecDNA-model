"""
ecDNA Copy-Number Kinetics Model - PDMP Dynamics
=================================================
Implements the flow (φ), jump intensities (λ), and transition kernel (Q).
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Dict, List
from cell import Cell
import config as cfg


# =============================================================================
# SECTION 4.1: Deterministic Flow Between Events
# =============================================================================

def flow_age(a: float, delta: float) -> float:
    """
    Age evolution: dA/dt = 1
    A(t + Δ) = a + Δ
    """
    return a + delta


def flow_phenotype(y: np.ndarray, e: int, c: int, s: int, x: int, delta: float) -> np.ndarray:
    """
    Phenomic dynamics (OU mean-reversion ODE):
    dY/dt = -B_{e,m}(Y - μ_{e,m})
    
    Closed-form solution:
    Y(t+Δ) = μ_{e,m} + exp(-B_{e,m}·Δ)·(Y(t) - μ_{e,m})
    """
    mu = cfg.get_mu(e, c, s, x)
    B = cfg.get_B(e, c, s, x)

    # Fast path: B is scalar * I (current default), use closed-form scaling.
    if np.isscalar(B):
        scale = np.exp(-B * delta)
        return mu + scale * (y - mu)
    if isinstance(B, np.ndarray) and B.ndim == 2 and B.shape[0] == B.shape[1]:
        b0 = B[0, 0]
        if np.all(B == np.eye(B.shape[0]) * b0):
            scale = np.exp(-b0 * delta)
            return mu + scale * (y - mu)

    # Fallback: matrix exponential for general B
    exp_neg_B_delta = expm(-B * delta)
    return mu + exp_neg_B_delta @ (y - mu)


def apply_flow(cell: Cell, delta: float) -> None:
    """
    Apply deterministic PDMP flow φ(z, Δ) to cell in-place.
    Updates age and phenotype; (E, M, K) remain constant.
    """
    cell.a = flow_age(cell.a, delta)
    cell.y = flow_phenotype(cell.y, cell.e, cell.c, cell.s, cell.x, delta)


# =============================================================================
# SECTION 4.2: Jump Channel Intensities
# =============================================================================

def sigmoid(x: float) -> float:
    """Sigmoid function σ(x) = 1/(1+exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


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
    
    # -------------------------------------------------------------------------
    # Section 4.2.1: CTMC Switching Rates
    # -------------------------------------------------------------------------
    
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
        
        return cfg.Q_MAX_SEN * sigmoid(np.log(base_rate * k_effect)) 
    
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
    
    # -------------------------------------------------------------------------
    # Section 4.2.3: Inter-division ecDNA Gain/Loss
    # -------------------------------------------------------------------------
    
    def ecdna_gain_rate(self, cell: Cell, j: int, t: float,
                        drug_conc: Dict[str, float] = None) -> float:
        """ecDNA gain rate μ^gain_{e,m,j}(k,y,a;u)."""
        if not cfg.ENABLE_INTERDIV_ECDNA:
            return 0.0
        
        # Base rate proportional to existing copy number
        base_rate = cfg.MU_GAIN_BASE * (1 + cell.k[j])
        
        # MYC expression promotes gain
        if cell.x == 2:  # MYC program
            base_rate *= 1.5
        
        # Drug modulation
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("ecdna_destabilizer", 0.0)
        if u > 0:
            base_rate *= drug_effect(u, cfg.DRUGS["ecdna_destabilizer"], "inhibit")
        
        return base_rate
    
    def ecdna_loss_rate(self, cell: Cell, j: int, t: float,
                        drug_conc: Dict[str, float] = None) -> float:
        """ecDNA loss rate μ^loss_{e,m,j}(k,y,a;u) * k_j (per-copy loss)."""
        if not cfg.ENABLE_INTERDIV_ECDNA:
            return 0.0
        if cell.k[j] == 0:
            return 0.0
        
        # Per-copy loss rate times number of copies
        base_rate = cfg.MU_LOSS_BASE * cell.k[j]
        
        # Drug modulation
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("ecdna_destabilizer", 0.0)
        if u > 0:
            base_rate *= drug_effect(u, cfg.DRUGS["ecdna_destabilizer"], "activate")
        
        return base_rate
    
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
    
    # -------------------------------------------------------------------------
    # Section 4.2.4: Division and Death Hazards
    # -------------------------------------------------------------------------
    
    def division_hazard(self, cell: Cell, t: float,
                        drug_conc: Dict[str, float] = None,
                        k_total: int = None) -> float:
        """
        Division hazard λ^div_i(a, y; u).
        Only G2M phase can divide.
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
        
        # ecDNA fitness effect
        if k_total is None:
            k_total = cell.total_ecdna()
        k_effect = 1.0 + cfg.ECDNA_FITNESS_EFFECT * k_total
        
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
                     drug_conc: Dict[str, float] = None) -> float:
        """
        Death hazard λ^death_i(a, y; u).
        """
        # Base death rate
        base_rate = cfg.DEATH_HAZARD_BASE
        
        # Senescence increases death risk
        sen_mult = cfg.DEATH_HAZARD_SEN_MULT.get(cell.s, 1.0)
        
        # Drug modulation (senolytic targets senescent cells)
        drug_mod = 1.0
        if drug_conc is None:
            drug_conc = self.get_drug_conc(t)
        u = drug_conc.get("senolytic", 0.0)
        if u > 0 and cell.s >= 1:  # senolytic targets pre-senescent and senescent
            drug_mod = drug_effect(u, cfg.DRUGS["senolytic"], "activate")
        
        eta = np.log(base_rate * sen_mult * drug_mod + 1e-10)
        return cfg.LAMBDA_DEATH_MAX * sigmoid(eta)
    
    # -------------------------------------------------------------------------
    # Total Intensity
    # -------------------------------------------------------------------------
    
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
        
        death_rate = self.death_hazard(cell, t, drug_conc=drug_conc)
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
                bound += cfg.MU_GAIN_BASE * (1 + cfg.K_MAX[j]) * 2  # gain
                bound += cfg.MU_LOSS_BASE * cfg.K_MAX[j] * 2       # loss
        
        # Terminal events
        bound += cfg.LAMBDA_DIV_MAX
        bound += cfg.LAMBDA_DEATH_MAX
        
        return bound


# =============================================================================
# SECTION 4.4: Transition Kernel (non-branching channels)
# =============================================================================

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
