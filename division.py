"""
ecDNA Copy-Number Kinetics Model - Division Kernel
ecDNA amplification, segregation, and daughter initialization.
"""

import numpy as np
from typing import Tuple
from cell import Cell
import config as cfg


def _drug_effect_hill(u: float, emax: float, ec50: float, hill_n: float, effect_type: str) -> float:
    """Compute drug effect using Hill/Emax form."""
    if u <= 0:
        return 1.0
    hill = (u ** hill_n) / (ec50 ** hill_n + u ** hill_n)
    if effect_type == "inhibit":
        return 1.0 - emax * hill
    else:  # activate
        return 1.0 + emax * hill


class DivisionKernel:
    """
    Division kernel K^{a,y}_{e,m,k} implementing:
    - State-dependent ecDNA amplification
    - Random segregation
    - Post-segregation copy loss
    - Daughter discrete state, phenotype, and age reset
    """
    
    def __init__(self, rng: np.random.Generator = None, drug_schedule: dict = None):
        self.rng = rng if rng is not None else np.random.default_rng(cfg.RANDOM_SEED)
        self.drug_schedule = drug_schedule or cfg.DRUG_SCHEDULE
    
    def _get_drug_conc(self, drug_name: str, t: float) -> float:
        """Get drug concentration at time t."""
        if drug_name in self.drug_schedule:
            return self.drug_schedule[drug_name](t)
        return 0.0
    
    
    #State-dependent ecDNA Amplification
    
    
    def sample_amplification(self, cell: Cell, j: int, t: float) -> int:
        """
        Sample amplification A_j ~ g^amp_{e,m,j}(· | k_j, y, a; u)
        
        Model: A_j ~ Poisson(λ_amp * k_j)
        Pre-segregation copy number: k̃_j = 2*k_j + A_j
        Drug modulation: ecDNA_destabilizer inhibits amplification
        """
        k_j = cell.k[j]
        
        # Base amplification rate
        lambda_amp = cfg.AMP_LAMBDA_PER_COPY * k_j
        
        # MYC expression increases amplification
        if cell.x == 2:
            lambda_amp *= 1.5
        
        # Drug modulation: ecDNA_destabilizer inhibits amplification (target: "amp")
        u = self._get_drug_conc("ecdna_destabilizer", t)
        if u > 0:
            drug = cfg.DRUGS["ecdna_destabilizer"]
            if "amp" in drug.targets:
                lambda_amp *= _drug_effect_hill(u, drug.emax, drug.ec50, drug.hill_n, drug.targets["amp"])
        
        # Sample extra copies
        A_j = self.rng.poisson(max(0, lambda_amp))
        
        return A_j
    
    def compute_pre_segregation(self, cell: Cell, t: float) -> np.ndarray:
        """
        Compute pre-segregation copy numbers for all species.
        k̃_j = 2*k_j + A_j (replication + amplification)
        """
        k_tilde = np.zeros(cfg.J_ECDNA, dtype=int)
        
        for j in range(cfg.J_ECDNA):
            A_j = self.sample_amplification(cell, j, t)
            k_tilde[j] = 2 * cell.k[j] + A_j
        
        return k_tilde
    
    
    # Random Segregation (Unbiased)
    
    
    def segregate(self, k_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random segregation with binomial distribution.
        K_{1j} | k̃_j ~ Binomial(k̃_j, 1/2)
        K_{2j} = k̃_j - K_{1j}
        """
        k1 = np.zeros(cfg.J_ECDNA, dtype=int)
        k2 = np.zeros(cfg.J_ECDNA, dtype=int)
        
        for j in range(cfg.J_ECDNA):
            k1[j] = self.rng.binomial(k_tilde[j], 0.5)
            k2[j] = k_tilde[j] - k1[j]
        
        return k1, k2
    
    
    # Post-segregation Copy Loss
    
    
    def apply_post_segregation_loss(self, k: np.ndarray, cell: Cell, t: float) -> np.ndarray:
        """
        Apply post-segregation loss.
        K*_{rj} | K_{rj} ~ Binomial(K_{rj}, 1 - ℓ_{e,m,j})
        
        ecDNA effect: High copy number causes physical hindrance during mitosis,
        leading to increased segregation errors and chromosome mis-segregation.
        Drug modulation: ecDNA_destabilizer activates loss
        """
        k_star = np.zeros(cfg.J_ECDNA, dtype=int)
        
        # Total ecDNA for segregation error calculation
        total_k = np.sum(k)
        
        for j in range(cfg.J_ECDNA):
            loss_prob = cfg.LOSS_PROB_POST_SEG
            
            # Stress/persister states may have higher loss
            if cell.x in [3, 4]:
                loss_prob *= 1.5
            
            # ecDNA-induced segregation errors (physical hindrance at mitosis)
            # High copy number causes chromosome bridges, lagging chromosomes
            k_above_seg = max(0, total_k - cfg.ECDNA_SEGREGATION_ERROR_THRESHOLD)
            if k_above_seg > 0:
                # Additional loss probability scales with excess copies
                seg_error_loss = cfg.ECDNA_SEGREGATION_ERROR_COEFF * k_above_seg
                loss_prob += seg_error_loss
            
            # Drug modulation: ecDNA_destabilizer activates loss (target: "loss")
            u = self._get_drug_conc("ecdna_destabilizer", t)
            if u > 0:
                drug = cfg.DRUGS["ecdna_destabilizer"]
                if "loss" in drug.targets:
                    # activation increases loss_prob
                    loss_mult = _drug_effect_hill(u, drug.emax, drug.ec50, drug.hill_n, drug.targets["loss"])
                    loss_prob = loss_prob * loss_mult
            
            # Cap loss probability at 0.95 to avoid complete elimination
            loss_prob = min(loss_prob, 0.95)
            
            # Each copy survives with probability (1 - loss_prob)
            k_star[j] = self.rng.binomial(k[j], 1.0 - loss_prob)
        
        # Truncate to K_max
        k_star = np.clip(k_star, 0, cfg.K_MAX)
        
        return k_star
    
    
    #Daughter Discrete State, Phenotype, and Age Reset
    
    
    def sample_daughter_cycle(self, parent_cell: Cell) -> int:
        """
        Sample daughter cycle phase.
        Typically resets to G1 (or small probability G0).
        """
        phases = list(cfg.DAUGHTER_CYCLE_PROBS.keys())
        probs = list(cfg.DAUGHTER_CYCLE_PROBS.values())
        return self.rng.choice(phases, p=probs)
    
    def sample_daughter_senescence(self, parent_cell: Cell, k_daughter: np.ndarray) -> int:
        """
        Sample daughter senescence status.
        Largely inherited, with small probability of progression.
        """
        s = parent_cell.s
        
        # High ecDNA may accelerate senescence
        if s < 2 and np.sum(k_daughter) > 50:
            if self.rng.random() < 0.1:
                s = min(s + 1, 2)
        
        return s
    
    def sample_daughter_expression(self, parent_cell: Cell, k_daughter: np.ndarray) -> int:
        """
        Sample daughter expression program.
        Partially inherited with rare reprogramming.
        """
        x = parent_cell.x
        
        # Small chance of reprogramming at division
        if self.rng.random() < 0.05:
            # Random switch with preference for basal
            if x != 0:
                x = 0 if self.rng.random() < 0.7 else x
            else:
                x = self.rng.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
        
        return x
    
    def sample_daughter_phenotype(self, parent_cell: Cell, m_daughter: Tuple[int, int, int]) -> np.ndarray:
        """
        Sample daughter phenotype.
        Y_r ~ H^{(e)}(· | m_r, k_r, y; u)
        
        Model: Y_daughter ~ N(Y_parent, σ²I) with pull toward new attractor.
        """
        c, s, x = m_daughter
        
        # Daughter inherits parent phenotype with noise
        y_daughter = parent_cell.y + self.rng.normal(0, cfg.DAUGHTER_Y_NOISE_STD, size=cfg.P_DIM)
        
        # Small pull toward new state's attractor
        mu_new = cfg.get_mu(parent_cell.e, c, s, x)
        y_daughter = 0.9 * y_daughter + 0.1 * mu_new
        
        return y_daughter
    
    
    # Main Division Method
    
    
    def divide(self, parent: Cell, t: float) -> Tuple[Cell, Cell]:
        """
        Execute division kernel to produce two daughter cells.
        
        Args:
            parent: Parent cell state
            t: Current time
            
        Returns:
            daughter1, daughter2: Two daughter cells
        """
        # Step 1: Amplification and pre-segregation
        k_tilde = self.compute_pre_segregation(parent, t)
        
        # Step 2: Binomial segregation
        k1, k2 = self.segregate(k_tilde)
        
        # Step 3: Post-segregation loss
        k1 = self.apply_post_segregation_loss(k1, parent, t)
        k2 = self.apply_post_segregation_loss(k2, parent, t)
        
        # Step 4: Sample daughter discrete states
        daughters = []
        for k_r in [k1, k2]:
            c_r = self.sample_daughter_cycle(parent)
            s_r = self.sample_daughter_senescence(parent, k_r)
            x_r = self.sample_daughter_expression(parent, k_r)
            
            # Sample phenotype
            y_r = self.sample_daughter_phenotype(parent, (c_r, s_r, x_r))
            
            # Create daughter cell (age reset to 0)
            daughter = Cell(
                e=parent.e,
                c=c_r,
                s=s_r,
                x=x_r,
                k=k_r.copy(),
                a=0.0,
                y=y_r,
                parent_id=parent.cell_id
            )
            daughters.append(daughter)
        
        return daughters[0], daughters[1]


#Random-daughter Marginal

def compute_sister_correlation(k1: np.ndarray, k2: np.ndarray) -> float:
    """
    Compute correlation between sister cells' ecDNA counts.
    Used for model validation against experimental data.
    """
    if len(k1) == 1:
        return 0.0 if (k1[0] + k2[0]) == 0 else 2 * min(k1[0], k2[0]) / (k1[0] + k2[0])
    
    # Multi-species correlation
    total1 = np.sum(k1)
    total2 = np.sum(k2)
    if total1 + total2 == 0:
        return 0.0
    return 2 * min(total1, total2) / (total1 + total2)
