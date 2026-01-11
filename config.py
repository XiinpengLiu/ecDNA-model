"""
ecDNA Copy-Number Kinetics Model - Configuration Parameters
All tunable parameters are centralized here for convenient modification.
"""

import numpy as np


# 1. STATE SPACE DEFINITIONS

# Environment stages E ∈ {0: baseline, 1: treatment}
ENV_STATES = [0, 1]
ENV_NAMES = {0: "baseline", 1: "treatment"}

# Cell-cycle phases C ∈ {0: G0, 1: G1, 2: S, 3: G2M}
CYCLE_STATES = [0, 1, 2, 3]
CYCLE_NAMES = {0: "G0", 1: "G1", 2: "S", 3: "G2M"}

# Senescence status S ∈ {0: normal, 1: pre-senescent, 2: senescent}
SEN_STATES = [0, 1, 2]
SEN_NAMES = {0: "normal", 1: "pre-sen", 2: "senescent"}

# Expression program X {0: basal, 1: MYC, 2: TP53, 3: MYC_TP53}
EXPR_STATES = [0, 1, 2, 3]
EXPR_NAMES = {0: "basal", 1: "MYC", 2: "TP53", 3: "MYC_TP53"}

# Number of ecDNA species
J_ECDNA = 1

# Truncation bounds K_max,j for each ecDNA species
K_MAX = np.array([100])  # shape: (J,)

# Phenomic state dimension P
P_DIM = 2
# Phenotype axes
Y_DDR_IDX = 0
Y_SURV_IDX = 1

# Expression program groups
MYC_STATES = (1, 3)
TP53_INACTIVE_STATES = (2, 3)


# 2. PHENOMIC DYNAMICS (OU Process Parameters)


# Attractor μ_{e,m} for each (env, cycle, sen, expr) combination
# Shape: (n_env, n_cycle, n_sen, n_expr, P)
# Simplified: use default attractors based on expression program
MU_DDR_BASE = 0.0
MU_SURV_BASE = 0.0
MU_DDR_BY_CYCLE = np.array([0.0, 0.1, 0.3, 0.2])
MU_DDR_BY_SEN = np.array([0.0, 0.2, 0.5])
MU_DDR_BY_EXPR = np.array([0.0, 0.6, 0.1, 0.7])
MU_DDR_BY_ENV = np.array([0.0, 0.3])
MU_DDR_K_LOG = 0.09

MU_SURV_BY_EXPR = np.array([0.0, 0.1, 0.6, 0.7])
MU_SURV_BY_SEN = np.array([0.0, 0.2, 0.4])
MU_SURV_BY_ENV = np.array([0.0, -0.2])

def get_mu(e, c, s, x, k_total=0):
    """State-dependent phenomic attractor."""
    mu_ddr = (
        MU_DDR_BASE
        + MU_DDR_BY_CYCLE[c]
        + MU_DDR_BY_SEN[s]
        + MU_DDR_BY_EXPR[x]
        + MU_DDR_K_LOG * np.log1p(k_total)
        + MU_DDR_BY_ENV[e]
    )
    mu_surv = (
        MU_SURV_BASE
        + MU_SURV_BY_EXPR[x]
        - MU_SURV_BY_SEN[s]
        + MU_SURV_BY_ENV[e]
    )
    return np.array([mu_ddr, mu_surv])

# Mean-reversion matrix B_{e,m} (relaxation rate)
# For simplicity, use scalar * identity (isotropic relaxation)
B_RELAX_RATE = 0.7  # eigenvalue magnitude
def get_B(e, c, s, x):
    """State-dependent relaxation matrix."""
    return B_RELAX_RATE * np.eye(P_DIM)

# Diffusion strength (diagonal)
SIGMA_DIFFUSION = np.array([0.25, 0.2])
def get_Sigma(e, c, s, x):
    """State-dependent diffusion (diagonal)."""
    return SIGMA_DIFFUSION


# 3. CTMC SWITCHING RATES (Baseline, without drug)


# Maximum switching rates (used for bounded parameterization)
Q_MAX_CYCLE = 1.0      # cell-cycle transitions
Q_MAX_SEN = 0.1        # senescence transitions
Q_MAX_EXPR = 0.05      # expression program switches

# Cell-cycle transition baseline rates (G0 <-> G1 -> S -> G2M -> G1)
CYCLE_RATES = {
    (0, 1): 0.2,   # G0 -> G1
    (1, 0): 0.05,  # G1 -> G0 (quiescence entry)
    (1, 2): 0.3,   # G1 -> S
    (2, 3): 0.4,   # S -> G2M
    # G2M -> G1 handled by division
}

# Senescence progression rates (normal -> pre-sen -> senescent)
SEN_RATES = {
    (0, 1): 0.01,  # normal -> pre-senescent
    (1, 2): 0.05,  # pre-senescent -> senescent
}

# Expression program switch rates (sparse, biology-constrained)
EXPR_RATES = {
    (0, 1): 0.02,  # basal -> MYC
    (0, 2): 0.02,  # basal -> TP53
    (1, 3): 0.01,  # MYC -> MYC_TP53
    (2, 3): 0.01,  # TP53 -> MYC_TP53
}


# 4. DIVISION AND DEATH HAZARDS


# Maximum hazard rates (for bounded sigmoid parameterization)
LAMBDA_DIV_MAX = 0.5   # max division rate per unit time
LAMBDA_DEATH_MAX = 0.3 # max death rate per unit time

# Division hazard parameters (depends on cycle phase)
# Only G2M phase can divide
DIV_HAZARD_BY_CYCLE = {
    0: 0.0,   # G0: no division
    1: 0.0,   # G1: no division
    2: 0.0,   # S: no division
    3: 0.8,   # G2M: high division competence
}

# Death hazard parameters (baseline + senescence effect)
DEATH_HAZARD_BASE = 0.1
DEATH_HAZARD_SEN_MULT = {0: 1.0, 1: 1.5, 2: 5.0}  # senescence multiplier
DEATH_BETA_DDR_BY_EXPR = np.array([0.5, 0.7, 0.2, 0.3])
DEATH_BETA_SURV = 0.5

# Senescence DDR effect
SEN_DDR_EFFECT = 0.2

# Age-dependence for hazards (Gompertz-like or constant)
USE_AGE_DEPENDENT_HAZARD = False
AGE_HAZARD_SCALE = 0.1  # how strongly age affects hazard


# 5. ecDNA DYNAMICS PARAMETERS


# Inter-division gain/loss (optional, can be disabled)
ENABLE_INTERDIV_ECDNA = True

# Bounded gain/loss rates (per unit time)
MU_GAIN_MAX = 0.12
MU_LOSS_MAX = 0.2

# Gain rate logits
GAIN_ETA_BASE = -2.0
GAIN_ETA_K = 0.3
GAIN_ETA_MYC = 0.6
GAIN_ETA_DDR_HUMP = 2.6
DDR_HUMP_THETA1 = 0.0
DDR_HUMP_THETA2 = 1.1
DDR_HUMP_WIDTH = 0.4

# Loss rate logits
LOSS_ETA_BASE = -1.5
LOSS_ETA_K = 0.3
LOSS_ETA_DDR = 0.9
LOSS_ETA_SURV = 0.4

# ecDNA effect on fitness (optional)
# Division: inverted-U (Gaussian) relationship with ecDNA
ECDNA_OPTIMAL_COPIES = 10      # optimal ecDNA copy number for division (peak of inverted-U)
ECDNA_FITNESS_WIDTH = 20       # width parameter (sigma) of the Gaussian curve
ECDNA_FITNESS_PEAK = 1.5       # peak fitness multiplier at optimal ecDNA (>1 means enhanced division)
ECDNA_FITNESS_BASELINE = 0.5   # baseline fitness when ecDNA=0 or very high

# Death: linear relationship with ecDNA (higher ecDNA -> higher death risk)
ECDNA_DEATH_EFFECT = 0.01      # per-copy death rate increase


# 6. DIVISION KERNEL PARAMETERS


# Amplification distribution g^amp: A_j ~ Poisson(lambda_amp * k_j)
# or A_j ~ NegBin, etc.
AMP_LAMBDA_PER_COPY = 0.1  # mean extra copies per existing copy

# Post-segregation loss probability
LOSS_PROB_POST_SEG = 0.02  # probability each copy is lost after segregation

# Daughter phenotype initialization
# Y_daughter ~ N(Y_parent, sigma^2 * I)
DAUGHTER_Y_NOISE_STD = 0.1

# Daughter cycle phase reset probabilities (from G2M -> new phase)
DAUGHTER_CYCLE_PROBS = {1: 0.95, 0: 0.05}  # mostly G1, small chance G0


# 7. TREATMENT / DRUG PARAMETERS


# Drug effect parameterization: Hill/Emax form
# r_c(z;u) = r_{c,0}(z) * (1 - Emax * u^n / (EC50^n + u^n))

class DrugParams:
    """Parameters for a single drug affecting specific channels."""
    def __init__(self, name, emax, ec50, hill_n, targets):
        self.name = name
        self.emax = emax      # maximum effect (0-1 for inhibition)
        self.ec50 = ec50      # half-maximal concentration
        self.hill_n = hill_n  # Hill coefficient
        self.targets = targets  # dict: {channel: effect_type}

# Example drugs
DRUGS = {
    "cell_cycle_inhibitor": DrugParams(
        name="CDK_inhibitor",
        emax=0.9,
        ec50=1.0,
        hill_n=2,
        targets={"div": "inhibit", "cycle_G1S": "inhibit"}
    ),
    "senolytic": DrugParams(
        name="Senolytic",
        emax=0.8,
        ec50=0.5,
        hill_n=1.5,
        targets={"death_sen": "activate"}
    ),
    "ecdna_destabilizer": DrugParams(
        name="ecDNA_destab",
        emax=0.7,
        ec50=1.0,
        hill_n=2,
        targets={"amp": "inhibit", "loss": "activate"}
    ),
}

# Default drug concentration schedule
DRUG_SCHEDULE = {
    "cell_cycle_inhibitor": lambda t: 0.0,  # no drug by default
    "senolytic": lambda t: 0.0,
    "ecdna_destabilizer": lambda t: 0.0,
}


# 8. SIMULATION PARAMETERS


# Random seed for reproducibility
RANDOM_SEED = 42

# Simulation time
T_MAX = 100.0  # total simulation time

# Initial population
N_INIT = 100   # initial number of cells

# Initial state distribution
def sample_initial_state(rng):
    """Sample initial state for a single cell."""
    e = 0  # baseline environment
    c = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # mostly G1
    s = 0  # normal (not senescent)
    x = rng.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])  # mostly basal
    #k = rng.poisson(5, size=J_ECDNA)  # Poisson-distributed ecDNA
    #k = np.clip(k, 2, K_MAX)
    k = rng.integers(20, 101, size=J_ECDNA)  # Uniform distribution [2, K_MAX]
    a = rng.exponential(5)  # age from last division
    y = rng.normal(0, 0.5, size=P_DIM)  # phenotype
    return e, c, s, x, k, a, y

# Maximum population size (for memory management)
MAX_POP_SIZE = 100000

# Recording interval for time series
RECORD_INTERVAL = 1.0


# 9. DERIVED QUANTITIES (computed from above)


# Total number of discrete cell states M
N_CYCLE = len(CYCLE_STATES)
N_SEN = len(SEN_STATES)
N_EXPR = len(EXPR_STATES)
N_M = N_CYCLE * N_SEN * N_EXPR

# Total number of types (without ecDNA)
N_ENV = len(ENV_STATES)

def m_to_tuple(m_idx):
    """Convert flat M index to (c, s, x) tuple."""
    x = m_idx % N_EXPR
    s = (m_idx // N_EXPR) % N_SEN
    c = m_idx // (N_EXPR * N_SEN)
    return c, s, x

def tuple_to_m(c, s, x):
    """Convert (c, s, x) tuple to flat M index."""
    return c * (N_SEN * N_EXPR) + s * N_EXPR + x
