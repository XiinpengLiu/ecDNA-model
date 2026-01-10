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

# Expression program X ∈ {0: basal, 1: EMT, 2: MYC, 3: stress, 4: persister}
EXPR_STATES = [0, 1, 2, 3, 4]
EXPR_NAMES = {0: "basal", 1: "EMT", 2: "MYC", 3: "stress", 4: "persister"}

# Number of ecDNA species
J_ECDNA = 1

# Truncation bounds K_max,j for each ecDNA species
K_MAX = np.array([100])  # shape: (J,)

# Phenomic state dimension P
P_DIM = 2


# 2. PHENOMIC DYNAMICS (OU Process Parameters)


# Attractor μ_{e,m} for each (env, cycle, sen, expr) combination
# Shape: (n_env, n_cycle, n_sen, n_expr, P)
# Simplified: use default attractors based on expression program
def get_mu(e, c, s, x):
    """State-dependent phenomic attractor."""
    base = np.array([0.0, 0.0])
    # Expression program shifts attractor
    expr_shift = {
        0: np.array([0.0, 0.0]),    # basal
        1: np.array([1.0, -0.5]),   # EMT
        2: np.array([0.5, 1.0]),    # MYC
        3: np.array([-0.5, 0.5]),   # stress
        4: np.array([-1.0, -1.0]),  # persister
    }
    # Senescence shifts
    sen_shift = {0: 0.0, 1: 0.2, 2: 0.5}
    return base + expr_shift[x] + np.array([sen_shift[s], 0])

# Mean-reversion matrix B_{e,m} (relaxation rate)
# For simplicity, use scalar * identity (isotropic relaxation)
B_RELAX_RATE = 0.5  # eigenvalue magnitude
def get_B(e, c, s, x):
    """State-dependent relaxation matrix."""
    return B_RELAX_RATE * np.eye(P_DIM)


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
    (0, 1): 0.01,  # basal -> EMT
    (0, 2): 0.02,  # basal -> MYC
    (0, 3): 0.005, # basal -> stress
    (1, 0): 0.01,  # EMT -> basal
    (2, 0): 0.015, # MYC -> basal
    (3, 0): 0.02,  # stress -> basal
    (3, 4): 0.01,  # stress -> persister
    (4, 3): 0.005, # persister -> stress
}


# 4. DIVISION AND DEATH HAZARDS


# Maximum hazard rates (for bounded sigmoid parameterization)
LAMBDA_DIV_MAX = 0.5   # max division rate per unit time
LAMBDA_DEATH_MAX = 2.0 # max death rate per unit time

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
DEATH_HAZARD_SEN_MULT = {0: 1.0, 1: 1.5, 2: 3.0}  # senescence multiplier

# Age-dependence for hazards (Gompertz-like or constant)
USE_AGE_DEPENDENT_HAZARD = True
AGE_HAZARD_SCALE = 0.1  # how strongly age affects hazard


# 5. ecDNA DYNAMICS PARAMETERS


# Inter-division gain/loss (optional, can be disabled)
ENABLE_INTERDIV_ECDNA = True

# Gain rate (per unit time)
MU_GAIN_BASE = 0.001  # baseline gain rate per copy

# Loss rate (per copy, per unit time)
MU_LOSS_BASE = 0.002  # baseline loss rate per copy

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
    x = rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15])  # mostly basal
    #k = rng.poisson(5, size=J_ECDNA)  # Poisson-distributed ecDNA
    #k = np.clip(k, 2, K_MAX)
    k = rng.integers(20, K_MAX + 1, size=J_ECDNA)  # Uniform distribution [2, K_MAX]
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
