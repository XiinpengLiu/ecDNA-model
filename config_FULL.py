from __future__ import annotations

import numpy as np

# =============================================================================
# Global defaults
# =============================================================================

# Cell state defaults (used when a field is not explicitly provided)
CELL_DEFAULTS = {
    "e": 0,                                 # environment index
    "m": 0,                                 # regulatory-state index
    "k": np.array([0, 0], dtype=int),        # ecDNA copy vector (length = n_ecdna)
    "y": np.array([0.0], dtype=float),       # phenotype vector (OU dimension)
    "a": 0.0,                                # age since last division
}

# Phenotype OU defaults (exactly propagated; diagonal OU)
OU_DEFAULTS = {
    "mean": np.zeros(1),                     # steady-state phenotype mean
    "rate": np.ones(1) * 0.5,                # mean-reversion rate (B>0)
    "diffusion": np.ones(1) * 0.2,           # diagonal diffusion (std)
}

# =============================================================================
# Model layer defaults (TOP-TIER B: biologically reasonable time scale)
# =============================================================================
# Design target:
#   - Start from 100 cells
#   - Each cell starts with 2 ecDNA species, k=[2,2]  (total K=4)
#   - Reach ~1e4 population size on a biologically reasonable time scale (tens of time units),
#     without explosive "instantaneous" growth.
#
# IMPORTANT:
#   - n_ecdna MUST match len(k_max) and any CellState.k length used in simulations.
#   - k_max acts as a hard truncation used for strict thinning bounds and to prevent runaway copies.

MODEL_DEFAULTS = {
    # Discrete state space sizes (kept identical across ALL examples)
    "n_env": 1,
    "n_reg": 1,
    "n_ecdna": 2,

    # ecDNA copy truncation (chosen so k=[2,2] is near the fitness optimum)
    # Higher k_max loosens the thinning bound and slows simulation (more proposals).
    "k_max": np.array([8, 8], dtype=int),

    # Event-channel maxima (bounded hazards; top-tier exact thinning)
    # These values are tuned for "B": gradual growth, reaching ~1e4 around t~50-80
    "div_rate_max": 0.30,
    "death_rate_max": 0.02,
    "gain_rate_max": 0.006,
    "loss_rate_max": 0.006,

    # Keep switching off for "full, consistent" examples (one environment, one regulatory state)
    "reg_switch_max": 0.0,
    "env_switch_max": 0.0,

    # Fitness curve (acts through logistic(fitness_score))
    # K = dot(weights, k). With weights=None => weights=ones => K = k1+k2.
    # We set K* = 5.0 to match initial k=[2,2].
    "fitness_k_star": 5.0,
    "fitness_alpha": 0.8,
    "fitness_beta": 0.06,
    "fitness_weights": None,

    # Division age gating: lambda_div ∝ (1 - exp(-age_effect_rate * a))
    "age_effect_rate": 0.2,

    # (unused when n_reg=1 / n_env=1, but kept for completeness)
    "reg_switch_slope": 0.5,
    "env_switch_bias": 0.0,

    # Division kernel (replication + amplification + segregation + post-loss)
    "amplification_scale": 0.05,         # Poisson mean = amplification_scale * k
    "division_copy_factor": 2,           # baseline replication (2*k)
    "segregation_prob": 0.5,             # unbiased segregation

    "post_segregation_loss_cap": 0.30,
    "post_segregation_loss_slope": 0.01,
    "post_segregation_loss_offset": 1.0,

    # Daughter phenotype inheritance noise
    "daughter_y_noise_std": 0.05,
}

# =============================================================================
# Simulation defaults
# =============================================================================

SIM_DEFAULTS = {
    "t_max": 100.0,
    "seed": 42,
    "max_cells": 500000,          # safeguard cap (should exceed 1e4 target)
    "record_interval": 2.0,      # snapshot spacing (tradeoff: resolution vs memory)
    "check_bounds": False,       # set True only for diagnostics
    "bound_tolerance": 1e-9,
}

# A lightweight "run preset" for a single formal full simulation
RUN_DEFAULTS = {
    "t_max": SIM_DEFAULTS["t_max"],
    "seed": SIM_DEFAULTS["seed"],
    "record_interval": SIM_DEFAULTS["record_interval"],
    "initial_k": np.array([2, 2], dtype=int),
    "initial_count": 100,
    "target_population": 10000,
}

ANALYSIS_DEFAULTS = {
    "growth_start_frac": 0.2,
    "extinction_alpha": 0.05,
}

PLOT_DEFAULTS = {
    "dpi": 150,
    "bbox_tight": "tight",
}

# =============================================================================
# Examples: FULL runs with identical model + identical growth settings
# =============================================================================
# Policy requested:
#   - Every example uses the SAME model parameters (growth, kernel, OU, etc.).
#   - Every example uses the SAME initial scale (100 cells, mean k around [2,2]).
#   - Every example runs on the SAME time horizon (t_max) and record interval.
#
# Note: examples.py currently samples initial k with Poisson(k_means).
#       Setting k_means=[2,2] makes the initial distribution centered at 2 per species.
#       If you want EXACTLY k=[2,2] for every initial cell, adjust sample_initial_cells
#       in examples.py to use a fixed vector (not Poisson).

EXAMPLES = {
    "output_dir": "figures",

    # Default model used by make_default_model()
    "default_model": {
        # OU
        "ou_mean": OU_DEFAULTS["mean"].copy(),
        "ou_rate": OU_DEFAULTS["rate"].copy(),
        "ou_diffusion": OU_DEFAULTS["diffusion"].copy(),

        # ecDNA dimensions + truncation
        "k_max": MODEL_DEFAULTS["k_max"].copy(),

        # bounded hazards + kernel + fitness (include all to make it self-contained)
        "n_env": MODEL_DEFAULTS["n_env"],
        "n_reg": MODEL_DEFAULTS["n_reg"],
        "div_rate_max": MODEL_DEFAULTS["div_rate_max"],
        "death_rate_max": MODEL_DEFAULTS["death_rate_max"],
        "gain_rate_max": MODEL_DEFAULTS["gain_rate_max"],
        "loss_rate_max": MODEL_DEFAULTS["loss_rate_max"],
        "reg_switch_max": MODEL_DEFAULTS["reg_switch_max"],
        "env_switch_max": MODEL_DEFAULTS["env_switch_max"],
        "fitness_k_star": MODEL_DEFAULTS["fitness_k_star"],
        "fitness_alpha": MODEL_DEFAULTS["fitness_alpha"],
        "fitness_beta": MODEL_DEFAULTS["fitness_beta"],
        "fitness_weights": MODEL_DEFAULTS["fitness_weights"],
        "age_effect_rate": MODEL_DEFAULTS["age_effect_rate"],
        "reg_switch_slope": MODEL_DEFAULTS["reg_switch_slope"],
        "env_switch_bias": MODEL_DEFAULTS["env_switch_bias"],
        "amplification_scale": MODEL_DEFAULTS["amplification_scale"],
        "division_copy_factor": MODEL_DEFAULTS["division_copy_factor"],
        "segregation_prob": MODEL_DEFAULTS["segregation_prob"],
        "post_segregation_loss_cap": MODEL_DEFAULTS["post_segregation_loss_cap"],
        "post_segregation_loss_slope": MODEL_DEFAULTS["post_segregation_loss_slope"],
        "post_segregation_loss_offset": MODEL_DEFAULTS["post_segregation_loss_offset"],
        "daughter_y_noise_std": MODEL_DEFAULTS["daughter_y_noise_std"],
    },

    # 1) Full population run
    "population": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),
        "summary_title_y": 1.02,
    },

    # 2) Full single-lineage run (starting from exactly k=[2,2])
    "lineage": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "initial_k": RUN_DEFAULTS["initial_k"].copy(),
        "figsize": (6, 3),
    },

    # 3) Sweep example: kept as a "degenerate sweep" so parameters remain identical.
    #    (One point in 1D and 2D so the plotting code still runs.)
    "sweep": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),

        "param_values": np.array([MODEL_DEFAULTS["div_rate_max"]]),
        "param1_values": np.array([MODEL_DEFAULTS["div_rate_max"]]),
        "param2_values": np.array([MODEL_DEFAULTS["death_rate_max"]]),

        "n_replicates_growth": 3,
        "n_replicates_extinction": 3,
        "n_replicates_k": 3,
        "n_replicates_2d": 3,
        "n_replicates_threshold": 3,

        "threshold": RUN_DEFAULTS["target_population"],

        "figsize_1d": (5, 3),
        "figsize_2d": (5, 4),
    },

    # 4) Thinning diagnostics: keep identical model; just enable bound checks and wider figures.
    "thinning": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "check_bounds": True,
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),
        "figsize": (12, 3),
    },

    # 5) Extinction analysis: same model, replicated. (May show low extinction under growth settings.)
    "extinction": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "n_replicates": 10,
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),
        "figsize": (12, 3),
    },

    # 6) Replicate variability: same model, multiple runs.
    "replicate": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "n_replicates": 10,
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),
        "figsize": (5, 3),
        "ci": 0.8,
    },

    # 7) Multi-ecDNA example: same model already has n_ecdna=2; keep identical settings.
    "multi_ecdna": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),

        "figsize_mean": (5, 3),
        "figsize_joint": (4, 3),

        # heatmap display controls (not model parameters)
        "heatmap_max_k": int(np.max(MODEL_DEFAULTS["k_max"])),
        "heatmap_time_resolution": 40,
    },

    # 8) Model atlas: same model; uses the same initial scale and checks bounds.
    "atlas": {
        "t_max": SIM_DEFAULTS["t_max"],
        "seed": SIM_DEFAULTS["seed"],
        "record_interval": SIM_DEFAULTS["record_interval"],
        "check_bounds": True,
        "n_cells": RUN_DEFAULTS["initial_count"],
        "k_means": RUN_DEFAULTS["initial_k"].astype(float),

        # plotting sizes
        "figsize_event_raster": (7, 2.5),
        "figsize_channel_stack": (6, 3),
        "figsize_acceptance_history": (6, 3),
        "figsize_proposal_hist": (5, 3),
        "figsize_waiting_time": (5, 3),
        "figsize_hazard_vs_age": (5, 3),
        "figsize_hazard_vs_k": (5, 3),
        "figsize_hazard_heatmap": (5, 3),
        "figsize_gain_loss": (5, 3),
        "figsize_amplification": (5, 3),
        "figsize_segregation": (5, 3),
        "figsize_daughter_scatter": (4, 3),

        "waiting_time_bins": 20,

        # hazard grids (note: model_atlas currently varies a scalar k; here we treat it as "total copy index")
        "hazard_k_range": np.arange(0, int(np.sum(MODEL_DEFAULTS["k_max"])) + 1),
        "hazard_age_grid": np.linspace(0, 8, 60),
        "hazard_k_value": RUN_DEFAULTS["initial_k"].copy().tolist(),

        # kernel diagnostics parent state and sample size
        "parent_k": RUN_DEFAULTS["initial_k"].copy(),
        "draw_samples": 400,
    },
}
