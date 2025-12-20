from __future__ import annotations

import numpy as np

CELL_DEFAULTS = {
    "e": 0,
    "m": 0,
    "k": np.array([0, 0], dtype=int),
    "y": np.array([0.0], dtype=float),
    "a": 0.0,
}

OU_DEFAULTS = {
    "mean": np.zeros(1),
    "rate": np.ones(1),
    "diffusion": np.zeros(1),
}

# 2) 模型层：两种 ecDNA；k_max 维度=2；并把 fitness_k_star 调到初始 K=4 附近
MODEL_DEFAULTS = {
    "n_env": 1,
    "n_reg": 1,
    "n_ecdna": 1,

    # 计算效率与上界(thinning bound)相关：过大的 k_max 会让上界变松、提案变密、变慢
    "k_max": np.array([50, 50], dtype=int),

    # 让系统具有正增长倾向（你也可以按需要再校准）
    "div_rate_max": 0.18,
    "death_rate_max": 0.03,

    # gain/loss 对上界也有影响；想先把群体推到 1e4，建议先设小一些，后续再做敏感性分析
    "gain_rate_max": 0.002,
    "loss_rate_max": 0.002,

    "reg_switch_max": 0.02,
    "env_switch_max": 0.0,

    # 关键：初始每细胞 k=[2,2]，若 weights=None 则 K=4
    "fitness_k_star": 4.0,
    "fitness_alpha": 1.0,
    "fitness_beta": 0.02,
    "fitness_weights": None,

    "age_effect_rate": 0.2,
    "reg_switch_slope": 0.5,
    "env_switch_bias": 0.0,

    # ecDNA 分裂核（保持你原模型思想，但别设太大，否则 k 容易爆、事件也会变多）
    "amplification_scale": 0.05,
    "division_copy_factor": 2,
    "segregation_prob": 0.5,

    "post_segregation_loss_cap": 0.5,
    "post_segregation_loss_slope": 0.01,
    "post_segregation_loss_offset": 1.0,
    "daughter_y_noise_std": 0.05,
}

# 3) 仿真层：给 max_cells 一个保险上限；record_interval 建议放大以减小 history 体积
SIM_DEFAULTS = {
    "t_max": 100.0,
    "seed": 42,
    "max_cells": 100000,      # 超过会直接抛错（保护内存/时间）
    "record_interval": 5.0,   # 记录快照越密，history(k_matrix 等)越大
    "check_bounds": False,
    "bound_tolerance": 1e-9,
}

# 4) 运行入口：从 100 个细胞开始，每个细胞 k=[2,2]；跑到 1e4 量级（t_max 取 100 常够用）
RUN_DEFAULTS = {
    "t_max": 100.0,
    "seed": 42,
    "record_interval": 5.0,
    "initial_k": np.array([2, 2], dtype=int),
    "initial_count": 100,
}

ANALYSIS_DEFAULTS = {
    "growth_start_frac": 0.2,
    "extinction_alpha": 0.05,
}

PLOT_DEFAULTS = {
    "dpi": 150,
    "bbox_tight": "tight",
}

EXAMPLES = {
    "output_dir": "figures",
    "default_model": {
        "ou_mean": np.zeros(1),
        "ou_rate": np.ones(1) * 0.5,
        "ou_diffusion": np.ones(1) * 0.2,
        "k_max": np.array([40], dtype=int),
        "div_rate_max": 0.15,
        "death_rate_max": 0.05,
    },
    "population": {
        "t_max": 30.0,
        "seed": 42,
        "record_interval": 1.0,
        "n_cells": 20,
        "k_means": np.array([8]),
        "summary_title_y": 1.02,
    },
    "lineage": {
        "t_max": 15.0,
        "seed": 42,
        "record_interval": 1.0,
        "initial_k": np.array([10]),
        "figsize": (6, 3),
    },
    "sweep": {
        "t_max": 10.0,
        "seed": 42,
        "record_interval": 1.0,
        "n_cells": 5,
        "k_means": np.array([5]),
        "param_values": np.linspace(0.05, 0.2, 4),
        "param1_values": np.linspace(0.08, 0.2, 4),
        "param2_values": np.linspace(0.03, 0.12, 4),
        "n_replicates_growth": 3,
        "n_replicates_extinction": 5,
        "n_replicates_k": 3,
        "n_replicates_2d": 3,
        "n_replicates_threshold": 3,
        "threshold": 40,
        "figsize_1d": (5, 3),
        "figsize_2d": (5, 4),
    },
    "thinning": {
        "ou_mean": np.zeros(1),
        "ou_rate": np.ones(1) * 0.4,
        "ou_diffusion": np.ones(1) * 0.2,
        "k_max": np.array([40]),
        "div_rate_max": 0.12,
        "death_rate_max": 0.05,
        "n_reg": 3,
        "n_env": 2,
        "reg_switch_max": 0.05,
        "env_switch_max": 0.03,
        "t_max": 20.0,
        "seed": 42,
        "record_interval": 1.0,
        "check_bounds": True,
        "n_cells": 20,
        "k_means": np.array([6]),
        "figsize": (12, 3),
    },
    "extinction": {
        "death_rate_max": 0.2,
        "t_max": 25.0,
        "seed": 42,
        "record_interval": 1.0,
        "n_replicates": 20,
        "n_cells": 10,
        "k_means": np.array([4]),
        "figsize": (12, 3),
    },
    "replicate": {
        "t_max": 20.0,
        "seed": 42,
        "record_interval": 1.0,
        "n_replicates": 15,
        "n_cells": 10,
        "k_means": np.array([6]),
        "figsize": (5, 3),
        "ci": 0.8,
    },
    "multi_ecdna": {
        "ou_mean": np.zeros(1),
        "ou_rate": np.ones(1) * 0.4,
        "ou_diffusion": np.ones(1) * 0.2,
        "n_ecdna": 2,
        "k_max": np.array([30, 20], dtype=int),
        "div_rate_max": 0.12,
        "death_rate_max": 0.04,
        "gain_rate_max": 0.015,
        "loss_rate_max": 0.02,
        "t_max": 20.0,
        "seed": 42,
        "record_interval": 1.0,
        "n_cells": 20,
        "k_means": np.array([6, 3]),
        "figsize_mean": (5, 3),
        "figsize_joint": (4, 3),
        "heatmap_max_k": 25,
        "heatmap_time_resolution": 40,
    },
    "atlas": {
        "n_reg": 3,
        "n_env": 2,
        "reg_switch_max": 0.05,
        "env_switch_max": 0.03,
        "gain_rate_max": 0.015,
        "loss_rate_max": 0.02,
        "t_max": 12.0,
        "seed": 42,
        "record_interval": 1.0,
        "check_bounds": True,
        "n_cells": 15,
        "k_means": np.array([6]),
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
        "hazard_k_range": np.arange(0, 41),
        "hazard_age_grid": np.linspace(0, 5, 40),
        "hazard_k_value": [10],
        "parent_k": np.array([10]),
        "draw_samples": 250,
    },
}
