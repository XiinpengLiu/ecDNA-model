"""Bulk partial-observation scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_COMPONENTS: tuple[str, ...] = ("ddpcr", "cell_count", "flow3", "prior", "biology")


def score_bulk_predictions(predictions: dict[str, pd.DataFrame], obs: dict, params: dict) -> dict:
    dd = _ddpcr_distance(predictions["ddpcr"], obs)
    cc = _cell_count_distance(predictions["cell_count"], obs)
    fl = _flow3_distance(predictions["flow3"], obs)
    prior = float(params.get("D_prior", 0.0))
    biology = float(params.get("D_biology", 0.0))
    total = dd + cc + fl + prior + biology
    return {"score": float(total), "contributions": {"ddpcr": dd, "cell_count": cc, "flow3": fl, "prior": prior, "biology": biology}}


def _ddpcr_distance(pred: pd.DataFrame, obs: dict) -> float:
    sigma_by_species = obs["ddpcr"]["log_sd_by_species"]
    values = pred["predicted_bulk_mean"].astype(float).clip(lower=1e-9)
    targets = pred["observed_bulk_mean"].astype(float).clip(lower=1e-9)
    sigma = np.asarray([float(sigma_by_species.get(str(species), obs["ddpcr"]["default_log_sd"])) for species in pred["species"]], dtype=float)
    residual = (np.log(values) - np.log(targets)) / np.clip(sigma, 1e-9, None)
    return float(np.sum(residual * residual))


def _cell_count_distance(pred: pd.DataFrame, obs: dict) -> float:
    sigma = float(obs["cell_count"]["log_sd"])
    residual = (np.log(pred["observed_cell_count"].astype(float).clip(lower=0.0) + 1.0) - np.log(pred["predicted_cell_count"].astype(float).clip(lower=0.0) + 1.0)) / max(1e-9, sigma)
    # Student-t-like robust distance.
    nu = float(obs["cell_count"].get("nu", 4))
    return float(np.sum((nu + 1.0) * np.log1p((residual * residual) / nu)))


def _flow3_distance(pred: pd.DataFrame, obs: dict) -> float:
    sd = float(obs["flow3"]["absolute_fraction_sd"])
    residual = (pred["predicted_fraction"].astype(float) - pred["target_fraction"].astype(float)) / max(1e-9, sd)
    return float(np.sum(residual * residual))
