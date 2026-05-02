"""Particle scoring for full conditional history reconstruction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import dirichlet_multinomial, entropy, wasserstein_distance

from fit.io_utils import read_json, read_table


SCORE_COMPONENTS: tuple[str, ...] = ("flow", "ectag", "qpcdr", "ddpcr", "cell_count", "lite_summary", "prior", "biology")


def score_particle_summary(
    particle_features: pd.DataFrame,
    lite_target: pd.DataFrame,
    distance_weights: dict,
    params: dict | None = None,
    biology_penalty: float = 0.0,
) -> dict:
    """Score one particle against raw-observation and lite-summary targets."""

    required = {"feature_id", "value"}
    missing = required.difference(particle_features.columns)
    if missing:
        raise ValueError(f"particle_features missing required columns: {sorted(missing)}")
    merged = lite_target.merge(
        particle_features[["feature_id", "value"]],
        on="feature_id",
        how="left",
        validate="one_to_one",
    )
    if merged["value"].isna().any():
        missing_ids = merged.loc[merged["value"].isna(), "feature_id"].head(5).tolist()
        raise ValueError(f"Particle summary is missing target features, examples: {missing_ids}")

    merged = merged.copy()
    feature_weights = distance_weights.get("feature_weights", {})
    merged["score_weight"] = [
        float(feature_weights.get(fid, weight)) for fid, weight in zip(merged["feature_id"], merged["weight"])
    ]
    log_channels = merged["channel"].isin(["qpcdr", "ddpcr", "cell_count"])
    merged["score_value"] = merged["value"].astype(float)
    merged["score_target"] = merged["target"].astype(float)
    merged.loc[log_channels, "score_value"] = np.log(merged.loc[log_channels, "score_value"].astype(float).clip(lower=1e-9))
    merged.loc[log_channels, "score_target"] = np.log(merged.loc[log_channels, "score_target"].astype(float).clip(lower=1e-9))
    contribution = {name: 0.0 for name in SCORE_COMPONENTS}
    for channel, group in merged.groupby("channel"):
        channel_name = str(channel)
        if channel_name == "ectag":
            contribution[channel_name] = _ectag_species_histogram_score(group)
        else:
            contribution[channel_name] = _weighted_mahalanobis_score(group)
    contribution["prior"] = _prior_penalty(params or {})
    contribution["biology"] = float(max(0.0, biology_penalty))
    total = float(sum(contribution.values()))
    return {
        "score": total,
        "contributions": contribution,
        "n_features": int(len(merged)),
    }


def score_particles_from_files(
    particle_features_path: str | Path,
    lite_target_path: str | Path,
    distance_weights_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    features = read_table(particle_features_path)
    target = read_table(lite_target_path)
    weights = read_json(distance_weights_path)
    rows = []
    for particle_id, group in features.groupby("particle_id"):
        score = score_particle_summary(group, target, weights)
        row = {"particle_id": particle_id, "score": score["score"], **score["contributions"]}
        rows.append(row)
    result = pd.DataFrame(rows)
    if output_path is not None:
        from fit.io_utils import write_table

        write_table(result, output_path)
    return result


def _weighted_mahalanobis_score(group: pd.DataFrame) -> float:
    values = group["score_value"].astype(float).to_numpy()
    targets = group["score_target"].astype(float).to_numpy()
    variances = group["variance"].astype(float).clip(lower=1e-9).to_numpy()
    weights = group["score_weight"].astype(float).clip(lower=1e-9).to_numpy()
    inv_cov = np.diag(weights / variances)
    return float(mahalanobis(values, targets, inv_cov) ** 2)


def _ectag_species_histogram_score(group: pd.DataFrame) -> float:
    required = {"week", "condition", "replicate", "state_gate", "species", "bin_label"}
    missing = required.difference(group.columns)
    if missing:
        raise ValueError(f"ecTAG score rows missing required grouping columns: {sorted(missing)}")
    total = 0.0
    keys = ["week", "condition", "replicate", "state_gate", "species"]
    for _, hist in group.groupby(keys, dropna=False):
        hist = hist.sort_values("bin_label", key=lambda labels: labels.map(_bin_sort_key))
        target_p = _probability_vector(hist["target"].to_numpy(dtype=float))
        particle_p = _probability_vector(hist["value"].to_numpy(dtype=float))
        n_cells = _histogram_cell_count(hist)
        counts = _integer_counts_from_probabilities(target_p, n_cells)
        concentration = max(float(n_cells), float(len(particle_p)))
        alpha = np.clip(particle_p * concentration, 1e-6, None)
        dm_nll = -float(dirichlet_multinomial.logpmf(counts, alpha, int(n_cells)))
        positions = np.asarray([_bin_sort_key(label) for label in hist["bin_label"]], dtype=float)
        wasserstein = float(wasserstein_distance(positions, positions, target_p, particle_p))
        support_span = float(np.ptp(positions)) if positions.size else 0.0
        support = max(1.0, support_span)
        kl = float(entropy(target_p, particle_p))
        weight = float(hist["score_weight"].astype(float).mean())
        total += weight * ((dm_nll / max(1, n_cells)) + kl + (wasserstein / support))
    return float(total)


def _probability_vector(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    total = float(np.sum(clipped))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(clipped.size, dtype=float) / max(1, clipped.size)
    return clipped / total


def _histogram_cell_count(hist: pd.DataFrame) -> int:
    if "n_cells" in hist.columns and hist["n_cells"].notna().any():
        return int(max(1, round(float(hist["n_cells"].dropna().astype(float).median()))))
    p = hist["target"].astype(float).to_numpy()
    v = hist["variance"].astype(float).clip(lower=1e-9).to_numpy()
    inferred = np.nanmedian(np.clip(p * (1.0 - p) / v, 1.0, None))
    return int(max(1, round(float(inferred)))) if np.isfinite(inferred) else 1


def _integer_counts_from_probabilities(probabilities: np.ndarray, n_cells: int) -> np.ndarray:
    raw = probabilities * int(n_cells)
    counts = np.floor(raw).astype(int)
    remainder = int(n_cells) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _bin_sort_key(label: object) -> int:
    text = str(label)
    if text.endswith("+"):
        return int(text[:-1])
    if "-" in text:
        return int(text.split("-", 1)[0])
    return int(text)


def _prior_penalty(params: dict) -> float:
    penalty = 0.0
    for key, value in params.items():
        if key.endswith("_scale") or key in {"particle_id", "seed"}:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            penalty += 0.05 * numeric * numeric
    return float(penalty)
