"""Particle scoring for full conditional history reconstruction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fit.io_utils import read_json, read_table


SCORE_COMPONENTS: tuple[str, ...] = ("flow", "ectag", "qpcdr", "ddpcr", "lite_summary", "prior", "biology")


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

    feature_weights = distance_weights.get("feature_weights", {})
    values = merged["value"].astype(float).to_numpy()
    targets = merged["target"].astype(float).to_numpy()
    variances = merged["variance"].astype(float).clip(lower=1e-9).to_numpy()
    weights = np.asarray([float(feature_weights.get(fid, weight)) for fid, weight in zip(merged["feature_id"], merged["weight"])])
    deltas = values - targets

    log_channels = merged["channel"].isin(["qpcdr", "ddpcr"]).to_numpy()
    positive = log_channels & (values > 0.0) & (targets > 0.0)
    deltas[positive] = np.log(values[positive]) - np.log(targets[positive])

    merged = merged.copy()
    merged["component_score"] = weights * (deltas * deltas) / variances
    contribution = {name: 0.0 for name in SCORE_COMPONENTS}
    for channel, group in merged.groupby("channel"):
        contribution[str(channel)] = float(group["component_score"].sum())
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
