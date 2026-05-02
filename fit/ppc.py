"""Posterior predictive check helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit.io_utils import ensure_dir, write_json, write_markdown_report, write_table


def run_full_ppc(full_dir: str | Path, lite_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    base_full = Path(full_dir)
    base_lite = Path(lite_dir)
    target_dir = base_full if output_dir is None else ensure_dir(output_dir)
    weights = pd.read_parquet(base_full / "particle_weights.parquet")
    target = pd.read_parquet(base_lite / "LITE_summary_target_vector.parquet")
    raw_like_path = base_full / "FULL_ppc_raw_observables.parquet"
    if raw_like_path.exists():
        raw_like = pd.read_parquet(raw_like_path)
        features = raw_like[["particle_id", "feature_id", "value"]].copy()
    else:
        features = pd.read_parquet(base_full / "particle_summary_features.parquet")
    accepted_weights = weights[weights["accepted"]].copy()
    if accepted_weights.empty:
        accepted_weights = weights.copy()
    total_weight = float(accepted_weights["weight"].astype(float).sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        accepted_weights["accepted_weight"] = 1.0 / max(1, len(accepted_weights))
    else:
        accepted_weights["accepted_weight"] = accepted_weights["weight"].astype(float) / total_weight
    merged = features.merge(accepted_weights[["particle_id", "accepted_weight"]], on="particle_id", how="inner")
    merged["weighted_value"] = merged["value"].astype(float) * merged["accepted_weight"].astype(float)
    posterior = merged.groupby("feature_id", as_index=False)["weighted_value"].sum().rename(columns={"weighted_value": "posterior_mean"})
    report = target.merge(posterior, on="feature_id", how="left")
    report["covered_by_two_sigma"] = (report["posterior_mean"] - report["target"]).abs() <= 2.0 * np.sqrt(report["variance"].astype(float))
    by_channel = report.groupby("channel", as_index=False)["covered_by_two_sigma"].mean().rename(columns={"covered_by_two_sigma": "coverage"})
    report = report.sort_values(["channel", "feature_id"]).reset_index(drop=True)
    payload = {
        "coverage_by_channel": dict(zip(by_channel["channel"], by_channel["coverage"].astype(float))),
        "weighted_history_ensemble": True,
        "particle_scope": "accepted_particles_only",
        "accepted_particles": int(accepted_weights["particle_id"].nunique()),
        "raw_like_channels": sorted(str(channel) for channel in by_channel["channel"].unique()),
    }
    write_table(by_channel, target_dir / "full_ppc_channel_coverage.parquet")
    write_table(report, target_dir / "full_ppc_feature_coverage.parquet")
    write_json(target_dir / "full_ppc_report.json", payload)
    write_markdown_report(
        target_dir / "full_ppc_report.md",
        "Full PPC Report",
        [
            ("Scope", "Posterior predictive summaries were computed from weighted full history particles."),
            ("Particle Scope", "Only accepted particles are used, with weights renormalized inside the accepted ensemble."),
            ("Coverage", ", ".join(f"{row.channel}: {row.coverage:.3f}" for row in by_channel.itertuples(index=False))),
        ],
    )
    return payload
