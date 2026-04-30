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
    features = pd.read_parquet(base_full / "particle_summary_features.parquet")
    weights = pd.read_parquet(base_full / "particle_weights.parquet")
    target = pd.read_parquet(base_lite / "LITE_summary_target_vector.parquet")
    merged = features.merge(weights[["particle_id", "weight"]], on="particle_id", how="left")
    merged["weighted_value"] = merged["value"].astype(float) * merged["weight"].astype(float)
    posterior = merged.groupby("feature_id", as_index=False)["weighted_value"].sum().rename(columns={"weighted_value": "posterior_mean"})
    report = target.merge(posterior, on="feature_id", how="left")
    report["covered_by_two_sigma"] = (report["posterior_mean"] - report["target"]).abs() <= 2.0 * np.sqrt(report["variance"].astype(float))
    by_channel = report.groupby("channel", as_index=False)["covered_by_two_sigma"].mean().rename(columns={"covered_by_two_sigma": "coverage"})
    payload = {
        "coverage_by_channel": dict(zip(by_channel["channel"], by_channel["coverage"].astype(float))),
        "weighted_history_ensemble": True,
    }
    write_table(by_channel, target_dir / "full_ppc_channel_coverage.parquet")
    write_json(target_dir / "full_ppc_report.json", payload)
    write_markdown_report(
        target_dir / "full_ppc_report.md",
        "Full PPC Report",
        [
            ("Scope", "Posterior predictive summaries were computed from weighted full history particles."),
            ("Coverage", ", ".join(f"{row.channel}: {row.coverage:.3f}" for row in by_channel.itertuples(index=False))),
        ],
    )
    return payload
