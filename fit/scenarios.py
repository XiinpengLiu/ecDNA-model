"""Scenario classification for bulk-compatible history ensembles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit.io_utils import ensure_dir, write_table


def classify_scenarios(events: pd.DataFrame, snapshots: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    del events, snapshots
    if weights.empty:
        return pd.DataFrame(columns=["particle_id", "scenario_class", "posterior_weight", "accepted"])
    rows = []
    for row in weights.itertuples(index=False):
        rows.append(
            {
                "particle_id": int(row.particle_id),
                "scenario_class": "mixed latent histories",
                "scenario_cluster_id": 1,
                "posterior_weight": float(row.weight),
                "accepted": bool(row.accepted),
                "interpretation": "bulk scenario class is model-dependent, not a unique microscopic mechanism",
            }
        )
    return pd.DataFrame(rows)


def classify_scenarios_from_files(full_dir: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    base = Path(full_dir)
    weights = pd.read_parquet(base / "FULL_particle_weights.parquet")
    result = classify_scenarios(pd.DataFrame(), pd.DataFrame(), weights)
    target = ensure_dir(output_dir or base)
    write_table(result, target / "FULL_scenario_classes.parquet")
    summary = result.groupby("scenario_class", as_index=False)["posterior_weight"].sum()
    write_table(summary, target / "FULL_scenario_summary.parquet")
    return result
