"""Scenario classification for accepted full history particles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit.io_utils import ensure_dir, write_markdown_report, write_table


def classify_scenarios(events: pd.DataFrame, snapshots: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        event_features = pd.DataFrame({"particle_id": weights["particle_id"], "gain": 0.0, "loss": 0.0, "division": 0.0, "death": 0.0, "transition": 0.0})
    else:
        pivot = events.pivot_table(index="particle_id", columns="event_type", values="count", aggfunc="sum", fill_value=0.0).reset_index()
        for name in ("gain", "loss", "division", "death", "transition"):
            if name not in pivot:
                pivot[name] = 0.0
        event_features = pivot[["particle_id", "gain", "loss", "division", "death", "transition"]]

    snap = _snapshot_change_features(snapshots)
    features = weights[["particle_id", "weight", "score", "accepted"]].merge(event_features, on="particle_id", how="left").merge(snap, on="particle_id", how="left").fillna(0.0)
    rows = []
    for row in features.itertuples(index=False):
        turnover = float(row.gain + row.loss)
        transition = float(row.transition + row.state_fraction_change)
        selection = float(row.division + max(0.0, row.tail_change))
        segregation = float(row.species_correlation_proxy)
        active = sum(value > 0.0 for value in (turnover, transition, selection, segregation))
        if float(row.score) > float(features["score"].quantile(0.9)):
            label = "measurement-conflict"
        elif active >= 3:
            label = "mixed"
        else:
            scores = {
                "turnover-dominant": turnover,
                "transition-dominant": transition,
                "selection-dominant": selection,
                "segregation-dominant": segregation,
            }
            label = max(scores.items(), key=lambda item: item[1])[0]
            if scores[label] <= 0.0:
                label = "mixed"
        rows.append(
            {
                "particle_id": int(row.particle_id),
                "scenario_class": label,
                "posterior_weight": float(row.weight),
                "accepted": bool(row.accepted),
                "turnover_score": turnover,
                "transition_score": transition,
                "selection_score": selection,
                "segregation_score": segregation,
            }
        )
    return pd.DataFrame(rows)


def classify_scenarios_from_files(full_dir: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    base = Path(full_dir)
    events = pd.read_parquet(base / "event_summaries.parquet")
    snapshots = pd.read_parquet(base / "full_snapshot_summaries.parquet")
    weights = pd.read_parquet(base / "particle_weights.parquet")
    result = classify_scenarios(events, snapshots, weights)
    target_dir = base if output_dir is None else ensure_dir(output_dir)
    write_table(result, target_dir / "scenario_classes.parquet")
    summary = result.groupby("scenario_class", as_index=False)["posterior_weight"].sum().sort_values("posterior_weight", ascending=False)
    write_table(summary, target_dir / "scenario_summary.parquet")
    write_markdown_report(
        target_dir / "FULL_scenario_summary.md",
        "Full Scenario Summary",
        [("Policy", "Scenario classes summarize compatible history ensembles rather than unique biological rates.")],
    )
    return result


def _snapshot_change_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for particle_id, group in snapshots.groupby("particle_id"):
        first_week = group["week"].min()
        last_week = group["week"].max()
        first = group[group["week"] == first_week]
        last = group[group["week"] == last_week]
        state_first = first.groupby("state_gate")["flow_fraction"].mean()
        state_last = last.groupby("state_gate")["flow_fraction"].mean()
        state_change = float((state_last - state_first).abs().sum())
        tail_change = float(last["tail_fraction"].mean() - first["tail_fraction"].mean())
        species_means = last.pivot_table(index="state_gate", columns="species", values="copy_mean", aggfunc="mean")
        corr_proxy = 0.0
        if species_means.shape[1] >= 2:
            corr = species_means.corr().to_numpy(dtype=float)
            corr_proxy = float(np.nanmean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
            if not np.isfinite(corr_proxy):
                corr_proxy = 0.0
        rows.append(
            {
                "particle_id": particle_id,
                "state_fraction_change": state_change,
                "tail_change": tail_change,
                "species_correlation_proxy": corr_proxy,
            }
        )
    return pd.DataFrame(rows)
