"""Scenario classification for accepted full history particles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

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
    features = _add_method_scenario_features(features)
    features["cluster_id"] = _cluster_scenario_features(features)
    labels = _label_clusters(features)
    rows = []
    for row in features.itertuples(index=False):
        label = labels.get(int(row.cluster_id), "mixed")
        if float(row.score) > float(features["score"].quantile(0.9)):
            label = "measurement-conflict"
        rows.append(
            {
                "particle_id": int(row.particle_id),
                "scenario_class": label,
                "scenario_cluster_id": int(row.cluster_id),
                "posterior_weight": float(row.weight),
                "accepted": bool(row.accepted),
                "turnover_score": float(row.turnover_feature),
                "transition_score": float(row.transition_flux_feature),
                "selection_score": float(row.high_copy_expansion_feature),
                "segregation_score": float(row.daughter_inheritance_feature),
                "zero_fraction_change": float(row.zero_fraction_change),
                "tail_change": float(row.tail_change),
                "species_correlation": float(row.species_correlation_proxy),
            }
        )
    return pd.DataFrame(rows)


def classify_scenarios_from_files(full_dir: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    base = Path(full_dir)
    events, snapshots, weights, source, provisional = _select_scenario_inputs(base)
    result = classify_scenarios(events, snapshots, weights)
    result["scenario_source"] = source
    result["provisional"] = bool(provisional)
    target_dir = base if output_dir is None else ensure_dir(output_dir)
    write_table(result, target_dir / "scenario_classes.parquet")
    summary = result.groupby("scenario_class", as_index=False)["posterior_weight"].sum().sort_values("posterior_weight", ascending=False)
    summary["scenario_source"] = source
    summary["provisional"] = bool(provisional)
    write_table(summary, target_dir / "scenario_summary.parquet")
    write_markdown_report(
        target_dir / "FULL_scenario_summary.md",
        "Full Scenario Summary",
        [
            ("Policy", "Scenario classes summarize compatible history ensembles rather than unique biological rates."),
            ("Source", f"{source}; provisional={bool(provisional)}"),
        ],
    )
    return result


def _select_scenario_inputs(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, bool]:
    exact_events = base / "FULL_exact_replay_event_summaries.parquet"
    exact_snapshots = base / "FULL_exact_replay_snapshot_summaries.parquet"
    exact_weights = base / "FULL_exact_replay_particle_weights.parquet"
    if exact_events.exists() and exact_snapshots.exists() and exact_weights.exists():
        weights = pd.read_parquet(exact_weights)
        selected = weights[weights["accepted"].astype(bool)].copy() if "accepted" in weights else weights.copy()
        if selected.empty:
            selected = weights.copy()
        selected_ids = set(selected["particle_id"].astype(int))
        events = pd.read_parquet(exact_events)
        snapshots = pd.read_parquet(exact_snapshots)
        return (
            events[events["particle_id"].astype(int).isin(selected_ids)].copy(),
            snapshots[snapshots["particle_id"].astype(int).isin(selected_ids)].copy(),
            selected,
            "exact_replay_event_summaries",
            False,
        )
    return (
        pd.read_parquet(base / "event_summaries.parquet"),
        pd.read_parquet(base / "full_snapshot_summaries.parquet"),
        pd.read_parquet(base / "particle_weights.parquet"),
        "simplified_smc_event_summaries",
        True,
    )


def _add_method_scenario_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result["turnover_feature"] = result["gain"].astype(float) + result["loss"].astype(float)
    result["transition_flux_feature"] = result["transition"].astype(float) + result["state_fraction_change"].astype(float)
    result["high_copy_expansion_feature"] = result["division"].astype(float) * np.clip(result["tail_change"].astype(float), 0.0, None)
    result["zero_fraction_feature"] = result["zero_fraction_change"].astype(float).abs()
    result["tail_feature"] = result["tail_change"].astype(float).abs()
    result["species_correlation_feature"] = result["species_correlation_proxy"].astype(float)
    result["daughter_inheritance_feature"] = result["division"].astype(float) * np.clip(result["species_correlation_proxy"].astype(float), 0.0, None)
    return result


def _cluster_scenario_features(features: pd.DataFrame) -> np.ndarray:
    columns = [
        "gain",
        "loss",
        "division",
        "death",
        "transition_flux_feature",
        "high_copy_expansion_feature",
        "zero_fraction_feature",
        "tail_feature",
        "species_correlation_feature",
        "daughter_inheritance_feature",
    ]
    if len(features) <= 1:
        return np.ones(len(features), dtype=int)
    matrix = features[columns].astype(float).to_numpy()
    scale = np.nanstd(matrix, axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (matrix - np.nanmean(matrix, axis=0)) / scale
    if not np.isfinite(standardized).all():
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(standardized, 0.0):
        return np.ones(len(features), dtype=int)
    n_clusters = min(5, max(1, len(features)))
    tree = linkage(standardized, method="ward")
    return fcluster(tree, t=n_clusters, criterion="maxclust").astype(int)


def _label_clusters(features: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    for cluster_id, group in features.groupby("cluster_id"):
        centroid = group[
            [
                "turnover_feature",
                "transition_flux_feature",
                "high_copy_expansion_feature",
                "daughter_inheritance_feature",
            ]
        ].astype(float).mean()
        scores = {
            "turnover-dominant": float(centroid["turnover_feature"]),
            "transition-dominant": float(centroid["transition_flux_feature"]),
            "selection-dominant": float(centroid["high_copy_expansion_feature"]),
            "segregation-dominant": float(centroid["daughter_inheritance_feature"]),
        }
        active = sum(value > 1e-9 for value in scores.values())
        label = max(scores.items(), key=lambda item: item[1])[0] if active else "mixed"
        if active >= 3:
            label = "mixed"
        labels[int(cluster_id)] = label
    return labels


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
        zero_change = float(last["zero_fraction"].mean() - first["zero_fraction"].mean()) if "zero_fraction" in last else 0.0
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
                "zero_fraction_change": zero_change,
                "species_correlation_proxy": corr_proxy,
            }
        )
    return pd.DataFrame(rows)
