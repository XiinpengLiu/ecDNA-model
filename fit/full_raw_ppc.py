"""Raw-table posterior predictive checks from accepted full histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.empirical import (
    _cell_count_summary,
    _ddpcr_summary,
    _ectag_histograms,
    _flow_summary,
    _qpcdr_summary,
    _snapshot_summary,
)
from fit.io_utils import ensure_dir, write_json, write_markdown_report, write_table
from fit.observation import load_observation_params
from fit.raw import validate_raw_tables
from fit.v4_lite import build_lite_summary_target_vector


SYNTHETIC_RAW_OUTPUTS: dict[str, str] = {
    "flow": "synthetic_flow_long.parquet",
    "qpcdr": "synthetic_qpcdr_long.parquet",
    "ectag": "synthetic_ectag_cell_long.parquet",
    "ddpcr": "synthetic_ddpcr_long.parquet",
    "cell_count": "synthetic_cell_count_long.parquet",
}


def generate_full_raw_table_ppc(
    full_dir: str | Path,
    obs_params_path: str | Path,
    lite_dir: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 1,
) -> dict[str, Path]:
    """Generate replicated raw observation tables from accepted histories."""

    base_full = Path(full_dir)
    out = ensure_dir(base_full if output_dir is None else output_dir)
    obs_params = load_observation_params(obs_params_path)
    target = pd.read_parquet(Path(lite_dir) / "LITE_summary_target_vector.parquet")
    history_path, weights_path, source = _select_history_source(base_full)
    histories = pd.read_parquet(history_path)
    weights = pd.read_parquet(weights_path)
    accepted_weights = _accepted_normalized_weights(weights)
    histories = histories[histories["particle_id"].astype(int).isin(set(accepted_weights["particle_id"].astype(int)))].copy()
    if histories.empty:
        raise ValueError("Cannot generate raw-table PPC: no accepted history rows are available")

    rng = np.random.default_rng(seed)
    tables = _simulate_raw_tables(histories, obs_params, rng)
    validate_raw_tables({name: table.copy() for name, table in tables.items()})
    paths = {name: out / filename for name, filename in SYNTHETIC_RAW_OUTPUTS.items()}
    for name, path in paths.items():
        write_table(tables[name], path)

    particle_features = _particle_summary_features(tables, obs_params, target)
    write_table(particle_features, out / "raw_table_ppc_particle_features.parquet")
    summary_coverage = _summary_coverage(particle_features, target, accepted_weights)
    replicate_diagnostics = _replicate_diagnostics(particle_features, target, accepted_weights)
    write_table(summary_coverage, out / "raw_table_ppc_summary_coverage.parquet")
    write_table(replicate_diagnostics, out / "raw_table_ppc_replicate_diagnostics.parquet")

    by_channel = summary_coverage.groupby("channel", as_index=False)["covered_by_two_sigma"].mean()
    payload = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "history_source": source,
        "particle_scope": "accepted_histories_only",
        "synthetic_raw_tables": {name: str(path) for name, path in paths.items()},
        "summary_coverage_by_channel": dict(zip(by_channel["channel"], by_channel["covered_by_two_sigma"].astype(float))),
        "replicate_diagnostic_rows": int(len(replicate_diagnostics)),
        "ddpcr_policy": "bulk_pooled_mean_only",
        "ectag_policy": "species_specific_single_cell_raw_table",
    }
    write_json(out / "raw_table_ppc_report.json", payload)
    write_markdown_report(
        out / "raw_table_ppc_report.md",
        "Full Raw-Table PPC Report",
        [
            ("Scope", "Generated complete synthetic raw tables from accepted posterior histories, then summarized them back to the fit target scale."),
            ("History Source", source),
            ("Summary Coverage", ", ".join(f"{row.channel}: {float(row.covered_by_two_sigma):.3f}" for row in by_channel.itertuples(index=False))),
            ("Replicate Diagnostics", f"rows={len(replicate_diagnostics)}; columns={', '.join(replicate_diagnostics.columns)}"),
            ("Guards", "ddPCR is generated only as a bulk pooled mean; ecTAG remains species-specific at cell level."),
        ],
    )
    return {**paths, "report": out / "raw_table_ppc_report.json"}


def _select_history_source(full_dir: Path) -> tuple[Path, Path, str]:
    exact_history = full_dir / "FULL_exact_replay_histories.parquet"
    exact_weights = full_dir / "FULL_exact_replay_particle_weights.parquet"
    if exact_history.exists() and exact_weights.exists():
        return exact_history, exact_weights, "exact_replay_accepted_histories"
    return full_dir / "FULL_single_cell_history_samples.parquet", full_dir / "FULL_particle_weights.parquet", "full_smc_accepted_histories"


def _accepted_normalized_weights(weights: pd.DataFrame) -> pd.DataFrame:
    result = weights.copy()
    if "accepted" in result:
        accepted = result[result["accepted"].astype(bool)].copy()
        if accepted.empty:
            accepted = result.copy()
    else:
        accepted = result.copy()
    total = float(accepted["weight"].astype(float).sum()) if "weight" in accepted else float(len(accepted))
    if total <= 0.0 or not np.isfinite(total):
        accepted["accepted_weight"] = 1.0 / max(1, len(accepted))
    else:
        accepted["accepted_weight"] = accepted["weight"].astype(float) / total
    return accepted[["particle_id", "accepted_weight"]].copy()


def _simulate_raw_tables(histories: pd.DataFrame, obs_params: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    flow_rows: list[dict] = []
    qpcdr_rows: list[dict] = []
    ectag_rows: list[dict] = []
    ddpcr_rows: list[dict] = []
    cell_count_rows: list[dict] = []
    purity = float(obs_params.get("flow", {}).get("purity", 0.95))
    count_dispersion = float(obs_params.get("cell_count", {}).get("dispersion", 1.0))

    keys = ["particle_id", "week", "condition", "replicate"]
    ordered = histories.sort_values(keys + ["cell_id"]).reset_index(drop=True)
    for key, group in ordered.groupby(keys, dropna=False):
        particle_id, week, condition, replicate = key
        group = group.copy()
        group["population_weight"] = group.get("population_weight", 1.0)
        weights = group["population_weight"].astype(float).clip(lower=0.0)
        represented_total = float(weights.sum())
        if represented_total <= 0.0 or not np.isfinite(represented_total):
            weights = pd.Series(np.ones(len(group), dtype=float), index=group.index)
            represented_total = float(len(group))
        observed_count_total = int(max(1, round(represented_total)))
        for state in schemas.STATE_NAMES:
            state_mask = group["state_gate"].astype(str) == state
            state_weight = float(weights[state_mask].sum())
            fraction = state_weight / max(1e-9, represented_total)
            pre_count = int(max(0, round(observed_count_total * fraction)))
            flow_rows.append(
                {
                    "particle_id": int(particle_id),
                    "week": int(week),
                    "condition": str(condition),
                    "replicate": str(replicate),
                    "state_gate": state,
                    "pre_sort_count": pre_count,
                    "post_sort_count": int(round(pre_count * purity)),
                    "fraction": float(fraction),
                    "sort_purity": purity,
                    "marker_panel": "synthetic-full-raw-ppc",
                    "batch_id": f"ppc-p{int(particle_id)}",
                }
            )
            state_group = group[state_mask]
            for species in schemas.SPECIES:
                values = state_group[f"K_{species}"].astype(float).to_numpy() if f"K_{species}" in state_group else np.asarray([], dtype=float)
                value_weights = weights[state_mask].to_numpy(dtype=float)
                mean_copy = float(np.average(values, weights=value_weights)) if values.size and value_weights.sum() > 0.0 else 0.0
                for technical_rep in (1, 2):
                    qpcdr_rows.append(
                        _qpcdr_raw_row(
                            int(particle_id),
                            int(week),
                            str(condition),
                            str(replicate),
                            state,
                            species,
                            technical_rep,
                            mean_copy,
                            obs_params,
                            rng,
                        )
                    )

        for row in group.itertuples(index=False):
            cell_label = f"p{int(particle_id)}-w{int(week)}-{row.condition}-{row.replicate}-c{int(row.cell_id)}"
            for species in schemas.SPECIES:
                true_count = max(0.0, float(getattr(row, f"K_{species}")))
                observed = int(rng.poisson(true_count)) if true_count > 0.0 else 0
                ectag_rows.append(
                    {
                        "particle_id": int(particle_id),
                        "week": int(week),
                        "condition": str(condition),
                        "replicate": str(replicate),
                        "state_gate": str(row.state_gate),
                        "cell_id": cell_label,
                        "species": species,
                        "ectag_count": observed,
                        "image_qc_pass": True,
                        "batch_id": f"ppc-p{int(particle_id)}",
                    }
                )
        for species in schemas.SPECIES:
            values = group[f"K_{species}"].astype(float).to_numpy()
            bulk_mean = float(np.average(values, weights=weights.to_numpy(dtype=float))) if len(values) else 0.0
            sigma = max(1e-9, float(obs_params["ddpcr"]["sigma_by_species"][species]))
            ddpcr_value = float(np.exp(rng.normal(np.log(max(1e-9, bulk_mean)), sigma)))
            ddpcr_rows.append(
                {
                    "particle_id": int(particle_id),
                    "week": int(week),
                    "condition": str(condition),
                    "replicate": str(replicate),
                    "species": species,
                    "ddpcr_copy_number": ddpcr_value,
                    "ddpcr_sd_or_ci": max(0.05, ddpcr_value * sigma),
                    "batch_id": f"ppc-p{int(particle_id)}",
                }
            )
        noisy_total = float(max(0.0, rng.normal(represented_total, count_dispersion)))
        cell_count_rows.append(
            {
                "particle_id": int(particle_id),
                "week": int(week),
                "condition": str(condition),
                "replicate": str(replicate),
                "total_cell_count": noisy_total,
                "viability": 0.95,
                "passage_info": "synthetic-full-raw-ppc",
                "batch_id": f"ppc-p{int(particle_id)}",
            }
        )

    tables = {
        "flow": pd.DataFrame(flow_rows),
        "qpcdr": pd.DataFrame(qpcdr_rows),
        "ectag": pd.DataFrame(ectag_rows),
        "ddpcr": pd.DataFrame(ddpcr_rows),
        "cell_count": pd.DataFrame(cell_count_rows),
    }
    return {name: table.sort_values([column for column in ("particle_id", "week", "condition", "replicate", "state_gate", "species", "cell_id") if column in table.columns]).reset_index(drop=True) for name, table in tables.items()}


def _qpcdr_raw_row(
    particle_id: int,
    week: int,
    condition: str,
    replicate: str,
    state: str,
    species: str,
    technical_rep: int,
    mean_copy: float,
    obs_params: dict,
    rng: np.random.Generator,
) -> dict:
    species_params = obs_params["qpcdr"]["by_species"][species]
    sigma = max(1e-9, float(species_params["sigma"]))
    mu = max(float(mean_copy), float(obs_params["qpcdr"].get("epsilon", 1e-9)))
    raw_ct = np.nan
    relative = np.nan
    if str(species_params["scale"]) == "relative_copy_number_log":
        value = float(np.exp(rng.normal(float(species_params["intercept"]) + float(species_params["slope"]) * np.log(mu), sigma)))
        relative = value
    else:
        raw_ct = float(rng.normal(float(species_params["intercept"]) - float(species_params["slope"]) * np.log10(mu), sigma))
    return {
        "particle_id": particle_id,
        "week": week,
        "condition": condition,
        "replicate": replicate,
        "state_gate": state,
        "species": species,
        "technical_rep": int(technical_rep),
        "raw_Ct_or_Cq": raw_ct,
        "relative_copy_number": relative,
        "plate_id": f"ppc-p{particle_id}",
        "batch_id": f"ppc-p{particle_id}",
    }


def _particle_summary_features(tables: dict[str, pd.DataFrame], obs_params: dict, target: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    particle_ids = sorted(int(value) for value in tables["flow"]["particle_id"].dropna().unique())
    for particle_id in particle_ids:
        particle_tables = {name: table[table["particle_id"].astype(int) == particle_id].copy() for name, table in tables.items()}
        flow = _flow_summary(particle_tables["flow"])
        hist = _ectag_histograms(particle_tables["ectag"], list(obs_params["ectag"]["bins"]), min_cells=1)
        snapshot = _snapshot_summary(particle_tables["ectag"], flow, list(obs_params["ectag"]["bins"]))
        empirical = {
            "flow": flow,
            "ectag_histograms": hist,
            "snapshot": snapshot,
            "qpcdr": _qpcdr_summary(particle_tables["qpcdr"]),
            "ddpcr": _ddpcr_summary(particle_tables["ddpcr"]),
            "cell_count": _cell_count_summary(particle_tables["cell_count"]),
        }
        features = build_lite_summary_target_vector(empirical, obs_params)
        features = features[features["feature_id"].isin(set(target["feature_id"]))].copy()
        features["particle_id"] = int(particle_id)
        features = features.rename(columns={"target": "value"})
        rows.append(features)
    if not rows:
        return pd.DataFrame(columns=["particle_id", "feature_id", "channel", "value"])
    return pd.concat(rows, ignore_index=True)


def _summary_coverage(particle_features: pd.DataFrame, target: pd.DataFrame, accepted_weights: pd.DataFrame) -> pd.DataFrame:
    weighted = particle_features.merge(accepted_weights, on="particle_id", how="inner")
    weighted["weighted_value"] = weighted["value"].astype(float) * weighted["accepted_weight"].astype(float)
    posterior = weighted.groupby("feature_id", as_index=False)["weighted_value"].sum().rename(columns={"weighted_value": "posterior_mean"})
    report = target.merge(posterior, on="feature_id", how="inner")
    report["abs_error"] = (report["posterior_mean"].astype(float) - report["target"].astype(float)).abs()
    report["covered_by_two_sigma"] = report["abs_error"] <= 2.0 * np.sqrt(report["variance"].astype(float).clip(lower=1e-9))
    return report.sort_values(["channel", "feature_id"]).reset_index(drop=True)


def _replicate_diagnostics(particle_features: pd.DataFrame, target: pd.DataFrame, accepted_weights: pd.DataFrame) -> pd.DataFrame:
    merged = particle_features.merge(target, on="feature_id", suffixes=("_synthetic", "_observed"), how="inner")
    merged = merged.merge(accepted_weights, on="particle_id", how="inner")
    merged["abs_error"] = (merged["value"].astype(float) - merged["target"].astype(float)).abs()
    merged["covered_by_two_sigma"] = merged["abs_error"] <= 2.0 * np.sqrt(merged["variance_observed"].astype(float).clip(lower=1e-9))
    rows = []
    group_cols = ["particle_id", "channel_observed", "condition_observed", "replicate_observed"]
    for key, group in merged.groupby(group_cols, dropna=False):
        particle_id, channel, condition, replicate = key
        rows.append(
            {
                "particle_id": int(particle_id),
                "posterior_weight": float(group["accepted_weight"].iloc[0]),
                "channel": str(channel),
                "condition": "" if pd.isna(condition) else str(condition),
                "replicate": "" if pd.isna(replicate) else str(replicate),
                "n_features": int(len(group)),
                "mean_abs_error": float(group["abs_error"].astype(float).mean()),
                "coverage": float(group["covered_by_two_sigma"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["channel", "condition", "replicate", "particle_id"]).reset_index(drop=True)
