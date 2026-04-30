"""v4-lite calibrated summary posterior generation.

v4-lite is a summary posterior over observed snapshots. It is not a reduced
agent-based simulator and it does not create synthetic observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, read_table, write_json, write_markdown_report, write_npz, write_table
from fit.observation import load_observation_params


def fit_v4_lite_summary_posterior(
    empirical_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
    seed: int = 1,
    posterior_draws: int = 64,
) -> dict[str, Path]:
    """Build all lite artifacts required by full reconstruction."""

    rng = np.random.default_rng(seed)
    empirical = _load_empirical(empirical_dir)
    obs_params = load_observation_params(obs_params_path)
    out = ensure_dir(output_dir)

    target = build_lite_summary_target_vector(empirical, obs_params)
    posterior = _draw_summary_posterior(target, posterior_draws, rng)
    covariance = np.diag(target["variance"].astype(float).clip(lower=1e-9).to_numpy())
    distance_weights = _distance_weights(target)
    sampler = _initial_population_sampler(empirical, obs_params)
    prior_scales = _lite_to_full_prior_scales(target, empirical)

    write_table(posterior, out / "LITE_snapshot_posterior.parquet")
    write_table(target, out / "LITE_summary_target_vector.parquet")
    write_npz(
        out / "LITE_summary_covariance.npz",
        feature_id=target["feature_id"].astype(str).to_numpy(),
        covariance=covariance,
    )
    write_json(out / "LITE_distance_weights.json", distance_weights)
    write_json(out / "LITE_initial_population_sampler.json", sampler)
    write_json(out / "LITE_to_FULL_prior_scales.json", prior_scales)
    write_json(
        out / "LITE_final_fit.json",
        {
            "schema_version": 1,
            "method_source": "markdown/fit_method.md",
            "role": "calibrated_summary_posterior",
            "posterior_draws": int(posterior_draws),
            "n_target_features": int(len(target)),
            "outputs": list(schemas.LITE_OUTPUTS),
            "not_a_fake_data_generator": True,
        },
    )
    write_markdown_report(
        out / "LITE_final_report.md",
        "v4-lite Summary Posterior Report",
        [
            (
                "Role",
                "v4-lite generated calibrated summary posterior artifacts for full reconstruction. It did not simulate fake raw data.",
            ),
            (
                "Full Bridge",
                "Wrote summary target vector, covariance, distance weights, initial population sampler, and broad prior scales.",
            ),
            (
                "Method Guards",
                "ddPCR targets are bulk pooled means. ecTAG targets remain species-specific histogram features.",
            ),
        ],
    )
    return {name: out / name for name in schemas.LITE_OUTPUTS}


def build_lite_summary_target_vector(empirical: dict[str, pd.DataFrame], obs_params: dict) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(_flow_target_rows(empirical["flow"]))
    rows.extend(_ectag_target_rows(empirical["ectag_histograms"]))
    rows.extend(_qpcdr_target_rows(empirical["qpcdr"], obs_params))
    rows.extend(_ddpcr_target_rows(empirical["ddpcr"], obs_params))
    rows.extend(_lite_summary_rows(empirical["snapshot"]))
    target = pd.DataFrame(rows)
    if target.empty:
        raise ValueError("Lite summary target vector is empty")
    target["variance"] = target["variance"].astype(float).clip(lower=1e-9)
    target["weight"] = target["weight"].astype(float).clip(lower=0.0)
    return target.sort_values(["channel", "feature_id"]).reset_index(drop=True)


def validate_lite_artifacts(lite_dir: str | Path) -> None:
    base = Path(lite_dir)
    missing = [name for name in schemas.LITE_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing lite artifacts: {', '.join(missing)}")
    target = read_table(base / "LITE_summary_target_vector.parquet")
    required = {"feature_id", "channel", "target", "variance", "weight"}
    schemas.validate_required_columns(set(target.columns), tuple(required), "LITE_summary_target_vector")
    channels = set(target["channel"])
    needed = {"flow", "ectag", "qpcdr", "ddpcr", "lite_summary"}
    if not needed.issubset(channels):
        raise ValueError(f"Lite target vector missing channels: {sorted(needed - channels)}")
    sampler = read_json(base / "LITE_initial_population_sampler.json")
    if sampler.get("ddpcr_policy") != "pooled_mean_anchor_only":
        raise ValueError("Lite sampler must preserve ddPCR as pooled mean anchor only")


def load_lite_artifacts(lite_dir: str | Path) -> dict:
    validate_lite_artifacts(lite_dir)
    base = Path(lite_dir)
    covariance = np.load(base / "LITE_summary_covariance.npz", allow_pickle=True)
    return {
        "target": read_table(base / "LITE_summary_target_vector.parquet"),
        "posterior": read_table(base / "LITE_snapshot_posterior.parquet"),
        "covariance_feature_id": covariance["feature_id"].astype(str),
        "covariance": covariance["covariance"],
        "distance_weights": read_json(base / "LITE_distance_weights.json"),
        "sampler": read_json(base / "LITE_initial_population_sampler.json"),
        "prior_scales": read_json(base / "LITE_to_FULL_prior_scales.json"),
    }


def _load_empirical(empirical_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(empirical_dir)
    return {
        "snapshot": read_table(base / "snapshot_summary.parquet"),
        "ectag_histograms": read_table(base / "ectag_histograms_species_specific.parquet"),
        "joint": read_table(base / "ectag_joint_species_summary.parquet"),
        "ddpcr": read_table(base / "ddpcr_bulk_anchor_summary.parquet"),
        "qpcdr": read_table(base / "qpcdr_state_species_summary.parquet"),
        "flow": read_table(base / "flow_fraction_summary.parquet"),
    }


def _flow_target_rows(flow: pd.DataFrame) -> list[dict]:
    rows = []
    for row in flow.itertuples(index=False):
        n = max(1.0, float(row.flow_count))
        f = float(row.fraction)
        rows.append(
            _target_row(
                "flow",
                "flow_fraction",
                f,
                max(1e-4, f * (1.0 - f) / n),
                1.0,
                week=row.week,
                condition=row.condition,
                replicate=row.replicate,
                state_gate=row.state_gate,
            )
        )
    return rows


def _ectag_target_rows(hist: pd.DataFrame) -> list[dict]:
    rows = []
    for row in hist.itertuples(index=False):
        p = float(row.probability)
        n = max(1.0, float(row.n_cells))
        rows.append(
            _target_row(
                "ectag",
                "species_histogram_probability",
                p,
                max(1e-5, p * (1.0 - p) / n),
                float(row.histogram_weight),
                week=row.week,
                condition=row.condition,
                replicate=row.replicate,
                state_gate=row.state_gate,
                species=row.species,
                bin_label=row.bin_label,
            )
        )
    return rows


def _qpcdr_target_rows(qpcdr: pd.DataFrame, obs_params: dict) -> list[dict]:
    rows = []
    by_species = obs_params["qpcdr"]["by_species"]
    for row in qpcdr.itertuples(index=False):
        sigma = float(by_species[str(row.species)]["sigma"])
        rows.append(
            _target_row(
                "qpcdr",
                "state_species_mean",
                float(row.qpcdr_mean),
                max(1e-6, sigma * sigma),
                1.0,
                week=row.week,
                condition=row.condition,
                replicate=row.replicate,
                state_gate=row.state_gate,
                species=row.species,
            )
        )
    return rows


def _ddpcr_target_rows(ddpcr: pd.DataFrame, obs_params: dict) -> list[dict]:
    rows = []
    sigma_by_species = obs_params["ddpcr"]["sigma_by_species"]
    for row in ddpcr.itertuples(index=False):
        sigma = float(sigma_by_species[str(row.species)])
        rows.append(
            _target_row(
                "ddpcr",
                "bulk_pooled_mean",
                float(row.ddpcr_copy_number),
                max(1e-6, sigma * sigma),
                1.0,
                week=row.week,
                condition=row.condition,
                replicate=row.replicate,
                species=row.species,
            )
        )
    return rows


def _lite_summary_rows(snapshot: pd.DataFrame) -> list[dict]:
    rows = []
    for row in snapshot.itertuples(index=False):
        n = max(1.0, float(row.n_cells))
        common = {
            "week": row.week,
            "condition": row.condition,
            "replicate": row.replicate,
            "state_gate": row.state_gate,
            "species": row.species,
        }
        for variable, value in (
            ("copy_mean", float(row.copy_mean)),
            ("zero_fraction", float(row.zero_fraction)),
            ("tail_fraction", float(row.tail_fraction)),
            ("copy_variance", float(row.copy_variance)),
        ):
            variance = max(1e-5, abs(float(value)) / n) if "fraction" not in variable else max(1e-5, value * (1.0 - value) / n)
            rows.append(_target_row("lite_summary", variable, value, variance, 1.0, **common))
    return rows


def _target_row(channel: str, variable: str, target: float, variance: float, weight: float, **parts) -> dict:
    feature_id = schemas.stable_feature_id(channel, variable=variable, **parts)
    row = {
        "feature_id": feature_id,
        "channel": channel,
        "variable": variable,
        "target": float(target),
        "variance": float(variance),
        "weight": float(weight),
    }
    row.update(parts)
    return row


def _draw_summary_posterior(target: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for draw in range(int(draws)):
        noise = rng.normal(0.0, np.sqrt(target["variance"].astype(float).to_numpy()))
        values = target["target"].astype(float).to_numpy() + noise
        fraction_mask = target["variable"].astype(str).str.contains("fraction|probability", regex=True).to_numpy()
        values[fraction_mask] = np.clip(values[fraction_mask], 0.0, 1.0)
        frame = target[["feature_id", "channel", "variable", "week", "condition", "replicate", "state_gate", "species", "bin_label"]].copy()
        frame["draw"] = draw
        frame["value"] = values
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _distance_weights(target: pd.DataFrame) -> dict:
    channel_weights = target.groupby("channel")["weight"].mean().to_dict()
    return {
        "schema_version": 1,
        "channel_weights": {str(key): float(value) for key, value in channel_weights.items()},
        "feature_weights": dict(zip(target["feature_id"].astype(str), target["weight"].astype(float))),
        "components_required_for_full_score": ["flow", "ectag", "qpcdr", "ddpcr", "lite_summary", "prior", "biology"],
    }


def _initial_population_sampler(empirical: dict[str, pd.DataFrame], obs_params: dict) -> dict:
    snapshot = empirical["snapshot"]
    flow = empirical["flow"]
    hist = empirical["ectag_histograms"]
    initial_week = int(min(flow["week"]))
    initial_flow = flow[flow["week"] == initial_week]
    state_mass = (
        initial_flow.groupby("state_gate")["fraction"].mean().reindex(schemas.STATE_NAMES).fillna(0.0).to_numpy(dtype=float)
    )
    state_probs = schemas.normalize_probabilities(state_mass, name="initial_state_probs")
    hist_initial = hist[hist["week"] == initial_week]
    distributions: dict[str, dict[str, dict[str, list]]] = {}
    for state in schemas.STATE_NAMES:
        distributions[state] = {}
        for species in schemas.SPECIES:
            subset = hist_initial[(hist_initial["state_gate"] == state) & (hist_initial["species"] == species)]
            if subset.empty:
                labels = [item["label"] for item in obs_params["ectag"]["bins"]]
                probs = np.ones(len(labels), dtype=float) / len(labels)
            else:
                labels = subset["bin_label"].astype(str).tolist()
                probs = schemas.normalize_probabilities(subset["probability"].to_numpy(dtype=float) + 1e-6, name=f"{state}-{species}")
            distributions[state][species] = {
                "bin_labels": labels,
                "probabilities": probs.tolist(),
            }
    return {
        "schema_version": 1,
        "initial_week": initial_week,
        "states": list(schemas.STATE_NAMES),
        "species": list(schemas.SPECIES),
        "state_probabilities": dict(zip(schemas.STATE_NAMES, state_probs.tolist())),
        "copy_number_bins": obs_params["ectag"]["bins"],
        "state_species_copy_distributions": distributions,
        "soft_state_policy": "dominant_gate_with_dirichlet_hybrid_weight",
        "same_cell_species_correlation_policy": "use_empirical_joint_summary_when_available",
        "ddpcr_policy": "pooled_mean_anchor_only",
        "snapshot_rows_used": int(len(snapshot)),
    }


def _lite_to_full_prior_scales(target: pd.DataFrame, empirical: dict[str, pd.DataFrame]) -> dict:
    del empirical
    by_channel = target.groupby("channel")["target"].agg(["mean", "std"]).fillna(0.0)
    return {
        "schema_version": 1,
        "role": "broad_plausibility_prior_scales_not_truth_parameters",
        "state_transition_scale": float(max(0.02, by_channel.loc["flow", "std"] if "flow" in by_channel.index else 0.05)),
        "copy_gain_scale": float(max(0.05, by_channel.loc["lite_summary", "std"] if "lite_summary" in by_channel.index else 0.1)),
        "copy_loss_scale": float(max(0.05, by_channel.loc["ectag", "std"] if "ectag" in by_channel.index else 0.1)),
        "division_scale": 0.05,
        "death_scale": 0.02,
        "segregation_scale": 0.10,
        "observation_slack_scale": 0.10,
        "reporting_policy": "full parameters are latent controls; report history/scenario ensemble first",
    }
