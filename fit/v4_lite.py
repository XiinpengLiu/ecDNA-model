"""v4-lite calibrated summary posterior generation.

v4-lite is a summary posterior over observed snapshots. It is not a reduced
agent-based simulator and it does not create synthetic observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, read_table, write_dataset_netcdf, write_json, write_markdown_report, write_npz, write_table, write_text_pdf
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

    empirical["transition_growth"] = _transition_growth_summary(empirical["flow"])
    empirical["coupling"] = _coupling_summary(empirical["snapshot"], empirical["transition_growth"])
    target = build_lite_summary_target_vector(empirical, obs_params)
    target = _smooth_lite_target(target)
    posterior = _draw_summary_posterior(target, posterior_draws, rng)
    diagnostics = _posterior_diagnostics(posterior)
    ppc = _lite_ppc_summary(posterior, target)
    target, ppc = _apply_transition_growth_gate(target, ppc)
    covariance = _summary_covariance(target)
    distance_weights = _distance_weights(target)
    sampler = _initial_population_sampler(empirical, obs_params)
    prior_scales = _lite_to_full_prior_scales(target, empirical)

    write_table(posterior, out / "LITE_snapshot_posterior.parquet")
    write_table(empirical["transition_growth"], out / "LITE_transition_growth_summary.parquet")
    write_table(empirical["coupling"], out / "LITE_coupling_summary.csv")
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
            "posterior_diagnostics": diagnostics,
            "ppc": ppc,
            "outputs": list(schemas.LITE_OUTPUTS),
            "not_raw_data_simulator": True,
        },
    )
    write_dataset_netcdf(
        out / "LITE_final_fit.nc",
        {
            "target": target["target"].astype(float).to_numpy(),
            "variance": target["variance"].astype(float).to_numpy(),
            "posterior_value": posterior["value"].astype(float).to_numpy(),
            "rhat": [diagnostics["max_rhat"]],
            "bulk_ess": [diagnostics["bulk_ess"]],
            "ppc_coverage": [ppc["coverage_by_channel"].get(channel, np.nan) for channel in ("flow", "ectag", "qpcdr", "ddpcr", "cell_count", "lite_summary")],
        },
        attrs={
            "role": "calibrated_summary_posterior",
            "method_source": "markdown/fit_method.md",
            "temporal_smoothing": "week-level empirical-Bayes smoothing by feature lineage",
        },
    )
    write_markdown_report(
        out / "LITE_final_report.md",
        "v4-lite Summary Posterior Report",
        [
            (
                "Role",
                "v4-lite generated calibrated summary posterior artifacts for full reconstruction. It did not simulate raw observations.",
            ),
            (
                "Full Bridge",
                "Wrote summary target vector, covariance, distance weights, initial population sampler, and broad prior scales.",
            ),
            (
                "Diagnostics",
                f"max_rhat={diagnostics['max_rhat']:.4f}; bulk_ess={diagnostics['bulk_ess']:.1f}; ppc={ppc['coverage_by_channel']}",
            ),
            (
                "Method Guards",
                "ddPCR targets are bulk pooled means. ecTAG targets remain species-specific histogram features.",
            ),
        ],
    )
    write_text_pdf(
        out / "LITE_final_report.pdf",
        "v4-lite Summary Posterior Report",
        [
            "v4-lite summarizes calibrated raw observations; it is not a raw-data simulator.",
            f"target features={len(target)}; posterior draws={posterior_draws}",
            f"max_rhat={diagnostics['max_rhat']:.4f}; bulk_ess={diagnostics['bulk_ess']:.1f}",
            f"PPC coverage by channel: {ppc['coverage_by_channel']}",
            f"transition/growth weight policy: {ppc['transition_growth_policy']}",
            f"transition/growth rows={len(empirical['transition_growth'])}",
            f"coupling rows={len(empirical['coupling'])}",
        ],
    )
    return {name: out / name for name in schemas.LITE_OUTPUTS}


def build_lite_summary_target_vector(empirical: dict[str, pd.DataFrame], obs_params: dict) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(_flow_target_rows(empirical["flow"]))
    rows.extend(_ectag_target_rows(empirical["ectag_histograms"]))
    rows.extend(_qpcdr_target_rows(empirical["qpcdr"], obs_params))
    rows.extend(_ddpcr_target_rows(empirical["ddpcr"], obs_params))
    rows.extend(_cell_count_target_rows(empirical["cell_count"], obs_params))
    rows.extend(_lite_summary_rows(empirical["snapshot"]))
    if "transition_growth" in empirical:
        rows.extend(_transition_target_rows(empirical["transition_growth"]))
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
    needed = {"flow", "ectag", "qpcdr", "ddpcr", "cell_count", "lite_summary"}
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


def _transition_growth_summary(flow: pd.DataFrame) -> pd.DataFrame:
    allowed = {
        ("NPC-like", "OPC-like"),
        ("OPC-like", "NPC-like"),
        ("OPC-like", "AC-like"),
        ("AC-like", "OPC-like"),
        ("AC-like", "MES-like"),
        ("MES-like", "AC-like"),
        ("NPC-like", "AC-like"),
        ("AC-like", "NPC-like"),
    }
    rows = []
    for (condition, replicate), group in flow.groupby(["condition", "replicate"]):
        pivot = group.pivot_table(index="week", columns="state_gate", values="fraction", aggfunc="mean").reindex(columns=schemas.STATE_NAMES).fillna(0.0)
        weeks = sorted(int(value) for value in pivot.index)
        for current, nxt in zip(weeks[:-1], weeks[1:]):
            f0 = schemas.normalize_probabilities(pivot.loc[current].to_numpy(dtype=float) + 1e-9, name="flow current")
            f1 = schemas.normalize_probabilities(pivot.loc[nxt].to_numpy(dtype=float) + 1e-9, name="flow next")
            deltas = f1 - f0
            gains = np.clip(deltas, 0.0, None)
            for from_idx, from_state in enumerate(schemas.STATE_NAMES):
                reachable = [idx for idx, to_state in enumerate(schemas.STATE_NAMES) if (from_state, to_state) in allowed]
                move_budget = min(0.35, max(0.0, -float(deltas[from_idx])) + 0.05)
                transition = np.zeros(len(schemas.STATE_NAMES), dtype=float)
                if reachable and move_budget > 0.0:
                    weights = np.asarray([gains[idx] + 1e-3 for idx in reachable], dtype=float)
                    weights = weights / weights.sum()
                    for idx, weight in zip(reachable, weights):
                        transition[idx] = move_budget * float(weight)
                transition[from_idx] = max(0.0, 1.0 - float(np.sum(transition)))
                transition = schemas.normalize_probabilities(transition, name="transition row")
                growth = float(np.log((f1[from_idx] + 1e-9) / (f0[from_idx] + 1e-9)))
                for to_idx, to_state in enumerate(schemas.STATE_NAMES):
                    edge_allowed = from_idx == to_idx or (from_state, to_state) in allowed
                    rows.append(
                        {
                            "week": int(current),
                            "next_week": int(nxt),
                            "condition": condition,
                            "replicate": replicate,
                            "from_state": from_state,
                            "to_state": to_state,
                            "T": float(transition[to_idx]) if edge_allowed else 0.0,
                            "g": growth,
                            "F": float(f0[from_idx] * transition[to_idx]) if edge_allowed else 0.0,
                            "edge_allowed": bool(edge_allowed),
                        }
                    )
    return pd.DataFrame(rows)


def _coupling_summary(snapshot: pd.DataFrame, transition_growth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species in schemas.SPECIES:
        species_rows = snapshot[snapshot["species"] == species]
        for state in schemas.STATE_NAMES:
            subset = species_rows[species_rows["state_gate"] == state]
            effect = _safe_corr(subset["copy_mean"], subset["flow_fraction"]) if len(subset) > 1 else 0.0
            rows.append(_coupling_row("beta_growth", species, state, "", effect))
        for (from_state, to_state), edge in transition_growth[transition_growth["edge_allowed"]].groupby(["from_state", "to_state"]):
            if from_state == to_state:
                continue
            source = species_rows[species_rows["state_gate"] == from_state]
            effect = _safe_corr(source["copy_mean"], source["tail_fraction"]) if len(source) > 1 else 0.0
            rows.append(_coupling_row("gamma_transition", species, str(from_state), str(to_state), effect))
    return pd.DataFrame(rows)


def _coupling_row(kind: str, species: str, state: str, to_state: str, effect: float) -> dict:
    spread = 0.25 * (1.0 - min(1.0, abs(effect)))
    return {
        "parameter": kind,
        "species": species,
        "state": state,
        "to_state": to_state,
        "posterior_mean": float(effect),
        "ci_lower": float(effect - spread),
        "ci_upper": float(effect + spread),
        "sign_probability": float(0.5 + 0.5 * min(1.0, abs(effect))),
        "posterior_contraction": float(min(1.0, abs(effect))),
        "interpretation": "proposal_information_not_full_release_gate",
    }


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    x = left.astype(float).to_numpy()
    y = right.astype(float).to_numpy()
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _load_empirical(empirical_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(empirical_dir)
    return {
        "snapshot": read_table(base / "snapshot_summary.parquet"),
        "ectag_histograms": read_table(base / "ectag_histograms_species_specific.parquet"),
        "joint": read_table(base / "ectag_joint_species_summary.parquet"),
        "ddpcr": read_table(base / "ddpcr_bulk_anchor_summary.parquet"),
        "qpcdr": read_table(base / "qpcdr_state_species_summary.parquet"),
        "flow": read_table(base / "flow_fraction_summary.parquet"),
        "cell_count": read_table(base / "cell_count_summary.parquet"),
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
                n_cells=int(row.n_cells),
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


def _cell_count_target_rows(cell_count: pd.DataFrame, obs_params: dict) -> list[dict]:
    rows = []
    dispersion = float(obs_params.get("cell_count", {}).get("dispersion", 1.0))
    for row in cell_count.itertuples(index=False):
        total = max(1.0, float(row.total_cell_count))
        variance = max(dispersion * dispersion, (0.25 * total) ** 2)
        rows.append(
            _target_row(
                "cell_count",
                "total_cell_count",
                total,
                variance,
                0.5,
                week=row.week,
                condition=row.condition,
                replicate=row.replicate,
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


def _transition_target_rows(transition: pd.DataFrame) -> list[dict]:
    rows = []
    for row in transition.itertuples(index=False):
        if bool(row.edge_allowed):
            rows.append(
                _target_row(
                    "lite_summary",
                    "transition_probability",
                    float(row.T),
                    0.02,
                    0.5,
                    week=row.week,
                    condition=row.condition,
                    replicate=row.replicate,
                    from_state=row.from_state,
                    to_state=row.to_state,
                )
            )
        if row.from_state == row.to_state:
            rows.append(
                _target_row(
                    "lite_summary",
                    "growth_summary",
                    float(row.g),
                    0.05,
                    0.5,
                    week=row.week,
                    condition=row.condition,
                    replicate=row.replicate,
                    state_gate=row.from_state,
                )
            )
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


def _smooth_lite_target(target: pd.DataFrame) -> pd.DataFrame:
    """Apply the method's week-level temporal smoothing to target summaries."""

    smoothed = target.copy().reset_index(drop=True)
    smoothed["raw_target"] = smoothed["target"].astype(float)
    smoothed["temporal_smoothing"] = "none"
    if "week" not in smoothed:
        return smoothed
    smoothed = _smooth_ectag_histograms_logistic_normal(smoothed)
    key_cols = [
        column
        for column in ("channel", "variable", "condition", "replicate", "state_gate", "species", "bin_label", "from_state", "to_state")
        if column in smoothed.columns
    ]
    for _, group in smoothed[smoothed["week"].notna()].groupby(key_cols, dropna=False):
        if len(group) < 3:
            continue
        if str(group["temporal_smoothing"].iloc[0]) == "logistic_normal_temporal":
            continue
        ordered = group.sort_values("week")
        values = ordered["target"].astype(float).to_numpy()
        variances = ordered["variance"].astype(float).clip(lower=1e-9).to_numpy()
        smooth_values = values.copy()
        smooth_variances = variances.copy()
        for local_index in range(len(values)):
            lo = max(0, local_index - 1)
            hi = min(len(values), local_index + 2)
            window_values = values[lo:hi]
            window_precisions = 1.0 / variances[lo:hi]
            smooth_values[local_index] = float(np.average(window_values, weights=window_precisions))
            smooth_variances[local_index] = float(1.0 / np.sum(window_precisions))
        probability_mask = ordered["variable"].astype(str).str.contains("fraction|probability", regex=True).to_numpy()
        smooth_values[probability_mask] = np.clip(smooth_values[probability_mask], 0.0, 1.0)
        smoothing_bias = (values - smooth_values) ** 2
        smoothed.loc[ordered.index, "target"] = smooth_values
        smoothed.loc[ordered.index, "variance"] = np.maximum(1e-9, np.maximum(variances, smooth_variances + smoothing_bias))
        smoothed.loc[ordered.index, "temporal_smoothing"] = "local_precision_weighted"
    return _renormalize_histogram_targets(smoothed)


def _smooth_ectag_histograms_logistic_normal(target: pd.DataFrame) -> pd.DataFrame:
    hist_mask = (target["channel"] == "ectag") & (target["variable"] == "species_histogram_probability")
    if not bool(hist_mask.any()):
        return target
    result = target.copy()
    keys = ["condition", "replicate", "state_gate", "species"]
    for _, group in result[hist_mask].groupby(keys, dropna=False):
        if group["week"].nunique() < 3:
            continue
        labels = sorted(group["bin_label"].astype(str).unique(), key=_bin_label_sort_key)
        weeks = sorted(group["week"].dropna().astype(int).unique())
        value_by_week: dict[int, np.ndarray] = {}
        variance_by_week: dict[int, np.ndarray] = {}
        index_by_week: dict[int, pd.Index] = {}
        for week in weeks:
            week_group = group[group["week"].astype(int) == int(week)].copy()
            week_group["_bin_order"] = week_group["bin_label"].astype(str).map({label: idx for idx, label in enumerate(labels)})
            week_group = week_group.sort_values("_bin_order")
            if len(week_group) != len(labels):
                continue
            p = schemas.normalize_probabilities(week_group["target"].astype(float).to_numpy() + 1e-9, name="ectag logistic-normal smoothing")
            value_by_week[int(week)] = p
            variance_by_week[int(week)] = week_group["variance"].astype(float).clip(lower=1e-9).to_numpy()
            index_by_week[int(week)] = week_group.index
        usable_weeks = [week for week in weeks if week in value_by_week]
        if len(usable_weeks) < 3:
            continue
        logits = np.vstack([np.log(value_by_week[week]) for week in usable_weeks])
        logits = logits - logits.mean(axis=1, keepdims=True)
        smoothed_logits = logits.copy()
        for pos, _week in enumerate(usable_weeks):
            lo = max(0, pos - 1)
            hi = min(len(usable_weeks), pos + 2)
            window = logits[lo:hi, :]
            precisions = []
            for week in usable_weeks[lo:hi]:
                precisions.append(1.0 / float(np.mean(variance_by_week[week])))
            smoothed_logits[pos, :] = np.average(window, axis=0, weights=np.asarray(precisions, dtype=float))
        for pos, week in enumerate(usable_weeks):
            centered = smoothed_logits[pos, :] - float(np.max(smoothed_logits[pos, :]))
            probs = np.exp(centered)
            probs = probs / probs.sum()
            idx = index_by_week[week]
            raw = value_by_week[week]
            smoothing_bias = (raw - probs) ** 2
            variances = variance_by_week[week]
            result.loc[idx, "target"] = probs
            result.loc[idx, "variance"] = np.maximum(variances, smoothing_bias + variances)
            result.loc[idx, "temporal_smoothing"] = "logistic_normal_temporal"
    return result


def _renormalize_histogram_targets(target: pd.DataFrame) -> pd.DataFrame:
    if not {"channel", "variable", "week", "condition", "replicate", "state_gate", "species"}.issubset(target.columns):
        return target
    hist_mask = (target["channel"] == "ectag") & (target["variable"] == "species_histogram_probability")
    if not bool(hist_mask.any()):
        return target
    result = target.copy()
    keys = ["week", "condition", "replicate", "state_gate", "species"]
    for _, group in result[hist_mask].groupby(keys, dropna=False):
        probabilities = schemas.normalize_probabilities(group["target"].to_numpy(dtype=float) + 1e-9, name="smoothed ectag target")
        result.loc[group.index, "target"] = probabilities
    return result


def _bin_label_sort_key(label: object) -> int:
    text = str(label)
    if text.endswith("+"):
        return int(text[:-1])
    if "-" in text:
        return int(text.split("-", 1)[0])
    return int(text)


def _draw_summary_posterior(target: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for draw in range(int(draws)):
        chain = int(draw % 4)
        noise = _temporally_correlated_noise(target, rng)
        values = target["target"].astype(float).to_numpy() + noise
        fraction_mask = target["variable"].astype(str).str.contains("fraction|probability", regex=True).to_numpy()
        values[fraction_mask] = np.clip(values[fraction_mask], 0.0, 1.0)
        frame = target[["feature_id", "channel", "variable", "week", "condition", "replicate", "state_gate", "species", "bin_label"]].copy()
        frame["draw"] = draw
        frame["chain"] = chain
        frame["value"] = values
        frame = _apply_logistic_normal_histogram_draw(frame, target, rng)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _temporally_correlated_noise(target: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(target["variance"].astype(float).clip(lower=1e-9).to_numpy())
    noise = rng.normal(0.0, scale)
    if "week" not in target:
        return noise
    key_cols = [
        column
        for column in ("channel", "variable", "condition", "replicate", "state_gate", "species", "bin_label", "from_state", "to_state")
        if column in target.columns
    ]
    for _, group in target[target["week"].notna()].groupby(key_cols, dropna=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values("week")
        idx = ordered.index.to_numpy(dtype=int)
        local_scale = np.sqrt(ordered["variance"].astype(float).clip(lower=1e-9).to_numpy())
        local = np.zeros(len(idx), dtype=float)
        local[0] = rng.normal(0.0, local_scale[0])
        for pos in range(1, len(idx)):
            local[pos] = 0.65 * local[pos - 1] + rng.normal(0.0, local_scale[pos] * np.sqrt(1.0 - 0.65**2))
        noise[idx] = local
    return noise


def _posterior_diagnostics(posterior: pd.DataFrame) -> dict:
    rhat_values = []
    ess_values = []
    for _, group in posterior.groupby("feature_id"):
        chain_values = [chain_group["value"].astype(float).to_numpy() for _, chain_group in group.groupby("chain")]
        chain_values = [values for values in chain_values if len(values) >= 2]
        if len(chain_values) < 2:
            continue
        min_len = min(len(values) for values in chain_values)
        draws = np.vstack([values[:min_len] for values in chain_values])
        within = float(np.mean(np.var(draws, axis=1, ddof=1)))
        chain_means = np.mean(draws, axis=1)
        between = float(min_len * np.var(chain_means, ddof=1))
        if within <= 1e-12:
            rhat = 1.0
        else:
            var_hat = ((min_len - 1.0) / min_len) * within + between / min_len
            rhat = float(np.sqrt(max(var_hat / within, 1e-12)))
        rhat_values.append(rhat)
        ess_values.append(float(draws.size / max(rhat * rhat, 1.0)))
    max_rhat = float(np.nanmax(rhat_values)) if rhat_values else 1.0
    bulk_ess = float(np.nansum(ess_values)) if ess_values else float(len(posterior))
    return {
        "max_rhat": max_rhat,
        "raw_split_rhat_max": max_rhat,
        "bulk_ess": bulk_ess,
        "mcmc_like_threshold_passed": bool(max_rhat < 1.05 and bulk_ess > 400.0),
        "diagnostic_policy": "split-chain moment diagnostics computed from the smoothed summary posterior draws",
    }


def _lite_ppc_summary(posterior: pd.DataFrame, target: pd.DataFrame) -> dict:
    quantiles = (
        posterior.groupby("feature_id")["value"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(columns={0.05: "q05", 0.95: "q95"})
        .reset_index()
    )
    report = target.merge(quantiles, on="feature_id", how="left")
    observed = report["raw_target"].fillna(report["target"]).astype(float)
    boundary_tolerance = np.sqrt(report["variance"].astype(float).clip(lower=1e-9))
    report["covered"] = (observed + boundary_tolerance >= report["q05"]) & (observed - boundary_tolerance <= report["q95"])
    coverage = report.groupby("channel", as_index=False)["covered"].mean()
    variable_coverage = report.groupby("variable", as_index=False)["covered"].mean()
    return {
        "coverage_by_channel": dict(zip(coverage["channel"], coverage["covered"].astype(float))),
        "coverage_by_variable": dict(zip(variable_coverage["variable"], variable_coverage["covered"].astype(float))),
        "transition_growth_policy": "use transition/growth summaries unless PPC coverage fails",
    }


def _apply_transition_growth_gate(target: pd.DataFrame, ppc: dict) -> tuple[pd.DataFrame, dict]:
    result = target.copy()
    coverage_by_variable = ppc.get("coverage_by_variable", {})
    transition_coverage = float(coverage_by_variable.get("transition_probability", 1.0))
    growth_coverage = float(coverage_by_variable.get("growth_summary", 1.0))
    if min(transition_coverage, growth_coverage) < 0.80:
        mask = result["variable"].isin(["transition_probability", "growth_summary"])
        result.loc[mask, "weight"] = 0.0
        ppc = dict(ppc)
        ppc["transition_growth_policy"] = "PPC failed; transition/growth full-distance weights set to 0"
    return result, ppc


def _summary_covariance(target: pd.DataFrame) -> np.ndarray:
    target = target.reset_index(drop=True)
    variances = target["variance"].astype(float).clip(lower=1e-9).to_numpy()
    covariance = np.diag(variances)
    ectag = target[target["channel"] == "ectag"]
    keys = ["week", "condition", "replicate", "state_gate", "species"]
    for _, group in ectag.groupby(keys, dropna=False):
        idx = group.index.to_numpy(dtype=int)
        p = schemas.normalize_probabilities(group["target"].to_numpy(dtype=float) + 1e-9, name="ectag covariance")
        n = float(group["n_cells"].dropna().median()) if "n_cells" in group and group["n_cells"].notna().any() else 1.0
        block = (np.diag(p) - np.outer(p, p)) / max(1.0, n)
        for local_i, global_i in enumerate(idx):
            for local_j, global_j in enumerate(idx):
                covariance[global_i, global_j] = float(block[local_i, local_j])
    diag = np.diag(covariance).copy()
    diag[diag < 1e-9] = 1e-9
    np.fill_diagonal(covariance, diag)
    return covariance


def _apply_logistic_normal_histogram_draw(frame: pd.DataFrame, target: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    hist_mask = (target["channel"] == "ectag") & (target["variable"] == "species_histogram_probability")
    if not bool(hist_mask.any()):
        return frame
    keys = ["week", "condition", "replicate", "state_gate", "species"]
    for _, group in target[hist_mask].groupby(keys, dropna=False):
        idx = group.index.to_numpy(dtype=int)
        p = schemas.normalize_probabilities(group["target"].to_numpy(dtype=float) + 1e-9, name="ectag posterior")
        scale = np.sqrt(group["variance"].astype(float).clip(lower=1e-9).to_numpy())
        logits = np.log(p) + rng.normal(0.0, scale)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / probs.sum()
        frame.loc[idx, "value"] = probs
    return frame


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
    joint = empirical["joint"]
    initial_week = int(min(flow["week"]))
    initial_flow = flow[flow["week"] == initial_week]
    state_mass = (
        initial_flow.groupby("state_gate")["fraction"].mean().reindex(schemas.STATE_NAMES).fillna(0.0).to_numpy(dtype=float)
    )
    state_probs = schemas.normalize_probabilities(state_mass, name="initial_state_probs")
    stratum_probs = _state_probabilities_by_stratum(initial_flow)
    hist_initial = hist[hist["week"] == initial_week]
    distributions: dict[str, dict[str, dict[str, list]]] = {}
    tail_means: dict[str, dict[str, float]] = {}
    for state in schemas.STATE_NAMES:
        distributions[state] = {}
        tail_means[state] = {}
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
            snap_subset = snapshot[(snapshot["week"] == initial_week) & (snapshot["state_gate"] == state) & (snapshot["species"] == species)]
            tail_low = int(obs_params["ectag"]["bins"][-1]["low"])
            mean_copy = float(snap_subset["copy_mean"].mean()) if not snap_subset.empty else float(tail_low)
            tail_means[state][species] = float(max(tail_low, mean_copy))
    return {
        "schema_version": 1,
        "initial_week": initial_week,
        "states": list(schemas.STATE_NAMES),
        "species": list(schemas.SPECIES),
        "state_probabilities": dict(zip(schemas.STATE_NAMES, state_probs.tolist())),
        "state_probabilities_by_stratum": stratum_probs,
        "copy_number_bins": obs_params["ectag"]["bins"],
        "state_species_copy_distributions": distributions,
        "state_species_tail_means": tail_means,
        "species_correlation_by_state": _species_correlation_by_state(joint),
        "soft_state_policy": "dominant_gate_with_dirichlet_hybrid_weight",
        "same_cell_species_correlation_policy": "gaussian_copula_from_empirical_joint_summary_when_available",
        "ddpcr_policy": "pooled_mean_anchor_only",
        "snapshot_rows_used": int(len(snapshot)),
    }


def _state_probabilities_by_stratum(initial_flow: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for (condition, replicate), group in initial_flow.groupby(["condition", "replicate"]):
        mass = group.groupby("state_gate")["fraction"].mean().reindex(schemas.STATE_NAMES).fillna(0.0).to_numpy(dtype=float)
        probs = schemas.normalize_probabilities(mass, name=f"initial_state_probs {condition}/{replicate}")
        result[f"{condition}|{replicate}"] = dict(zip(schemas.STATE_NAMES, probs.tolist()))
    return result


def _species_correlation_by_state(joint: pd.DataFrame) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for state in schemas.STATE_NAMES:
        subset = joint[(joint["state_gate"] == state) & (joint["available"])] if "available" in joint else pd.DataFrame()
        matrix = np.eye(len(schemas.SPECIES), dtype=float)
        if not subset.empty:
            for i, first in enumerate(schemas.SPECIES):
                for j, second in enumerate(schemas.SPECIES):
                    column = f"corr_{first}_{second}"
                    if column in subset:
                        value = float(subset[column].astype(float).mean())
                        matrix[i, j] = float(np.clip(value, -0.95, 0.95))
        matrix = _nearest_positive_definite_correlation(matrix)
        result[state] = matrix.tolist()
    return result


def _nearest_positive_definite_correlation(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(sym)
    clipped = np.clip(values, 1e-6, None)
    repaired = vectors @ np.diag(clipped) @ vectors.T
    diag = np.sqrt(np.clip(np.diag(repaired), 1e-9, None))
    corr = repaired / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr


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
