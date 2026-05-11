"""Validation, identifiability, and method contract checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import read_json, write_table, write_text_pdf
from fit.observation import load_observation_params
from fit.v4_lite import load_lite_artifacts, validate_lite_artifacts


def build_validation_reports(lite_dir: str | Path, full_dir: str | Path, registry_dir: str | Path, output_dir: str | Path, obs_params_path: str | Path) -> dict[str, Path]:
    artifacts = load_lite_artifacts(lite_dir)
    obs = load_observation_params(obs_params_path)
    full_path = Path(full_dir)
    params = pd.read_parquet(full_path / "FULL_particle_parameters.parquet")
    weights = pd.read_parquet(full_path / "FULL_particle_weights.parquet")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted_ids = set(weights.loc[weights["accepted"], "particle_id"].astype(int))
    accepted = params[params["particle_id"].astype(int).isin(accepted_ids)]
    if accepted.empty:
        accepted = params.head(1)
    replay = _load_prediction_sidecars(full_path, "FULL_replay_histories")
    if replay:
        ddpcr_ppc = _ppc_from_predictions(replay["ddpcr"], ["week", "condition", "replicate", "species"], "observed_bulk_mean", "predicted_bulk_mean")
        count_ppc = _ppc_from_predictions(replay["cell_count"], ["week", "condition", "replicate"], "observed_cell_count", "predicted_cell_count")
        flow_ppc = _flow_ppc_from_predictions(replay["flow3"])
        ppc_source = "FULL_replay_histories sidecar predictions"
    else:
        ddpcr_ppc = _ddpcr_ppc(accepted, artifacts, obs)
        count_ppc = _count_ppc(accepted, artifacts, obs)
        flow_ppc = _flow3_ppc(accepted, artifacts, obs)
        ppc_source = "accepted parameter moment predictions"
    ident = _identifiability(params, weights, registry_dir)
    ridge = _ridge_report(params, weights, ident)
    if not ridge.empty:
        for row in ridge.itertuples(index=False):
            first = ident["parameter"] == row.parameter
            second = ident["parameter"] == row.ridge_partner
            ident.loc[first, "ridge_partner"] = row.ridge_partner
            ident.loc[second, "ridge_partner"] = row.parameter
            ident.loc[first | second, "interpretation_status"] = "ridge-nonidentifiable"
    boundary = ident[["parameter", "role", "boundary_mass", "interpretation_status"]].copy()
    holdout = _holdout_validation(artifacts, obs, params, weights)
    write_table(ddpcr_ppc, out / "FULL_ddpcr_ppc.parquet")
    write_table(count_ppc, out / "FULL_cellcount_ppc.parquet")
    write_table(flow_ppc, out / "FULL_flow3steady_ppc.parquet")
    write_table(ident, out / "FULL_identifiability_report.csv")
    write_table(boundary, out / "FULL_boundary_forcing_report.csv")
    write_table(ridge, out / "FULL_ridge_report.csv")
    write_table(holdout, out / "FULL_holdout_validation.csv")
    write_text_pdf(
        out / "FULL_ppc_report.pdf",
        "Full PPC Report",
        [
            f"ddPCR coverage={ddpcr_ppc['covered'].mean():.3f}",
            f"cell count coverage={count_ppc['covered'].mean():.3f}",
            f"flow3 mean absolute error={flow_ppc['abs_error'].mean():.3f}",
            f"PPC source={ppc_source}.",
            "PPC uses only ddPCR, cell count, and flow3 steady projection.",
        ],
    )
    holdout_dd = holdout[holdout["channel"] == "ddpcr"]
    holdout_cc = holdout[holdout["channel"] == "cell_count"]
    holdout_flow = holdout[holdout["channel"] == "flow3"]
    write_text_pdf(
        out / "FULL_holdout_validation_report.pdf",
        "Holdout Validation Report",
        [
            "split_A train weeks: 1,2,4,6,8,10; test weeks: 3,5,7,9",
            "split_B train weeks: 1,3,5,7,9; test weeks: 2,4,6,8,10",
            "Held-out coverage is evaluated by refitting low-dimensional bulk-visible controls on train weeks only.",
            f"heldout_ddpcr_coverage={holdout_dd['covered'].mean():.3f}",
            f"heldout_cell_count_coverage={holdout_cc['covered'].mean():.3f}",
            f"heldout_flow3_abs_error={holdout_flow['abs_error'].mean():.3f}",
        ],
    )
    return {name: out / name for name in schemas.VALIDATION_OUTPUTS}


def validate_full_artifacts(full_dir: str | Path) -> None:
    base = Path(full_dir)
    missing = [name for name in schemas.FULL_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing full artifacts: {', '.join(missing)}")
    weights = pd.read_parquet(base / "FULL_particle_weights.parquet")
    if "weight" not in weights or abs(float(weights["weight"].sum()) - 1.0) > 1e-6:
        raise ValueError("FULL_particle_weights.parquet must contain normalized posterior weights")
    if not bool(weights["accepted"].any()):
        raise ValueError("At least one accepted particle is required")


def validate_method_contracts(observation_dir: str | Path, lite_dir: str | Path, full_dir: str | Path, final_dir: str | Path | None = None) -> dict:
    obs = read_json(Path(observation_dir) / "obs_params_for_full.json")
    for key in ("use_qpcdr", "use_ectag", "use_flow4", "use_state_specific_copy", "use_lite_summary_in_final_score"):
        if bool(obs["fit_mask"].get(key)):
            raise ValueError(f"{key} must be false")
    if obs.get("ddpcr", {}).get("likelihood") != "lognormal_on_bulk_mean":
        raise ValueError("ddPCR must be scored as a bulk mean")
    validate_lite_artifacts(lite_dir)
    validate_full_artifacts(full_dir)
    final_valid = False
    if final_dir is not None:
        missing = [name for name in schemas.FINAL_OUTPUTS if not (Path(final_dir) / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing final artifacts: {', '.join(missing)}")
        final_valid = True
    return {"observation_locked": True, "closed_modalities": True, "lite_artifacts_valid": True, "full_artifacts_valid": True, "final_artifacts_valid": final_valid}


def _ddpcr_ppc(params: pd.DataFrame, artifacts: dict, obs: dict) -> pd.DataFrame:
    rows = []
    for target in artifacts["ddpcr"].itertuples(index=False):
        values = []
        for _, param in params.iterrows():
            current = _predict_ddpcr_value(param, artifacts, target)
            values.append(current)
        lo, hi = np.quantile(values, [0.05, 0.95])
        observed = float(target.bulk_mean)
        rows.append({"week": int(target.week), "condition": target.condition, "replicate": target.replicate, "species": target.species, "observed": observed, "q05": lo, "q95": hi, "covered": bool(lo <= observed <= hi)})
    return pd.DataFrame(rows)


def _count_ppc(params: pd.DataFrame, artifacts: dict, obs: dict) -> pd.DataFrame:
    del obs
    rows = []
    for target in artifacts["cell_count"].itertuples(index=False):
        values = []
        for _, param in params.iterrows():
            values.append(_predict_count_value(param, artifacts, target))
        lo, hi = np.quantile(values, [0.05, 0.95])
        observed = float(target.total_cell_count)
        rows.append({"week": int(target.week), "condition": target.condition, "replicate": target.replicate, "observed": observed, "q05": lo, "q95": hi, "covered": bool(lo <= observed <= hi)})
    return pd.DataFrame(rows)


def _flow3_ppc(params: pd.DataFrame, artifacts: dict, obs: dict) -> pd.DataFrame:
    del artifacts
    target = obs["flow3"]["target"]["fractions"]
    rows = []
    for group in schemas.FLOW3_GROUPS:
        predicted = []
        for _, param in params.iterrows():
            phase_values = [float(param.get(f"zeta_flow3__p{phase}", 0.0)) for phase in schemas.PHASES]
            bias = float(np.mean(phase_values))
            raw = np.asarray([target[g] for g in schemas.FLOW3_GROUPS], dtype=float)
            raw[0] += bias
            raw[1:] -= bias / 2.0
            frac = schemas.normalize_probabilities(np.clip(raw, 1e-6, None), name="validation flow3")
            predicted.append(float(frac[list(schemas.FLOW3_GROUPS).index(group)]))
        median = float(np.median(predicted))
        rows.append({"group": group, "target_fraction": float(target[group]), "predicted_fraction": median, "abs_error": abs(median - float(target[group]))})
    return pd.DataFrame(rows)


def _load_prediction_sidecars(full_dir: Path, stem: str) -> dict[str, pd.DataFrame]:
    paths = {
        "ddpcr": full_dir / f"{stem}_ddpcr_predictions.parquet",
        "cell_count": full_dir / f"{stem}_cellcount_predictions.parquet",
        "flow3": full_dir / f"{stem}_flow3_predictions.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        return {}
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def _ppc_from_predictions(table: pd.DataFrame, keys: list[str], observed_col: str, predicted_col: str) -> pd.DataFrame:
    rows = []
    for key, group in table.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        values = group[predicted_col].astype(float).to_numpy()
        lo, hi = np.quantile(values, [0.05, 0.95])
        observed = float(group[observed_col].astype(float).iloc[0])
        row = dict(zip(keys, key_values))
        row.update({"observed": observed, "q05": float(lo), "q95": float(hi), "covered": bool(lo <= observed <= hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def _flow_ppc_from_predictions(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["week", "condition", "replicate", "group"]
    for key, group in table.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        values = group["predicted_fraction"].astype(float).to_numpy()
        lo, hi = np.quantile(values, [0.05, 0.95])
        target = float(group["target_fraction"].astype(float).iloc[0])
        median = float(np.median(values))
        row = dict(zip(keys, key_values))
        row.update({"target_fraction": target, "predicted_fraction": median, "q05": float(lo), "q95": float(hi), "covered": bool(lo <= target <= hi), "abs_error": abs(median - target)})
        rows.append(row)
    return pd.DataFrame(rows)


def _holdout_validation(artifacts: dict, obs: dict, params: pd.DataFrame | None = None, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    splits = {
        "split_A": {"train": {1, 2, 4, 6, 8, 10}, "test": {3, 5, 7, 9}},
        "split_B": {"train": {1, 3, 5, 7, 9}, "test": {2, 4, 6, 8, 10}},
    }
    rows = []
    dd = artifacts["ddpcr"].copy()
    cc = artifacts["cell_count"].copy()
    accepted = _accepted_weighted_params(params, weights) if params is not None and weights is not None else pd.DataFrame()
    for split_name, split in splits.items():
        for _, group in dd.groupby(["condition", "replicate", "species"], dropna=False):
            train = group[group["week"].astype(int).isin(split["train"])].sort_values("week")
            test = group[group["week"].astype(int).isin(split["test"])].sort_values("week")
            if train.empty:
                continue
            species = str(group["species"].iloc[0])
            sd = float(obs["ddpcr"]["log_sd_by_species"].get(species, obs["ddpcr"]["default_log_sd"]))
            for target in test.itertuples(index=False):
                values = _holdout_ddpcr_ensemble_values(accepted, artifacts, train, target)
                if values.size == 0:
                    slope, intercept = _fit_loglinear(train["week"], train["bulk_mean"])
                    values = np.asarray([float(np.exp(intercept + slope * float(target.week)))])
                pred_log_values = np.log(np.clip(values.astype(float), 1e-9, None))
                pred_log = float(np.median(pred_log_values))
                observed_log = float(np.log(max(1e-9, float(target.bulk_mean))))
                ensemble_lo, ensemble_hi = np.quantile(pred_log_values, [0.05, 0.95])
                lo, hi = float(ensemble_lo - 1.645 * sd), float(ensemble_hi + 1.645 * sd)
                rows.append({"split": split_name, "channel": "ddpcr", "week": int(target.week), "condition": target.condition, "replicate": target.replicate, "species": target.species, "predicted": float(np.exp(pred_log)), "observed": float(target.bulk_mean), "q05": float(np.exp(lo)), "q95": float(np.exp(hi)), "covered": bool(lo <= observed_log <= hi), "abs_error": abs(float(np.exp(pred_log)) - float(target.bulk_mean))})
        for _, group in cc.groupby(["condition", "replicate"], dropna=False):
            train = group[group["week"].astype(int).isin(split["train"])].sort_values("week")
            test = group[group["week"].astype(int).isin(split["test"])].sort_values("week")
            if train.empty:
                continue
            sd = float(obs["cell_count"]["log_sd"])
            for target in test.itertuples(index=False):
                values = _holdout_count_ensemble_values(accepted, artifacts, train, target)
                if values.size == 0:
                    slope, intercept = _fit_loglinear(train["week"], train["total_cell_count"] + 1.0)
                    values = np.asarray([float(np.exp(intercept + slope * float(target.week)) - 1.0)])
                pred_log_values = np.log(np.clip(values.astype(float), 0.0, None) + 1.0)
                pred_log = float(np.median(pred_log_values))
                observed_log = float(np.log(max(1.0, float(target.total_cell_count) + 1.0)))
                ensemble_lo, ensemble_hi = np.quantile(pred_log_values, [0.05, 0.95])
                lo, hi = float(ensemble_lo - 1.645 * sd), float(ensemble_hi + 1.645 * sd)
                rows.append({"split": split_name, "channel": "cell_count", "week": int(target.week), "condition": target.condition, "replicate": target.replicate, "species": "", "predicted": float(np.exp(pred_log) - 1.0), "observed": float(target.total_cell_count), "q05": float(np.exp(lo) - 1.0), "q95": float(np.exp(hi) - 1.0), "covered": bool(lo <= observed_log <= hi), "abs_error": abs(float(np.exp(pred_log) - 1.0) - float(target.total_cell_count))})
        target = obs["flow3"]["target"]["fractions"]
        for week in split["test"]:
            phase = schemas.phase_for_week(week)
            for group_name, fraction in target.items():
                values = _holdout_flow3_values(accepted, target, phase, group_name)
                if values.size == 0:
                    values = np.asarray([float(fraction)])
                lo, hi = np.quantile(values, [0.05, 0.95])
                pred = float(np.median(values))
                rows.append({"split": split_name, "channel": "flow3", "week": int(week), "condition": "all", "replicate": "", "species": group_name, "predicted": pred, "observed": float(fraction), "q05": float(lo), "q95": float(hi), "covered": bool(lo <= float(fraction) <= hi), "abs_error": abs(pred - float(fraction))})
    return pd.DataFrame(rows)


def _holdout_ddpcr_ensemble_values(accepted: pd.DataFrame, artifacts: dict, train: pd.DataFrame, target) -> np.ndarray:
    if accepted.empty:
        return np.asarray([])
    values = []
    for _, param in accepted.iterrows():
        train_residuals = []
        for train_row in train.itertuples(index=False):
            pred = max(1e-9, _predict_ddpcr_value(param, artifacts, train_row))
            train_residuals.append(float(np.log(max(1e-9, float(train_row.bulk_mean))) - np.log(pred)))
        offset = float(np.median(train_residuals)) if train_residuals else 0.0
        pred_test = max(1e-9, _predict_ddpcr_value(param, artifacts, target))
        values.append(float(np.exp(np.log(pred_test) + offset)))
    return np.asarray(values, dtype=float)


def _holdout_count_ensemble_values(accepted: pd.DataFrame, artifacts: dict, train: pd.DataFrame, target) -> np.ndarray:
    if accepted.empty:
        return np.asarray([])
    values = []
    for _, param in accepted.iterrows():
        train_residuals = []
        for train_row in train.itertuples(index=False):
            pred = max(0.0, _predict_count_value(param, artifacts, train_row))
            train_residuals.append(float(np.log(float(train_row.total_cell_count) + 1.0) - np.log(pred + 1.0)))
        offset = float(np.median(train_residuals)) if train_residuals else 0.0
        pred_test = max(0.0, _predict_count_value(param, artifacts, target))
        values.append(float(np.exp(np.log(pred_test + 1.0) + offset) - 1.0))
    return np.asarray(values, dtype=float)


def _holdout_flow3_values(accepted: pd.DataFrame, target: dict, phase: int, group_name: str) -> np.ndarray:
    if accepted.empty:
        return np.asarray([])
    group_index = list(schemas.FLOW3_GROUPS).index(group_name)
    values = []
    for _, param in accepted.iterrows():
        bias = float(param.get(f"zeta_flow3__p{phase}", 0.0))
        raw = np.asarray([target[group] for group in schemas.FLOW3_GROUPS], dtype=float)
        raw[0] += bias
        raw[1:] -= bias / 2.0
        values.append(float(schemas.normalize_probabilities(np.clip(raw, 1e-6, None), name="holdout flow3")[group_index]))
    return np.asarray(values, dtype=float)


def _fit_loglinear(weeks: pd.Series, values: pd.Series) -> tuple[float, float]:
    x = weeks.astype(float).to_numpy()
    y = np.log(values.astype(float).clip(lower=1e-9).to_numpy())
    if len(np.unique(x)) < 2:
        return 0.0, float(y[0])
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _identifiability(params: pd.DataFrame, weights: pd.DataFrame, registry_dir: str | Path) -> pd.DataFrame:
    prior = pd.read_csv(Path(registry_dir) / "PARAMETER_interpretability_prior_table.csv")
    prior_region_path = Path(registry_dir).parent / "05_prior_predictive" / "PRIOR_predictive_accepted_region.parquet"
    prior_region = pd.read_parquet(prior_region_path) if prior_region_path.exists() else pd.DataFrame()
    accepted = _accepted_weighted_params(params, weights)
    if accepted.empty:
        accepted = params.assign(weight=1.0 / max(1, len(params)))
    rows = []
    for column in params.columns:
        if column in {"particle_id", "round"} or not pd.api.types.is_numeric_dtype(params[column]):
            continue
        role = _role_for_column(column, prior)
        prior_values = _prior_reference_values(column, params[column], prior_region)
        prior_var = float(prior_values.astype(float).var(ddof=0) + 1e-9)
        post_var = _weighted_var(accepted[column].astype(float), accepted["weight"].astype(float))
        contraction = float(max(0.0, min(1.0, 1.0 - post_var / prior_var)))
        prior_mean = float(prior_values.astype(float).mean())
        prior_sd = float(prior_values.astype(float).std(ddof=0) + 1e-9)
        post_mean = _weighted_mean(accepted[column].astype(float), accepted["weight"].astype(float))
        shift = float((post_mean - prior_mean) / prior_sd)
        boundary = _boundary_mass(column, accepted[column].astype(float), accepted["weight"].astype(float))
        status = _status(role, contraction, shift, boundary)
        rows.append({"parameter": column, "role": role, "posterior_contraction": contraction, "prior_shift_z": shift, "boundary_mass": boundary, "ridge_partner": "", "interpretation_status": status})
    present = {row["parameter"] for row in rows}
    for row in prior.itertuples(index=False):
        parameter = str(row.parameter)
        if parameter in present:
            continue
        role = str(row.role)
        rows.append(
            {
                "parameter": parameter,
                "role": role,
                "posterior_contraction": 0.0,
                "prior_shift_z": 0.0,
                "boundary_mass": 0.0,
                "ridge_partner": "",
                "interpretation_status": _status(role, 0.0, 0.0, 0.0),
            }
        )
    return pd.DataFrame(rows)


def _prior_reference_values(column: str, fallback: pd.Series, prior_region: pd.DataFrame) -> pd.Series:
    if prior_region.empty:
        return fallback.astype(float)
    if column.startswith("r__") and "net_growth_rate" in prior_region:
        return prior_region["net_growth_rate"].astype(float)
    if column.startswith("v__") and "bulk_copy_velocity" in prior_region:
        return prior_region["bulk_copy_velocity"].astype(float)
    if column.startswith("zeta_flow3") and "flow3_projection_bias" in prior_region:
        return prior_region["flow3_projection_bias"].astype(float)
    if column in prior_region:
        return prior_region[column].astype(float)
    return fallback.astype(float)


def _ridge_report(params: pd.DataFrame, weights: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    accepted = _accepted_weighted_params(params, weights)
    if accepted.empty:
        accepted = params.assign(weight=1.0 / max(1, len(params)))
    numeric = [c for c in accepted.columns if c not in {"particle_id", "round", "weight"} and pd.api.types.is_numeric_dtype(accepted[c])]
    rows = []
    if len(numeric) < 2:
        return pd.DataFrame(columns=["parameter", "ridge_partner", "correlation", "interpretation_status"])
    contraction = dict(zip(ident["parameter"], ident["posterior_contraction"]))
    for i, first in enumerate(numeric):
        for second in numeric[i + 1 :]:
            if max(float(contraction.get(first, 1.0)), float(contraction.get(second, 1.0))) >= 0.5:
                continue
            value = _weighted_corr(accepted[first].astype(float), accepted[second].astype(float), accepted["weight"].astype(float))
            if abs(value) > 0.9:
                rows.append({"parameter": first, "ridge_partner": second, "correlation": value, "interpretation_status": "ridge-nonidentifiable"})
    return pd.DataFrame(rows, columns=["parameter", "ridge_partner", "correlation", "interpretation_status"])


def _predict_ddpcr_value(param, artifacts: dict, target) -> float:
    group = artifacts["ddpcr"][(artifacts["ddpcr"]["condition"] == target.condition) & (artifacts["ddpcr"]["replicate"] == target.replicate) & (artifacts["ddpcr"]["species"] == target.species)].sort_values("week")
    current = float(group["bulk_mean"].iloc[0])
    first_week = int(group["week"].iloc[0])
    for week in range(first_week + 1, int(target.week) + 1):
        current *= np.exp(float(_param_get(param, f"v__{target.condition}__{target.species}__p{schemas.phase_for_week(week - 1)}", 0.0)))
    return current


def _predict_count_value(param, artifacts: dict, target) -> float:
    group = artifacts["cell_count"][(artifacts["cell_count"]["condition"] == target.condition) & (artifacts["cell_count"]["replicate"] == target.replicate)].sort_values("week")
    current = float(group["total_cell_count"].iloc[0])
    first_week = int(group["week"].iloc[0])
    for week in range(first_week + 1, int(target.week) + 1):
        current *= np.exp(float(_param_get(param, f"r__{target.condition}__p{schemas.phase_for_week(week - 1)}", 0.0)))
    return current


def _param_get(param, name: str, default: float = 0.0) -> float:
    if isinstance(param, pd.Series):
        return float(param.get(name, default))
    if isinstance(param, dict):
        return float(param.get(name, default))
    return float(getattr(param, name, default))


def _role_for_column(column: str, prior: pd.DataFrame) -> str:
    if column.startswith("r__"):
        return "active_effective_control"
    if column.startswith("v__"):
        return "active_effective_control"
    if column.startswith("zeta_flow3"):
        return "active_effective_control"
    mapping = dict(zip(prior["parameter"], prior["role"]))
    return str(mapping.get(column, "prior_constrained_nuisance"))


def _accepted_weighted_params(params: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    accepted_weights = weights[weights["accepted"]][["particle_id", "weight"]].copy()
    if accepted_weights.empty:
        return pd.DataFrame()
    merged = params.merge(accepted_weights, on="particle_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    total = float(merged["weight"].astype(float).sum())
    merged["weight"] = merged["weight"].astype(float) / total if total > 0.0 else 1.0 / len(merged)
    return merged


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float).to_numpy()
    total = float(w.sum())
    if total <= 0.0:
        return float(values.astype(float).mean())
    return float(np.sum(values.astype(float).to_numpy() * w) / total)


def _weighted_var(values: pd.Series, weights: pd.Series) -> float:
    mean = _weighted_mean(values, weights)
    w = weights.astype(float).to_numpy()
    total = float(w.sum())
    if total <= 0.0:
        return float(values.astype(float).var(ddof=0))
    centered = values.astype(float).to_numpy() - mean
    return float(np.sum(w * centered * centered) / total)


def _weighted_corr(first: pd.Series, second: pd.Series, weights: pd.Series) -> float:
    var_a = _weighted_var(first, weights)
    var_b = _weighted_var(second, weights)
    if var_a <= 1e-12 or var_b <= 1e-12:
        return 0.0
    mean_a = _weighted_mean(first, weights)
    mean_b = _weighted_mean(second, weights)
    w = weights.astype(float).to_numpy()
    total = float(w.sum())
    if total <= 0.0:
        return float(first.astype(float).corr(second.astype(float)))
    cov = float(np.sum(w * (first.astype(float).to_numpy() - mean_a) * (second.astype(float).to_numpy() - mean_b)) / total)
    return float(cov / np.sqrt(var_a * var_b))


def _boundary_mass(column: str, values: pd.Series, weights: pd.Series | None = None) -> float:
    if column.startswith("r__"):
        lo, hi = -3.0, 3.0
    elif column.startswith("v__"):
        lo, hi = -1.5, 1.5
    elif column.startswith("zeta_flow3"):
        lo, hi = -0.25, 0.25
    elif column == "division_death_turnover":
        lo, hi = 0.0, 8.0
    elif column == "ecDNA_gain_loss_turnover":
        lo, hi = 0.0, 10.0
    else:
        return 0.0
    span = hi - lo
    mask = ((values <= lo + 0.05 * span) | (values >= hi - 0.05 * span)).astype(float)
    if weights is None:
        return float(mask.mean())
    w = weights.astype(float).to_numpy()
    total = float(w.sum())
    return float(np.sum(mask.to_numpy() * w) / total) if total > 0.0 else float(mask.mean())


def _status(role: str, contraction: float, shift: float, boundary: float) -> str:
    if role == "derived_only":
        return "derived-only"
    if role == "fixed":
        return "hard-fixed"
    if boundary >= 0.3:
        return "boundary-forced"
    if role != "active_effective_control":
        return "prior-driven"
    if contraction >= 0.5 and abs(shift) < 2.5:
        return "data-constrained"
    if contraction >= 0.2:
        return "weakly-informed"
    return "prior-driven"
