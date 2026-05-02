"""Final report layer and method-layout compatibility artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_table, write_text_pdf
from fit.ppc import run_full_ppc
from fit.scenarios import classify_scenarios_from_files


def build_final_report_layer(
    observation_dir: str | Path,
    lite_dir: str | Path,
    full_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Build the `FINAL_*` report layer described in fit_method.md."""

    obs_dir = Path(observation_dir)
    lite = Path(lite_dir)
    full = Path(full_dir)
    out = ensure_dir(output_dir)

    classify_scenarios_from_files(full, full)
    ppc_payload = run_full_ppc(full, lite, out)
    raw_table_ppc = read_json(full / "raw_table_ppc_report.json") if (full / "raw_table_ppc_report.json").exists() else {}
    write_text_pdf(
        out / "FINAL_raw_ppc_report.pdf",
        "Final Raw-Compatible PPC Report",
        [
            "Weighted full histories were replayed against flow, qPCDR, ecTAG, ddPCR, cell count, and lite summary targets.",
            "ddPCR is interpreted only as a bulk pooled mean anchor.",
            "ecTAG summaries remain species-specific for MYC, CDK4, and PDGFRA.",
            f"raw_like_channels={ppc_payload.get('raw_like_channels', [])}",
            f"coverage_by_channel={ppc_payload.get('coverage_by_channel', {})}",
            f"raw_table_history_source={raw_table_ppc.get('history_source', '')}",
            f"raw_table_summary_coverage={raw_table_ppc.get('summary_coverage_by_channel', {})}",
        ],
    )

    _copy_store(full / "FULL_particles_final.zarr", out / "FINAL_single_cell_histories.zarr")
    event_summary = _final_event_summary(full)
    write_table(event_summary, out / "FINAL_event_history_summary.parquet")

    scenarios = pd.read_parquet(full / "scenario_classes.parquet")
    write_table(scenarios, out / "FULL_scenario_classes.parquet")
    scenario_summary = (
        scenarios[scenarios["accepted"]]
        .groupby("scenario_class", as_index=False)["posterior_weight"]
        .sum()
        .sort_values("posterior_weight", ascending=False)
    )
    if scenario_summary.empty:
        scenario_summary = scenarios.groupby("scenario_class", as_index=False)["posterior_weight"].sum()
    write_table(scenario_summary, out / "FINAL_scenario_summary.parquet")
    write_text_pdf(
        out / "FINAL_scenario_summary.pdf",
        "Final Scenario Summary",
        [
            "Scenario classes summarize compatible history ensembles, not unique biological rates.",
            *[
                f"{row.scenario_class}: posterior_weight={float(row.posterior_weight):.4f}"
                for row in scenario_summary.itertuples(index=False)
            ],
        ],
    )

    appendix = _parameter_appendix(obs_dir, lite, full)
    write_table(appendix, out / "FINAL_parameter_appendix.csv")
    write_json(
        out / "FINAL_report_manifest.json",
        {
            "schema_version": 1,
            "method_source": "markdown/fit_method.md",
            "outputs": list(schemas.FINAL_OUTPUTS),
            "history_ensemble": str(out / "FINAL_single_cell_histories.zarr"),
            "parameters_are_latent_controls": True,
        },
    )
    return {name: out / name for name in schemas.FINAL_OUTPUTS}


def materialize_method_layout(output_root: str | Path) -> dict[str, Path]:
    """Mirror CLI outputs into the directory names used in fit_method.md."""

    layout = schemas.ResultLayout(Path(output_root))
    paths = {
        "method_lite": _copy_artifact_tree(layout.lite, layout.method_lite),
        "method_full_init": _copy_artifact_tree(layout.full_init, layout.method_full_init),
        "method_full_history": _copy_artifact_tree(layout.full_smc, layout.full_history),
    }
    return paths


def validate_final_artifacts(final_dir: str | Path) -> None:
    base = Path(final_dir)
    missing = [name for name in schemas.FINAL_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing final report artifacts: {', '.join(missing)}")
    scenarios = pd.read_parquet(base / "FULL_scenario_classes.parquet")
    if "scenario_class" not in scenarios:
        raise ValueError("FULL_scenario_classes.parquet must include scenario_class")
    appendix = pd.read_csv(base / "FINAL_parameter_appendix.csv")
    required = {"name", "parameter_type", "reporting_policy"}
    schemas.validate_required_columns(set(appendix.columns), tuple(required), "FINAL_parameter_appendix")


def _final_event_summary(full: Path) -> pd.DataFrame:
    exact_events = full / "FULL_exact_replay_event_summaries.parquet"
    exact_weights = full / "FULL_exact_replay_particle_weights.parquet"
    if exact_events.exists() and exact_weights.exists():
        events = pd.read_parquet(exact_events)
        weights = pd.read_parquet(exact_weights)[["particle_id", "weight", "accepted"]]
    else:
        events = pd.read_parquet(full / "FULL_event_summaries.parquet")
        weights = pd.read_parquet(full / "FULL_particle_weights.parquet")[["particle_id", "weight", "accepted"]]
    if events.empty:
        return pd.DataFrame(columns=["event_type", "species", "weighted_count_mean", "accepted_weighted_count_mean"])
    merged = events.merge(weights, on="particle_id", how="left")
    merged["weighted_count"] = merged["count"].astype(float) * merged["weight"].astype(float)
    merged["accepted_weighted_count"] = np.where(merged["accepted"], merged["weighted_count"], 0.0)
    return (
        merged.groupby(["event_type", "species"], as_index=False)
        .agg(weighted_count_mean=("weighted_count", "sum"), accepted_weighted_count_mean=("accepted_weighted_count", "sum"))
        .sort_values(["event_type", "species"])
    )


def _parameter_appendix(obs_dir: Path, lite: Path, full: Path) -> pd.DataFrame:
    rows: list[dict] = []
    obs = read_json(obs_dir / "obs_params_for_full.json")
    rows.append(
        {
            "name": "observation_calibration",
            "parameter_type": "hard-fixed",
            "source": str(obs_dir / "obs_params_for_full.json"),
            "reporting_policy": "fixed before full reconstruction; not re-estimated by full biology",
            "value_summary": f"locked_for_full={obs.get('locked_for_full')}",
        }
    )
    prior = read_json(lite / "LITE_to_FULL_prior_scales.json")
    for name, value in prior.items():
        if name in {"schema_version", "role", "reporting_policy"}:
            continue
        rows.append(
            {
                "name": str(name),
                "parameter_type": "plausibility-free",
                "source": str(lite / "LITE_to_FULL_prior_scales.json"),
                "reporting_policy": "broad prior scale for latent full controls; not interpreted as a true biological rate",
                "value_summary": str(value),
            }
        )
    params = pd.read_parquet(full / "FULL_particle_parameters.parquet")
    for column in params.columns:
        if column in {"particle_id", "smc_round", "smc_tolerance"}:
            continue
        values = params[column].astype(float)
        rows.append(
            {
                "name": str(column),
                "parameter_type": "plausibility-free",
                "source": str(full / "FULL_particle_parameters.parquet"),
                "reporting_policy": "posterior range is a latent control diagnostic; biological interpretation uses histories and scenarios",
                "value_summary": f"median={values.median():.6g}; q05={values.quantile(0.05):.6g}; q95={values.quantile(0.95):.6g}",
            }
        )
    for name in ("B", "P", "Q"):
        rows.append(
            {
                "name": name,
                "parameter_type": "derived-only",
                "source": str(full / "FULL_derived_Q.parquet"),
                "reporting_policy": "computed from accepted histories; not directly fitted as a full parameter",
                "value_summary": "see FULL_derived_Q.parquet",
            }
        )
    return pd.DataFrame(rows)


def _copy_store(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_artifact_tree(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Cannot mirror missing method artifact tree: {src}")
    ensure_dir(dst)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists() and target.is_file():
                target.unlink()
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            ensure_dir(target.parent)
            shutil.copy2(item, target)
    return dst
