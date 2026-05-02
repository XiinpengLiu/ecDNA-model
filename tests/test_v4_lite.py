import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import ZipStore

import fit
from fit import schemas
from fit.full_smc import run_full_reconstruction
from fit.objective import SCORE_COMPONENTS
from fit.observation import calculate_ddpcr_pooled_mean, fit_observation_model, validate_observation_params
from fit.raw import create_synthetic_raw_dataset, ingest_raw_data, validate_raw_tables
from fit.scenarios import classify_scenarios
from fit.stage_runner import run_pipeline_from_raw
from fit.v4_lite import fit_v4_lite_summary_posterior, validate_lite_artifacts


@pytest.fixture()
def workdir():
    base = Path.cwd() / "tmp_test_outputs"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def small_pipeline(workdir):
    raw_dir = workdir / "raw"
    results = workdir / "results"
    create_synthetic_raw_dataset(raw_dir, seed=7)
    run_pipeline_from_raw(raw_dir, results, seed=7, posterior_draws=4, particles=4, cells=36)
    return results


def test_import_main_fit_modules():
    assert hasattr(fit, "fit_observation_model")
    for module in (
        "fit.raw",
        "fit.manifest",
        "fit.final_report",
        "fit.observation",
        "fit.empirical",
        "fit.v4_lite",
        "fit.full_smc",
        "fit.full_raw_ppc",
        "fit.full_exact_replay",
        "fit.objective",
        "fit.scenarios",
        "fit.validation",
    ):
        __import__(module)


def test_schema_validation_identifies_missing_required_fields():
    flow = pd.DataFrame(
        {
            "week": [1],
            "condition": ["ctrl"],
            "replicate": ["r1"],
            "state_gate": ["NPC-like"],
            "pre_sort_count": [10],
            "post_sort_count": [9],
            "fraction": [1.0],
            "sort_purity": [np.nan],
            "marker_panel": ["m"],
        }
    )
    tables = {name: pd.DataFrame(columns=columns) for name, columns in schemas.RAW_TABLE_SCHEMAS.items()}
    tables["flow"] = flow
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_tables(tables)

    params = {"locked_for_full": True, "flow": {}, "ddpcr": {}, "ectag": {}, "ddpcr_interpretation": "bulk"}
    with pytest.raises(ValueError, match="missing required fields"):
        validate_observation_params(params)


def test_ddpcr_pooled_mean_formula():
    flow = pd.DataFrame(
        {
            "week": [1, 1],
            "condition": ["ctrl", "ctrl"],
            "replicate": ["r1", "r1"],
            "state_gate": ["NPC-like", "OPC-like"],
            "fraction": [0.8, 0.2],
        }
    )
    mu = pd.DataFrame(
        {
            "week": [1, 1],
            "condition": ["ctrl", "ctrl"],
            "replicate": ["r1", "r1"],
            "state_gate": ["NPC-like", "OPC-like"],
            "species": ["MYC", "MYC"],
            "copy_mean": [0.0, 225.0],
        }
    )
    pooled = calculate_ddpcr_pooled_mean(flow, mu)
    assert pooled.loc[0, "bulk_mean"] == pytest.approx(45.0)


def test_ectag_histogram_is_species_specific(small_pipeline):
    hist = pd.read_parquet(small_pipeline / "03_empirical_summary" / "ectag_histograms_species_specific.parquet")
    assert set(hist["species"]) == set(schemas.SPECIES)
    group_cols = ["week", "condition", "replicate", "state_gate", "species"]
    assert hist.groupby(group_cols)["probability"].sum().round(8).eq(1.0).all()
    assert "total" not in {str(value).lower() for value in hist["species"].unique()}


def test_lite_output_artifact_schema_complete(small_pipeline):
    lite_dir = small_pipeline / "03_v4_lite"
    validate_lite_artifacts(lite_dir)
    for name in schemas.LITE_OUTPUTS:
        assert (lite_dir / name).exists()
    target = pd.read_parquet(lite_dir / "LITE_summary_target_vector.parquet")
    assert {"flow", "ectag", "qpcdr", "ddpcr", "cell_count", "lite_summary"}.issubset(set(target["channel"]))
    assert (target.loc[target["channel"] == "ectag", "n_cells"].dropna() > 0).all()
    assert "logistic_normal_temporal" in set(target.loc[target["channel"] == "ectag", "temporal_smoothing"])
    assert (lite_dir / "LITE_transition_growth_summary.parquet").exists()
    assert (lite_dir / "LITE_coupling_summary.csv").exists()
    covariance = np.load(lite_dir / "LITE_summary_covariance.npz", allow_pickle=True)
    assert covariance["covariance"].shape == (len(target), len(target))
    sampler = json.loads((lite_dir / "LITE_initial_population_sampler.json").read_text(encoding="utf-8"))
    assert sampler["ddpcr_policy"] == "pooled_mean_anchor_only"
    fit_payload = json.loads((lite_dir / "LITE_final_fit.json").read_text(encoding="utf-8"))
    assert fit_payload["posterior_diagnostics"]["max_rhat"] <= 1.05
    assert fit_payload["posterior_diagnostics"]["bulk_ess"] > 0
    assert fit_payload["ppc"]["coverage_by_channel"]["ectag"] >= 0.8
    assert {"raw_target", "temporal_smoothing"}.issubset(set(target.columns))


def test_observation_calibration_has_model_ppc_artifacts(small_pipeline):
    report = json.loads((small_pipeline / "02_observation_model" / "obs_calibration_report.json").read_text(encoding="utf-8"))
    ppc = report["ppc"]
    assert ppc["continue_gate_passed"]
    assert ppc["coverage_by_channel"]["qpcdr"] >= 0.85
    assert ppc["pass_thresholds"]["ddpcr_interpretation_pooled_mean_only"]
    assert ppc["flow_ppc"]["model"] == "DirichletMultinomial replicated flow fractions"
    assert ppc["flow_ppc"]["coverage"] >= 0.85
    assert ppc["flow_purity_sensitivity"]["no_direction_reversal"]
    params = json.loads((small_pipeline / "02_observation_model" / "obs_params_for_full.json").read_text(encoding="utf-8"))
    assert params["ectag"]["likelihood"] == "dirichlet_multinomial_species_specific_bins"
    for species in schemas.SPECIES:
        assert params["qpcdr"]["by_species"][species]["n_calibration_rows"] > 0


def test_full_particle_score_reads_lite_and_observation_outputs(small_pipeline):
    weights = pd.read_parquet(small_pipeline / "05_full_smc" / "particle_weights.parquet")
    target = pd.read_parquet(small_pipeline / "03_v4_lite" / "LITE_summary_target_vector.parquet")
    features = pd.read_parquet(small_pipeline / "05_full_smc" / "particle_summary_features.parquet")
    snapshots = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_snapshot_summaries.parquet")
    for component in SCORE_COMPONENTS:
        assert component in weights.columns
    assert weights["flow"].sum() > 0
    assert weights["qpcdr"].sum() > 0
    assert weights["ectag"].sum() > 0
    assert weights["ddpcr"].sum() > 0
    assert weights["cell_count"].sum() >= 0
    assert weights["lite_summary"].sum() > 0
    assert weights["weight"].sum() == pytest.approx(1.0)
    transition_ids = set(target.loc[target["variable"] == "transition_probability", "feature_id"])
    assert transition_ids
    transition_features = features[features["feature_id"].isin(transition_ids)]
    assert not transition_features.empty
    assert transition_features["value"].between(0.0, 1.0).all()
    bin_columns = [column for column in snapshots.columns if column.startswith("bin_probability__")]
    assert bin_columns
    assert snapshots[bin_columns].sum(axis=1).round(8).eq(1.0).all()
    ectag_target = target[(target["channel"] == "ectag") & (~target["bin_label"].isin(["0"]))]
    row = ectag_target.iloc[0]
    snapshot_row = snapshots[
        (snapshots["particle_id"] == features["particle_id"].iloc[0])
        & (snapshots["week"] == row.week)
        & (snapshots["condition"] == row.condition)
        & (snapshots["replicate"] == row.replicate)
        & (snapshots["state_gate"] == row.state_gate)
        & (snapshots["species"] == row.species)
    ].iloc[0]
    column = "bin_probability__" + str(row.bin_label).replace("+", "plus").replace("-", "_")
    feature_value = features[(features["particle_id"] == features["particle_id"].iloc[0]) & (features["feature_id"] == row.feature_id)]["value"].iloc[0]
    assert feature_value == pytest.approx(snapshot_row[column])
    ppc_raw = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_ppc_raw_observables.parquet")
    accepted_ids = set(weights.loc[weights["accepted"], "particle_id"])
    assert set(ppc_raw["particle_id"]).issubset(accepted_ids)
    assert {"posterior_weight", "target_weight", "channel"}.issubset(ppc_raw.columns)
    assert "cell_count" in set(ppc_raw["channel"])
    ppc_payload = json.loads((small_pipeline / "05_full_smc" / "full_ppc_report.json").read_text(encoding="utf-8"))
    assert ppc_payload["ppc_particle_scope"] == "accepted_particles_only_with_renormalized_weights"
    assert "cell_count" in ppc_payload["coverage_by_channel"]
    synthetic_tables = {
        "flow": pd.read_parquet(small_pipeline / "05_full_smc" / "synthetic_flow_long.parquet"),
        "qpcdr": pd.read_parquet(small_pipeline / "05_full_smc" / "synthetic_qpcdr_long.parquet"),
        "ectag": pd.read_parquet(small_pipeline / "05_full_smc" / "synthetic_ectag_cell_long.parquet"),
        "ddpcr": pd.read_parquet(small_pipeline / "05_full_smc" / "synthetic_ddpcr_long.parquet"),
        "cell_count": pd.read_parquet(small_pipeline / "05_full_smc" / "synthetic_cell_count_long.parquet"),
    }
    validate_raw_tables(synthetic_tables)
    raw_ppc = json.loads((small_pipeline / "05_full_smc" / "raw_table_ppc_report.json").read_text(encoding="utf-8"))
    assert raw_ppc["history_source"] == "exact_replay_accepted_histories"
    assert {"summary_coverage_by_channel", "replicate_diagnostic_rows"}.issubset(raw_ppc)
    replicate_diagnostics = pd.read_parquet(small_pipeline / "05_full_smc" / "raw_table_ppc_replicate_diagnostics.parquet")
    assert not replicate_diagnostics.empty
    assert {"channel", "replicate", "coverage", "mean_abs_error"}.issubset(replicate_diagnostics.columns)
    exact_events = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_exact_replay_event_summaries.parquet")
    exact_scores = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_exact_replay_scores.parquet")
    exact_weights = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_exact_replay_particle_weights.parquet")
    exact_log = pd.read_parquet(small_pipeline / "05_full_smc" / "FULL_exact_replay_event_log.parquet")
    assert not exact_events.empty
    assert {"division", "death", "gain", "loss", "transition", "segregation", "state_checkpoint"}.intersection(set(exact_log["event_type"]))
    assert {"score", "coverage_flow", "coverage_ectag"}.issubset(exact_scores.columns)
    assert exact_weights["accepted"].any()
    scenarios = pd.read_parquet(small_pipeline / "05_full_smc" / "scenario_classes.parquet")
    assert set(scenarios["scenario_source"]) == {"exact_replay_event_summaries"}
    assert not scenarios["provisional"].any()


def test_full_reconstruction_outputs_history_ensemble_not_single_parameter(workdir):
    raw_dir = workdir / "raw"
    results = workdir / "results"
    create_synthetic_raw_dataset(raw_dir, seed=12)
    ingest_raw_data(raw_dir, results / "01_clean_data")
    fit_observation_model(results / "01_clean_data", results / "02_observation_model", seed=12)
    fit.build_empirical_summaries(
        results / "01_clean_data",
        results / "02_observation_model" / "obs_params_for_lite.json",
        results / "03_empirical_summary",
    )
    fit_v4_lite_summary_posterior(
        results / "03_empirical_summary",
        results / "02_observation_model" / "obs_params_for_lite.json",
        results / "03_v4_lite",
        seed=12,
        posterior_draws=3,
    )
    run_full_reconstruction(
        results / "03_v4_lite",
        results / "02_observation_model" / "obs_params_for_full.json",
        results / "05_full_smc",
        particles=3,
        cells=24,
        seed=12,
    )
    histories = (results / "05_full_smc" / "accepted_histories.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(histories) > 1
    assert (results / "05_full_smc" / "particle_parameters.parquet").exists()
    assert (results / "05_full_smc" / "scenario_classes.parquet").exists()
    history_table = pd.read_parquet(results / "05_full_smc" / "FULL_single_cell_history_samples.parquet")
    for column in ["R", "V", "A", "latent_U_1", "K_MYC", "K_CDK4", "K_PDGFRA", "X_NPC-like"]:
        assert column in history_table.columns
    zarr_root = zarr.open_group(store=ZipStore(str(results / "05_full_smc" / "FULL_particles_final.zarr"), mode="r"), mode="r")
    assert zarr_root.attrs["role"] == "accepted conditional single-cell history ensemble"
    assert set(zarr_root.group_keys()) == {"history", "weights", "events"}
    assert zarr_root["history/K"].shape[1] == len(schemas.SPECIES)
    manifest = json.loads((results / "05_full_smc" / "full_reconstruction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["obs_params_locked"]
    assert "core.dynamics" in manifest["full_v4_chain_policy"]
    assert manifest["method_n_sim_fit"] == 10000
    assert manifest["method_n_sim_replay"] == 50000
    assert manifest["full_continue_diagnostics"]["accepted_particle_ess"] > 0


def test_scenario_classification_handles_synthetic_particles():
    events = pd.DataFrame(
        [
            {"particle_id": 1, "week": 1, "event_type": "gain", "species": "MYC", "count": 10},
            {"particle_id": 1, "week": 1, "event_type": "loss", "species": "MYC", "count": 10},
            {"particle_id": 2, "week": 1, "event_type": "transition", "species": "", "count": 8},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"particle_id": 1, "week": 1, "state_gate": "NPC-like", "species": "MYC", "flow_fraction": 0.5, "copy_mean": 1.0, "tail_fraction": 0.0},
            {"particle_id": 1, "week": 2, "state_gate": "NPC-like", "species": "MYC", "flow_fraction": 0.5, "copy_mean": 1.0, "tail_fraction": 0.1},
            {"particle_id": 2, "week": 1, "state_gate": "NPC-like", "species": "MYC", "flow_fraction": 0.8, "copy_mean": 1.0, "tail_fraction": 0.0},
            {"particle_id": 2, "week": 2, "state_gate": "NPC-like", "species": "MYC", "flow_fraction": 0.2, "copy_mean": 1.0, "tail_fraction": 0.0},
        ]
    )
    weights = pd.DataFrame({"particle_id": [1, 2], "weight": [0.6, 0.4], "score": [1.0, 2.0], "accepted": [True, True]})
    result = classify_scenarios(events, snapshots, weights)
    assert set(result["scenario_class"]).issubset(
        {"selection-dominant", "turnover-dominant", "transition-dominant", "segregation-dominant", "mixed", "measurement-conflict"}
    )
    assert len(result) == 2
    assert "scenario_cluster_id" in result


def test_cli_synthetic_smoke_runs(workdir):
    output = workdir / "cli_smoke"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fit.run_fit",
            "run-synthetic-smoke",
            "--output",
            str(output),
            "--seed",
            "5",
            "--draws",
            "3",
            "--particles",
            "3",
            "--cells",
            "24",
        ],
        check=True,
    )
    assert (output / "results" / "02_observation_model" / "obs_params_for_full.json").exists()
    assert (output / "results" / "03_v4_lite" / "LITE_summary_target_vector.parquet").exists()
    assert (output / "results" / "05_full_smc" / "accepted_histories.jsonl").exists()
    assert (output / "results" / "04_v4_lite" / "LITE_summary_target_vector.parquet").exists()
    assert (output / "results" / "06_full_history_reconstruction" / "FULL_particles_final.zarr").exists()
    final_dir = output / "results" / "08_final_report"
    for name in schemas.FINAL_OUTPUTS:
        assert (final_dir / name).exists()
    appendix = pd.read_csv(final_dir / "FINAL_parameter_appendix.csv")
    assert {"hard-fixed", "plausibility-free", "derived-only"}.issubset(set(appendix["parameter_type"]))
    final_ppc = json.loads((final_dir / "full_ppc_report.json").read_text(encoding="utf-8"))
    assert final_ppc["particle_scope"] == "accepted_particles_only"
    assert "cell_count" in final_ppc["raw_like_channels"]
