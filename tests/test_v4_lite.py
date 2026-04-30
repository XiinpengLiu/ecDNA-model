import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
        "fit.observation",
        "fit.empirical",
        "fit.v4_lite",
        "fit.full_smc",
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
    target = pd.read_parquet(lite_dir / "LITE_summary_target_vector.parquet")
    assert {"flow", "ectag", "qpcdr", "ddpcr", "lite_summary"}.issubset(set(target["channel"]))
    covariance = np.load(lite_dir / "LITE_summary_covariance.npz", allow_pickle=True)
    assert covariance["covariance"].shape == (len(target), len(target))
    sampler = json.loads((lite_dir / "LITE_initial_population_sampler.json").read_text(encoding="utf-8"))
    assert sampler["ddpcr_policy"] == "pooled_mean_anchor_only"


def test_full_particle_score_reads_lite_and_observation_outputs(small_pipeline):
    weights = pd.read_parquet(small_pipeline / "05_full_smc" / "particle_weights.parquet")
    for component in SCORE_COMPONENTS:
        assert component in weights.columns
    assert weights["flow"].sum() > 0
    assert weights["qpcdr"].sum() > 0
    assert weights["ectag"].sum() > 0
    assert weights["ddpcr"].sum() > 0
    assert weights["lite_summary"].sum() > 0
    assert weights["weight"].sum() == pytest.approx(1.0)


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
