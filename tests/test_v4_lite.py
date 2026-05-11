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
from fit import full_smc
from fit import schemas
from fit.objective import SCORE_COMPONENTS
from fit.full_smc import _score_candidate_moments, _weights_from_scores
from fit.observation import calculate_ddpcr_pooled_mean, validate_observation_params
from fit.parameter_registry import run_prior_predictive_gate
from fit.raw import create_synthetic_raw_dataset, validate_raw_tables
from fit.stage_runner import run_pipeline_from_raw
from fit.validation import validate_method_contracts
from fit.v4_lite import validate_lite_artifacts


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
        "fit.observation",
        "fit.v4_lite",
        "fit.parameter_registry",
        "fit.full_smc",
        "fit.objective",
        "fit.validation",
        "fit.final_report",
        "fit.full_raw_ppc",
        "fit.full_exact_replay",
        "fit.ppc",
    ):
        __import__(module)


def test_schema_validation_identifies_missing_required_fields():
    tables = {
        "ddpcr": pd.DataFrame(columns=["week", "condition", "replicate", "ddpcr_copy_number", "ddpcr_sd_or_ci", "batch_id"]),
        "cell_count": pd.DataFrame(columns=schemas.RAW_TABLE_SCHEMAS["cell_count"]),
        "flow": pd.DataFrame({"week": [1], "condition": ["ctrl"], "replicate": ["r1"], "fraction": [1.0], "batch_id": ["b"]}),
    }
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_tables(tables)
    with pytest.raises(ValueError, match="missing required fields"):
        validate_observation_params({"locked_for_full": True, "ddpcr": {}})


def test_ddpcr_pooled_mean_formula():
    flow = pd.DataFrame(
        {"week": [1, 1], "condition": ["ctrl", "ctrl"], "replicate": ["r1", "r1"], "state_gate": ["NPC-like", "OPC-like"], "fraction": [0.8, 0.2]}
    )
    mu = pd.DataFrame(
        {"week": [1, 1], "condition": ["ctrl", "ctrl"], "replicate": ["r1", "r1"], "state_gate": ["NPC-like", "OPC-like"], "species": ["MYC", "MYC"], "copy_mean": [0.0, 225.0]}
    )
    pooled = calculate_ddpcr_pooled_mean(flow, mu)
    assert pooled.loc[0, "bulk_mean"] == pytest.approx(45.0)


def test_manifest_and_clean_outputs_close_unavailable_modalities(small_pipeline):
    mask = json.loads((small_pipeline / "00_manifest" / "available_data_mask.json").read_text(encoding="utf-8"))
    assert mask == {
        "ddpcr_bulk": True,
        "cell_count": True,
        "flow_3group_early": True,
        "flow_4state": False,
        "qpcdr_sorted": False,
        "ectag_single_cell": False,
    }
    for name in schemas.CLEAN_OUTPUTS:
        assert (small_pipeline / "01_clean_data" / name).exists()
    assert json.loads((small_pipeline / "01_clean_data" / "qpcdr_unavailable.json").read_text(encoding="utf-8"))["likelihood_weight"] == 0
    flow3 = pd.read_parquet(small_pipeline / "01_clean_data" / "flow3_early_long.parquet")
    assert set(flow3["group"]) == set(schemas.FLOW3_GROUPS)
    assert flow3.groupby(["week", "condition", "replicate"])["fraction"].sum().round(8).eq(1.0).all()


def test_observation_model_bulk_mask_and_projection(small_pipeline):
    obs = json.loads((small_pipeline / "02_observation_model" / "obs_params_for_full.json").read_text(encoding="utf-8"))
    assert obs["locked_for_full"]
    assert obs["fit_mask"] == schemas.FIT_MASK
    assert obs["ddpcr"]["likelihood"] == "lognormal_on_bulk_mean"
    assert "never" in obs["ddpcr"]["interpretation"]
    projection = np.load(small_pipeline / "02_observation_model" / "flow3_projection_matrix.npy")
    assert projection.shape == (3, 4)
    assert not obs["fit_mask"]["use_qpcdr"]
    assert not obs["fit_mask"]["use_ectag"]


def test_bulk_lite_outputs_only_allowed_quantities(small_pipeline):
    lite_dir = small_pipeline / "03_v4_lite_bulk"
    validate_lite_artifacts(lite_dir)
    for name in schemas.LITE_OUTPUTS:
        assert (lite_dir / name).exists()
    mask = json.loads((lite_dir / "BULK_LITE_to_FULL_fit_mask.json").read_text(encoding="utf-8"))
    assert mask["use_lite_summary_in_final_score"] is False
    assert mask["use_qpcdr"] is False and mask["use_ectag"] is False
    copy = pd.read_parquet(lite_dir / "BULK_LITE_copy_velocity.parquet")
    growth = pd.read_parquet(lite_dir / "BULK_LITE_growth_velocity.parquet")
    assert {"v_center", "phase", "species"}.issubset(copy.columns)
    assert {"r_center", "phase"}.issubset(growth.columns)
    assert set(copy["species"]) == set(schemas.SPECIES)


def test_parameter_registry_layers_and_blocks(small_pipeline):
    table = pd.read_csv(small_pipeline / "04_parameter_registry" / "PARAMETER_interpretability_prior_table.csv")
    roles = set(table["role"])
    assert {"active_effective_control", "prior_constrained_nuisance", "prior_only", "fixed", "derived_only"}.issubset(roles)
    active = json.loads((small_pipeline / "04_parameter_registry" / "PARAMETER_active_blocks.json").read_text(encoding="utf-8"))
    assert set(active) == {"growth_block", "copy_MYC_block", "copy_CDK4_block", "copy_PDGFRA_block", "flow3_projection_block"}
    nuisance = json.loads((small_pipeline / "04_parameter_registry" / "PARAMETER_nuisance_blocks.json").read_text(encoding="utf-8"))
    assert "division_death_turnover" in nuisance
    assert "ecDNA_gain_loss_turnover" in nuisance


def test_dynamic_condition_parameter_columns_are_not_attribute_dependent():
    artifacts = {
        "growth_velocity": pd.DataFrame({"condition": ["drugA__0.1uM"], "phase": [1], "r_center": [0.2]}),
        "copy_velocity": pd.DataFrame({"condition": ["drugA__0.1uM"], "species": ["MYC"], "phase": [1], "v_center": [0.1]}),
        "prior_scales": {"r_center_sd": 0.1, "v_center_sd": 0.1, "flow3_bias_sd": 0.05},
    }
    candidates = pd.DataFrame(
        {
            "particle_id": [0],
            "r__drugA__0.1uM__p1": [0.2],
            "v__drugA__0.1uM__MYC__p1": [0.1],
            "zeta_flow3__p1": [0.0],
            "zeta_flow3__p2": [0.0],
            "zeta_flow3__p3": [0.0],
            "division_death_turnover": [1.0],
            "ecDNA_gain_loss_turnover": [1.0],
            "hidden_npc_opc_split": [0.5],
        }
    )
    scores = _score_candidate_moments(candidates, artifacts)
    assert scores.loc[0, "D_count"] == pytest.approx(0.0)
    assert scores.loc[0, "D_ddPCR"] == pytest.approx(0.0)


def test_candidate_moment_parallel_scores_match_serial():
    artifacts = {
        "growth_velocity": pd.DataFrame({"condition": ["ctrl"], "phase": [1], "r_center": [0.1]}),
        "copy_velocity": pd.DataFrame(
            {
                "condition": ["ctrl", "ctrl", "ctrl"],
                "species": ["MYC", "CDK4", "PDGFRA"],
                "phase": [1, 1, 1],
                "v_center": [0.2, -0.1, 0.05],
            }
        ),
        "prior_scales": {"r_center_sd": 0.2, "v_center_sd": 0.2, "flow3_bias_sd": 0.05},
    }
    candidates = pd.DataFrame(
        {
            "particle_id": [0, 1, 2, 3],
            "r__ctrl__p1": [0.1, 0.15, -0.2, 0.3],
            "v__ctrl__MYC__p1": [0.2, 0.1, 0.0, -0.2],
            "v__ctrl__CDK4__p1": [-0.1, -0.05, 0.2, 0.0],
            "v__ctrl__PDGFRA__p1": [0.05, 0.0, 0.15, -0.05],
            "zeta_flow3__p1": [0.0, 0.01, -0.01, 0.02],
            "zeta_flow3__p2": [0.0, 0.0, 0.0, 0.0],
            "zeta_flow3__p3": [0.0, 0.0, 0.0, 0.0],
            "division_death_turnover": [1.0, 1.1, 1.2, 1.3],
            "ecDNA_gain_loss_turnover": [1.0, 1.1, 1.2, 1.3],
            "hidden_npc_opc_split": [0.5, 0.4, 0.6, 0.55],
        }
    )
    serial = _score_candidate_moments(candidates, artifacts, workers=1)
    parallel = _score_candidate_moments(candidates, artifacts, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_fit_cli_accepts_workers_for_parallel_fit_stages():
    from fit.run_fit import build_parser

    parser = build_parser()
    moment = parser.parse_args(
        [
            "run-moment-prescreen",
            "--lite-dir",
            "lite",
            "--prior-dir",
            "prior",
            "--output",
            "out",
            "--workers",
            "3",
        ]
    )
    full = parser.parse_args(
        [
            "run-full-reconstruction",
            "--lite-dir",
            "lite",
            "--obs-params",
            "obs.json",
            "--output",
            "out",
            "--workers",
            "4",
        ]
    )
    run_all = parser.parse_args(["run-all", "--raw-dir", "raw", "--output", "out", "--workers", "5"])
    assert moment.workers == 3
    assert full.workers == 4
    assert run_all.workers == 5


def test_prior_gate_stops_after_failed_active_relaxation(workdir):
    lite = workdir / "lite"
    out = workdir / "prior"
    lite.mkdir()
    (workdir / "PARAMETER_registry_resolved.yaml").write_text("parameters: {}\n", encoding="utf-8")
    (lite / "BULK_LITE_to_FULL_prior_scales.json").write_text(json.dumps({"r_center_sd": 100.0, "v_center_sd": 100.0}), encoding="utf-8")
    (lite / "BULK_LITE_initial_population_sampler.json").write_text(
        json.dumps(
            {
                "cell_count_anchor": [{"condition": "ctrl", "replicate": "r1", "total_cell_count": 1000.0}],
                "ddpcr_bulk_anchor": [{"condition": "ctrl", "replicate": "r1", "species": "MYC", "ddpcr_copy_number": 20.0}],
            }
        ),
        encoding="utf-8",
    )
    (workdir / "obs.json").write_text(
        json.dumps({"flow3": {"target": {"fractions": {"OLIG2-high": 0.7, "AC": 0.2, "MES": 0.1}}}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="accepted fraction <1%"):
        run_prior_predictive_gate(workdir, lite, workdir / "obs.json", out, seed=11, candidates=1000)
    assert (out / "PRIOR_region_incompatible_report.md").exists()
    rejection = pd.read_csv(out / "PRIOR_predictive_rejection_reasons.csv")
    assert set(rejection["active_control_relaxation"]) == {1.2}


def test_moment_prescreen_expands_to_method_candidate_floor(workdir, monkeypatch, capsys):
    prior_dir = workdir / "prior"
    prior_dir.mkdir()
    pd.DataFrame({"accepted": [True], "division_death_turnover": [1.0], "ecDNA_gain_loss_turnover": [1.0]}).to_parquet(
        prior_dir / "PRIOR_predictive_accepted_region.parquet",
        index=False,
    )
    calls = []

    def fake_candidate_table(artifacts, prior, rng, n_candidates):
        del artifacts, prior, rng
        calls.append(int(n_candidates))
        rows = 100 if int(n_candidates) < full_smc.METHOD_MOMENT_MAX_CANDIDATES else full_smc.METHOD_MOMENT_MIN_TOP
        return pd.DataFrame({"particle_id": range(rows), "random_stream_id": range(rows)})

    def fake_scores(candidates, artifacts, workers=1):
        del artifacts, workers
        return pd.DataFrame(
            {
                "particle_id": candidates["particle_id"].astype(int),
                "D_moment": np.arange(len(candidates), dtype=float),
                "D_prior": 0.0,
                "D_biology": 0.0,
            }
        )

    monkeypatch.setattr(full_smc, "load_lite_artifacts", lambda _: {})
    monkeypatch.setattr(full_smc, "_candidate_parameter_table", fake_candidate_table)
    monkeypatch.setattr(full_smc, "_score_candidate_moments", fake_scores)

    full_smc.run_moment_prescreen(workdir / "lite", prior_dir, workdir / "moment", n_candidates=100, keep_top=10000)

    progress = capsys.readouterr().out
    assert "[fit] moment prescreen start" in progress
    assert "[fit] moment prescreen result" in progress
    assert calls == [100, full_smc.METHOD_MOMENT_MAX_CANDIDATES]
    keep = pd.read_parquet(workdir / "moment" / "MOMENT_keep_top_particles.parquet")
    assert len(keep) == full_smc.METHOD_MOMENT_MIN_TOP


def test_final_weights_do_not_fallback_when_cutoff_rejects_all():
    scores = pd.DataFrame(
        {
            "particle_id": [1, 2],
            "round": [3, 3],
            "score": [10.0, 11.0],
            "early_rejected": [False, False],
        }
    )
    with pytest.raises(RuntimeError, match="final score cutoff"):
        _weights_from_scores(scores, final_round=3, cutoff=1.0, gate_particle_ids={1, 2})


def test_smc_outputs_have_required_adaptive_features_and_closed_scores(small_pipeline):
    full = small_pipeline / "08_full_smc"
    for name in schemas.FULL_OUTPUTS:
        assert (full / name).exists()
    scores = pd.read_parquet(full / "FULL_particle_scores.parquet")
    for component in SCORE_COMPONENTS:
        assert f"D_{component}" in scores.columns or component in {"prior", "biology"}
    assert "D_ddpcr" in scores and "D_cell_count" in scores and "D_flow3" in scores
    assert "D_qpcdr" not in scores and "D_ectag" not in scores
    params = pd.read_parquet(full / "FULL_particle_parameters.parquet")
    weights = pd.read_parquet(full / "FULL_particle_weights.parquet")
    assert weights.loc[~weights["accepted"], "weight"].sum() == pytest.approx(0.0)
    assert weights.loc[weights["accepted"], "weight"].sum() == pytest.approx(1.0)
    forbidden = {"copy_gain_rate", "copy_loss_rate", "division_rate", "death_rate", "state_specific_copy_enrichment"}
    assert forbidden.isdisjoint(params.columns)
    assert "updated_block" in params
    assert "random_stream_id" in params
    assert params["D_prior"].nunique() > 1
    log = pd.read_parquet(full / "FULL_smc_adaptation_log.parquet")
    assert {"epsilon", "acceptance_rate", "proposal_scale_growth", "proposal_scale_copy_MYC", "proposal_scale_flow3", "median_D_flow3"}.issubset(log.columns)
    assert {"n_sim_cells", "n_sim_cells_next"}.issubset(log.columns)
    early = pd.read_parquet(full / "FULL_early_rejection_log.parquet")
    assert {"partial_data_distance", "partial_total_distance", "screen_stage", "simulated_full"}.issubset(early.columns)
    mc = pd.read_csv(full / "FULL_monte_carlo_noise_report.csv")
    assert {"estimated_mc_sd_log", "threshold", "n_sim_cells_current", "n_sim_cells_next", "repeats", "round"}.issubset(mc.columns)
    manifest = json.loads((full / "full_reconstruction_manifest.json").read_text(encoding="utf-8"))
    assert "moment_prescreen" in manifest["smc_features"]
    assert "blockwise_update" in manifest["smc_features"]
    assert manifest["fit_mask"] == schemas.FIT_MASK
    assert "core_full_simulator_replay" in manifest["smc_features"]
    moment = pd.read_parquet(small_pipeline / "06_moment_prescreen" / "MOMENT_keep_top_particles.parquet")
    assert "D_prior" in moment.columns and "D_prior_x" not in moment.columns and "D_prior_y" not in moment.columns
    assert params.filter(regex="^zeta_flow3").abs().le(0.25).all().all()
    init_root = zarr.open_group(store=ZipStore(str(small_pipeline / "07_full_initialization" / "FULL_initial_population.zarr"), mode="r"), mode="r")
    assert "initial_copy_numbers" in init_root.array_keys()
    root = zarr.open_group(store=ZipStore(str(full / "FULL_particles_final.zarr"), mode="r"), mode="r")
    assert root.attrs["full_simulator"] == "core.simulation.run_simulation"
    assert "history_population_size" in root.array_keys()
    replay = zarr.open_group(store=ZipStore(str(full / "FULL_replay_histories.zarr"), mode="r"), mode="r")
    assert replay.attrs["n_sim_cells"] >= 24
    assert (full / "FULL_replay_histories_ddpcr_predictions.parquet").exists()


def test_validation_and_final_outputs_are_method_shaped(small_pipeline):
    validation = small_pipeline / "09_validation"
    for name in schemas.VALIDATION_OUTPUTS:
        assert (validation / name).exists()
    ident = pd.read_csv(validation / "FULL_identifiability_report.csv")
    required = {"parameter", "role", "posterior_contraction", "prior_shift_z", "boundary_mass", "ridge_partner", "interpretation_status"}
    assert required.issubset(ident.columns)
    assert {"state_specific_copy_enrichment", "co_segregation_strength", "single_cell_copy_distribution_shape"}.issubset(set(ident["parameter"]))
    holdout = pd.read_csv(validation / "FULL_holdout_validation.csv")
    assert {"split", "channel", "covered"}.issubset(holdout.columns)
    assert {"ddpcr", "cell_count", "flow3"}.issubset(set(holdout["channel"]))
    final = small_pipeline / "10_final_report"
    for name in schemas.FINAL_OUTPUTS:
        assert (final / name).exists()
    data = pd.read_csv(final / "FINAL_data_constrained_results.csv")
    assert set(data["result_type"]).issuperset({"bulk ddPCR trajectory", "cell count trajectory", "flow3 steady projection"})
    latent = pd.read_csv(final / "FINAL_latent_model_dependent_results.csv")
    assert latent["interpretation"].str.contains("current data do not directly identify").all()
    validate_method_contracts(small_pipeline / "02_observation_model", small_pipeline / "03_v4_lite_bulk", small_pipeline / "08_full_smc", final)


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
    results = output / "results"
    assert (results / "00_manifest" / "available_data_mask.json").exists()
    assert (results / "03_v4_lite_bulk" / "BULK_LITE_to_FULL_fit_mask.json").exists()
    assert (results / "08_full_smc" / "FULL_particles_final.zarr").exists()
    assert (results / "10_final_report" / "FINAL_method_manifest.json").exists()


def test_cli_run_all_raw_to_final_report(workdir):
    raw = workdir / "raw"
    results = workdir / "run_all_results"
    create_synthetic_raw_dataset(raw, seed=17)
    subprocess.run(
        [sys.executable, "-m", "fit.run_fit", "run-all", "--raw-dir", str(raw), "--output", str(results), "--seed", "17", "--draws", "2", "--particles", "2", "--cells", "12"],
        check=True,
    )
    assert (results / "00_manifest" / "available_data_mask.json").exists()
    assert (results / "10_final_report" / "FINAL_method_manifest.json").exists()
