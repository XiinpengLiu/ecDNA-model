from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as cfg
from analysis.export_tables import write_simulation_tables
from analysis.plotting import _pooled_state_group_copy_stats, plot_t87_treatment_comparison_suite
from core.simulation import SimulationResult, run_simulation
from main import build_parser, run_condition


@pytest.fixture()
def workdir() -> Path:
    base = Path.cwd() / "tmp_test_outputs"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _cell(cell_id: int, parent_id: int | None, copy_base: int = 1) -> dict:
    return {
        "cell_id": cell_id,
        "parent_id": parent_id,
        "cycle_state": "G1",
        "cycle_index": 1,
        "copy_numbers": [copy_base, copy_base + 1, copy_base + 2],
        "soft_state": [0.60, 0.20, 0.10, 0.10],
        "latent_state": [0.1, 0.2, 0.3],
        "stress_score": 0.4,
        "survival_score": 0.5,
        "age": 0.6,
        "last_update_time": 0.0,
        "dominant_state": "NPC-like",
        "dominant_state_index": 0,
        "last_D_C": 0.0,
        "last_D_P": 0.0,
        "division_hazard": 0.7,
        "death_hazard": 0.8,
        "derived_report_only": {
            "local_transition_generator": [
                [-0.1, 0.1, 0.0, 0.0],
                [0.1, -0.2, 0.1, 0.0],
                [0.0, 0.1, -0.2, 0.1],
                [0.0, 0.0, 0.1, -0.1],
            ]
        },
    }


def _truth_snapshot() -> dict:
    state_values = {state: [1.0, 2.0, 3.0] for state in cfg.STATE_NAMES}
    return {
        "population_size": 1,
        "soft_state_fractions": [0.60, 0.20, 0.10, 0.10],
        "cycle_fractions": [0.0, 1.0, 0.0, 0.0],
        "bulk_copy_means": [1.0, 2.0, 3.0],
        "mean_stress_score": 0.4,
        "mean_survival_score": 0.5,
        "mean_division_hazard": 0.7,
        "mean_death_hazard": 0.8,
        "dominant_state_counts": [1, 0, 0, 0],
        "dominant_state_fractions": [1.0, 0.0, 0.0, 0.0],
        "copy_means_by_gate": state_values,
        "copy_vars_by_gate": state_values,
        "zero_fraction_by_gate": state_values,
        "tail_fraction_by_gate": state_values,
    }


def _observation_snapshot() -> dict:
    state_values = {state: [1.0, 2.0, 3.0] for state in cfg.STATE_NAMES}
    return {
        "observed_count": 1,
        "latent_gate_counts": [1, 0, 0, 0],
        "latent_gate_fractions": [1.0, 0.0, 0.0, 0.0],
        "flow_counts": [1, 0, 0, 0],
        "flow_fractions": [1.0, 0.0, 0.0, 0.0],
        "sorted_state_counts": {state: 1 if state == "NPC-like" else 0 for state in cfg.STATE_NAMES},
        "pooled_qpcdr_means": [1.0, 2.0, 3.0],
        "pooled_ecTAG_means": [1.0, 2.0, 3.0],
        "sorted_bulk_copy_means": state_values,
        "sorted_qpcdr": {"means": state_values},
        "sorted_ecTAG": {"means": state_values},
    }


def _test_safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return token or "value"


def _result_with_division_event() -> SimulationResult:
    result = SimulationResult(stop_time=9.0, stop_reason="t_max")
    for idx in range(9):
        result.times.append(float(idx + 1))
        result.population_sizes.append(1)
        result.soft_state_fractions.append(pd.Series([0.60, 0.20, 0.10, 0.10]).to_numpy())
        result.cycle_fractions.append(pd.Series([0.0, 1.0, 0.0, 0.0]).to_numpy())
        result.bulk_copy_means.append(pd.Series([1.0, 2.0, 3.0]).to_numpy())
        result.mean_stress_scores.append(0.4)
        result.mean_survival_scores.append(0.5)
        result.mean_division_hazard.append(0.7)
        result.mean_death_hazard.append(0.8)
        result.exposures.append({"D_C": 0.0, "D_P": 0.0, "a": 0.0, "m": 0.0})
        result.truth_snapshots.append(_truth_snapshot())
        result.observations.append(_observation_snapshot())
        result.cell_snapshots.append([_cell(100 + idx, None, copy_base=idx + 1)])
    result.events.append(
        (
            1.5,
            "division",
            10,
            {
                "state_pre": _cell(10, None, copy_base=2),
                "daughter_one": _cell(11, 10, copy_base=3),
                "daughter_two": _cell(12, 10, copy_base=4),
            },
        )
    )
    return result


def _ensemble_dir(base: Path) -> Path:
    return base / "ensemble_id=ENS_000001"


def _run_dir(base: Path, condition: str) -> Path:
    return _ensemble_dir(base) / "runs" / f"sim_id=SIM_FULL_{condition.upper()}_REP001"


def _required_package_files(run_dir: Path) -> list[Path]:
    return [
        run_dir / "manifest.json",
        run_dir / "parameters" / "parameter_table.parquet",
        run_dir / "parameters" / "parameter_blocks.parquet",
        run_dir / "root" / "observables_long.parquet",
        run_dir / "root" / "copy_vector.parquet",
        run_dir / "root" / "cell_registry.parquet",
        run_dir / "root" / "cell_snapshot",
        run_dir / "root" / "cell_terminal_state.parquet",
        run_dir / "root" / "event_log.parquet",
        run_dir / "root" / "lineage_edges.parquet",
        run_dir / "root" / "division_inheritance.parquet",
        run_dir / "root" / "virtual_assay_draws.parquet",
        run_dir / "cache" / "population_summary.parquet",
        run_dir / "cache" / "state_copy_summary.parquet",
        run_dir / "cache" / "founder_t_summary.parquet",
        run_dir / "cache" / "copy_distribution_summary.parquet",
        run_dir / "cache" / "event_summary.parquet",
        run_dir / "cache" / "lineage_family_summary.parquet",
        run_dir / "qc" / "output_integrity_report.json",
        run_dir / "qc" / "id_consistency_report.parquet",
    ]


def _assert_no_forbidden_columns(frame: pd.DataFrame) -> None:
    forbidden = {"week", "day", "time_day", "tau", "simulation_time"}
    for column in frame.columns:
        parts = set(str(column).lower().split("_"))
        assert not (parts & forbidden), column
        assert str(column).lower() not in forbidden


def test_write_simulation_tables_exports_complete_ensemble_package(workdir: Path) -> None:
    result = _result_with_division_event()
    outputs = write_simulation_tables(result, workdir, condition="P10", seed=7, metadata={"source": "test"})

    ensemble_dir = _ensemble_dir(workdir)
    run_dir = _run_dir(workdir, "P10")
    assert outputs["ensemble_manifest"] == ensemble_dir / "ensemble_manifest.json"
    assert outputs["cell_snapshot"] == run_dir / "root" / "cell_snapshot"

    ensemble_expected = [
        ensemble_dir / "ensemble_manifest.json",
        ensemble_dir / "run_index.parquet",
        ensemble_dir / "metadata" / "conditions.parquet",
        ensemble_dir / "metadata" / "model_variants.parquet",
        ensemble_dir / "metadata" / "initial_conditions.parquet",
        ensemble_dir / "metadata" / "t_grid.parquet",
        ensemble_dir / "metadata" / "species.parquet",
        ensemble_dir / "metadata" / "state_definitions.parquet",
        ensemble_dir / "metadata" / "assay_definitions.parquet",
        ensemble_dir / "metadata" / "copy_bins.parquet",
        ensemble_dir / "metadata" / "event_type_definitions.parquet",
    ]
    for path in [*ensemble_expected, *_required_package_files(run_dir)]:
        assert path.exists(), path
    assert not (workdir / "tables").exists()
    assert not (workdir / "simulation_data").exists()

    ensemble_manifest = json.loads((ensemble_dir / "ensemble_manifest.json").read_text(encoding="utf-8"))
    assert ensemble_manifest["time_variable"] == "t"
    assert ensemble_manifest["uses_real_time"] is False
    assert ensemble_manifest["uses_week_labels"] is False

    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["condition_id"] == "CDK4i_10nM"
    assert run_manifest["time_variable"] == "t"
    assert run_manifest["records_all_cells_ever_born"] is True

    conditions = pd.read_parquet(ensemble_dir / "metadata" / "conditions.parquet")
    assert "CDK4i_10nM" in set(conditions["condition_id"])
    assert {"treatment_start_t", "treatment_end_t", "dose_value", "dose_unit"} <= set(conditions.columns)
    _assert_no_forbidden_columns(conditions)

    t_grid = pd.read_parquet(ensemble_dir / "metadata" / "t_grid.parquet")
    assert {"t", "t_index", "t_grid_id"} <= set(t_grid.columns)
    _assert_no_forbidden_columns(t_grid)

    registry = pd.read_parquet(run_dir / "root" / "cell_registry.parquet")
    required_cell_columns = {
        "cell_id",
        "cell_uid",
        "founder_id",
        "founder_uid",
        "parent_id",
        "parent_uid",
        "birth_t",
        "death_t",
        "final_status",
    }
    assert required_cell_columns <= set(registry.columns)
    assert registry["cell_uid"].is_unique
    _assert_no_forbidden_columns(registry)

    root_cells = pd.read_parquet(run_dir / "root" / "cell_snapshot")
    assert root_cells["alive"].all()
    assert {"t", "t_index", "cell_uid", "founder_uid", "coarse_state", "k_myc", "k_cdk4", "k_pdgfra"} <= set(root_cells.columns)
    assert (
        run_dir
        / "root"
        / "cell_snapshot"
        / "condition_id=CDK4i_10nM"
        / "replicate_id=REP001"
        / "t_index=0"
        / "part-000.parquet"
    ).exists()
    assert not (run_dir / "root" / "cell_snapshot" / "t_index=0").exists()
    assert set(root_cells["condition_id"]) == {"CDK4i_10nM"}
    assert set(root_cells["replicate_id"]) == {"REP001"}
    _assert_no_forbidden_columns(root_cells)

    event_log = pd.read_parquet(run_dir / "root" / "event_log.parquet")
    assert {"k_myc_before", "k_myc_after", "daughter1_uid", "daughter2_uid"} <= set(event_log.columns)
    assert set(event_log["event_type"]) == {"division"}

    lineage = pd.read_parquet(run_dir / "root" / "lineage_edges.parquet")
    assert {"division_event_id", "parent_uid", "child_uid", "t_birth"} <= set(lineage.columns)
    assert lineage.shape[0] == 2

    inheritance = pd.read_parquet(run_dir / "root" / "division_inheritance.parquet")
    assert {"segregation_pool_myc", "imbalance_cdk4", "daughter1_k_pdgfra"} <= set(inheritance.columns)
    assert inheritance.shape[0] == 1

    observables = pd.read_parquet(run_dir / "root" / "observables_long.parquet")
    assert {"cell_count", "ddpcr", "flow"} <= set(observables["assay"])
    _assert_no_forbidden_columns(observables)

    qc = json.loads((run_dir / "qc" / "output_integrity_report.json").read_text(encoding="utf-8"))
    assert qc["all_checks_passed"] is True


def test_package_uses_t_for_r500_without_day_or_week_mapping(workdir: Path) -> None:
    result = _result_with_division_event()
    write_simulation_tables(result, workdir, condition="R500", seed=11, metadata={"source": "test"})

    ensemble_dir = _ensemble_dir(workdir)
    run_dir = _run_dir(workdir, "R500")
    ensemble_manifest = json.loads((ensemble_dir / "ensemble_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert ensemble_manifest["time_variable"] == "t"
    assert run_manifest["condition_id"] == "PDGFRAi_500nM"
    assert "time_mapping" not in run_manifest

    for path in [
        ensemble_dir / "metadata" / "t_grid.parquet",
        run_dir / "root" / "cell_snapshot",
        run_dir / "root" / "event_log.parquet",
        run_dir / "cache" / "population_summary.parquet",
    ]:
        _assert_no_forbidden_columns(pd.read_parquet(path))


def test_recorded_simulation_snapshots_include_full_cell_state(workdir: Path) -> None:
    raw_dir = workdir / "raw"
    raw_dir.mkdir()
    rows = [
        {"week": 1, "condition": "ctrl", "species": species, "ddpcr_copy_number": 2.0}
        for species in cfg.SPECIES
    ]
    pd.DataFrame(rows).to_csv(raw_dir / "ddpcr.csv", index=False)

    params = replace(
        cfg.DEFAULT_MODEL_PARAMETERS,
        simulation=replace(
            cfg.DEFAULT_MODEL_PARAMETERS.simulation,
            t_max=1.0,
            record_times=(1.0,),
            n_init=4,
            target_population_size=None,
            max_pop_size=50,
            random_seed=7,
            record_full_snapshots=True,
            record_events=True,
        ),
    )
    row = run_condition(
        "ctrl",
        params=params,
        raw_dir=raw_dir,
        output_dir=workdir / "run",
        seed=7,
        rows_per_state=8,
        plots=False,
        verbose=False,
    )

    condition_dir = Path(row["result_dir"])
    run_dir = condition_dir / "ensemble_id=ENS_000001" / "runs" / "sim_id=SIM_FULL_CTRL_REP001"
    assert (run_dir / "root" / "cell_snapshot").exists()
    assert not (condition_dir / "tables").exists()
    assert not (condition_dir / "simulation_data").exists()
    assert not list(condition_dir.glob("*.png"))

    cells = pd.read_parquet(run_dir / "root" / "cell_snapshot")
    assert {"parent_id", "parent_uid", "cell_cycle_state", "hard_state", "birth_t"} <= set(cells.columns)


def test_small_simulation_exports_t0_to_t12_and_passes_qc(workdir: Path) -> None:
    simulation = replace(
        cfg.DEFAULT_MODEL_PARAMETERS.simulation,
        time_unit="t",
        t_max=12.0,
        record_times=tuple(float(t) for t in range(13)),
        n_init=5,
        target_population_size=None,
        max_pop_size=1000,
        random_seed=20260515,
        record_full_snapshots=True,
        record_events=True,
    )
    hazard = replace(
        cfg.DEFAULT_MODEL_PARAMETERS.hazard,
        lambda_div_ceiling=0.05,
        lambda_death_ceiling=0.02,
    )
    params = replace(cfg.DEFAULT_MODEL_PARAMETERS, simulation=simulation, hazard=hazard)
    result = run_simulation(
        params=params,
        input_schedules=cfg.t87_input_schedules_for_condition("P10"),
        seed=20260515,
        verbose=False,
    )
    assert result.times == [float(t) for t in range(13)]

    write_simulation_tables(
        result,
        workdir / "sim_outputs",
        condition="P10",
        seed=20260515,
        metadata={"simulation": {"t_max": 12.0, "record_times": list(result.times)}},
    )
    ensemble_dir = workdir / "sim_outputs" / "ensemble_id=ENS_000001"
    run_dir = ensemble_dir / "runs" / "sim_id=SIM_FULL_P10_REP001"

    for path in _required_package_files(run_dir):
        assert path.exists(), path

    snapshot = pd.read_parquet(run_dir / "root" / "cell_snapshot")
    assert sorted(snapshot["t"].astype(float).unique().tolist()) == [float(t) for t in range(13)]
    assert snapshot["alive"].all()
    for column in ("k_myc", "k_cdk4", "k_pdgfra"):
        assert (pd.to_numeric(snapshot[column]) >= 0).all()
    assert np.allclose(snapshot[["x_npc", "x_opc", "x_ac", "x_mes"]].sum(axis=1), 1.0)

    registry = pd.read_parquet(run_dir / "root" / "cell_registry.parquet")
    assert registry["cell_uid"].is_unique
    assert set(registry["founder_id"].dropna().astype(int)) <= set(registry["cell_id"].astype(int))
    assert set(registry["parent_id"].dropna().astype(int)) <= set(registry["cell_id"].astype(int))

    population = pd.read_parquet(run_dir / "cache" / "population_summary.parquet")
    snapshot_counts = snapshot.groupby("t").size().reset_index(name="snapshot_n")
    merged = population.merge(snapshot_counts, on="t")
    assert (merged["n_alive_cells"].astype(int) == merged["snapshot_n"].astype(int)).all()

    qc = json.loads((run_dir / "qc" / "output_integrity_report.json").read_text(encoding="utf-8"))
    assert qc["all_checks_passed"] is True
    id_report = pd.read_parquet(run_dir / "qc" / "id_consistency_report.parquet")
    assert id_report["passed"].all()


def test_record_full_snapshots_cli_flag_was_removed() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--record-full-snapshots"])


def test_t87_treatment_comparison_suite_writes_batch_plots(workdir: Path) -> None:
    output_dir = workdir / "t87"
    raw_dir = workdir / "raw"
    raw_dir.mkdir()
    conditions = ("ctrl", "P10", "P50", "P250", "R20", "R100", "R500")
    pd.DataFrame(
        [
            {"week": 1, "condition": condition, "total_cell_count": 1000.0 + idx * 100.0}
            for idx, condition in enumerate(conditions)
        ]
    ).to_csv(raw_dir / "cell_count.csv", index=False)

    for condition_idx, condition in enumerate(conditions):
        table_dir = output_dir / condition / "tables"
        table_dir.mkdir(parents=True)
        rows = []
        for time_idx, time in enumerate((1.0, 3.0, 5.0)):
            row = {
                "time": time,
                "population_size": 100 + 10 * time_idx,
            }
            counts = [40 + time_idx, 30 + time_idx, 20 + time_idx, 10 + time_idx]
            for state_name, count in zip(cfg.STATE_NAMES, counts):
                row[f"dominant_count_{_test_safe_token(state_name)}"] = count
            for species_idx, species_name in enumerate(cfg.SPECIES):
                row[f"mean_copy_{species_name}"] = 90.0 + condition_idx + species_idx + time
                for state_idx, state_name in enumerate(cfg.STATE_NAMES):
                    token = _test_safe_token(state_name)
                    row[f"state_mean_copy_{token}_{species_name}"] = 80.0 + 4.0 * state_idx + species_idx + condition_idx
                    row[f"state_var_copy_{token}_{species_name}"] = 4.0 + state_idx + species_idx
            rows.append(row)
        pd.DataFrame(rows).to_csv(table_dir / "time_summary.csv", index=False)
        (table_dir / "metadata.json").write_text(
            json.dumps({"condition": condition, "simulation": {"n_init": 100}}),
            encoding="utf-8",
        )

    outputs = plot_t87_treatment_comparison_suite(output_dir, raw_dir=raw_dir, conditions=conditions)

    assert set(outputs) == {
        "01_log10_state_counts_by_condition.png",
        "02_ecdna_copy_number_by_treatment.png",
        "03_state_group_ecdna_endpoint_points.png",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_pooled_state_group_copy_stats_uses_counts_means_and_variances() -> None:
    row = pd.Series(
        {
            "dominant_count_NPC_like": 2,
            "dominant_count_OPC_like": 4,
            "state_mean_copy_NPC_like_MYC": 10.0,
            "state_mean_copy_OPC_like_MYC": 20.0,
            "state_var_copy_NPC_like_MYC": 1.0,
            "state_var_copy_OPC_like_MYC": 4.0,
        }
    )

    mean, sd = _pooled_state_group_copy_stats(row, ("NPC-like", "OPC-like"), "MYC")

    assert mean == pytest.approx((2 * 10.0 + 4 * 20.0) / 6)
    assert sd == pytest.approx(((2 * (1.0 + 10.0**2) + 4 * (4.0 + 20.0**2)) / 6 - mean**2) ** 0.5)
