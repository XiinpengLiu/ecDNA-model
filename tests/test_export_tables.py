from __future__ import annotations

import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import config as cfg
from analysis.export_tables import write_simulation_tables
from core.simulation import SimulationResult
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


def test_write_simulation_tables_exports_complete_r_tables(workdir: Path) -> None:
    result = _result_with_division_event()
    outputs = write_simulation_tables(result, workdir, condition="Control", seed=7, metadata={"source": "test"})

    expected = {
        "time_summary",
        "cell_snapshots",
        "events",
        "lineage_edges",
        "observations",
        "selected_plot_timepoints",
        "metadata",
        "manifest",
    }
    assert expected <= set(outputs)
    assert not (workdir / "simulation_data").exists()

    cells = pd.read_parquet(outputs["cell_snapshots"])
    assert cells.shape[0] == sum(len(snapshot) for snapshot in result.cell_snapshots)
    for column in (
        "parent_id",
        "latent_1",
        "latent_2",
        "latent_3",
        "copy_MYC",
        "copy_CDK4",
        "copy_PDGFRA",
        "soft_NPC_like",
        "division_hazard",
        "death_hazard",
        "transition_NPC_like_to_OPC_like",
    ):
        assert column in cells.columns

    lineage = pd.read_parquet(outputs["lineage_edges"])
    assert lineage.shape[0] == 2
    assert set(lineage["parent_id"]) == {10}
    assert set(lineage["child_id"]) == {11, 12}

    selected = pd.read_csv(outputs["selected_plot_timepoints"])
    assert selected.shape[0] == 8
    assert selected["time"].iloc[-1] == result.times[-1]

    for legacy_name in ("summary.csv", "snapshots.jsonl", "events.jsonl"):
        assert not (workdir / "tables" / legacy_name).exists()


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
    assert (condition_dir / "tables" / "cell_snapshots.parquet").exists()
    assert not (condition_dir / "simulation_data").exists()
    assert not list(condition_dir.glob("*.png"))

    cells = pd.read_parquet(condition_dir / "tables" / "cell_snapshots.parquet")
    assert {"parent_id", "latent_1", "cycle_index", "dominant_state_index", "last_update_time"} <= set(cells.columns)


def test_record_full_snapshots_cli_flag_was_removed() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--record-full-snapshots"])
