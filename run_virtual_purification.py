"""
Run virtual purification simulations from a control single-cell export table.

Each purified run starts from 2,000 cells sampled without replacement from one
dominant four-state compartment in the control export snapshot. The sampled
cell-state tuples are used directly as the initial population.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as cfg
from analysis.export_tables import write_simulation_tables
from analysis.treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts
from core.cell import Cell, CellPopulation
from core.simulation import HybridOgataSimulator, SimulationResult


STATE_RUNS = tuple(
    (state_name.replace("-like", "_like"), state_name, state_idx)
    for state_idx, state_name in enumerate(cfg.STATE_NAMES)
)

LEGACY_SOFT_COLUMNS = ("soft_NPC_like", "soft_OPC_like", "soft_AC_like", "soft_MES_like")
PACKAGE_SOFT_COLUMNS = ("x_npc", "x_opc", "x_ac", "x_mes")
LEGACY_COPY_COLUMNS = ("copy_MYC", "copy_CDK4", "copy_PDGFRA")
PACKAGE_COPY_COLUMNS = ("k_myc", "k_cdk4", "k_pdgfra")
LEGACY_LATENT_COLUMNS = ("latent_1", "latent_2", "latent_3")
PACKAGE_LATENT_COLUMNS = ("u1", "u2", "u3")
SOURCE_T_INDEX = 24


def _optional_population_size(value: int) -> int | None:
    return None if int(value) == 0 else int(value)


def _build_model_parameters(args: argparse.Namespace) -> cfg.ModelParameters:
    base = cfg.DEFAULT_MODEL_PARAMETERS
    t_max = 11.0
    record_step = 0.5
    simulation = replace(
        base.simulation,
        time_unit="t",
        t_max=t_max,
        record_times=tuple(index * record_step for index in range(int(t_max / record_step) + 1)),
        n_init=int(args.sample_size),
        target_population_size=_optional_population_size(int(args.target_population_size)),
        max_pop_size=int(args.max_pop_size),
        random_seed=int(args.seed),
        record_full_snapshots=True,
        record_events=not bool(args.no_record_events),
    )
    return replace(base, simulation=simulation)


def _part_file_or_directory(path: Path) -> Path:
    part_files = sorted(path.glob("part-*.parquet"))
    return part_files[0] if len(part_files) == 1 else path


def _t_index_snapshot_path(cell_snapshot_dir: Path) -> Path | None:
    preferred = (
        cell_snapshot_dir
        / "condition_id=CTRL"
        / "replicate_id=REP001"
        / f"t_index={SOURCE_T_INDEX}"
    )
    if preferred.exists():
        return _part_file_or_directory(preferred)

    matches = sorted(cell_snapshot_dir.glob(f"condition_id=*/replicate_id=*/t_index={SOURCE_T_INDEX}"))
    if matches:
        return _part_file_or_directory(matches[0])
    return None


def _resolve_ctrl_snapshot_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.exists() and path.name == f"t_index={SOURCE_T_INDEX}":
        return _part_file_or_directory(path)
    if path.exists() and path.name == "cell_snapshot":
        t_index_path = _t_index_snapshot_path(path)
        if t_index_path is None:
            raise FileNotFoundError(f"Could not find t_index={SOURCE_T_INDEX} under {path}.")
        return t_index_path

    candidates = [
        path / "root" / "cell_snapshot",
        path / "tables" / "cell_snapshots.parquet",
        path / "ctrl" / "tables" / "cell_snapshots.parquet",
    ]
    candidates.extend(sorted(path.glob("ensemble_id=*/runs/sim_id=*CTRL*/root/cell_snapshot")))
    candidates.extend(sorted((path / "ctrl").glob("ensemble_id=*/runs/sim_id=*CTRL*/root/cell_snapshot")))
    for candidate in candidates:
        if candidate.exists() and candidate.name == "cell_snapshot":
            t_index_path = _t_index_snapshot_path(candidate)
            if t_index_path is None:
                raise FileNotFoundError(f"Could not find t_index={SOURCE_T_INDEX} under {candidate}.")
            return t_index_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a ctrl cell snapshot export under {path}.")


def _time_column(frame: pd.DataFrame) -> str:
    if "t" in frame.columns:
        return "t"
    if "time" in frame.columns:
        return "time"
    raise ValueError("Control export table must contain either 't' or 'time'.")


def _column_set(frame: pd.DataFrame, choices: tuple[tuple[str, ...], ...], purpose: str) -> tuple[str, ...]:
    for columns in choices:
        if set(columns) <= set(frame.columns):
            return columns
    raise ValueError(f"Control export table is missing {purpose} columns.")


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...], purpose: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Control export table is missing {purpose}.")


def _source_snapshot(frame: pd.DataFrame, source_time: float | None) -> tuple[pd.DataFrame, float]:
    time_col = _time_column(frame)
    times = pd.to_numeric(frame[time_col], errors="raise").astype(float)
    selected_time = float(times.max()) if source_time is None else float(source_time)
    mask = np.isclose(times.to_numpy(dtype=float), selected_time, rtol=0.0, atol=1e-9)
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No control export rows found at source time {selected_time}.")
    return selected, selected_time


def _sample_initial_rows(
    snapshot: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    soft_columns = _column_set(snapshot, (LEGACY_SOFT_COLUMNS, PACKAGE_SOFT_COLUMNS), "soft-state")
    soft_scores = snapshot.loc[:, soft_columns].to_numpy(dtype=float)
    dominant_state = np.argmax(soft_scores, axis=1)
    rng = np.random.default_rng(int(seed))

    rows_by_state: dict[str, pd.DataFrame] = {}
    for state_token, state_name, state_idx in STATE_RUNS:
        state_rows = snapshot.loc[dominant_state == state_idx].copy()
        if len(state_rows) < int(sample_size):
            raise ValueError(
                f"{state_name} has only {len(state_rows)} cells at the selected source time; "
                f"cannot sample {sample_size} without replacement."
            )
        selected_index = rng.choice(state_rows.index.to_numpy(), size=int(sample_size), replace=False)
        sampled = snapshot.loc[selected_index].copy()
        sampled.insert(0, "virtual_purified_state", state_name)
        sampled.insert(1, "virtual_source_row", sampled.index.astype(str))
        rows_by_state[state_token] = sampled.reset_index(drop=True)
    return rows_by_state


def _cycle_index_from_row(row: pd.Series) -> int:
    if "cycle_index" in row and pd.notna(row["cycle_index"]):
        return int(row["cycle_index"])
    cycle_column = "cell_cycle_state" if "cell_cycle_state" in row else "cycle_state"
    value = row[cycle_column]
    if isinstance(value, str):
        return int(cfg.CYCLE_INDEX[value])
    return int(value)


def _rows_to_population(params: cfg.ModelParameters, rows: pd.DataFrame) -> CellPopulation:
    copy_columns = _column_set(rows, (LEGACY_COPY_COLUMNS, PACKAGE_COPY_COLUMNS), "copy-number")
    soft_columns = _column_set(rows, (LEGACY_SOFT_COLUMNS, PACKAGE_SOFT_COLUMNS), "soft-state")
    latent_columns = _column_set(rows, (LEGACY_LATENT_COLUMNS, PACKAGE_LATENT_COLUMNS), "latent-state")
    age_column = _first_column(rows, ("age", "age_t"), "cell age")
    stress_column = _first_column(rows, ("stress_score", "r_stress"), "stress score")
    survival_column = _first_column(rows, ("survival_score", "v_survival"), "survival score")
    cell_id_column = "cell_id" if "cell_id" in rows.columns else None

    population = CellPopulation(params, cfg.DEFAULT_INITIALIZATION_PARAMETERS, np.random.default_rng(0))
    max_cell_id = -1
    for offset, (_, row) in enumerate(rows.iterrows()):
        cell_id = int(row[cell_id_column]) if cell_id_column is not None and pd.notna(row[cell_id_column]) else offset
        max_cell_id = max(max_cell_id, cell_id)
        cell = Cell(
            cell_id=cell_id,
            parent_id=None,
            cycle_state=_cycle_index_from_row(row),
            copy_numbers=row.loc[list(copy_columns)].to_numpy(dtype=int),
            soft_state=row.loc[list(soft_columns)].to_numpy(dtype=float),
            latent_state=row.loc[list(latent_columns)].to_numpy(dtype=float),
            stress_score=float(row[stress_column]),
            survival_score=float(row[survival_column]),
            age=float(row[age_column]),
            last_update_time=0.0,
            last_D_C=params.exposure.D_C0,
            last_D_P=params.exposure.D_P0,
        )
        cell.validate()
        population.cells.append(cell)
    population.next_id = max_cell_id + 1
    return population


def _build_run_metadata(
    *,
    state_token: str,
    state_name: str,
    seed: int,
    sample_size: int,
    source_path: Path,
    source_time: float,
    params: cfg.ModelParameters,
) -> dict[str, Any]:
    simulation = params.simulation
    return {
        "condition": "ctrl",
        "drug": "vehicle",
        "dose_nM": 0.0,
        "seed": int(seed),
        "virtual_purification": {
            "purified_state": state_name,
            "sample_size": int(sample_size),
            "source_condition": "ctrl",
            "source_export": str(source_path),
            "source_time": float(source_time),
            "state_call": "argmax over the four soft-state scores in the source export row",
        },
        "sim_id": f"SIM_FULL_CTRL_PURIFIED_{state_token.upper()}_REP001",
        "simulation": {
            "time_unit": simulation.time_unit,
            "t_max": simulation.t_max,
            "record_times": list(simulation.record_times),
            "n_init": simulation.n_init,
            "target_population_size": simulation.target_population_size,
            "max_pop_size": simulation.max_pop_size,
            "record_full_snapshots": simulation.record_full_snapshots,
            "record_events": simulation.record_events,
        },
    }


def _write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.joinpath("run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _summary_row(state_token: str, state_name: str, result: SimulationResult, condition_dir: Path) -> dict[str, Any]:
    trends = compute_bulk_copy_trends(result)
    terminal_counts = compute_terminal_event_counts(result)
    final_bulk = result.bulk_copy_means[-1] if result.bulk_copy_means else [float("nan")] * cfg.N_SPECIES
    return {
        "purified_state": state_name,
        "run_id": state_token,
        "condition": "ctrl",
        "drug": "vehicle",
        "dose_nM": 0.0,
        "stop_reason": result.stop_reason,
        "stop_time": result.stop_time,
        "final_population_size": result.population_sizes[-1] if result.population_sizes else 0,
        "late_growth_rate": compute_growth_rate(result),
        "final_mean_MYC": float(final_bulk[cfg.MYC]),
        "final_mean_CDK4": float(final_bulk[cfg.CDK4]),
        "final_mean_PDGFRA": float(final_bulk[cfg.PDGFRA]),
        "MYC_trend": trends["MYC"],
        "CDK4_trend": trends["CDK4"],
        "PDGFRA_trend": trends["PDGFRA"],
        "division_events": terminal_counts["division"],
        "death_events": terminal_counts["death"],
        "result_dir": str(condition_dir),
    }


def run_purified_state(
    state_token: str,
    state_name: str,
    rows: pd.DataFrame,
    *,
    params: cfg.ModelParameters,
    output_dir: Path,
    source_path: Path,
    source_time: float,
    seed: int,
    plots: bool,
    verbose: bool,
) -> dict[str, Any]:
    condition_dir = output_dir / state_token
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(condition_dir / "sampled_initial_cells.parquet", index=False)

    seed_sequence = np.random.SeedSequence(int(seed))
    _init_seed, event_seed, observation_seed = seed_sequence.spawn(3)
    population = _rows_to_population(params, rows)
    simulator = HybridOgataSimulator(
        params=params,
        observation_params=cfg.DEFAULT_OBSERVATION_PARAMETERS,
        input_schedules=cfg.t87_input_schedules_for_condition("ctrl"),
        seed=int(seed),
        event_rng=np.random.default_rng(event_seed),
        observation_rng=np.random.default_rng(observation_seed),
    )
    result = simulator.simulate(population=population, verbose=verbose)

    metadata = _build_run_metadata(
        state_token=state_token,
        state_name=state_name,
        seed=seed,
        sample_size=len(rows),
        source_path=source_path,
        source_time=source_time,
        params=params,
    )
    write_simulation_tables(result, condition_dir, condition="ctrl", seed=seed, metadata=metadata)
    if plots:
        from analysis.plotting import plot_single_run_diagnostic_suite

        plot_single_run_diagnostic_suite(result, condition_dir)
    _write_run_metadata(condition_dir, metadata)
    return _summary_row(state_token, state_name, result, condition_dir)


def _write_batch_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with output_dir.joinpath("batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    defaults = cfg.DEFAULT_MODEL_PARAMETERS.simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ctrl-export",
        type=Path,
        default=Path("outputs") / "ensemble_id=ENS_000001" / "runs" / "sim_id=SIM_FULL_CTRL_REP001",
        help="Control export directory or cell snapshot parquet/dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_v4") / "virtual_purification",
        help="Directory for virtual purification results.",
    )
    parser.add_argument(
        "--source-time",
        type=float,
        default=None,
        help="Control snapshot time to sample. Default uses the maximum exported time.",
    )
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--target-population-size",
        type=int,
        default=defaults.target_population_size if defaults.target_population_size is not None else 0,
        help="Use 0 to disable target-population stopping.",
    )
    parser.add_argument("--max-pop-size", type=int, default=defaults.max_pop_size)
    parser.add_argument("--no-record-events", action="store_true", help="Do not save event logs.")
    parser.add_argument(
        "--plots",
        dest="plots",
        action="store_true",
        default=True,
        help="Write the single-run diagnostic PNG suite. This is enabled by default.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip diagnostic PNGs for faster data-only runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(STATE_RUNS),
        help="State worker processes. Default runs one process per purified state.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress simulator progress lines.")
    return parser


def _print_state_summary(state_name: str, row: dict[str, Any]) -> None:
    print(
        f"{state_name}: stop={row['stop_reason']} at t={float(row['stop_time']):.2f}, "
        f"final_pop={row['final_population_size']}"
    )


def _run_states_sequential(
    rows_by_state: dict[str, pd.DataFrame],
    *,
    params: cfg.ModelParameters,
    output_dir: Path,
    source_path: Path,
    source_time: float,
    seed: int,
    plots: bool,
    verbose: bool,
) -> dict[str, dict[str, Any]]:
    rows_by_run: dict[str, dict[str, Any]] = {}
    for state_token, state_name, _state_idx in STATE_RUNS:
        print(f"Running purified {state_name}...")
        row = run_purified_state(
            state_token,
            state_name,
            rows_by_state[state_token],
            params=params,
            output_dir=output_dir,
            source_path=source_path,
            source_time=source_time,
            seed=seed,
            plots=plots,
            verbose=verbose,
        )
        rows_by_run[state_token] = row
        _print_state_summary(state_name, row)
    return rows_by_run


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg.require(int(args.sample_size) > 0, "sample_size must be strictly positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = _resolve_ctrl_snapshot_path(Path(args.ctrl_export))
    ctrl_export = pd.read_parquet(source_path)
    source_snapshot, source_time = _source_snapshot(ctrl_export, args.source_time)
    rows_by_state = _sample_initial_rows(
        source_snapshot,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )
    print(f"Sampling source: {source_path.resolve()} at time {source_time:g}")

    params = _build_model_parameters(args)
    cfg.validate_model_parameters(params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    rows_by_run: dict[str, dict[str, Any]]
    max_workers = min(max(1, int(args.workers)), len(STATE_RUNS))
    if max_workers == 1:
        rows_by_run = _run_states_sequential(
            rows_by_state,
            params=params,
            output_dir=output_dir,
            source_path=source_path,
            source_time=source_time,
            seed=int(args.seed),
            plots=bool(args.plots),
            verbose=not bool(args.quiet),
        )
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except PermissionError:
            print("Process pool unavailable; falling back to sequential purified-state runs.")
            rows_by_run = _run_states_sequential(
                rows_by_state,
                params=params,
                output_dir=output_dir,
                source_path=source_path,
                source_time=source_time,
                seed=int(args.seed),
                plots=bool(args.plots),
                verbose=not bool(args.quiet),
            )
        else:
            rows_by_run = {}
            with executor:
                futures = {}
                for state_token, state_name, _state_idx in STATE_RUNS:
                    print(f"Running purified {state_name}...")
                    futures[
                        executor.submit(
                            run_purified_state,
                            state_token,
                            state_name,
                            rows_by_state[state_token],
                            params=params,
                            output_dir=output_dir,
                            source_path=source_path,
                            source_time=source_time,
                            seed=int(args.seed),
                            plots=bool(args.plots),
                            verbose=not bool(args.quiet),
                        )
                    ] = (state_token, state_name)

                for future in as_completed(futures):
                    state_token, state_name = futures[future]
                    row = future.result()
                    rows_by_run[state_token] = row
                    _print_state_summary(state_name, row)

    rows = [rows_by_run[state_token] for state_token, _state_name, _state_idx in STATE_RUNS]
    _write_batch_summary(output_dir, rows)
    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
