"""
One-command runner for the T87 drug-condition simulations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import config as cfg
from analysis.export_tables import write_simulation_tables
from analysis.treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts
from core.simulation import SimulationResult, run_simulation


T87_CONDITIONS = tuple(cfg.T87_CONDITION_TREATMENTS.keys())


def _parse_conditions(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return T87_CONDITIONS
    conditions = tuple(token.strip() for token in raw.split(",") if token.strip())
    cfg.require(bool(conditions), "At least one condition is required.")
    unknown = [condition for condition in conditions if condition not in cfg.T87_CONDITION_TREATMENTS]
    cfg.require(not unknown, f"Unsupported T87 condition(s): {unknown}.")
    return tuple(dict.fromkeys(conditions))


def _optional_population_size(value: int) -> int | None:
    return None if int(value) == 0 else int(value)


def _build_model_parameters(args: argparse.Namespace) -> cfg.ModelParameters:
    base = cfg.DEFAULT_MODEL_PARAMETERS
    t_max = 12.0
    record_step = 0.5
    simulation = replace(
        base.simulation,
        time_unit="t",
        t_max=t_max,
        record_times=tuple(index * record_step for index in range(int(t_max / record_step) + 1)),
        n_init=int(args.n_init),
        target_population_size=_optional_population_size(int(args.target_population_size)),
        max_pop_size=int(args.max_pop_size),
        random_seed=int(args.seed),
        record_full_snapshots=True,
        record_events=not bool(args.no_record_events),
    )
    return replace(base, simulation=simulation)


def _build_run_metadata(
    *,
    condition: str,
    seed: int,
    rows_per_state: int,
    params: cfg.ModelParameters,
) -> dict[str, object]:
    drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    simulation = params.simulation
    return {
        "condition": condition,
        "drug": drug,
        "dose_nM": dose,
        "seed": int(seed),
        "rows_per_state": int(rows_per_state),
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


def _write_run_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.joinpath("run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@contextmanager
def _directory_lock(path: Path, *, timeout_s: float = 3600.0, poll_s: float = 0.1):
    deadline = time.monotonic() + timeout_s
    acquired = False
    while not acquired:
        try:
            path.mkdir()
            acquired = True
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for export lock: {path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            path.rmdir()
        except OSError:
            pass


def _summary_row(condition: str, result: SimulationResult, condition_dir: Path) -> dict[str, object]:
    drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    trends = compute_bulk_copy_trends(result)
    terminal_counts = compute_terminal_event_counts(result)
    final_bulk = result.bulk_copy_means[-1] if result.bulk_copy_means else [float("nan")] * cfg.N_SPECIES
    return {
        "condition": condition,
        "drug": drug,
        "dose_nM": dose,
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


def run_condition(
    condition: str,
    *,
    params: cfg.ModelParameters,
    raw_dir: Path,
    output_dir: Path,
    seed: int,
    rows_per_state: int,
    plots: bool,
    verbose: bool,
    export_base_dir: Path | None = None,
    export_lock_path: Path | None = None,
) -> dict[str, object]:
    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    initialization = cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=raw_dir / "ddpcr.csv",
        seed=seed,
        rows_per_state=rows_per_state,
    )
    result = run_simulation(
        params=params,
        initialization=initialization,
        input_schedules=cfg.t87_input_schedules_for_condition(condition),
        seed=seed,
        verbose=verbose,
    )
    metadata = _build_run_metadata(
        condition=condition,
        seed=seed,
        rows_per_state=rows_per_state,
        params=params,
    )
    package_base_dir = condition_dir if export_base_dir is None else Path(export_base_dir)
    if export_lock_path is None:
        write_simulation_tables(result, package_base_dir, condition=condition, seed=seed, metadata=metadata)
    else:
        with _directory_lock(Path(export_lock_path)):
            write_simulation_tables(result, package_base_dir, condition=condition, seed=seed, metadata=metadata)
    if plots:
        from analysis.plotting import plot_single_run_diagnostic_suite

        plot_single_run_diagnostic_suite(result, condition_dir)
    _write_run_metadata(condition_dir, metadata)
    return _summary_row(condition, result, condition_dir)


def _write_batch_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
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
        "--conditions",
        default="all",
        help="Comma-separated T87 conditions to run, or 'all'.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw") / "t87_drug_bulkfit",
        help="Directory containing T87 ddpcr.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_v4") / "t87_conditions",
        help="Directory for per-condition results.",
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--rows-per-state", type=int, default=512)
    parser.add_argument("--n-init", type=int, default=defaults.n_init)
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
        default=len(T87_CONDITIONS),
        help="Condition worker processes. Default runs one process per condition.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress simulator progress lines.")
    return parser


def _print_condition_summary(condition: str, row: dict[str, object]) -> None:
    print(
        f"{condition}: stop={row['stop_reason']} at t={float(row['stop_time']):.2f}, "
        f"final_pop={row['final_population_size']}"
    )


def _run_conditions_sequential(
    conditions: tuple[str, ...],
    *,
    params: cfg.ModelParameters,
    raw_dir: Path,
    output_dir: Path,
    seed: int,
    rows_per_state: int,
    plots: bool,
    verbose: bool,
) -> dict[str, dict[str, object]]:
    rows_by_condition: dict[str, dict[str, object]] = {}
    for condition in conditions:
        print(f"Running {condition}...")
        row = run_condition(
            condition,
            params=params,
            raw_dir=raw_dir,
            output_dir=output_dir,
            seed=seed,
            rows_per_state=rows_per_state,
            plots=plots,
            verbose=verbose,
            export_base_dir=output_dir,
        )
        rows_by_condition[condition] = row
        _print_condition_summary(condition, row)
    return rows_by_condition


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    conditions = _parse_conditions(str(args.conditions))
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _build_model_parameters(args)
    cfg.validate_model_parameters(params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    rows_by_condition: dict[str, dict[str, object]] = {}
    max_workers = min(max(1, int(args.workers)), len(conditions))
    if max_workers == 1:
        rows_by_condition = _run_conditions_sequential(
            conditions,
            params=params,
            raw_dir=raw_dir,
            output_dir=output_dir,
            seed=int(args.seed),
            rows_per_state=int(args.rows_per_state),
            plots=bool(args.plots),
            verbose=not bool(args.quiet),
        )
    else:
        export_lock_path = output_dir / f".export_lock_{os.getpid()}"
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except PermissionError:
            print("Process pool unavailable; falling back to sequential condition runs.")
            rows_by_condition = _run_conditions_sequential(
                conditions,
                params=params,
                raw_dir=raw_dir,
                output_dir=output_dir,
                seed=int(args.seed),
                rows_per_state=int(args.rows_per_state),
                plots=bool(args.plots),
                verbose=not bool(args.quiet),
            )
        else:
            with executor:
                futures = {}
                for condition in conditions:
                    print(f"Running {condition}...")
                    futures[
                        executor.submit(
                            run_condition,
                            condition,
                            params=params,
                            raw_dir=raw_dir,
                            output_dir=output_dir,
                            seed=int(args.seed),
                            rows_per_state=int(args.rows_per_state),
                            plots=bool(args.plots),
                            verbose=not bool(args.quiet),
                            export_base_dir=output_dir,
                            export_lock_path=export_lock_path,
                        )
                    ] = condition

                for future in as_completed(futures):
                    condition = futures[future]
                    row = future.result()
                    rows_by_condition[condition] = row
                    _print_condition_summary(condition, row)

    rows = [rows_by_condition[condition] for condition in conditions]
    _write_batch_summary(output_dir, rows)
    if args.plots:
        from analysis.plotting import plot_t87_treatment_comparison_suite

        comparison_plots = plot_t87_treatment_comparison_suite(output_dir, raw_dir=raw_dir, conditions=conditions)
        if comparison_plots:
            print(f"T87 comparison plots written to: {(output_dir / 't87_comparison_plots').resolve()}")
    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
