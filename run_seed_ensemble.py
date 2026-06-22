"""
Run T87 drug-condition simulations across multiple random seeds.

This separate entry point keeps main.py unchanged and writes all runs into one
ensemble package, for example:

    <output-dir>/ensemble_id=seeds_run/runs/sim_id=SIM_FULL_P10_SEED000001
"""

from __future__ import annotations

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import config as cfg
import main as t87_main
from analysis.export_tables import write_simulation_tables
from analysis.treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts
from core.simulation import SimulationResult, run_simulation


DEFAULT_ENSEMBLE_ID = "seeds_run"


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in (item.strip() for item in raw.split(",")):
        if not token:
            continue
        try:
            seed = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid seed in --seeds: {token!r}.") from exc
        cfg.require(seed >= 0, "Seeds must be non-negative integers.")
        seeds.append(seed)
    cfg.require(bool(seeds), "--seeds must contain at least one integer seed.")
    return tuple(dict.fromkeys(seeds))


def _seed_token(seed: int) -> str:
    return f"SEED{int(seed):06d}"


def _seeded_sim_id(condition: str, seed: int) -> str:
    return f"SIM_FULL_{condition.upper()}_{_seed_token(seed)}"


def _seeded_replicate_id(seed: int) -> str:
    return f"REP_{_seed_token(seed)}"


def _build_seed_parameters(args: argparse.Namespace, seed: int) -> cfg.ModelParameters:
    seed_args = copy.copy(args)
    seed_args.seed = int(seed)
    return t87_main._build_model_parameters(seed_args)


def _build_seed_metadata(
    *,
    condition: str,
    seed: int,
    rows_per_state: int,
    params: cfg.ModelParameters,
    ensemble_id: str,
    sim_id: str,
) -> dict[str, Any]:
    metadata = t87_main._build_run_metadata(
        condition=condition,
        seed=seed,
        rows_per_state=rows_per_state,
        params=params,
    )
    metadata.update(
        {
            "ensemble_id": ensemble_id,
            "sim_id": sim_id,
            "replicate_id": _seeded_replicate_id(seed),
        }
    )
    return metadata


def _summary_row(
    *,
    condition: str,
    seed: int,
    sim_id: str,
    ensemble_id: str,
    result: SimulationResult,
    diagnostic_dir: Path,
) -> dict[str, Any]:
    drug, dose = cfg.T87_CONDITION_TREATMENTS[condition]
    trends = compute_bulk_copy_trends(result)
    terminal_counts = compute_terminal_event_counts(result)
    final_bulk = result.bulk_copy_means[-1] if result.bulk_copy_means else [float("nan")] * cfg.N_SPECIES
    return {
        "ensemble_id": ensemble_id,
        "sim_id": sim_id,
        "condition": condition,
        "seed": int(seed),
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
        "result_dir": str(diagnostic_dir),
    }


def run_seed_condition(
    condition: str,
    seed: int,
    *,
    args: argparse.Namespace,
    raw_dir: Path,
    output_dir: Path,
    export_lock_path: Path | None = None,
) -> dict[str, Any]:
    params = _build_seed_parameters(args, seed)
    cfg.validate_model_parameters(params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    sim_id = _seeded_sim_id(condition, seed)
    diagnostic_dir = output_dir / "diagnostics" / condition / f"seed={int(seed)}"
    initialization = cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=raw_dir / "ddpcr.csv",
        seed=seed,
        rows_per_state=int(args.rows_per_state),
    )
    result = run_simulation(
        params=params,
        initialization=initialization,
        input_schedules=cfg.t87_input_schedules_for_condition(condition),
        seed=seed,
        verbose=not bool(args.quiet),
    )
    metadata = _build_seed_metadata(
        condition=condition,
        seed=seed,
        rows_per_state=int(args.rows_per_state),
        params=params,
        ensemble_id=str(args.ensemble_id),
        sim_id=sim_id,
    )

    if export_lock_path is None:
        write_simulation_tables(result, output_dir, condition=condition, seed=seed, metadata=metadata)
    else:
        with t87_main._directory_lock(export_lock_path):
            write_simulation_tables(result, output_dir, condition=condition, seed=seed, metadata=metadata)

    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        from analysis.plotting import plot_single_run_diagnostic_suite

        plot_single_run_diagnostic_suite(result, diagnostic_dir)
    t87_main._write_run_metadata(diagnostic_dir, metadata)
    return _summary_row(
        condition=condition,
        seed=seed,
        sim_id=sim_id,
        ensemble_id=str(args.ensemble_id),
        result=result,
        diagnostic_dir=diagnostic_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = cfg.DEFAULT_MODEL_PARAMETERS.simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default="all", help="Comma-separated T87 conditions to run, or 'all'.")
    parser.add_argument("--seeds", required=True, help="Comma-separated non-negative integer seeds, for example: 1,2,3.")
    parser.add_argument("--ensemble-id", default=DEFAULT_ENSEMBLE_ID, help="Default: seeds_run.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw") / "t87_drug_bulkfit")
    parser.add_argument("--output-dir", type=Path, default=Path("results_v4") / "t87_seed_runs")
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
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--workers", type=int, default=len(t87_main.T87_CONDITIONS))
    parser.add_argument("--quiet", action="store_true")
    return parser


def _print_run_summary(row: dict[str, Any]) -> None:
    print(
        f"{row['condition']} seed={row['seed']}: "
        f"stop={row['stop_reason']} at t={float(row['stop_time']):.2f}, "
        f"final_pop={row['final_population_size']}, sim_id={row['sim_id']}"
    )


def _run_sequential(
    tasks: list[tuple[str, int]],
    *,
    args: argparse.Namespace,
    raw_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for condition, seed in tasks:
        print(f"Running {condition} seed={seed}...")
        row = run_seed_condition(condition, seed, args=args, raw_dir=raw_dir, output_dir=output_dir)
        rows.append(row)
        _print_run_summary(row)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    conditions = t87_main._parse_conditions(str(args.conditions))
    seeds = _parse_seeds(str(args.seeds))
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(condition, seed) for seed in seeds for condition in conditions]
    max_workers = min(max(1, int(args.workers)), len(tasks))
    if max_workers == 1:
        rows = _run_sequential(tasks, args=args, raw_dir=raw_dir, output_dir=output_dir)
    else:
        export_lock_path = output_dir / f".export_lock_{os.getpid()}"
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except PermissionError:
            print("Process pool unavailable; falling back to sequential runs.")
            rows = _run_sequential(tasks, args=args, raw_dir=raw_dir, output_dir=output_dir)
        else:
            rows = []
            with executor:
                futures = {
                    executor.submit(
                        run_seed_condition,
                        condition,
                        seed,
                        args=args,
                        raw_dir=raw_dir,
                        output_dir=output_dir,
                        export_lock_path=export_lock_path,
                    ): (condition, seed)
                    for condition, seed in tasks
                }
                for condition, seed in tasks:
                    print(f"Running {condition} seed={seed}...")
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    _print_run_summary(row)

    row_by_key = {(str(row["condition"]), int(row["seed"])): row for row in rows}
    ordered_rows = [row_by_key[(condition, seed)] for condition, seed in tasks]
    t87_main._write_batch_summary(output_dir, ordered_rows)
    print(f"Seed ensemble written to: {(output_dir / f'ensemble_id={args.ensemble_id}').resolve()}")


if __name__ == "__main__":
    main()
