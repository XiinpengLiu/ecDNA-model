"""
Run T87 ablation simulations across the fixed seed panel.

Outputs are written under:

    <output-dir>/ensemble_id=ablation/runs/sim_id=<...>
"""

from __future__ import annotations

import argparse
import copy
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ablation_config as ab_cfg
import config as cfg
import main as t87_main
from analysis.export_tables import write_simulation_tables
from analysis.treatment import compute_bulk_copy_trends, compute_growth_rate, compute_terminal_event_counts
from core.simulation import SimulationResult, run_simulation


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
    seeds = list(dict.fromkeys(seeds))
    cfg.require(len(seeds) == 4, "--seeds must contain exactly four distinct seeds.")
    cfg.require(24 in seeds, "--seeds must include 24.")
    return tuple(seeds)


def _parse_condition(raw: str) -> str:
    condition = raw.strip()
    cfg.require(condition in cfg.T87_CONDITION_TREATMENTS, f"Unsupported T87 condition: {condition!r}.")
    return condition


def _seed_token(seed: int) -> str:
    return f"SEED{int(seed):06d}"


def _ablation_sim_id(ablation_name: str, condition: str, seed: int) -> str:
    return f"SIM_{ablation_name}_{condition.upper()}_{_seed_token(seed)}"


def _replicate_id(seed: int) -> str:
    return f"REP_{_seed_token(seed)}"


def _build_seed_base_parameters(args: argparse.Namespace, seed: int) -> cfg.ModelParameters:
    seed_args = copy.copy(args)
    seed_args.seed = int(seed)
    return t87_main._build_model_parameters(seed_args)


@contextmanager
def _temporary_export_parameters(params: cfg.ModelParameters) -> Iterator[None]:
    """Make export-derived parameter tables use the active ablation parameters."""

    original = cfg.DEFAULT_MODEL_PARAMETERS
    cfg.DEFAULT_MODEL_PARAMETERS = params
    try:
        yield
    finally:
        cfg.DEFAULT_MODEL_PARAMETERS = original


def _build_ablation_metadata(
    *,
    ablation_name: str,
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
    metadata.update(ab_cfg.metadata_for_ablation(ablation_name))
    metadata.update(
        {
            "ensemble_id": ensemble_id,
            "sim_id": sim_id,
            "replicate_id": _replicate_id(seed),
            "model_variant": ablation_name,
            "parameter_set_id": f"PARAM_{ablation_name}",
        }
    )
    return metadata


def _summary_row(
    *,
    ablation_name: str,
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
        "ablation_name": ablation_name,
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


def run_ablation(
    ablation_name: str,
    seed: int,
    *,
    args: argparse.Namespace,
    condition: str,
    raw_dir: Path,
    output_dir: Path,
    export_lock_path: Path | None = None,
) -> dict[str, Any]:
    base_params = _build_seed_base_parameters(args, seed)
    params = ab_cfg.build_model_parameters(ablation_name, base_params)
    cfg.validate_observation_parameters(cfg.DEFAULT_OBSERVATION_PARAMETERS)

    sim_id = _ablation_sim_id(ablation_name, condition, seed)
    initialization = ab_cfg.build_initialization_parameters(
        ablation_name,
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
    metadata = _build_ablation_metadata(
        ablation_name=ablation_name,
        condition=condition,
        seed=seed,
        rows_per_state=int(args.rows_per_state),
        params=params,
        ensemble_id=str(args.ensemble_id),
        sim_id=sim_id,
    )

    with _temporary_export_parameters(params):
        if export_lock_path is None:
            write_simulation_tables(result, output_dir, condition=condition, seed=seed, metadata=metadata)
        else:
            with t87_main._directory_lock(export_lock_path):
                write_simulation_tables(result, output_dir, condition=condition, seed=seed, metadata=metadata)

    diagnostic_dir = output_dir / "diagnostics" / ablation_name / f"seed={int(seed)}"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        from analysis.plotting import plot_single_run_diagnostic_suite

        plot_single_run_diagnostic_suite(result, diagnostic_dir)
    t87_main._write_run_metadata(diagnostic_dir, metadata)
    return _summary_row(
        ablation_name=ablation_name,
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
    parser.add_argument("--ablations", default="all", help="Comma-separated ablation names, or 'all'.")
    parser.add_argument("--condition", default=ab_cfg.DEFAULT_ABLATION_CONDITION, help="Single T87 condition to run.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in ab_cfg.DEFAULT_ABLATION_SEEDS),
        help="Exactly four comma-separated seeds. Must include 24.",
    )
    parser.add_argument("--ensemble-id", default=ab_cfg.DEFAULT_ABLATION_ENSEMBLE_ID)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw") / "t87_drug_bulkfit")
    parser.add_argument("--output-dir", type=Path, default=Path("results_v4") / "t87_ablation")
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
    parser.add_argument("--workers", type=int, default=len(ab_cfg.ABLATION_NAMES))
    parser.add_argument("--quiet", action="store_true")
    return parser


def _print_run_summary(row: dict[str, Any]) -> None:
    print(
        f"{row['ablation_name']} seed={row['seed']}: "
        f"stop={row['stop_reason']} at t={float(row['stop_time']):.2f}, "
        f"final_pop={row['final_population_size']}, sim_id={row['sim_id']}"
    )


def _run_sequential(
    tasks: list[tuple[str, int]],
    *,
    args: argparse.Namespace,
    condition: str,
    raw_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for ablation_name, seed in tasks:
        print(f"Running {ablation_name} {condition} seed={seed}...")
        row = run_ablation(
            ablation_name,
            seed,
            args=args,
            condition=condition,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )
        rows.append(row)
        _print_run_summary(row)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    ablations = ab_cfg.parse_ablation_names(str(args.ablations))
    seeds = _parse_seeds(str(args.seeds))
    condition = _parse_condition(str(args.condition))
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(ablation_name, seed) for ablation_name in ablations for seed in seeds]
    max_workers = min(max(1, int(args.workers)), len(tasks))
    if max_workers == 1:
        rows = _run_sequential(tasks, args=args, condition=condition, raw_dir=raw_dir, output_dir=output_dir)
    else:
        export_lock_path = output_dir / f".export_lock_{os.getpid()}"
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except PermissionError:
            print("Process pool unavailable; falling back to sequential runs.")
            rows = _run_sequential(tasks, args=args, condition=condition, raw_dir=raw_dir, output_dir=output_dir)
        else:
            rows = []
            with executor:
                futures = {
                    executor.submit(
                        run_ablation,
                        ablation_name,
                        seed,
                        args=args,
                        condition=condition,
                        raw_dir=raw_dir,
                        output_dir=output_dir,
                        export_lock_path=export_lock_path,
                    ): (ablation_name, seed)
                    for ablation_name, seed in tasks
                }
                for ablation_name, seed in tasks:
                    print(f"Running {ablation_name} {condition} seed={seed}...")
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    _print_run_summary(row)

    row_by_key = {(str(row["ablation_name"]), int(row["seed"])): row for row in rows}
    ordered_rows = [row_by_key[(ablation_name, seed)] for ablation_name, seed in tasks]
    t87_main._write_batch_summary(output_dir, ordered_rows)
    print(f"Ablation ensemble written to: {(output_dir / f'ensemble_id={args.ensemble_id}').resolve()}")


if __name__ == "__main__":
    main()
