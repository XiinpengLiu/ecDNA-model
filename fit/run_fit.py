"""Command line interface for the fit_method.md pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fit.empirical import build_empirical_summaries
from fit.full_smc import aggregate_accepted_histories, create_full_initial_particles, run_full_reconstruction
from fit.objective import score_particles_from_files
from fit.observation import fit_observation_model
from fit.ppc import run_full_ppc
from fit.raw import create_synthetic_raw_dataset, ingest_raw_data
from fit.scenarios import classify_scenarios_from_files
from fit.schemas import ResultLayout
from fit.stage_runner import run_pipeline_from_raw
from fit.validation import validate_method_contracts
from fit.v4_lite import fit_v4_lite_summary_posterior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ecDNA fit pipeline aligned to markdown/fit_method.md")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-synthetic-raw", help="Create a small deterministic raw-data fixture.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("ingest-raw", help="Standardize raw flow/qPCDR/ecTAG/ddPCR/cell-count tables.")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("fit-observation-model", help="Fit and lock observation calibration.")
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("build-empirical-summaries", help="Build empirical snapshot summaries.")
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-ectag-cells-for-hist", type=int, default=50)

    p = sub.add_parser("fit-lite", help="Generate v4-lite calibrated summary posterior artifacts.")
    p.add_argument("--empirical-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=64)

    p = sub.add_parser("create-full-initial-particles", help="Sample full week-1 representative particles from lite sampler.")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--particles", type=int, default=16)
    p.add_argument("--cells", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("run-full-reconstruction", help="Run full conditional particle reconstruction and scoring.")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--particles", type=int, default=32)
    p.add_argument("--cells", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("score-particles", help="Score existing particle summary features against lite/raw targets.")
    p.add_argument("--particle-features", type=Path, required=True)
    p.add_argument("--lite-target", type=Path, required=True)
    p.add_argument("--distance-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("aggregate-accepted-histories", help="Aggregate accepted full histories.")
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("run-ppc", help="Run posterior predictive checks from full particles.")
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("classify-scenarios", help="Classify scenario labels from full histories.")
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("run-all", help="Run the complete file-based pipeline from raw inputs.")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--particles", type=int, default=32)
    p.add_argument("--cells", type=int, default=300)

    p = sub.add_parser("run-synthetic-smoke", help="Run a complete synthetic smoke pipeline.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--cells", type=int, default=80)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic-raw":
        create_synthetic_raw_dataset(args.output, seed=args.seed)
        return
    if args.command == "ingest-raw":
        ingest_raw_data(args.raw_dir, args.output)
        return
    if args.command == "fit-observation-model":
        fit_observation_model(args.clean_dir, args.output, seed=args.seed)
        return
    if args.command == "build-empirical-summaries":
        build_empirical_summaries(args.clean_dir, args.obs_params, args.output, args.min_ectag_cells_for_hist)
        return
    if args.command == "fit-lite":
        fit_v4_lite_summary_posterior(args.empirical_dir, args.obs_params, args.output, seed=args.seed, posterior_draws=args.draws)
        return
    if args.command == "create-full-initial-particles":
        create_full_initial_particles(args.lite_dir, args.output, particles=args.particles, cells=args.cells, seed=args.seed)
        return
    if args.command == "run-full-reconstruction":
        run_full_reconstruction(args.lite_dir, args.obs_params, args.output, particles=args.particles, cells=args.cells, seed=args.seed)
        return
    if args.command == "score-particles":
        score_particles_from_files(args.particle_features, args.lite_target, args.distance_weights, args.output)
        return
    if args.command == "aggregate-accepted-histories":
        aggregate_accepted_histories(args.full_dir, args.output)
        return
    if args.command == "run-ppc":
        run_full_ppc(args.full_dir, args.lite_dir, args.output)
        return
    if args.command == "classify-scenarios":
        classify_scenarios_from_files(args.full_dir, args.output)
        return
    if args.command == "run-all":
        run_pipeline_from_raw(args.raw_dir, args.output, seed=args.seed, posterior_draws=args.draws, particles=args.particles, cells=args.cells)
        layout = ResultLayout(args.output)
        validate_method_contracts(layout.observation, layout.lite, layout.full_smc)
        return
    if args.command == "run-synthetic-smoke":
        raw_dir = args.output / "raw_fixture"
        create_synthetic_raw_dataset(raw_dir, seed=args.seed)
        run_pipeline_from_raw(raw_dir, args.output / "results", seed=args.seed, posterior_draws=args.draws, particles=args.particles, cells=args.cells)
        layout = ResultLayout(args.output / "results")
        validate_method_contracts(layout.observation, layout.lite, layout.full_smc)
        return
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
