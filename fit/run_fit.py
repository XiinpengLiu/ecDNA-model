"""Command line interface for the bulk-only fit_method.md pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fit.final_report import build_final_report_layer
from fit.full_smc import aggregate_accepted_histories, create_full_initial_particles, run_full_reconstruction, run_moment_prescreen
from fit.manifest import build_run_manifest
from fit.observation import fit_observation_model
from fit.parameter_registry import build_parameter_registry, run_prior_predictive_gate
from fit.raw import create_synthetic_raw_dataset, ingest_raw_data
from fit.schemas import ResultLayout
from fit.stage_runner import run_pipeline_from_raw
from fit.validation import build_validation_reports, validate_method_contracts
from fit.v4_lite import fit_v4_lite_summary_posterior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ecDNA bulk fit pipeline aligned to markdown/fit_method.md")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-synthetic-raw")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("ingest-raw")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build-manifest")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment-config", type=Path)
    p.add_argument("--model-schema", type=Path)
    p = sub.add_parser("fit-observation-model")
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("fit-lite")
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=64)
    p = sub.add_parser("build-parameter-registry")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run-prior-predictive-gate")
    p.add_argument("--registry-dir", type=Path, required=True)
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("run-moment-prescreen")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-candidates", type=int, default=200000)
    p.add_argument("--keep-top", type=int, default=10000)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("create-full-initial-particles")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--moment-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--particles", type=int, default=16)
    p.add_argument("--cells", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("run-full-reconstruction")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--moment-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--particles", type=int, default=3000)
    p.add_argument("--cells", type=int, default=10000)
    p.add_argument("--smc-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("run-validation")
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--registry-dir", type=Path, required=True)
    p.add_argument("--obs-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("aggregate-accepted-histories")
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build-final-report")
    p.add_argument("--observation-dir", type=Path, required=True)
    p.add_argument("--lite-dir", type=Path, required=True)
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--validation-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run-all")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--particles", type=int, default=3000)
    p.add_argument("--cells", type=int, default=10000)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("run-synthetic-smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--cells", type=int, default=80)
    p.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic-raw":
        create_synthetic_raw_dataset(args.output, seed=args.seed)
    elif args.command == "ingest-raw":
        ingest_raw_data(args.raw_dir, args.output)
    elif args.command == "build-manifest":
        build_run_manifest(args.raw_dir, args.output, args.experiment_config, args.model_schema)
    elif args.command == "fit-observation-model":
        fit_observation_model(args.clean_dir, args.output, seed=args.seed)
    elif args.command == "fit-lite":
        fit_v4_lite_summary_posterior(args.clean_dir, args.obs_params, args.output, seed=args.seed, posterior_draws=args.draws)
    elif args.command == "build-parameter-registry":
        build_parameter_registry(args.lite_dir, args.output)
    elif args.command == "run-prior-predictive-gate":
        run_prior_predictive_gate(args.registry_dir, args.lite_dir, args.obs_params, args.output, seed=args.seed)
    elif args.command == "run-moment-prescreen":
        run_moment_prescreen(args.lite_dir, args.prior_dir, args.output, seed=args.seed, n_candidates=args.n_candidates, keep_top=args.keep_top, workers=args.workers)
    elif args.command == "create-full-initial-particles":
        create_full_initial_particles(args.lite_dir, args.output, particles=args.particles, cells=args.cells, seed=args.seed, moment_dir=args.moment_dir)
    elif args.command == "run-full-reconstruction":
        run_full_reconstruction(args.lite_dir, args.obs_params, args.output, particles=args.particles, cells=args.cells, seed=args.seed, smc_steps=args.smc_steps, moment_dir=args.moment_dir, workers=args.workers)
    elif args.command == "run-validation":
        build_validation_reports(args.lite_dir, args.full_dir, args.registry_dir, args.output, args.obs_params)
    elif args.command == "aggregate-accepted-histories":
        aggregate_accepted_histories(args.full_dir, args.output)
    elif args.command == "build-final-report":
        build_final_report_layer(args.observation_dir, args.lite_dir, args.full_dir, args.output, validation_dir=args.validation_dir)
    elif args.command == "run-all":
        run_pipeline_from_raw(args.raw_dir, args.output, seed=args.seed, posterior_draws=args.draws, particles=args.particles, cells=args.cells, workers=args.workers)
        layout = ResultLayout(args.output)
        validate_method_contracts(layout.observation, layout.lite, layout.full_smc, layout.final_report)
    elif args.command == "run-synthetic-smoke":
        raw_dir = args.output / "raw_fixture"
        create_synthetic_raw_dataset(raw_dir, seed=args.seed)
        results = args.output / "results"
        run_pipeline_from_raw(raw_dir, results, seed=args.seed, posterior_draws=args.draws, particles=args.particles, cells=args.cells, workers=args.workers)
        layout = ResultLayout(results)
        validate_method_contracts(layout.observation, layout.lite, layout.full_smc, layout.final_report)
    else:
        raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
