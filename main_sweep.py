"""
CLI entry point for the external untreated ecDNA v4 screening sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import v4_config as cfg
from v4_sweep import ScreeningEngine, ScreeningExecutionPlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the external ecDNA v4 screening sweep.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("results_v4") / "sweep_outputs"),
        help="Directory for tables, plots, and reports.",
    )
    parser.add_argument("--protocol-name", default="untreated", help="Screening protocol name.")
    parser.add_argument("--t-max", type=float, default=72.0, help="Simulation end time for screening mode.")
    parser.add_argument("--record-interval", type=float, default=1.0, help="Record interval for screening mode.")
    parser.add_argument("--n-init", type=int, default=80, help="Initial population size for screening mode.")
    parser.add_argument(
        "--target-population-size",
        type=int,
        default=500,
        help="Target population size for screening mode. Use 0 to disable target-based stopping.",
    )
    parser.add_argument("--max-pop-size", type=int, default=2000, help="Maximum population size for screening mode.")
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106",
        help="Comma-separated seeds for screening mode.",
    )
    parser.add_argument("--oat-points", type=int, default=5, help="Number of OAT points per parameter.")
    parser.add_argument("--top-per-category", type=int, default=4, help="Top ranked parameters per category.")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size per axis for two-parameter maps.")
    parser.add_argument("--top-pairs-per-group", type=int, default=2, help="Top parameter pairs per group.")
    parser.add_argument(
        "--representative-top",
        type=int,
        default=4,
        help="Number of representative top parameters for response and trajectory plots.",
    )
    return parser


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in raw.split(",")]
    cfg.require(all(tokens), "Seed list must not contain empty entries.")
    seeds = tuple(int(token) for token in tokens)
    cfg.require(bool(seeds), "At least one screening seed is required.")
    return seeds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    target_population_size = None if int(args.target_population_size) == 0 else int(args.target_population_size)
    plan = ScreeningExecutionPlan(
        protocol_name=args.protocol_name,
        t_max=float(args.t_max),
        record_interval=float(args.record_interval),
        n_init=int(args.n_init),
        target_population_size=target_population_size,
        max_pop_size=int(args.max_pop_size),
        baseline_seeds=_parse_seed_list(args.seeds),
        oat_points_per_parameter=int(args.oat_points),
        ranking_top_parameters_per_category=int(args.top_per_category),
        two_param_grid_size=int(args.grid_size),
        top_pairs_per_group=int(args.top_pairs_per_group),
        representative_top_parameters=int(args.representative_top),
    )
    engine = ScreeningEngine(output_dir=args.output_dir, plan=plan)
    phases = engine.run()
    print(f"Screening completed. Outputs written to: {Path(args.output_dir).resolve()}")
    print(f"Completed phases: {', '.join(phases.keys())}")


if __name__ == "__main__":
    main()
