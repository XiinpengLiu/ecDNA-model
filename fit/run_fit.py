"""Monolithic CLI for the config-centered local ABC-SMC fit.

Usage:
    python -m fit.run_fit --conditions all --output results/fit_local_abc
    python -m fit.run_fit --conditions ctrl,P10 --generations 4 --n-per-generation 50 --accepted 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

from fit.engine import FitConfig, run_local_abc_fit  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "results" / "fit_local_abc"
DEFAULT_RAW_DIR = ROOT / "raw" / "t87_drug_bulkfit"


def _parse_conditions(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return tuple(cfg.T87_CONDITION_TREATMENTS.keys())
    values = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    unknown = [c for c in values if c not in cfg.T87_CONDITION_TREATMENTS]
    cfg.require(not unknown, f"Unsupported condition(s): {unknown}")
    cfg.require(bool(values), "At least one condition is required.")
    return tuple(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    defaults = cfg.DEFAULT_MODEL_PARAMETERS.simulation
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing ddpcr.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", default="all", help="Comma-separated T87 conditions, or 'all'.")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--n-per-generation", type=int, default=50)
    parser.add_argument("--accepted", type=int, default=20)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--n-init", type=int, default=defaults.n_init)
    parser.add_argument("--rows-per-state", type=int, default=512)
    parser.add_argument(
        "--target-population-size",
        type=int,
        default=defaults.target_population_size if defaults.target_population_size is not None else 0,
        help="Use 0 to disable target-population stopping (free-running growth).",
    )
    parser.add_argument("--max-pop-size", type=int, default=defaults.max_pop_size)
    parser.add_argument("--verbose", action="store_true", help="Verbose simulator output.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    conditions = _parse_conditions(str(args.conditions))
    target_pop = None if int(args.target_population_size) <= 0 else int(args.target_population_size)
    verbose = bool(args.verbose) and not bool(args.quiet)

    config = FitConfig(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        conditions=conditions,
        generations=int(args.generations),
        n_per_generation=int(args.n_per_generation),
        accepted_count=int(args.accepted),
        seed=int(args.seed),
        n_init=int(args.n_init),
        rows_per_state=int(args.rows_per_state),
        target_population_size=target_pop,
        max_pop_size=int(args.max_pop_size),
        verbose=verbose,
    )
    run_local_abc_fit(config)


if __name__ == "__main__":
    main()
