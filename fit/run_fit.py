"""Command line entry point for v4-lite fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config as cfg
from fit.data import CanonicalFitDataset, ConditionSpec, CountRecord, EcTAGRecord, FlowRecord, QPCDRRecord
from fit.full_calibration import FullCalibrationRunner, FullCalibrationSettings, write_full_calibration_reports
from fit.v4_lite import V4LiteFitRunner, V4LiteOptimizationSettings


def _synthetic_dataset() -> CanonicalFitDataset:
    conditions = {"ctrl": ConditionSpec("ctrl")}
    flow = []
    for week, fractions in ((1, (0.4, 0.3, 0.2, 0.1)), (2, (0.35, 0.35, 0.2, 0.1)), (3, (0.3, 0.38, 0.22, 0.1))):
        for state, fraction in zip(cfg.STATE_NAMES, fractions):
            flow.append(FlowRecord("ctrl", week, state, int(round(1000 * fraction)), fraction, 1000, "r1"))
    counts = (CountRecord("ctrl", 2, 1100.0, "r1"), CountRecord("ctrl", 3, 1200.0, "r1"))
    qpcdr = tuple(QPCDRRecord("ctrl", week, state, species, 2.0 + 0.1 * week, "r1") for week in (2, 3) for state in cfg.STATE_NAMES for species in cfg.SPECIES)
    ectag = tuple(EcTAGRecord("ctrl", week, state, species, f"{species}-cell{i}", i % 6, "r1") for week in (2, 3) for state in cfg.STATE_NAMES for species in cfg.SPECIES for i in range(8))
    week1 = {
        "ctrl": {
            state: np.asarray([[1 + idx, 2, 3], [2 + idx, 3, 4], [3 + idx, 4, 5], [4 + idx, 5, 6]], dtype=int)
            for idx, state in enumerate(cfg.STATE_NAMES)
        }
    }
    return CanonicalFitDataset(conditions=conditions, flow=tuple(flow), counts=counts, qpcdr=qpcdr, ectag=ectag, week1_copy_distributions=week1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ecDNA v4-lite fitting pipeline.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fit_outputs/smoke"))
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--full-bridge", action="store_true", help="Also run the full-simulator bridge against the v4-lite projection.")
    parser.add_argument("--full-formal", action="store_true", help="Run restricted full raw-summary refinement after the bridge.")
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--posterior-backend", choices=("auto", "emcee", "laplace"), default="auto")
    parser.add_argument("--emcee-walkers", type=int, default=0)
    parser.add_argument("--emcee-steps", type=int, default=0)
    parser.add_argument("--emcee-burnin", type=int, default=16)
    parser.add_argument("--synthetic-recovery-datasets", type=int, default=50)
    parser.add_argument("--sbc-datasets", type=int, default=12)
    parser.add_argument("--full-max-pop-size", type=int, default=200000)
    parser.add_argument("--full-n-init", type=int)
    parser.add_argument("--full-smc-particles", type=int, default=32)
    parser.add_argument("--full-smc-steps", type=int, default=2)
    args = parser.parse_args()
    if args.synthetic_smoke:
        dataset = _synthetic_dataset()
    elif args.manifest is not None:
        dataset = CanonicalFitDataset.from_manifest(args.manifest)
    else:
        raise SystemExit("Provide --manifest or --synthetic-smoke.")
    runner = V4LiteFitRunner(
        dataset,
        output_dir=args.output,
        optimization_settings=V4LiteOptimizationSettings(
            maxiter=args.maxiter,
            posterior_draws=16,
            posterior_backend=args.posterior_backend,
            emcee_walkers=args.emcee_walkers,
            emcee_steps=args.emcee_steps,
            emcee_burnin=args.emcee_burnin,
            synthetic_recovery_datasets=args.synthetic_recovery_datasets,
            sbc_datasets=args.sbc_datasets,
        ),
    )
    result = runner.run_all()
    if args.full_bridge or args.full_formal:
        if result.projection_targets is None:
            raise SystemExit("v4-lite projection target was not produced.")
        full_runner = FullCalibrationRunner(
            dataset,
            result.projection_targets,
            structure=result.tensor.structure,
            settings=FullCalibrationSettings(
                maxiter=max(1, min(args.maxiter, 4)),
                formal_maxiter=max(1, min(args.maxiter, 2)),
                run_formal_raw_refinement=bool(args.full_formal),
                max_pop_size=args.full_max_pop_size,
                n_init=args.full_n_init,
                smc_particles=args.full_smc_particles,
                smc_steps=args.full_smc_steps,
            ),
        )
        full_result = full_runner.run_all_stages()
        full_output = args.output / "full_bridge"
        write_full_calibration_reports(full_output, full_result)
        print(f"[fit] full bridge reports written to {full_output}")


if __name__ == "__main__":
    main()
