# Fit Implementation Status

Current status: aligned to `markdown/fit_method.md` bulk-only workflow for the Python CLI and Snakemake entrypoint.

Implemented DAG:

1. `00_manifest`
2. `01_clean_data`
3. `02_observation_model`
4. `03_v4_lite_bulk`
5. `04_parameter_registry`
6. `05_prior_predictive`
7. `06_moment_prescreen`
8. `07_full_initialization`
9. `08_full_smc`
10. `09_validation`
11. `10_final_report`

Opened likelihoods: bulk ddPCR, cell count, flow3 steady projection.

Closed likelihoods: qPCDR, ecTAG, flow4, state-specific copy, zero/tail single-cell summaries, and old lite-summary double counting.

Active fitted controls: net growth rate, bulk copy velocity, and flow3 projection bias.

Nuisance/prior-only controls: division/death turnover, gain/loss turnover, hidden NPC/OPC split, state-specific copy enrichment, co-segregation, and single-cell copy distribution shape.

Validation command:

```text
python -m pytest tests\test_v4_lite.py -q
```

Additional enforced gates:

- Prior predictive accepted fraction must be at least 1%; if not, active-control bounds are relaxed once by 20%, nuisance bounds remain fixed, and persistent failure writes `PRIOR_region_incompatible_report.md` and stops.
- Full SMC acceptance requires finite biological hard-bound distance, acceptable prior distance, bounded active controls including flow3 projection bias, and no fallback particle if the gates fail.
- Full SMC/replay artifacts are generated through `core.simulation.run_simulation`; the fit layer maps only bulk-visible effective controls and prior-constrained nuisance values.
- Initial copy-number distributions use mean-matched ZINB priors; ddPCR remains only the bulk mean anchor, not a single-cell distribution.
- Monte Carlo noise estimates feed the next SMC round's representative cell count and are recorded per round.
- Holdout validation uses accepted full-parameter ensemble predictions with train-week-only log-offset calibration for held-out weeks.
- Final particle weights are normalized only over accepted final-round particles; non-final and rejected rows carry zero posterior weight.
- Historical `workflow/rules/*.smk` files were removed because `workflow/Snakefile` now owns the method-aligned 00-10 DAG.
- Final compatibility reporting writes `FULL_bulkfit_incompatible_under_biological_priors.md` when PPC, active boundary mass, ESS, prior penalty, or biology gates fail.
- `run_pipeline_from_raw` and `python -m fit.run_fit run-all --raw-dir ... --output ...` run deterministically from raw tables through `10_final_report`.
