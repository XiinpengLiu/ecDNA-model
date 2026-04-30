# Fit Implementation Status

Source of truth: `markdown/fit_method.md`.

## Status Summary

All non-optional core requirements in `fit/implementation_checklist.json` are implemented.

The refactored `fit/` package now follows this execution contract:

1. Ingest raw flow, qPCDR, ecTAG, ddPCR, and cell-count tables into clean file-based artifacts.
2. Fit and lock observation calibration before any full reconstruction.
3. Build empirical snapshot summaries with species-specific ecTAG histograms.
4. Generate v4-lite calibrated summary posterior artifacts.
5. Bridge lite artifacts to full initial particle populations and broad prior scales.
6. Run full conditional single-cell history particle reconstruction.
7. Score particles with flow, qPCDR, ecTAG, ddPCR, lite-summary, prior, and biology terms.
8. Save weighted accepted history ensembles and scenario classes.
9. Run posterior predictive checks from the weighted history ensemble.

## Requirement Mapping

| Core requirement | Implementation | Tests | Status |
| --- | --- | --- | --- |
| Raw and clean data ingestion | `fit/raw.py` | `test_schema_validation_identifies_missing_required_fields`, CLI smoke | Complete |
| Observation calibration fixed before full | `fit/observation.py`, `fit/validation.py` | lite schema and CLI smoke tests | Complete |
| ddPCR pooled mean only | `calculate_ddpcr_pooled_mean`, `fit/full_smc.py` ddPCR scoring | `test_ddpcr_pooled_mean_formula`, full score test | Complete |
| Species-specific ecTAG histograms | `fit/empirical.py`, `fit/v4_lite.py` | `test_ectag_histogram_is_species_specific` | Complete |
| Total burden derived only | `snapshot_summary.derived_total_burden` | ecTAG species-specific test | Complete |
| v4-lite summary posterior, not raw-data simulation | `fit/v4_lite.py` | lite artifact schema test | Complete |
| Lite-to-full bridge artifacts | `LITE_initial_population_sampler.json`, `LITE_summary_target_vector.parquet`, `LITE_summary_covariance.npz`, `LITE_distance_weights.json`, `LITE_to_FULL_prior_scales.json` | lite artifact schema test | Complete |
| Full conditional history ensemble | `fit/full_smc.py` | full ensemble test | Complete |
| Full score contribution traceability | `fit/objective.py`, `particle_weights.parquet` | full score contribution test | Complete |
| Scenario classification | `fit/scenarios.py` | scenario classification test | Complete |
| Posterior predictive checks | `fit/ppc.py`, `full_ppc_report.*` | CLI smoke test | Complete |
| Workflow-callable CLI stages | `fit/run_fit.py` | CLI smoke test | Complete |
| Synthetic small smoke pipeline | `run-synthetic-smoke` | CLI smoke test | Complete |

## Output Contract

Observation outputs:

- `results/02_observation_model/obs_params_for_lite.json`
- `results/02_observation_model/obs_params_for_full.json`
- `results/02_observation_model/obs_calibration_report.md`
- `results/02_observation_model/obs_calibration_report.json`

Lite outputs:

- `results/03_v4_lite/LITE_final_fit.json`
- `results/03_v4_lite/LITE_snapshot_posterior.parquet`
- `results/03_v4_lite/LITE_summary_target_vector.parquet`
- `results/03_v4_lite/LITE_summary_covariance.npz`
- `results/03_v4_lite/LITE_distance_weights.json`
- `results/03_v4_lite/LITE_initial_population_sampler.json`
- `results/03_v4_lite/LITE_to_FULL_prior_scales.json`
- `results/03_v4_lite/LITE_final_report.md`

Full reconstruction outputs:

- `results/05_full_smc/accepted_histories.jsonl`
- `results/05_full_smc/particle_parameters.parquet`
- `results/05_full_smc/particle_weights.parquet`
- `results/05_full_smc/full_snapshot_summaries.parquet`
- `results/05_full_smc/event_summaries.parquet`
- `results/05_full_smc/scenario_classes.parquet`
- `results/05_full_smc/full_ppc_report.md`
- `results/05_full_smc/full_ppc_report.json`

## Method Guards

- ddPCR is represented only as a bulk pooled mean anchor, `sum_s f[w,c,r,s] * mu[w,c,r,s,j]`.
- ecTAG primary histogram features are keyed by species; MYC, CDK4, and PDGFRA are never summed as the primary ecTAG likelihood.
- Total ecDNA burden is stored only as a derived snapshot summary.
- Observation calibration artifacts are locked for full reconstruction.
- Full parameters are reported as latent control variables; the primary output is a weighted ensemble of possible single-cell histories and scenario classes.
- The full reconstruction writes particle histories, particle parameters, event summaries, scores, weights, and scenario labels.

## Remaining Items

No non-optional method requirement is blocked. Report formats currently use `.md` and `.json` for text artifacts so they remain deterministic and lightweight in smoke runs; the method's wildcard report requirement is satisfied by these files.

Compatibility-only modules from the old fit layout were removed after the new pipeline stabilized. The remaining `fit/` source files are directly used by the new method, CLI, validation, tests, or required status artifacts.
