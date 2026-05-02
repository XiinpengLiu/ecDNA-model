# Fit Implementation Status

Source of truth: `markdown/fit_method.md`.

## Status Summary

All non-optional requirements in `fit/implementation_checklist.json` are implemented.

The refactored `fit/` package follows this execution contract:

1. Build a run manifest and locked analysis index.
2. Ingest raw flow, qPCDR, ecTAG, ddPCR, and cell-count tables into clean file-based artifacts.
3. Fit and lock observation calibration before full reconstruction.
4. Build empirical snapshot summaries with species-specific ecTAG histograms.
5. Generate a v4-lite calibrated summary posterior and lite-to-full bridge artifacts.
6. Run full conditional single-cell SMC-ABC style history reconstruction.
7. Score particles with flow, qPCDR, ecTAG, ddPCR, lite-summary, prior, and biology terms.
8. Save weighted accepted history ensembles, event summaries, particle weights, and scenario classes.
9. Run posterior predictive checks from the weighted history ensemble.
10. Build the `08_final_report` layer and mirror outputs into the directory names used in `fit_method.md`.

## Requirement Mapping

| Core requirement | Implementation | Tests | Status |
| --- | --- | --- | --- |
| Step 0 manifest and schema lock | `fit/manifest.py`, `fit/stage_runner.py`, `fit/run_fit.py` | CLI smoke | Complete |
| Raw and clean data ingestion | `fit/raw.py`, `fit/schemas.py` | schema validation, CLI smoke | Complete |
| Observation calibration fixed before full | `fit/observation.py`, `fit/validation.py` | lite schema, CLI smoke | Complete |
| ddPCR pooled mean only | `calculate_ddpcr_pooled_mean`, `fit/objective.py`, `fit/full_smc.py` | ddPCR formula, full score test | Complete |
| Species-specific ecTAG histograms | `fit/empirical.py`, `fit/v4_lite.py`, `fit/objective.py` | ecTAG species-specific test | Complete |
| Total burden derived only | `snapshot_summary.derived_total_burden`, `FULL_derived_Q.parquet` | ecTAG species-specific test | Complete |
| v4-lite summary posterior | `fit/v4_lite.py` | lite artifact schema test | Complete |
| transition/growth and coupling summaries | `fit/v4_lite.py` | lite artifact schema test | Complete |
| Lite-to-full bridge artifacts | `LITE_*` bridge files, `create_full_initial_particles` | lite artifact schema test | Complete |
| Full conditional history ensemble | `fit/full_smc.py`, real zarr ZipStore artifact | full ensemble test | Complete |
| Full score contribution traceability | `fit/objective.py`, `particle_weights.parquet` | full score contribution test | Complete |
| SMC-ABC rounds, resample, perturb, tolerance | `run_full_reconstruction(..., smc_steps=...)` | full score and CLI smoke tests | Complete |
| Scenario classification | `fit/scenarios.py` | scenario classification test | Complete |
| Posterior predictive checks | `fit/ppc.py`, `full_ppc_report.*`, `FULL_ppc_raw_observables.parquet` | CLI smoke | Complete |
| Workflow-callable CLI stages | `fit/run_fit.py` | CLI smoke | Complete |
| Synthetic small smoke pipeline | `run-synthetic-smoke` | CLI smoke | Complete |
| Snakemake workflow | `workflow/Snakefile`, `workflow/rules/*.smk` | Snakemake dry-run | Complete |
| Method directory names | `materialize_method_layout` | CLI smoke | Complete |
| Final report layer | `fit/final_report.py` | CLI smoke and final validation | Complete |
| Species copula and tail initialization | `fit/v4_lite.py`, `fit/full_smc.py` | lite schema and full ensemble tests | Complete |
| Softplus R/V expression | `fit/full_smc.py`, `fit/schemas.py` | full ensemble test | Complete |

## Output Contract

Manifest outputs:

- `results/00_manifest/run_manifest.json`
- `results/00_manifest/analysis_index.parquet`
- `results/00_manifest/schema_check_report.md`

Observation outputs:

- `results/02_observation_model/obs_params_for_lite.json`
- `results/02_observation_model/obs_params_for_full.json`
- `results/02_observation_model/obs_calibration_fit.nc`
- `results/02_observation_model/obs_calibration_ppc.pdf`
- `results/02_observation_model/obs_calibration_report.md`
- `results/02_observation_model/obs_calibration_report.json`

Lite outputs:

- `results/03_v4_lite/LITE_final_fit.nc`
- `results/03_v4_lite/LITE_final_fit.json`
- `results/03_v4_lite/LITE_snapshot_posterior.parquet`
- `results/03_v4_lite/LITE_transition_growth_summary.parquet`
- `results/03_v4_lite/LITE_coupling_summary.csv`
- `results/03_v4_lite/LITE_summary_target_vector.parquet`
- `results/03_v4_lite/LITE_summary_covariance.npz`
- `results/03_v4_lite/LITE_distance_weights.json`
- `results/03_v4_lite/LITE_initial_population_sampler.json`
- `results/03_v4_lite/LITE_to_FULL_prior_scales.json`
- `results/03_v4_lite/LITE_final_report.md`
- `results/03_v4_lite/LITE_final_report.pdf`

Full reconstruction outputs:

- `results/05_full_smc/accepted_histories.jsonl`
- `results/05_full_smc/particle_parameters.parquet`
- `results/05_full_smc/particle_weights.parquet`
- `results/05_full_smc/full_snapshot_summaries.parquet`
- `results/05_full_smc/event_summaries.parquet`
- `results/05_full_smc/scenario_classes.parquet`
- `results/05_full_smc/full_ppc_report.md`
- `results/05_full_smc/full_ppc_report.json`
- `results/05_full_smc/FULL_particles_final.zarr`
- `results/05_full_smc/FULL_particle_parameters.parquet`
- `results/05_full_smc/FULL_particle_weights.parquet`
- `results/05_full_smc/FULL_snapshot_summaries.parquet`
- `results/05_full_smc/FULL_event_summaries.parquet`
- `results/05_full_smc/FULL_single_cell_history_samples.parquet`
- `results/05_full_smc/FULL_derived_Q.parquet`
- `results/05_full_smc/FULL_ppc_raw_observables.parquet`
- `results/05_full_smc/FULL_history_reconstruction_report.pdf`

Method-layout mirrors:

- `results/04_v4_lite/*`
- `results/05_full_initialization/*`
- `results/06_full_history_reconstruction/*`

Final report outputs:

- `results/08_final_report/FINAL_raw_ppc_report.pdf`
- `results/08_final_report/FINAL_single_cell_histories.zarr`
- `results/08_final_report/FINAL_event_history_summary.parquet`
- `results/08_final_report/FINAL_scenario_summary.pdf`
- `results/08_final_report/FINAL_scenario_summary.parquet`
- `results/08_final_report/FINAL_parameter_appendix.csv`
- `results/08_final_report/FULL_scenario_classes.parquet`
- `results/08_final_report/FINAL_report_manifest.json`

## Method Guards

- ddPCR is represented only as a bulk pooled mean anchor, `sum_s f[w,c,r,s] * mu[w,c,r,s,j]`.
- ecTAG primary histogram features are keyed by species; MYC, CDK4, and PDGFRA are never summed as the primary ecTAG likelihood.
- Total ecDNA burden is stored only as a derived snapshot or full-history summary.
- Observation calibration artifacts are locked for full reconstruction.
- Full parameters are reported as latent control variables; the primary output is a weighted ensemble of possible single-cell histories and scenario classes.
- Full scoring has explicit flow, qPCDR, ecTAG, ddPCR, lite-summary, prior, and biology contribution columns.
- Full reconstruction saves histories, event summaries, particle parameters, scores, weights, scenario labels, PPC artifacts, and a real zarr ZipStore history ensemble.
- Full initialization uses stratum-specific state probabilities, empirical same-cell species correlation via a Gaussian copula when available, and calibrated open-tail copy means.
- `R` and `V` are stored as non-negative softplus-style expressions from latent raw values.

## Dependencies

The fit pipeline intentionally uses standard scientific packages for general numerical/statistical work:

- `numpy`, `pandas`, `pyarrow`
- `scipy` for Mahalanobis distance, Dirichlet-multinomial scoring, KL entropy, and Wasserstein distance
- `xarray` for NetCDF artifacts
- `zarr` for the full accepted-history ensemble
- `matplotlib` for deterministic PDF reports
- `snakemake` for workflow DAG execution

These dependencies are declared in `requirements-fit.txt`.

## Remaining Items

No non-optional method requirement is blocked.

The repository currently keeps only method-aligned `fit/` source files: shared schemas/IO, raw ingestion, manifest, observation calibration, empirical summaries, v4-lite, full SMC, scoring, PPC, scenario classification, validation, CLI, stage runner, and implementation status artifacts.
