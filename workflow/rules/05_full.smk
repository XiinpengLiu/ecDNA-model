rule create_full_initial_particles:
    input:
        sampler=f"{RESULTS}/03_v4_lite/LITE_initial_population_sampler.json"
    output:
        particles=f"{RESULTS}/04_full_initialization/initial_particles.parquet",
        manifest=f"{RESULTS}/04_full_initialization/initial_particles_manifest.json"
    shell:
        "python -m fit.run_fit create-full-initial-particles --lite-dir {RESULTS}/03_v4_lite --output {RESULTS}/04_full_initialization --particles {config[particles]} --cells {config[cells]} --seed {config[seed]}"

rule run_full_reconstruction:
    input:
        init=f"{RESULTS}/04_full_initialization/initial_particles.parquet",
        obs=f"{RESULTS}/02_observation_model/obs_params_for_full.json",
        target=f"{RESULTS}/03_v4_lite/LITE_summary_target_vector.parquet",
        covariance=f"{RESULTS}/03_v4_lite/LITE_summary_covariance.npz",
        weights=f"{RESULTS}/03_v4_lite/LITE_distance_weights.json"
    output:
        histories=f"{RESULTS}/05_full_smc/accepted_histories.jsonl",
        parameters=f"{RESULTS}/05_full_smc/particle_parameters.parquet",
        weights=f"{RESULTS}/05_full_smc/particle_weights.parquet",
        snapshots=f"{RESULTS}/05_full_smc/full_snapshot_summaries.parquet",
        events=f"{RESULTS}/05_full_smc/event_summaries.parquet",
        scenarios=f"{RESULTS}/05_full_smc/scenario_classes.parquet",
        zarr=f"{RESULTS}/05_full_smc/FULL_particles_final.zarr",
        full_parameters=f"{RESULTS}/05_full_smc/FULL_particle_parameters.parquet",
        full_weights=f"{RESULTS}/05_full_smc/FULL_particle_weights.parquet",
        full_snapshots=f"{RESULTS}/05_full_smc/FULL_snapshot_summaries.parquet",
        full_events=f"{RESULTS}/05_full_smc/FULL_event_summaries.parquet",
        full_histories=f"{RESULTS}/05_full_smc/FULL_single_cell_history_samples.parquet",
        full_derived=f"{RESULTS}/05_full_smc/FULL_derived_Q.parquet",
        ppc=f"{RESULTS}/05_full_smc/full_ppc_report.json",
        raw_ppc=f"{RESULTS}/05_full_smc/raw_table_ppc_report.json",
        synthetic_flow=f"{RESULTS}/05_full_smc/synthetic_flow_long.parquet",
        synthetic_qpcdr=f"{RESULTS}/05_full_smc/synthetic_qpcdr_long.parquet",
        synthetic_ectag=f"{RESULTS}/05_full_smc/synthetic_ectag_cell_long.parquet",
        synthetic_ddpcr=f"{RESULTS}/05_full_smc/synthetic_ddpcr_long.parquet",
        synthetic_cell_count=f"{RESULTS}/05_full_smc/synthetic_cell_count_long.parquet"
    shell:
        "python -m fit.run_fit run-full-reconstruction --lite-dir {RESULTS}/03_v4_lite --obs-params {input.obs} --output {RESULTS}/05_full_smc --particles {config[particles]} --cells {config[cells]} --smc-steps {config[smc_steps]} --seed {config[seed]}"

rule run_full_exact_replay:
    input:
        histories=f"{RESULTS}/05_full_smc/FULL_single_cell_history_samples.parquet",
        parameters=f"{RESULTS}/05_full_smc/FULL_particle_parameters.parquet",
        weights=f"{RESULTS}/05_full_smc/FULL_particle_weights.parquet",
        obs=f"{RESULTS}/02_observation_model/obs_params_for_full.json",
        target=f"{RESULTS}/03_v4_lite/LITE_summary_target_vector.parquet"
    output:
        histories=f"{RESULTS}/05_full_smc/FULL_exact_replay_histories.parquet",
        snapshots=f"{RESULTS}/05_full_smc/FULL_exact_replay_snapshot_summaries.parquet",
        event_log=f"{RESULTS}/05_full_smc/FULL_exact_replay_event_log.parquet",
        events=f"{RESULTS}/05_full_smc/FULL_exact_replay_event_summaries.parquet",
        scores=f"{RESULTS}/05_full_smc/FULL_exact_replay_scores.parquet",
        weights=f"{RESULTS}/05_full_smc/FULL_exact_replay_particle_weights.parquet",
        report=f"{RESULTS}/05_full_smc/FULL_exact_replay_report.md",
        manifest=f"{RESULTS}/05_full_smc/FULL_exact_replay_manifest.json"
    shell:
        "python -m fit.run_fit run-exact-replay --full-dir {RESULTS}/05_full_smc --lite-dir {RESULTS}/03_v4_lite --obs-params {input.obs} --output {RESULTS}/05_full_smc --seed {config[seed]}"
