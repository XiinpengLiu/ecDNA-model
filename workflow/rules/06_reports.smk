rule build_final_report:
    input:
        obs=f"{RESULTS}/02_observation_model/obs_params_for_full.json",
        lite=f"{RESULTS}/03_v4_lite/LITE_summary_target_vector.parquet",
        full=f"{RESULTS}/05_full_smc/FULL_particles_final.zarr",
        scenarios=f"{RESULTS}/05_full_smc/scenario_classes.parquet",
        exact_events=f"{RESULTS}/05_full_smc/FULL_exact_replay_event_summaries.parquet"
    output:
        raw_ppc=f"{RESULTS}/08_final_report/FINAL_raw_ppc_report.pdf",
        histories=f"{RESULTS}/08_final_report/FINAL_single_cell_histories.zarr",
        events=f"{RESULTS}/08_final_report/FINAL_event_history_summary.parquet",
        scenarios=f"{RESULTS}/08_final_report/FINAL_scenario_summary.pdf",
        appendix=f"{RESULTS}/08_final_report/FINAL_parameter_appendix.csv",
        manifest=f"{RESULTS}/08_final_report/FINAL_report_manifest.json"
    shell:
        "python -m fit.run_fit build-final-report --observation-dir {RESULTS}/02_observation_model --lite-dir {RESULTS}/03_v4_lite --full-dir {RESULTS}/05_full_smc --output {RESULTS}/08_final_report"

rule materialize_method_layout:
    input:
        final=f"{RESULTS}/08_final_report/FINAL_report_manifest.json"
    output:
        lite=f"{RESULTS}/04_v4_lite/LITE_summary_target_vector.parquet",
        init=f"{RESULTS}/05_full_initialization/initial_particles.parquet",
        full=f"{RESULTS}/06_full_history_reconstruction/FULL_particles_final.zarr"
    shell:
        "python -m fit.run_fit materialize-method-layout --output-root {RESULTS}"
