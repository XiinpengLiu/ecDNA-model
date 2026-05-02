rule fit_lite:
    input:
        snapshot=f"{RESULTS}/03_empirical_summary/snapshot_summary.parquet",
        hist=f"{RESULTS}/03_empirical_summary/ectag_histograms_species_specific.parquet",
        ddpcr=f"{RESULTS}/03_empirical_summary/ddpcr_bulk_anchor_summary.parquet",
        cell_count=f"{RESULTS}/03_empirical_summary/cell_count_summary.parquet",
        obs=f"{RESULTS}/02_observation_model/obs_params_for_lite.json"
    output:
        fit=f"{RESULTS}/03_v4_lite/LITE_final_fit.nc",
        posterior=f"{RESULTS}/03_v4_lite/LITE_snapshot_posterior.parquet",
        target=f"{RESULTS}/03_v4_lite/LITE_summary_target_vector.parquet",
        covariance=f"{RESULTS}/03_v4_lite/LITE_summary_covariance.npz",
        weights=f"{RESULTS}/03_v4_lite/LITE_distance_weights.json",
        sampler=f"{RESULTS}/03_v4_lite/LITE_initial_population_sampler.json",
        priors=f"{RESULTS}/03_v4_lite/LITE_to_FULL_prior_scales.json",
        transition=f"{RESULTS}/03_v4_lite/LITE_transition_growth_summary.parquet",
        coupling=f"{RESULTS}/03_v4_lite/LITE_coupling_summary.csv",
        report=f"{RESULTS}/03_v4_lite/LITE_final_report.pdf"
    shell:
        "python -m fit.run_fit fit-lite --empirical-dir {RESULTS}/03_empirical_summary --obs-params {input.obs} --output {RESULTS}/03_v4_lite --seed {config[seed]} --draws {config[posterior_draws]}"
