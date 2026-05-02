rule build_empirical:
    input:
        obs=f"{RESULTS}/02_observation_model/obs_params_for_lite.json",
        flow=f"{RESULTS}/01_clean_data/flow_long.parquet",
        qpcdr=f"{RESULTS}/01_clean_data/qpcdr_long.parquet",
        ectag=f"{RESULTS}/01_clean_data/ectag_cell_long.parquet",
        ddpcr=f"{RESULTS}/01_clean_data/ddpcr_long.parquet",
        cell_count=f"{RESULTS}/01_clean_data/cell_count_long.parquet"
    output:
        snapshot=f"{RESULTS}/03_empirical_summary/snapshot_summary.parquet",
        hist=f"{RESULTS}/03_empirical_summary/ectag_histograms_species_specific.parquet",
        joint=f"{RESULTS}/03_empirical_summary/ectag_joint_species_summary.parquet",
        ddpcr=f"{RESULTS}/03_empirical_summary/ddpcr_bulk_anchor_summary.parquet",
        cell_count=f"{RESULTS}/03_empirical_summary/cell_count_summary.parquet",
        plots=f"{RESULTS}/03_empirical_summary/empirical_summary_plots.pdf"
    shell:
        "python -m fit.run_fit build-empirical-summaries --clean-dir {RESULTS}/01_clean_data --obs-params {input.obs} --output {RESULTS}/03_empirical_summary"
