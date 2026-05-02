rule fit_observation:
    input:
        flow=f"{RESULTS}/01_clean_data/flow_long.parquet",
        qpcdr=f"{RESULTS}/01_clean_data/qpcdr_long.parquet",
        ectag=f"{RESULTS}/01_clean_data/ectag_cell_long.parquet",
        ddpcr=f"{RESULTS}/01_clean_data/ddpcr_long.parquet",
        cell_count=f"{RESULTS}/01_clean_data/cell_count_long.parquet"
    output:
        lite=f"{RESULTS}/02_observation_model/obs_params_for_lite.json",
        full=f"{RESULTS}/02_observation_model/obs_params_for_full.json",
        fit=f"{RESULTS}/02_observation_model/obs_calibration_fit.nc",
        ppc=f"{RESULTS}/02_observation_model/obs_calibration_ppc.pdf",
        report=f"{RESULTS}/02_observation_model/obs_calibration_report.md"
    shell:
        "python -m fit.run_fit fit-observation-model --clean-dir {RESULTS}/01_clean_data --output {RESULTS}/02_observation_model --seed {config[seed]}"
