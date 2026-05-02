rule ingest_raw:
    input:
        manifest=f"{RESULTS}/00_manifest/run_manifest.json",
        index=f"{RESULTS}/00_manifest/analysis_index.parquet"
    output:
        flow=f"{RESULTS}/01_clean_data/flow_long.parquet",
        qpcdr=f"{RESULTS}/01_clean_data/qpcdr_long.parquet",
        ectag=f"{RESULTS}/01_clean_data/ectag_cell_long.parquet",
        ddpcr=f"{RESULTS}/01_clean_data/ddpcr_long.parquet",
        cell_count=f"{RESULTS}/01_clean_data/cell_count_long.parquet",
        report=f"{RESULTS}/01_clean_data/raw_data_qc_report.md"
    shell:
        "python -m fit.run_fit ingest-raw --raw-dir {config[raw_dir]} --output {RESULTS}/01_clean_data"
