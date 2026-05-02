rule build_manifest:
    input:
        raw_dir=config["raw_dir"]
    output:
        manifest=f"{RESULTS}/00_manifest/run_manifest.json",
        index=f"{RESULTS}/00_manifest/analysis_index.parquet",
        report=f"{RESULTS}/00_manifest/schema_check_report.md"
    shell:
        "python -m fit.run_fit build-manifest --raw-dir {input.raw_dir} --output {RESULTS}/00_manifest"
