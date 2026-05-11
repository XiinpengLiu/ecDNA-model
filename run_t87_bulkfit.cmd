@echo off
snakemake --snakefile workflow/Snakefile --configfile configs/t87_drug_bulkfit.yaml --cores 8
