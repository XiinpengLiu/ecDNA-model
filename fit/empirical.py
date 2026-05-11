"""Bulk empirical summaries retained for compatibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit.io_utils import ensure_dir, write_markdown_report, write_table
from fit.raw import load_clean_tables


def build_empirical_summaries(
    clean_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    del obs_params_path
    tables = load_clean_tables(clean_dir)
    out = ensure_dir(output_dir)
    dd = tables["ddpcr"].copy()
    cc = tables["cell_count"].copy()
    flow = tables["flow3"].copy()
    paths = {
        "ddpcr_bulk_anchor_summary": out / "ddpcr_bulk_anchor_summary.parquet",
        "cell_count_summary": out / "cell_count_summary.parquet",
        "flow3_fraction_summary": out / "flow3_fraction_summary.parquet",
        "report": out / "empirical_summary_report.md",
    }
    write_table(dd, paths["ddpcr_bulk_anchor_summary"])
    write_table(cc, paths["cell_count_summary"])
    write_table(flow, paths["flow3_fraction_summary"])
    write_markdown_report(
        paths["report"],
        "Bulk Empirical Summary Report",
        [
            ("Scope", "Summaries are limited to ddPCR bulk means, cell counts, and flow3 fractions."),
            ("Closed Modalities", "qPCDR and ecTAG are not summarized as fit likelihoods."),
        ],
    )
    return paths


def _flow_summary(flow3: pd.DataFrame) -> pd.DataFrame:
    return flow3.copy()


def _ddpcr_summary(ddpcr: pd.DataFrame) -> pd.DataFrame:
    return ddpcr.copy()


def _cell_count_summary(cell_count: pd.DataFrame) -> pd.DataFrame:
    return cell_count.copy()
