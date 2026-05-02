"""Empirical snapshot summary construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, write_markdown_report, write_table, write_text_pdf
from fit.observation import load_observation_params
from fit.raw import load_clean_tables


def build_empirical_summaries(
    clean_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
    min_ectag_cells_for_hist: int = 50,
) -> dict[str, Path]:
    tables = load_clean_tables(clean_dir)
    obs_params = load_observation_params(obs_params_path)
    bins = list(obs_params["ectag"]["bins"])
    out = ensure_dir(output_dir)

    flow_summary = _flow_summary(tables["flow"])
    ectag_hist = _ectag_histograms(tables["ectag"], bins, min_ectag_cells_for_hist)
    snapshot = _snapshot_summary(tables["ectag"], flow_summary, bins)
    qpcdr_summary = _qpcdr_summary(tables["qpcdr"])
    ddpcr_summary = _ddpcr_summary(tables["ddpcr"])
    cell_count_summary = _cell_count_summary(tables["cell_count"])
    joint = _joint_species_summary(tables["ectag"])

    paths = {
        "snapshot_summary": out / "snapshot_summary.parquet",
        "ectag_histograms_species_specific": out / "ectag_histograms_species_specific.parquet",
        "ectag_joint_species_summary": out / "ectag_joint_species_summary.parquet",
        "ddpcr_bulk_anchor_summary": out / "ddpcr_bulk_anchor_summary.parquet",
        "qpcdr_state_species_summary": out / "qpcdr_state_species_summary.parquet",
        "flow_fraction_summary": out / "flow_fraction_summary.parquet",
        "cell_count_summary": out / "cell_count_summary.parquet",
        "plots": out / "empirical_summary_plots.pdf",
        "report": out / "empirical_summary_report.md",
    }
    write_table(snapshot, paths["snapshot_summary"])
    write_table(ectag_hist, paths["ectag_histograms_species_specific"])
    write_table(joint, paths["ectag_joint_species_summary"])
    write_table(ddpcr_summary, paths["ddpcr_bulk_anchor_summary"])
    write_table(qpcdr_summary, paths["qpcdr_state_species_summary"])
    write_table(flow_summary, paths["flow_fraction_summary"])
    write_table(cell_count_summary, paths["cell_count_summary"])
    write_text_pdf(
        paths["plots"],
        "Empirical Summary Plots",
        [
            "Snapshot summaries were generated from raw observations.",
            "ecTAG histograms are species-specific.",
            "ddPCR is retained as a bulk anchor table.",
            f"snapshot rows={len(snapshot)}, histogram rows={len(ectag_hist)}",
        ],
    )
    write_markdown_report(
        paths["report"],
        "Empirical Summary Report",
        [
            ("Scope", "Constructed snapshot summaries without replacing species-specific ecTAG likelihoods."),
            (
                "ecTAG",
                "Histograms are keyed by week, condition, replicate, state_gate, species, and bin_label. No cross-species summed histogram is used as a primary likelihood.",
            ),
            (
                "ddPCR",
                "ddPCR is preserved as a bulk anchor table only. Single-cell copy-number distribution summaries come from ecTAG.",
            ),
            (
                "Rows",
                (
                    f"snapshot_summary={len(snapshot)}, ectag_histograms={len(ectag_hist)}, "
                    f"qpcdr={len(qpcdr_summary)}, ddpcr={len(ddpcr_summary)}, cell_count={len(cell_count_summary)}"
                ),
            ),
        ],
    )
    return paths


def _flow_summary(flow: pd.DataFrame) -> pd.DataFrame:
    rows = (
        flow.groupby(["week", "condition", "replicate", "state_gate"], as_index=False)
        .agg(fraction=("fraction", "mean"), flow_count=("pre_sort_count", "sum"))
        .sort_values(["week", "condition", "replicate", "state_gate"])
    )
    return rows


def _ectag_histograms(ectag: pd.DataFrame, bins: list[dict], min_cells: int) -> pd.DataFrame:
    df = ectag.copy()
    df["bin_label"] = [schemas.assign_copy_bin(value, bins) for value in df["ectag_count"]]
    group_cols = ["week", "condition", "replicate", "state_gate", "species"]
    counts = df.groupby(group_cols + ["bin_label"], as_index=False).size().rename(columns={"size": "count"})
    all_keys = df[group_cols].drop_duplicates()
    labels = pd.DataFrame({"bin_label": [str(item["label"]) for item in bins]})
    expanded = all_keys.merge(labels, how="cross")
    merged = expanded.merge(counts, on=group_cols + ["bin_label"], how="left")
    merged["count"] = merged["count"].fillna(0).astype(int)
    totals = merged.groupby(group_cols)["count"].transform("sum")
    merged["n_cells"] = totals.astype(int)
    merged["probability"] = np.where(totals > 0, merged["count"] / totals, 0.0)
    merged["histogram_weight"] = np.where(merged["n_cells"] >= int(min_cells), 1.0, 0.25)
    merged["is_species_specific"] = True
    return merged.sort_values(group_cols + ["bin_label"]).reset_index(drop=True)


def _snapshot_summary(ectag: pd.DataFrame, flow_summary: pd.DataFrame, bins: list[dict]) -> pd.DataFrame:
    top_label = str(bins[-1]["label"])
    df = ectag.copy()
    df["bin_label"] = [schemas.assign_copy_bin(value, bins) for value in df["ectag_count"]]
    group_cols = ["week", "condition", "replicate", "state_gate", "species"]
    grouped = df.groupby(group_cols)
    rows = grouped.agg(
        n_cells=("ectag_count", "size"),
        copy_mean=("ectag_count", "mean"),
        copy_variance=("ectag_count", "var"),
        zero_fraction=("ectag_count", lambda values: float(np.mean(np.asarray(values) == 0))),
        tail_fraction=("bin_label", lambda values: float(np.mean(np.asarray(values) == top_label))),
    ).reset_index()
    rows["copy_variance"] = rows["copy_variance"].fillna(0.0)
    rows = rows.merge(
        flow_summary[["week", "condition", "replicate", "state_gate", "fraction", "flow_count"]],
        on=["week", "condition", "replicate", "state_gate"],
        how="left",
        validate="many_to_one",
    )
    rows = rows.rename(columns={"fraction": "flow_fraction"})
    rows["derived_total_burden"] = rows.groupby(["week", "condition", "replicate", "state_gate"])["copy_mean"].transform("sum")
    return rows.sort_values(group_cols).reset_index(drop=True)


def _qpcdr_summary(qpcdr: pd.DataFrame) -> pd.DataFrame:
    df = qpcdr.copy()
    df["qpcdr_value"] = np.where(df["relative_copy_number"].notna(), df["relative_copy_number"], df["raw_Ct_or_Cq"])
    df["qpcdr_scale"] = np.where(df["relative_copy_number"].notna(), "relative_copy_number", "ct_or_cq")
    group_cols = ["week", "condition", "replicate", "state_gate", "species"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(qpcdr_mean=("qpcdr_value", "mean"), qpcdr_sd=("qpcdr_value", "std"), qpcdr_scale=("qpcdr_scale", "first"), technical_replicates=("qpcdr_value", "size"))
        .fillna({"qpcdr_sd": 0.0})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _ddpcr_summary(ddpcr: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["week", "condition", "replicate", "species"]
    return (
        ddpcr.groupby(group_cols, as_index=False)
        .agg(ddpcr_copy_number=("ddpcr_copy_number", "mean"), ddpcr_sd_or_ci=("ddpcr_sd_or_ci", "mean"))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _cell_count_summary(cell_count: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["week", "condition", "replicate"]
    return (
        cell_count.groupby(group_cols, as_index=False)
        .agg(total_cell_count=("total_cell_count", "mean"), viability=("viability", "mean"))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _joint_species_summary(ectag: pd.DataFrame) -> pd.DataFrame:
    keys = ["week", "condition", "replicate", "state_gate", "cell_id"]
    pivot = ectag.pivot_table(index=keys, columns="species", values="ectag_count", aggfunc="mean").reset_index()
    rows = []
    for group_key, group in pivot.groupby(["week", "condition", "replicate", "state_gate"]):
        available = all(species in group.columns for species in schemas.SPECIES)
        row = dict(zip(["week", "condition", "replicate", "state_gate"], group_key))
        row["available"] = bool(available and len(group) >= 2)
        if row["available"]:
            matrix = group[list(schemas.SPECIES)].astype(float)
            corr = matrix.corr().fillna(0.0)
            for first in schemas.SPECIES:
                for second in schemas.SPECIES:
                    row[f"corr_{first}_{second}"] = float(corr.loc[first, second])
        rows.append(row)
    return pd.DataFrame(rows)
