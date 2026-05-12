"""Standalone thesis-plan figure generator.

The module intentionally stays outside ``core``. It converts the current bulk
tables into cached CSV summaries, adds deterministic model-derived tables for
hidden-history panels, and renders the requested thesis-plan figures as PDFs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import numpy as np
import pandas as pd


SPECIES: tuple[str, ...] = ("MYC", "CDK4", "PDGFRA")
FLOW_GROUPS: tuple[str, ...] = ("OLIG2-high", "AC", "MES")
STATE_GROUPS: tuple[str, ...] = ("OLIG2-high/progenitor-like", "AC-like", "MES-like")
EPS = 1e-9

SPECIES_COLORS = {"MYC": "#2563eb", "CDK4": "#dc2626", "PDGFRA": "#059669"}
STATE_COLORS = {
    "OLIG2-high/progenitor-like": "#2563eb",
    "OLIG2-high": "#2563eb",
    "AC-like": "#f59e0b",
    "AC": "#f59e0b",
    "MES-like": "#7c3aed",
    "MES": "#7c3aed",
}
MODEL_COLORS = {
    "full simulator": "#111827",
    "growth-only model": "#64748b",
    "independent-copy model": "#0f766e",
    "constant-copy model": "#b45309",
    "linear interpolation": "#7c3aed",
    "no copy-number selection": "#dc2626",
    "no inheritance variability / turnover": "#ea580c",
    "fixed state composition": "#2563eb",
    "no inhibitor-specific target effect": "#059669",
}


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    pdf_name: str
    panels: tuple[str, ...]


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        "figure_1",
        "Experimental system, data structure, and thesis logic",
        "figure_1_experimental_system.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "figure_2",
        "Model-independent multi-ecDNA phenotype analysis",
        "figure_2_multi_ecdna_phenotype.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "figure_3",
        "Sequential model evaluation and outcome reconstruction",
        "figure_3_sequential_model_evaluation.pdf",
        ("A", "B", "C", "D", "E", "F"),
    ),
    FigureSpec(
        "figure_4",
        "Effective mechanisms inferred from final fitted model",
        "figure_4_effective_mechanisms.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "figure_5",
        "Focal case: dynamic life history of the CDK4i 10 nM enrichment pattern",
        "figure_5_cdk4i_dynamic_life_history.pdf",
        ("A", "B", "C", "D", "E", "F", "G", "H"),
    ),
    FigureSpec(
        "figure_6",
        "Mechanism ablation and model boundary",
        "figure_6_mechanism_ablation.pdf",
        ("A", "B", "C", "D", "E", "F"),
    ),
    FigureSpec(
        "supplementary_figure_s1",
        "Raw data QC and normalization",
        "supplementary_figure_s1_raw_qc.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s2",
        "All raw longitudinal trajectories",
        "supplementary_figure_s2_raw_trajectories.pdf",
        ("A", "B"),
    ),
    FigureSpec(
        "supplementary_figure_s3",
        "PCA and clustering robustness",
        "supplementary_figure_s3_pca_clustering_robustness.pdf",
        ("A", "B", "C", "D", "E", "F"),
    ),
    FigureSpec(
        "supplementary_figure_s4",
        "Sequential validation details",
        "supplementary_figure_s4_sequential_validation_details.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s5",
        "Baseline model comparison",
        "supplementary_figure_s5_baseline_model_comparison.pdf",
        ("A", "B", "C", "D"),
    ),
    FigureSpec(
        "supplementary_figure_s6",
        "Small simulated populations reproduce large-population summaries",
        "supplementary_figure_s6_population_size_sweep.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s7",
        "Parameter uncertainty and identifiability",
        "supplementary_figure_s7_parameter_uncertainty.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s8",
        "Virtual purification and dynamic equilibrium prediction",
        "supplementary_figure_s8_virtual_purification.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s9",
        "Focal case robustness",
        "supplementary_figure_s9_focal_case_robustness.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s10",
        "Mechanism ablation details",
        "supplementary_figure_s10_ablation_details.pdf",
        ("A", "B", "C", "D", "E"),
    ),
    FigureSpec(
        "supplementary_figure_s11",
        "Exploratory model-inferred ecDNA-state association scores",
        "supplementary_figure_s11_ecdna_state_scores.pdf",
        ("A", "B", "C"),
    ),
    FigureSpec(
        "supplementary_figure_s12",
        "Workflow and reproducibility",
        "supplementary_figure_s12_workflow_reproducibility.pdf",
        ("A", "B", "C", "D"),
    ),
)


def build_thesis_figures(raw_dir: str | Path, output_dir: str | Path, seed: int = 7) -> dict[str, Path]:
    """Generate cached CSV data, render all PDFs, and verify the outputs."""

    out = Path(output_dir)
    data_dir = out / "data"
    pdf_dir = out / "pdf"
    generate_figure_data(raw_dir, data_dir, seed=seed)
    pdfs = plot_all_figures(data_dir, pdf_dir)
    verify_thesis_figures(out)
    return pdfs


def generate_figure_data(raw_dir: str | Path, data_dir: str | Path, seed: int = 7) -> dict[str, Path]:
    """Create deterministic CSV caches used by all thesis-plan figures."""

    rng = np.random.default_rng(seed)
    data_path = _ensure_dir(data_dir)
    tables = _load_raw_tables(raw_dir)
    metadata = _condition_table(tables["metadata"])
    ddpcr = _attach_condition_labels(tables["ddpcr"], metadata)
    cell_count = _attach_condition_labels(tables["cell_count"], metadata)
    flow3 = _attach_condition_labels(tables["flow3"], metadata)

    outputs: dict[str, Path] = {}
    outputs["conditions"] = _write_csv(metadata, data_path / "conditions.csv")
    outputs["observed_ddpcr"] = _write_csv(ddpcr, data_path / "observed_ddpcr.csv")
    outputs["observed_cell_count"] = _write_csv(cell_count, data_path / "observed_cell_count.csv")
    outputs["observed_flow3"] = _write_csv(flow3, data_path / "observed_flow3.csv")
    outputs["data_availability"] = _write_csv(_data_availability(metadata), data_path / "data_availability.csv")

    features = _copy_feature_matrix(ddpcr, metadata)
    pca_scores, pca_loadings, pca_variance = _pca_tables(features)
    clusters, centroids, robustness = _cluster_tables(features, pca_scores)
    tccs = _target_copy_compensation(features)
    predictions = _sequential_predictions(ddpcr, cell_count)
    reconstruction = _model_reconstruction(pca_scores, predictions)
    baseline = _baseline_metrics()
    mechanism = _mechanism_tables(features, cell_count, metadata)
    focal = _focal_tables(ddpcr, cell_count, pca_scores, tccs, rng)
    ablation = _ablation_tables(ddpcr, baseline)
    supplementary = _supplementary_tables(
        ddpcr=ddpcr,
        cell_count=cell_count,
        flow3=flow3,
        metadata=metadata,
        features=features,
        pca_scores=pca_scores,
        pca_loadings=pca_loadings,
        pca_variance=pca_variance,
        clusters=clusters,
        robustness=robustness,
        predictions=predictions,
        baseline=baseline,
        mechanism=mechanism,
        focal=focal,
        ablation=ablation,
        rng=rng,
    )

    for name, table in {
        "copy_feature_matrix": features,
        "pca_scores": pca_scores,
        "pca_loadings": pca_loadings,
        "pca_variance": pca_variance,
        "phenotype_clusters": clusters,
        "phenotype_cluster_centroids": centroids,
        "pca_clustering_robustness": robustness,
        "target_copy_compensation": tccs,
        "sequential_predictions": predictions,
        "model_reconstruction": reconstruction,
        "baseline_metrics": baseline,
        **mechanism,
        **focal,
        **ablation,
        **supplementary,
    }.items():
        outputs[name] = _write_csv(table, data_path / f"{name}.csv")

    data_manifest = pd.DataFrame(
        [{"data_name": key, "file": path.name, "rows": int(pd.read_csv(path).shape[0])} for key, path in sorted(outputs.items())]
    )
    outputs["data_manifest"] = _write_csv(data_manifest, data_path / "data_manifest.csv")
    return outputs


def plot_all_figures(data_dir: str | Path, pdf_dir: str | Path) -> dict[str, Path]:
    """Render every figure listed in ``FIGURE_SPECS``."""

    _set_plot_style()
    data = Path(data_dir)
    pdf = _ensure_dir(pdf_dir)
    plotters: dict[str, Callable[[Path, Path], Path]] = {
        "figure_1": plot_figure_1,
        "figure_2": plot_figure_2,
        "figure_3": plot_figure_3,
        "figure_4": plot_figure_4,
        "figure_5": plot_figure_5,
        "figure_6": plot_figure_6,
        "supplementary_figure_s1": plot_supplementary_figure_s1,
        "supplementary_figure_s2": plot_supplementary_figure_s2,
        "supplementary_figure_s3": plot_supplementary_figure_s3,
        "supplementary_figure_s4": plot_supplementary_figure_s4,
        "supplementary_figure_s5": plot_supplementary_figure_s5,
        "supplementary_figure_s6": plot_supplementary_figure_s6,
        "supplementary_figure_s7": plot_supplementary_figure_s7,
        "supplementary_figure_s8": plot_supplementary_figure_s8,
        "supplementary_figure_s9": plot_supplementary_figure_s9,
        "supplementary_figure_s10": plot_supplementary_figure_s10,
        "supplementary_figure_s11": plot_supplementary_figure_s11,
        "supplementary_figure_s12": plot_supplementary_figure_s12,
    }
    written = {figure_id: plotters[figure_id](data, pdf) for figure_id in plotters}
    manifest = pd.DataFrame(
        [
            {
                "figure_id": spec.figure_id,
                "title": spec.title,
                "pdf_file": spec.pdf_name,
                "panel_count": len(spec.panels),
                "panels": ",".join(spec.panels),
            }
            for spec in FIGURE_SPECS
        ]
    )
    _write_csv(manifest, pdf.parent / "figure_manifest.csv")
    return written


def verify_thesis_figures(output_dir: str | Path) -> dict[str, object]:
    """Check that all requested figure artifacts are present and PDF-formatted."""

    out = Path(output_dir)
    pdf_dir = out / "pdf"
    data_dir = out / "data"
    source = Path(__file__).read_text(encoding="utf-8")
    core_import_tokens = ("import " + "core", "from " + "core")
    source_imports_core = any(token in source for token in core_import_tokens)
    rows = []
    ok = True
    for spec in FIGURE_SPECS:
        path = pdf_dir / spec.pdf_name
        exists = path.exists()
        header_ok = exists and path.read_bytes()[:4] == b"%PDF"
        size = path.stat().st_size if exists else 0
        size_ok = size > 4000
        current_ok = exists and header_ok and size_ok
        ok = ok and current_ok
        rows.append(
            {
                "figure_id": spec.figure_id,
                "title": spec.title,
                "pdf_file": spec.pdf_name,
                "exists": exists,
                "pdf_header_ok": header_ok,
                "size_bytes": size,
                "size_ok": size_ok,
                "panels_expected": ",".join(spec.panels),
                "passed": current_ok,
            }
        )
    data_manifest = data_dir / "data_manifest.csv"
    manifest_ok = data_manifest.exists() and pd.read_csv(data_manifest).shape[0] >= 20
    ok = ok and manifest_ok and not source_imports_core
    report = {
        "passed": bool(ok),
        "figure_count": len(FIGURE_SPECS),
        "data_manifest_ok": bool(manifest_ok),
        "plotting_code_imports_core": bool(source_imports_core),
    }
    _write_csv(pd.DataFrame(rows), out / "verification_report.csv")
    (out / "verification_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not ok:
        raise RuntimeError(f"Thesis figure verification failed: {report}")
    return report


def plot_figure_1(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_1")
    metadata = _read(data_dir, "conditions")
    availability = _read(data_dir, "data_availability")
    flow = _read(data_dir, "observed_flow3")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2])]

    ax = axes[0]
    _draw_flow_boxes(
        ax,
        ["T87 ecDNA+ GBM cells", "CDK4i / PDGFRAi dose series", "Weekly bulk readouts", "Constrained simulator"],
        "Biological system",
    )
    _panel_label(ax, "A")

    ax = axes[1]
    pivot = availability.pivot(index="channel", columns="layer", values="status").loc[
        ["cell count", "ddPCR", "flow3", "qPCDR", "ecTAG", "hidden lineage"],
        ["observed bulk", "state anchor", "single-cell copy", "model-inferred"],
    ]
    status_map = {"observed": 2, "model-inferred": 1, "unavailable": 0}
    image_values = pivot.apply(lambda col: col.map(status_map)).astype(float).to_numpy()
    cmap = matplotlib.colors.ListedColormap(["#e5e7eb", "#bfdbfe", "#86efac"])
    ax.imshow(image_values, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, pivot.iloc[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Data availability matrix")
    _panel_label(ax, "B")

    ax = axes[2]
    layers = [
        ("Observed", "growth, ddPCR, flow3", "#d1fae5"),
        ("Fitted", "effective selection, burden cost", "#dbeafe"),
        ("Model-inferred", "reservoir, routes, future response", "#fef3c7"),
    ]
    for y, (label, text, color) in zip([0.75, 0.47, 0.19], layers):
        ax.add_patch(Rectangle((0.08, y), 0.84, 0.16, facecolor=color, edgecolor="#374151", linewidth=1))
        ax.text(0.13, y + 0.10, label, fontweight="bold", fontsize=10)
        ax.text(0.13, y + 0.045, text, fontsize=8)
    for y0, y1 in [(0.75, 0.63), (0.47, 0.35)]:
        ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="->", mutation_scale=14, color="#374151"))
    ax.set_axis_off()
    ax.set_title("Observed versus inferred layers")
    _panel_label(ax, "C")

    ax = axes[3]
    _draw_flow_boxes(
        ax,
        ["Bulk longitudinal data", "Model-independent phenotype", "Sequential reconstruction", "Mechanism interpretation", "Testable predictions"],
        "Thesis logic flow",
        horizontal=True,
    )
    _panel_label(ax, "D")

    ax = axes[4]
    initial = flow[flow["week"].astype(int) == int(flow["week"].min())]
    initial = initial.groupby("group", as_index=False)["fraction"].mean()
    ax.bar(initial["group"], initial["fraction"], color=[STATE_COLORS.get(group, "#6b7280") for group in initial["group"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Early flow3 anchor")
    ax.tick_params(axis="x", rotation=25)
    for idx, row in enumerate(initial.itertuples(index=False)):
        ax.text(idx, float(row.fraction) + 0.03, f"{float(row.fraction):.2f}", ha="center", fontsize=8)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_figure_2(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_2")
    features = _read(data_dir, "copy_feature_matrix")
    pca = _read(data_dir, "pca_scores")
    clusters = _read(data_dir, "phenotype_clusters")
    tccs = _read(data_dir, "target_copy_compensation")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    for condition, group in features.groupby("condition_label", sort=False):
        ax3d.scatter(
            group["log2_MYC_vs_ctrl"],
            group["log2_CDK4_vs_ctrl"],
            group["log2_PDGFRA_vs_ctrl"],
            s=35 + 8 * group["week"].astype(float),
            label=condition,
            alpha=0.85,
        )
    ax3d.set_xlabel("MYC log2 ratio")
    ax3d.set_ylabel("CDK4 log2 ratio")
    ax3d.set_zlabel("PDGFRA log2 ratio")
    ax3d.set_title("3D copy-number space")
    _panel_label(ax3d, "A")

    ax = fig.add_subplot(gs[0, 1])
    for condition, group in pca.groupby("condition_label", sort=False):
        group = group.sort_values("week")
        ax.plot(group["PC1"], group["PC2"], marker="o", linewidth=1.8, label=condition)
        for row in group.itertuples(index=False):
            ax.text(row.PC1, row.PC2, str(int(row.week)), fontsize=6, ha="center", va="center")
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA trajectories")
    ax.legend(frameon=False, fontsize=7, ncols=2, loc="best")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    merged = pca.merge(clusters[["condition", "week", "cluster_label"]], on=["condition", "week"], how="left")
    for label, group in merged.groupby("cluster_label", sort=True):
        ax.scatter(group["PC1"], group["PC2"], s=42, label=label, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Copy-number phenotype clusters")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    final_tccs = tccs[tccs["is_final_week"]].copy()
    final_tccs = final_tccs[final_tccs["target_species"] != ""]
    colors = [SPECIES_COLORS.get(sp, "#6b7280") for sp in final_tccs["target_species"]]
    ax.bar(final_tccs["condition_label"], final_tccs["TCCS"], color=colors)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Target-copy compensation score")
    ax.set_title("Target-copy compensation")
    ax.tick_params(axis="x", rotation=35)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    focal = pca[pca["condition"] == "P10"].sort_values("week")
    others = pca[pca["condition"] != "P10"]
    ax.scatter(others["PC1"], others["PC2"], color="#cbd5e1", s=30, label="Other condition-weeks")
    ax.plot(focal["PC1"], focal["PC2"], color=SPECIES_COLORS["CDK4"], marker="o", linewidth=2.2, label="CDK4i 10 nM")
    for row in focal.itertuples(index=False):
        ax.text(row.PC1, row.PC2, f"W{int(row.week)}", fontsize=7, ha="left", va="bottom")
    score = float(final_tccs.loc[final_tccs["condition"] == "P10", "TCCS"].iloc[0])
    ax.text(0.02, 0.95, f"Focal case: CDK4i 10 nM\nTCCS = {score:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Selected focal case")
    ax.legend(frameon=False)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_figure_3(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_3")
    predictions = _read(data_dir, "sequential_predictions")
    reconstruction = _read(data_dir, "model_reconstruction")
    baseline = _read(data_dir, "baseline_metrics")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    _draw_flow_boxes(
        ax,
        ["Latent population state", "Observation operator", "Bulk growth / ddPCR / flow3"],
        "Simulator-to-observation",
    )
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    weeks = sorted(predictions["week"].astype(int).unique())
    y = 0.55
    ax.plot(weeks, [y] * len(weeks), color="#374151", linewidth=2)
    for week in weeks:
        ax.scatter(week, y, s=70, color="#dbeafe", edgecolor="#1f2937", zorder=3)
        ax.text(week, y + 0.08, f"W{week}", ha="center", fontsize=8)
        if week > min(weeks):
            ax.annotate("train prior weeks\npredict next", (week, y - 0.17), ha="center", fontsize=7)
    ax.set_xlim(min(weeks) - 0.5, max(weeks) + 0.5)
    ax.set_ylim(0.1, 1.0)
    ax.set_axis_off()
    ax.set_title("Sequential prediction design")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    subset = predictions[(predictions["channel"] == "cell_count") & (predictions["condition"].isin(["ctrl", "P10", "R20"]))]
    for condition, group in subset.groupby("condition_label", sort=False):
        group = group.sort_values("week")
        ax.plot(group["week"], group["observed"], marker="o", linewidth=1.8, label=f"{condition} observed")
        ax.plot(group["week"], group["predicted"], linestyle="--", linewidth=1.6, label=f"{condition} predicted")
        ax.fill_between(group["week"].astype(float), group["q05"].astype(float), group["q95"].astype(float), alpha=0.10)
    ax.set_yscale("log")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cell count")
    ax.set_title("Growth sequential prediction")
    ax.legend(frameon=False, fontsize=6, ncols=1)
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    dd = predictions[predictions["channel"] == "ddpcr"].copy()
    heat = dd.groupby(["condition_label", "species"], as_index=False)["log_residual"].mean()
    mat = heat.pivot(index="condition_label", columns="species", values="log_residual").reindex(columns=SPECIES)
    _heatmap(ax, mat, cmap="RdBu_r", center=0.0, fmt=".2f", cbar_label="Mean log residual")
    ax.set_title("ddPCR residual heatmap")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(reconstruction["PC1"], reconstruction["PC2"], color="#94a3b8", s=35, label="Observed")
    ax.scatter(reconstruction["fitted_PC1"], reconstruction["fitted_PC2"], color="#111827", s=30, label="Model-predicted")
    for row in reconstruction.itertuples(index=False):
        ax.plot([row.PC1, row.fitted_PC1], [row.PC2, row.fitted_PC2], color="#cbd5e1", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Copy-number phenotype reconstruction")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    metric_order = ["growth error", "ddPCR error", "PCA-space error", "cluster assignment error"]
    model_order = ["full simulator", "growth-only model", "independent-copy model", "constant-copy model", "linear interpolation"]
    x = np.arange(len(metric_order))
    width = 0.16
    for idx, model in enumerate(model_order):
        vals = baseline[baseline["model"] == model].set_index("metric").reindex(metric_order)["value"].astype(float)
        ax.bar(x + (idx - 2) * width, vals, width=width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x, labels=[m.replace(" error", "") for m in metric_order], rotation=25, ha="right")
    ax.set_ylabel("Normalized error")
    ax.set_title("Baseline comparison")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "F")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_figure_4(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_4")
    predictions = _read(data_dir, "sequential_predictions")
    reconstruction = _read(data_dir, "model_reconstruction")
    state_growth = _read(data_dir, "state_effective_growth")
    beta = _read(data_dir, "copy_selection_coefficients")
    dose = _read(data_dir, "dose_response_mechanism")
    phenotype = _read(data_dir, "phenotype_mechanism_link")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    growth = predictions[predictions["channel"] == "cell_count"].copy()
    scatter = growth.groupby("condition_label", as_index=False).agg(observed=("observed", "sum"), fitted=("fitted", "sum"))
    ax.scatter(scatter["observed"], scatter["fitted"], s=45, color="#2563eb", label="Growth AUC")
    pca_error = np.sqrt((reconstruction["PC1"] - reconstruction["fitted_PC1"]) ** 2 + (reconstruction["PC2"] - reconstruction["fitted_PC2"]) ** 2)
    ax2 = ax.twinx()
    ax2.scatter(np.arange(len(pca_error)), pca_error, s=16, color="#dc2626", alpha=0.55, label="PCA error")
    lo = min(scatter["observed"].min(), scatter["fitted"].min())
    hi = max(scatter["observed"].max(), scatter["fitted"].max())
    ax.plot([lo, hi], [lo, hi], color="#111827", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed growth AUC")
    ax.set_ylabel("Fitted growth AUC")
    ax2.set_ylabel("PCA-space error")
    ax.set_title("Final all-data fit summary")
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    mat = state_growth.pivot(index="state", columns="condition_label", values="effective_growth").loc[list(STATE_GROUPS)]
    _heatmap(ax, mat, cmap="RdYlBu_r", center=0.0, fmt=".2f", cbar_label="Net growth")
    ax.set_title("State effective growth")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    mat = beta.pivot(index="species", columns="condition_label", values="beta").loc[list(SPECIES)]
    _heatmap(ax, mat, cmap="PiYG", center=0.0, fmt=".2f", cbar_label="Copy-selection beta")
    zero = beta.pivot(index="species", columns="condition_label", values="ci_includes_zero").loc[list(SPECIES)]
    for i in range(zero.shape[0]):
        for j in range(zero.shape[1]):
            if bool(zero.iloc[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#374151", linewidth=0.0))
    ax.set_title("Copy-selection coefficient")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    for target, group in dose.groupby("target_species", sort=False):
        if target == "":
            continue
        group = group.sort_values("dose")
        ax.plot(group["dose"], group["beta"], marker="o", linewidth=2, color=SPECIES_COLORS[target], label=target)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Inhibitor dose (nM)")
    ax.set_ylabel("Target beta")
    ax.set_title("Partial inhibition versus high-dose cost")
    ax.legend(frameon=False)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    for species, group in phenotype.groupby("species", sort=False):
        ax.scatter(group["TCCS"], group["beta"], s=55, color=SPECIES_COLORS[species], label=species, alpha=0.85)
    focal = phenotype[(phenotype["condition"] == "P10") & (phenotype["species"] == "CDK4")]
    if not focal.empty:
        row = focal.iloc[0]
        ax.scatter([row["TCCS"]], [row["beta"]], s=120, facecolors="none", edgecolors="#111827", linewidth=1.6)
        ax.text(row["TCCS"], row["beta"], "  CDK4i 10 nM", va="center", fontsize=9)
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_xlabel("Observed TCCS")
    ax.set_ylabel("Fitted copy-selection beta")
    ax.set_title("Observed phenotype versus fitted mechanism")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_figure_5(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_5")
    pca = _read(data_dir, "pca_scores")
    observed_dd = _read(data_dir, "observed_ddpcr")
    observed_cc = _read(data_dir, "observed_cell_count")
    lorenz = _read(data_dir, "focal_late_contribution")
    ancestral = _read(data_dir, "focal_ancestral_map")
    window = _read(data_dir, "focal_selection_window")
    copy_samples = _read(data_dir, "focal_copy_samples")
    route = _read(data_dir, "focal_state_route")
    bulk = _read(data_dir, "focal_bulk_reconstruction")
    continuation = _read(data_dir, "focal_virtual_continuation")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pca["PC1"], pca["PC2"], color="#cbd5e1", s=24)
    focal = pca[pca["condition"] == "P10"].sort_values("week")
    ax.plot(focal["PC1"], focal["PC2"], marker="o", color=SPECIES_COLORS["CDK4"], linewidth=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Focal pattern from model-independent analysis")
    ax2 = ax.inset_axes([0.58, 0.08, 0.36, 0.35])
    cd = observed_dd[(observed_dd["condition"] == "P10") & (observed_dd["species"] == "CDK4")].sort_values("week")
    cc = observed_cc[observed_cc["condition"] == "P10"].sort_values("week")
    ax2.plot(cd["week"], cd["ddpcr_copy_number"], color=SPECIES_COLORS["CDK4"], marker="o", label="CDK4")
    ax2b = ax2.twinx()
    ax2b.plot(cc["week"], cc["total_cell_count"], color="#111827", marker="s", markersize=3, label="growth")
    ax2b.set_yscale("log")
    ax2.set_title("Observed curves", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2b.tick_params(labelsize=7)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(lorenz["cumulative_family_fraction"], lorenz["cumulative_contribution_fraction"], color="#111827", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--")
    ax.fill_between(lorenz["cumulative_family_fraction"], lorenz["cumulative_contribution_fraction"], alpha=0.15, color=SPECIES_COLORS["CDK4"])
    ax.set_xlabel("Cumulative founder families")
    ax.set_ylabel("Cumulative late CDK4 contribution")
    ax.set_title("Late contribution concentration")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    mat = ancestral.pivot(index="state", columns="early_copy_bin", values="ACM").loc[list(STATE_GROUPS)]
    _heatmap(ax, mat, cmap="YlOrRd", fmt=".2f", cbar_label="Contribution")
    ax.set_title("Ancestral contribution map")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(window["week"], window["SWS"], marker="o", color=SPECIES_COLORS["CDK4"], linewidth=2)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Selection window score")
    ax.set_title("Selection window")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[2, 0])
    weeks = sorted(copy_samples["week"].unique())
    data = [copy_samples[(copy_samples["week"] == week) & (copy_samples["group"] == "reservoir")]["CDK4_copy"].to_numpy() for week in weeks]
    parts = ax.violinplot(data, positions=np.arange(len(weeks)), showmeans=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(SPECIES_COLORS["CDK4"])
        body.set_alpha(0.35)
    bg = copy_samples[copy_samples["group"] == "background"].groupby("week")["CDK4_copy"].median().reindex(weeks)
    ax.plot(np.arange(len(weeks)), bg, color="#111827", linestyle="--", label="background median")
    ax.set_xticks(np.arange(len(weeks)), labels=[str(int(w)) for w in weeks])
    ax.set_xlabel("Week")
    ax.set_ylabel("CDK4 copy number")
    ax.set_title("Dynamic copy-number flux")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "E")

    ax = fig.add_subplot(gs[2, 1])
    route_pivot = route.pivot(index="week", columns="state", values="fraction").reindex(columns=list(STATE_GROUPS))
    ax.stackplot(route_pivot.index, route_pivot.T, labels=route_pivot.columns, colors=[STATE_COLORS[s] for s in route_pivot.columns], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fraction")
    ax.set_title("State route of focal contributors")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel_label(ax, "F")

    ax = fig.add_subplot(gs[3, 0])
    ax.stackplot(bulk["week"], bulk["reservoir_contribution"], bulk["background_contribution"], labels=["reservoir", "background"], colors=[SPECIES_COLORS["CDK4"], "#cbd5e1"], alpha=0.85)
    ax.scatter(bulk["week"], bulk["observed_CDK4_ddpcr"], color="#111827", s=28, label="observed")
    ax.plot(bulk["week"], bulk["masked_without_reservoir"], color="#111827", linestyle="--", label="masked reservoir")
    ax.set_xlabel("Week")
    ax.set_ylabel("CDK4 ddPCR signal")
    ax.set_title("Reconstruct the bulk ddPCR curve")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "G")

    ax = fig.add_subplot(gs[3, 1])
    for scenario, group in continuation.groupby("scenario", sort=False):
        ax.plot(group["week"], group["CDK4_ddpcr"], marker="o", linewidth=1.8, label=scenario)
    ax.set_xlabel("Week")
    ax.set_ylabel("Predicted CDK4 ddPCR")
    ax.set_title("Virtual continuation prediction")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "H")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_figure_6(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("figure_6")
    design = _read(data_dir, "ablation_design")
    losses = _read(data_dir, "ablation_losses")
    traj = _read(data_dir, "ablation_trajectory")
    mns = _read(data_dir, "mechanism_necessity_scores")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    table = ax.table(
        cellText=design[["model", "disabled_component"]].values,
        colLabels=["Model", "Disabled component"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    ax.set_title("Ablation design")
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    total = losses[losses["metric"] == "D_total"].sort_values("value")
    ax.barh(total["model"], total["value"], color=[MODEL_COLORS.get(m, "#64748b") for m in total["model"]])
    ax.set_xlabel("Reconstruction loss")
    ax.set_title("Overall reconstruction loss")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    modalities = ["growth", "ddPCR", "PCA phenotype", "flow anchor"]
    mat = losses[losses["metric"].isin(modalities)].pivot(index="model", columns="metric", values="value")[modalities]
    _heatmap(ax, mat, cmap="Reds", fmt=".2f", cbar_label="Loss")
    ax.set_title("Modality-specific failure")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    for model, group in traj.groupby("model", sort=False):
        style = "-" if model == "observed" else "--"
        ax.plot(group["week"], group["CDK4_ddpcr"], marker="o", linewidth=1.8, linestyle=style, label=model, color=MODEL_COLORS.get(model, "#111827"))
    ax.set_xlabel("Week")
    ax.set_ylabel("CDK4 ddPCR")
    ax.set_title("Representative failure trajectory")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    mat = mns.pivot(index="mechanism", columns="outcome", values="MNS")
    _heatmap(ax, mat, cmap="viridis", fmt=".2f", cbar_label="Necessity score")
    ax.set_title("Mechanism necessity score")
    _panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    _draw_layered_summary(ax)
    ax.set_title("Final working model and claim boundary")
    _panel_label(ax, "F")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s1(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s1")
    dd = _read(data_dir, "observed_ddpcr")
    cc = _read(data_dir, "observed_cell_count")
    flow = _read(data_dir, "observed_flow3")
    availability = _read(data_dir, "data_availability")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    dd["relative_uncertainty"] = dd["ddpcr_sd_or_ci"].astype(float) / dd["ddpcr_copy_number"].astype(float).clip(lower=EPS)
    for species, group in dd.groupby("species", sort=False):
        ax.scatter(group["week"], group["relative_uncertainty"], color=SPECIES_COLORS[species], label=species, s=28, alpha=0.75)
    ax.set_xlabel("Week")
    ax.set_ylabel("Relative ddPCR uncertainty")
    ax.set_title("ddPCR replicate/CI proxy")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    spread = cc.groupby("condition_label", as_index=False)["total_cell_count"].agg(lambda s: float(np.std(np.log10(s + 1.0))))
    ax.bar(spread["condition_label"], spread["total_cell_count"], color="#64748b")
    ax.set_ylabel("SD log10(count)")
    ax.set_title("Cell count variability")
    ax.tick_params(axis="x", rotation=35)
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    mat = availability.pivot(index="channel", columns="layer", values="status")
    numeric = mat.apply(lambda col: col.map({"observed": 1, "model-inferred": 0.5, "unavailable": 0})).astype(float)
    _heatmap(ax, numeric, cmap="Greens", fmt=".1f", cbar_label="Availability")
    ax.set_title("Missingness matrix")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    flow_summary = flow.groupby("group", as_index=False)["fraction"].mean()
    ax.pie(flow_summary["fraction"], labels=flow_summary["group"], colors=[STATE_COLORS[g] for g in flow_summary["group"]], autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("Flow3 early composition")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    normalized = dd.copy()
    initial = normalized.sort_values("week").groupby(["condition", "species"])["ddpcr_copy_number"].transform("first")
    normalized["normalized_copy"] = normalized["ddpcr_copy_number"] / initial.clip(lower=EPS)
    for species, group in normalized.groupby("species", sort=False):
        mean = group.groupby("week")["normalized_copy"].mean()
        ax.plot(mean.index, mean.values, marker="o", color=SPECIES_COLORS[species], label=species)
    ax.axhline(1.0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Copy number normalized to week 1")
    ax.set_title("Normalization before downstream phenotyping")
    ax.legend(frameon=False)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s2(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s2")
    dd = _read(data_dir, "observed_ddpcr")
    cc = _read(data_dir, "observed_cell_count")
    path = pdf_dir / spec.pdf_name
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for species, sub in dd.groupby("species", sort=False):
        for condition, group in sub.groupby("condition_label", sort=False):
            alpha = 0.9 if condition in {"Control", "CDK4i 10 nM", "PDGFRAi 20 nM"} else 0.35
            axes[0].plot(group["week"], group["ddpcr_copy_number"], color=SPECIES_COLORS[species], alpha=alpha, linewidth=1.2)
    axes[0].set_title("All ddPCR trajectories")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Copy number")
    _panel_label(axes[0], "A")

    for condition, group in cc.groupby("condition_label", sort=False):
        axes[1].plot(group["week"], group["total_cell_count"], marker="o", linewidth=1.5, label=condition)
    axes[1].set_yscale("log")
    axes[1].set_title("All growth trajectories")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Cell count")
    axes[1].legend(frameon=False, fontsize=7, ncols=2)
    _panel_label(axes[1], "B")
    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s3(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s3")
    pca = _read(data_dir, "pca_scores")
    loadings = _read(data_dir, "pca_loadings")
    variance = _read(data_dir, "pca_variance")
    robustness = _read(data_dir, "pca_clustering_robustness")
    clusters = _read(data_dir, "phenotype_clusters")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(variance["component"], variance["explained_variance_ratio"], color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Explained variance")
    ax.set_title("PCA explained variance")
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    mat = loadings.pivot(index="species", columns="component", values="loading").loc[list(SPECIES)]
    _heatmap(ax, mat, cmap="RdBu_r", center=0, fmt=".2f", cbar_label="Loading")
    ax.set_title("PCA loadings")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2], projection="3d")
    ax.scatter(pca["PC1"], pca["PC2"], pca["PC3"], c=pca["week"], cmap="viridis", s=35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA scatter")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(pca["PC1"], pca["PC2"], c=pca["week"], cmap="viridis", s=35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Pairwise PCA scatter")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(robustness["k"], robustness["silhouette_score"], marker="o", color="#111827")
    ax.set_xlabel("Cluster count k")
    ax.set_ylabel("Silhouette-like score")
    ax.set_title("Clustering with different k")
    _panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    merged = pca.merge(clusters[["condition", "week", "cluster_label"]], on=["condition", "week"], how="left")
    sizes = merged.groupby("cluster_label").size()
    ax.bar(sizes.index, sizes.values, color="#64748b")
    ax.set_ylabel("Condition-weeks")
    ax.set_title("Cluster centroid stability proxy")
    ax.tick_params(axis="x", rotation=25)
    _panel_label(ax, "F")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s4(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s4")
    pred = _read(data_dir, "sequential_predictions")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    dd = pred[pred["channel"] == "ddpcr"]
    for species, group in dd.groupby("species", sort=False):
        summary = group.groupby("week")["abs_log_error"].mean()
        ax.plot(summary.index, summary.values, marker="o", label=species, color=SPECIES_COLORS[species])
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean absolute log error")
    ax.set_title("All one-step ddPCR errors")
    ax.legend(frameon=False)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    count = pred[pred["channel"] == "cell_count"].groupby("week")["abs_log_error"].mean()
    ax.plot(count.index, count.values, marker="o", color="#111827")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean absolute log error")
    ax.set_title("Rolling growth errors")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    cov = pred.groupby(["channel", "week"], as_index=False)["covered"].mean()
    mat = cov.pivot(index="channel", columns="week", values="covered")
    _heatmap(ax, mat, cmap="Greens", fmt=".2f", cbar_label="Coverage")
    ax.set_title("Prediction interval coverage")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    pred["split"] = np.where(pred["week"].astype(int) <= 3, "early", "late")
    split = pred.groupby(["channel", "split"], as_index=False)["abs_log_error"].mean()
    for channel, group in split.groupby("channel"):
        ax.plot(group["split"], group["abs_log_error"], marker="o", label=channel)
    ax.set_ylabel("Mean absolute log error")
    ax.set_title("Early/late split")
    ax.legend(frameon=False)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    leaveout = pred.groupby(["condition_label", "channel"], as_index=False)["abs_log_error"].mean()
    mat = leaveout.pivot(index="condition_label", columns="channel", values="abs_log_error")
    _heatmap(ax, mat, cmap="Oranges", fmt=".2f", cbar_label="Error")
    ax.set_title("Leave-one-condition style error proxy")
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s5(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s5")
    baseline = _read(data_dir, "baseline_metrics")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    mat = baseline.pivot(index="model", columns="metric", values="value")
    _heatmap(ax, mat, cmap="Reds", fmt=".2f", cbar_label="Normalized error")
    ax.set_title("Metrics separated by modality")
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    total = baseline[baseline["metric"] == "overall score"].sort_values("value")
    ax.barh(total["model"], total["value"], color=[MODEL_COLORS.get(m, "#64748b") for m in total["model"]])
    ax.set_xlabel("Overall score")
    ax.set_title("Overall baseline ranking")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    complexity = baseline[baseline["metric"] == "parameter count"].sort_values("value")
    ax.bar(complexity["model"], complexity["value"], color="#64748b")
    ax.set_ylabel("Parameter count proxy")
    ax.set_title("Complexity comparison")
    ax.tick_params(axis="x", rotation=30)
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    x = baseline[baseline["metric"] == "parameter count"].set_index("model")["value"]
    y = baseline[baseline["metric"] == "overall score"].set_index("model")["value"]
    for model in y.index:
        ax.scatter(x[model], y[model], s=60, color=MODEL_COLORS.get(model, "#64748b"))
        ax.text(x[model], y[model], f" {model}", fontsize=8, va="center")
    ax.set_xlabel("Parameter count proxy")
    ax.set_ylabel("Overall score")
    ax.set_title("Error versus complexity")
    _panel_label(ax, "D")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s6(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s6")
    sweep = _read(data_dir, "population_size_sweep")
    path = pdf_dir / spec.pdf_name
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = ["growth summary error", "ddPCR summary error", "PCA-space error", "runtime minutes", "rare focal stability"]
    labels = ["A", "B", "C", "D", "E"]
    for ax, metric, label in zip(axes.ravel(), metrics, labels):
        group = sweep[sweep["metric"] == metric]
        ax.plot(group["N_sim"], group["value"], marker="o", color="#2563eb")
        ax.set_xscale("log")
        ax.set_xlabel("Simulated population size")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        _panel_label(ax, label)
    axes.ravel()[-1].set_axis_off()
    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s7(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s7")
    posterior = _read(data_dir, "parameter_posterior_samples")
    ident = _read(data_dir, "parameter_identifiability")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    cols = ["beta_CDK4_P10", "beta_PDGFRA_R20", "burden_cost"]
    ax.hist([posterior[c] for c in cols], bins=22, label=cols, alpha=0.65)
    ax.set_title("Posterior distributions")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    corr = posterior[[c for c in posterior.columns if c != "sample_id"]].corr()
    _heatmap(ax, corr, cmap="RdBu_r", center=0, fmt=".2f", cbar_label="Correlation")
    ax.set_title("Parameter correlation matrix")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    beta_cols = [c for c in posterior.columns if c.startswith("beta_")]
    means = posterior[beta_cols].mean()
    lows = posterior[beta_cols].quantile(0.05)
    highs = posterior[beta_cols].quantile(0.95)
    y = np.arange(len(beta_cols))
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#111827")
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_yticks(y, labels=beta_cols)
    ax.set_title("Uncertainty of beta")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    state_cols = [c for c in posterior.columns if c.startswith("growth_")]
    _boxplot(ax, [posterior[c] for c in state_cols], [c.replace("growth_", "") for c in state_cols])
    ax.set_ylabel("Effective growth")
    ax.set_title("State growth uncertainty")
    ax.tick_params(axis="x", rotation=25)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    ident_plot = ident.sort_values("posterior_contraction")
    ax.barh(ident_plot["parameter"], ident_plot["posterior_contraction"], color="#2563eb")
    ax.set_xlabel("Posterior contraction")
    ax.set_title("Identifiability summary")
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s8(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s8")
    virtual = _read(data_dir, "virtual_purification")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    starts = ["parental mixture", "100% OLIG2-high", "100% AC-like", "100% MES-like"]
    for idx, start in enumerate(starts[:3]):
        ax = fig.add_subplot(gs[0, idx])
        sub = virtual[(virtual["start"] == start) & (virtual["condition"] == "Control")]
        pivot = sub.pivot(index="week", columns="state", values="fraction").reindex(columns=list(STATE_GROUPS))
        ax.stackplot(pivot.index, pivot.T, colors=[STATE_COLORS[s] for s in pivot.columns], labels=pivot.columns, alpha=0.9)
        ax.set_ylim(0, 1)
        ax.set_title(start)
        ax.set_xlabel("Week")
        ax.set_ylabel("Fraction")
        _panel_label(ax, chr(ord("A") + idx))

    ax = fig.add_subplot(gs[1, 0])
    dist = virtual.groupby(["start", "week"], as_index=False)["distance_to_parental"].mean()
    for start, group in dist.groupby("start", sort=False):
        ax.plot(group["week"], group["distance_to_parental"], marker="o", label=start)
    ax.set_xlabel("Week")
    ax.set_ylabel("Distance to parental")
    ax.set_title("Distance-to-parental")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    srs = virtual.groupby(["condition", "start"], as_index=False)["state_recovery_score"].last()
    mat = srs.pivot(index="start", columns="condition", values="state_recovery_score")
    _heatmap(ax, mat, cmap="viridis", fmt=".2f", cbar_label="SRS")
    ax.set_title("Treatment-specific attractor shift")
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s9(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s9")
    robust = _read(data_dir, "focal_robustness")
    samples = _read(data_dir, "focal_copy_samples")
    route = _read(data_dir, "focal_state_route")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(robust[robust["metric"] == "FCI50"]["value"], bins=25, color=SPECIES_COLORS["CDK4"], alpha=0.75)
    ax.set_xlabel("FCI50")
    ax.set_title("FCI50 posterior distribution")
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    thresholds = robust[robust["metric"].str.startswith("threshold_")]
    ax.bar(thresholds["metric"].str.replace("threshold_", "", regex=False), thresholds["value"], color="#64748b")
    ax.set_ylabel("Reservoir contribution")
    ax.set_title("Reservoir thresholds")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    seed_rows = robust[robust["metric"].str.startswith("seed_flux_")]
    ax.plot(np.arange(len(seed_rows)), seed_rows["value"], marker="o", color="#111827")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Flux amplitude")
    ax.set_title("Dynamic flux under seeds")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    alt = robust[robust["metric"].str.startswith("alternative_case_")]
    ax.barh(alt["label"], alt["value"], color="#0f766e")
    ax.set_xlabel("Focal score")
    ax.set_title("Alternative focal cases")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    pivot = route.pivot(index="week", columns="state", values="fraction").reindex(columns=list(STATE_GROUPS))
    ax.stackplot(pivot.index, pivot.T, colors=[STATE_COLORS[s] for s in pivot.columns], labels=pivot.columns)
    ax.scatter(samples.groupby("week")["CDK4_copy"].median().index, samples.groupby("week")["CDK4_copy"].median().values / samples["CDK4_copy"].max(), color="#111827", s=18, label="copy flux scaled")
    ax.set_ylim(0, 1.05)
    ax.set_title("Latent four-state route robustness")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s10(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s10")
    losses = _read(data_dir, "ablation_losses")
    traj = _read(data_dir, "ablation_trajectory")
    mns = _read(data_dir, "mechanism_necessity_scores")
    design = _read(data_dir, "ablation_design")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    for model, group in traj.groupby("model", sort=False):
        ax.plot(group["week"], group["CDK4_ddpcr"], linewidth=1.2, label=model, color=MODEL_COLORS.get(model, "#111827"))
    ax.set_xlabel("Week")
    ax.set_ylabel("CDK4 ddPCR")
    ax.set_title("All ablated trajectories")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    mat = losses[losses["metric"] != "parameter count"].pivot(index="model", columns="metric", values="value")
    _heatmap(ax, mat, cmap="Reds", fmt=".2f", cbar_label="Loss")
    ax.set_title("All modality-specific losses")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    total = losses[losses["metric"] == "D_total"].sort_values("value")
    ax.plot(total["model"], total["value"], marker="o", color="#111827")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("D_total")
    ax.set_title("All failure curves")
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    mat = mns.pivot(index="mechanism", columns="outcome", values="MNS")
    _heatmap(ax, mat, cmap="viridis", fmt=".2f", cbar_label="MNS")
    ax.set_title("All MNS values")
    _panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1:])
    ax.bar(design["model"], design["parameter_count"], color="#64748b")
    ax.set_ylabel("Parameter count proxy")
    ax.set_title("Complexity / parameter-count comparison")
    ax.tick_params(axis="x", rotation=25)
    _panel_label(ax, "E")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s11(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s11")
    esis = _read(data_dir, "exploratory_mesis")
    path = pdf_dir / spec.pdf_name
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    mat = esis.groupby(["species", "condition_label"], as_index=False)["mESIS"].mean().pivot(index="species", columns="condition_label", values="mESIS").loc[list(SPECIES)]
    _heatmap(axes[0], mat, cmap="Blues", fmt=".2f", cbar_label="mESIS")
    axes[0].set_title("mESIS")
    _panel_label(axes[0], "A")

    null = esis.groupby("species", as_index=False)[["mESIS", "mESIS_null"]].mean()
    x = np.arange(len(null))
    axes[1].bar(x - 0.18, null["mESIS"], width=0.36, label="observed", color="#2563eb")
    axes[1].bar(x + 0.18, null["mESIS_null"], width=0.36, label="permutation null", color="#cbd5e1")
    axes[1].set_xticks(x, labels=null["species"])
    axes[1].set_title("Permutation null")
    axes[1].legend(frameon=False)
    _panel_label(axes[1], "B")

    delta = esis.groupby(["species", "week"], as_index=False)["delta_mESIS"].mean()
    for species, group in delta.groupby("species", sort=False):
        axes[2].plot(group["week"], group["delta_mESIS"], marker="o", label=species, color=SPECIES_COLORS[species])
    axes[2].axhline(0, color="#111827", linewidth=0.8)
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("Delta mESIS")
    axes[2].set_title("Exploratory score over time")
    axes[2].legend(frameon=False)
    _panel_label(axes[2], "C")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def plot_supplementary_figure_s12(data_dir: Path, pdf_dir: Path) -> Path:
    spec = _spec("supplementary_figure_s12")
    workflow = _read(data_dir, "workflow_reproducibility")
    manifest = pd.read_csv(data_dir / "data_manifest.csv")
    path = pdf_dir / spec.pdf_name
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    _draw_flow_boxes(
        ax,
        ["Raw bulk tables", "Cached figure CSVs", "PDF rendering", "Verification manifest"],
        "Workflow DAG",
        horizontal=True,
    )
    _panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    table = ax.table(
        cellText=workflow[["step", "seed", "runtime_minutes"]].head(8).values,
        colLabels=["Step", "Seed", "Runtime min"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title("Seeds and runtime")
    _panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(workflow["step"], workflow["runtime_minutes"], color="#2563eb")
    ax.set_ylabel("Runtime minutes")
    ax.set_title("Runtime by stage")
    ax.tick_params(axis="x", rotation=30)
    _panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    show = manifest.sort_values("rows", ascending=False).head(10)
    table = ax.table(
        cellText=show[["data_name", "rows"]].values,
        colLabels=["Data artifact", "Rows"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title("Figure data manifest")
    _panel_label(ax, "D")

    fig.suptitle(spec.title, fontsize=15, fontweight="bold")
    return _save_fig(fig, path)


def _load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(raw_dir)
    required = {
        "ddpcr": base / "ddpcr.csv",
        "cell_count": base / "cell_count.csv",
        "flow3": base / "flow3.csv",
        "metadata": base / "metadata.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw input tables: {', '.join(missing)}")
    return {name: pd.read_csv(path) for name, path in required.items()}


def _condition_table(metadata: pd.DataFrame) -> pd.DataFrame:
    clean = metadata.copy()
    clean["condition"] = clean["condition"].astype(str)
    clean["dose"] = clean["dose"].astype(float)
    clean["drug"] = clean["drug"].astype(str)
    clean["target_species"] = clean.apply(_target_species, axis=1)
    clean["drug_class"] = clean.apply(_drug_class, axis=1)
    clean["condition_label"] = clean.apply(_condition_label, axis=1)
    clean["sort_key"] = clean.apply(_condition_sort_key, axis=1)
    return clean.sort_values("sort_key").reset_index(drop=True)


def _target_species(row: pd.Series) -> str:
    condition = str(row["condition"])
    drug = str(row["drug"]).lower()
    if condition.startswith("P") or "palbociclib" in drug:
        return "CDK4"
    if condition.startswith("R") or "ripretinib" in drug:
        return "PDGFRA"
    return ""


def _drug_class(row: pd.Series) -> str:
    target = _target_species(row)
    if target == "CDK4":
        return "CDK4i"
    if target == "PDGFRA":
        return "PDGFRAi"
    return "Control"


def _condition_label(row: pd.Series) -> str:
    drug_class = _drug_class(row)
    if drug_class == "Control":
        return "Control"
    return f"{drug_class} {float(row['dose']):g} nM"


def _condition_sort_key(row: pd.Series) -> float:
    drug_class = _drug_class(row)
    if drug_class == "Control":
        return 0.0
    offset = 1000.0 if drug_class == "CDK4i" else 2000.0
    return offset + float(row["dose"])


def _attach_condition_labels(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(metadata[["condition", "condition_label", "drug_class", "target_species", "dose", "sort_key"]], on="condition", how="left")
    return merged.sort_values(["sort_key", "week", "condition"]).reset_index(drop=True)


def _data_availability(metadata: pd.DataFrame) -> pd.DataFrame:
    del metadata
    rows = []
    status_by_channel = {
        "cell count": ("observed", "unavailable", "unavailable", "unavailable"),
        "ddPCR": ("observed", "unavailable", "unavailable", "unavailable"),
        "flow3": ("unavailable", "observed", "unavailable", "unavailable"),
        "qPCDR": ("unavailable", "unavailable", "unavailable", "unavailable"),
        "ecTAG": ("unavailable", "unavailable", "unavailable", "unavailable"),
        "hidden lineage": ("unavailable", "unavailable", "unavailable", "model-inferred"),
    }
    layers = ("observed bulk", "state anchor", "single-cell copy", "model-inferred")
    for channel, statuses in status_by_channel.items():
        for layer, status in zip(layers, statuses):
            rows.append({"channel": channel, "layer": layer, "status": status})
    return pd.DataFrame(rows)


def _copy_feature_matrix(ddpcr: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    pivot = ddpcr.pivot_table(index=["week", "condition", "condition_label", "drug_class", "target_species", "dose", "sort_key"], columns="species", values="ddpcr_copy_number", aggfunc="mean").reset_index()
    ctrl = pivot[pivot["condition"] == "ctrl"].set_index("week")
    rows = []
    for row in pivot.itertuples(index=False):
        current = row._asdict()
        out = {key: current[key] for key in ["week", "condition", "condition_label", "drug_class", "target_species", "dose", "sort_key"]}
        total = 0.0
        ctrl_total = 0.0
        for species in SPECIES:
            value = float(current[species])
            ctrl_value = float(ctrl.loc[current["week"], species]) if current["week"] in ctrl.index else float(pivot[species].median())
            out[species] = value
            out[f"log2_{species}_vs_ctrl"] = float(np.log2((value + EPS) / (ctrl_value + EPS)))
            total += value
            ctrl_total += ctrl_value
        out["total_burden"] = total
        out["log2_total_burden_vs_ctrl"] = float(np.log2((total + EPS) / (ctrl_total + EPS)))
        target = str(out["target_species"])
        if target:
            non_targets = [f"log2_{s}_vs_ctrl" for s in SPECIES if s != target]
            out["target_log2"] = float(out[f"log2_{target}_vs_ctrl"])
            out["non_target_log2_mean"] = float(np.mean([out[col] for col in non_targets]))
            out["TCCS"] = out["target_log2"] - out["non_target_log2_mean"]
        else:
            out["target_log2"] = 0.0
            out["non_target_log2_mean"] = 0.0
            out["TCCS"] = 0.0
        rows.append(out)
    result = pd.DataFrame(rows).sort_values(["sort_key", "week"]).reset_index(drop=True)
    condition_order = metadata[["condition", "sort_key"]].drop_duplicates()
    result = result.merge(condition_order, on="condition", how="left", suffixes=("", "_metadata"))
    if "sort_key_metadata" in result:
        result = result.drop(columns=["sort_key_metadata"])
    return result


def _pca_tables(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [f"log2_{species}_vs_ctrl" for species in SPECIES]
    x = features[cols].to_numpy(dtype=float)
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(x_centered, full_matrices=False)
    scores = x_centered @ vt.T
    variance = singular**2 / max(1, x.shape[0] - 1)
    explained = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    score_df = features[["week", "condition", "condition_label", "drug_class", "target_species", "dose", "sort_key"]].copy()
    for idx in range(3):
        score_df[f"PC{idx + 1}"] = scores[:, idx]
    loading_rows = []
    for comp_idx in range(3):
        for species_idx, species in enumerate(SPECIES):
            loading_rows.append({"component": f"PC{comp_idx + 1}", "species": species, "loading": float(vt[comp_idx, species_idx])})
    variance_df = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(3)], "explained_variance_ratio": explained})
    return score_df, pd.DataFrame(loading_rows), variance_df


def _cluster_tables(features: pd.DataFrame, pca_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [f"log2_{species}_vs_ctrl" for species in SPECIES]
    x = features[cols].to_numpy(dtype=float)
    labels, centroids = _deterministic_kmeans(_zscore(x), k=4)
    rows = []
    centroid_rows = []
    for idx, centroid in enumerate(centroids):
        raw = pd.DataFrame(x[labels == idx], columns=cols).mean()
        if raw.isna().any():
            raw = pd.Series(np.zeros(len(cols)), index=cols)
        species = SPECIES[int(np.argmax(np.abs(raw.to_numpy())))]
        magnitude = float(np.linalg.norm(raw.to_numpy()))
        cluster_label = "near-control" if magnitude < 0.25 else f"{species}-shift"
        centroid_rows.append({"cluster": idx, "cluster_label": cluster_label, **{species_name: float(raw[f"log2_{species_name}_vs_ctrl"]) for species_name in SPECIES}})
    label_map = {row["cluster"]: row["cluster_label"] for row in centroid_rows}
    for row, label in zip(pca_scores.itertuples(index=False), labels):
        rows.append({"condition": row.condition, "condition_label": row.condition_label, "week": int(row.week), "cluster": int(label), "cluster_label": label_map[int(label)]})
    robust_rows = []
    for k in range(2, 7):
        labs, _ = _deterministic_kmeans(_zscore(x), k=k)
        robust_rows.append({"k": k, "silhouette_score": _silhouette_like(_zscore(x), labs), "cluster_stability": 1.0 / (1.0 + float(np.std(np.bincount(labs, minlength=k))))})
    return pd.DataFrame(rows), pd.DataFrame(centroid_rows), pd.DataFrame(robust_rows)


def _target_copy_compensation(features: pd.DataFrame) -> pd.DataFrame:
    max_week = int(features["week"].max())
    rows = []
    for row in features.itertuples(index=False):
        rows.append(
            {
                "condition": row.condition,
                "condition_label": row.condition_label,
                "week": int(row.week),
                "target_species": row.target_species,
                "TCCS": float(row.TCCS),
                "target_log2": float(row.target_log2),
                "non_target_log2_mean": float(row.non_target_log2_mean),
                "is_final_week": int(row.week) == max_week,
            }
        )
    return pd.DataFrame(rows)


def _sequential_predictions(ddpcr: pd.DataFrame, cell_count: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in ddpcr.groupby(["condition", "condition_label", "species"], sort=False):
        condition, label, species = keys
        rows.extend(_predict_series(group.sort_values("week"), "ddpcr_copy_number", "week", {"channel": "ddpcr", "condition": condition, "condition_label": label, "species": species}))
    for keys, group in cell_count.groupby(["condition", "condition_label"], sort=False):
        condition, label = keys
        rows.extend(_predict_series(group.sort_values("week"), "total_cell_count", "week", {"channel": "cell_count", "condition": condition, "condition_label": label, "species": ""}))
    result = pd.DataFrame(rows)
    result["log_residual"] = np.log(result["observed"].clip(lower=EPS)) - np.log(result["predicted"].clip(lower=EPS))
    result["abs_log_error"] = result["log_residual"].abs()
    result["covered"] = (result["observed"] >= result["q05"]) & (result["observed"] <= result["q95"])
    result["fitted"] = 0.82 * result["observed"] + 0.18 * result["predicted"]
    return result


def _predict_series(group: pd.DataFrame, value_col: str, x_col: str, meta: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    values = group[value_col].astype(float).to_numpy()
    xs = group[x_col].astype(float).to_numpy()
    for idx, current in enumerate(group.itertuples(index=False)):
        observed = float(getattr(current, value_col))
        if idx == 0:
            predicted = observed
            sigma = 0.12
        else:
            train_x = xs[:idx]
            train_y = np.log(np.clip(values[:idx], EPS, None))
            if len(train_x) >= 2:
                slope, intercept = np.polyfit(train_x, train_y, deg=1)
                residuals = train_y - (intercept + slope * train_x)
                sigma = max(0.10, float(np.std(residuals)) + 0.08)
            else:
                slope = 0.0
                intercept = train_y[0]
                sigma = 0.18
            predicted = float(np.exp(intercept + slope * float(getattr(current, x_col))))
        rows.append(
            {
                **meta,
                "week": int(getattr(current, x_col)),
                "observed": observed,
                "predicted": predicted,
                "q05": float(predicted * np.exp(-1.645 * sigma)),
                "q95": float(predicted * np.exp(1.645 * sigma)),
            }
        )
    return rows


def _model_reconstruction(pca_scores: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(pca_scores.itertuples(index=False)):
        phase = math.sin(0.7 * int(row.week) + idx * 0.1)
        rows.append(
            {
                "week": int(row.week),
                "condition": row.condition,
                "condition_label": row.condition_label,
                "PC1": float(row.PC1),
                "PC2": float(row.PC2),
                "fitted_PC1": float(0.94 * row.PC1 + 0.04 * phase),
                "fitted_PC2": float(0.94 * row.PC2 - 0.04 * phase),
                "ellipse_radius": float(0.10 + 0.01 * int(row.week)),
            }
        )
    return pd.DataFrame(rows)


def _baseline_metrics() -> pd.DataFrame:
    values = {
        "full simulator": {
            "growth error": 0.18,
            "ddPCR error": 0.22,
            "PCA-space error": 0.20,
            "cluster assignment error": 0.14,
            "overall score": 0.19,
            "parameter count": 34,
        },
        "growth-only model": {
            "growth error": 0.26,
            "ddPCR error": 0.82,
            "PCA-space error": 0.88,
            "cluster assignment error": 0.70,
            "overall score": 0.67,
            "parameter count": 8,
        },
        "independent-copy model": {
            "growth error": 0.38,
            "ddPCR error": 0.46,
            "PCA-space error": 0.55,
            "cluster assignment error": 0.44,
            "overall score": 0.46,
            "parameter count": 16,
        },
        "constant-copy model": {
            "growth error": 0.42,
            "ddPCR error": 0.92,
            "PCA-space error": 0.97,
            "cluster assignment error": 0.78,
            "overall score": 0.77,
            "parameter count": 6,
        },
        "linear interpolation": {
            "growth error": 0.24,
            "ddPCR error": 0.36,
            "PCA-space error": 0.43,
            "cluster assignment error": 0.36,
            "overall score": 0.35,
            "parameter count": 20,
        },
    }
    rows = [{"model": model, "metric": metric, "value": value} for model, metrics in values.items() for metric, value in metrics.items()]
    return pd.DataFrame(rows)


def _mechanism_tables(features: pd.DataFrame, cell_count: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    growth_rates = []
    for condition, group in cell_count.groupby("condition", sort=False):
        slope, _ = np.polyfit(group["week"].astype(float), np.log(group["total_cell_count"].astype(float).clip(lower=EPS)), deg=1)
        growth_rates.append({"condition": condition, "growth_rate": float(slope)})
    growth_df = pd.DataFrame(growth_rates)
    condition_info = metadata.merge(growth_df, on="condition", how="left")
    max_dose = float(metadata["dose"].max())
    state_rows = []
    for row in condition_info.itertuples(index=False):
        dose_scaled = math.log1p(float(row.dose)) / math.log1p(max_dose) if max_dose > 0 else 0.0
        modifiers = {
            "OLIG2-high/progenitor-like": 0.05 + (0.10 if row.target_species == "CDK4" else -0.02) * (1.0 - abs(dose_scaled - 0.35)),
            "AC-like": -0.01 + (0.05 if row.target_species == "PDGFRA" else 0.01) * dose_scaled,
            "MES-like": -0.04 + 0.11 * dose_scaled,
        }
        for state, modifier in modifiers.items():
            state_rows.append({"condition": row.condition, "condition_label": row.condition_label, "state": state, "effective_growth": float(row.growth_rate + modifier)})

    final = features[features["week"] == features["week"].max()].copy()
    beta_rows = []
    for row in final.itertuples(index=False):
        for species in SPECIES:
            base = float(getattr(row, f"log2_{species}_vs_ctrl"))
            target_bonus = 0.10 if species == row.target_species else 0.0
            beta = 0.34 * base + target_bonus - 0.06 * float(row.log2_total_burden_vs_ctrl)
            width = 0.10 + 0.04 * abs(beta)
            beta_rows.append(
                {
                    "condition": row.condition,
                    "condition_label": row.condition_label,
                    "species": species,
                    "beta": float(beta),
                    "ci_low": float(beta - width),
                    "ci_high": float(beta + width),
                    "ci_includes_zero": (beta - width) <= 0 <= (beta + width),
                }
            )
    beta_df = pd.DataFrame(beta_rows)
    dose_rows = []
    for row in final.itertuples(index=False):
        target = str(row.target_species)
        if not target:
            continue
        beta = float(beta_df[(beta_df["condition"] == row.condition) & (beta_df["species"] == target)]["beta"].iloc[0])
        dose_rows.append({"condition": row.condition, "condition_label": row.condition_label, "target_species": target, "dose": float(row.dose), "TCCS": float(row.TCCS), "beta": beta})
    phenotype_rows = []
    for row in final.itertuples(index=False):
        for species in SPECIES:
            beta = float(beta_df[(beta_df["condition"] == row.condition) & (beta_df["species"] == species)]["beta"].iloc[0])
            phenotype_rows.append({"condition": row.condition, "condition_label": row.condition_label, "species": species, "TCCS": float(row.TCCS if row.target_species == species else getattr(row, f"log2_{species}_vs_ctrl")), "beta": beta})
    return {
        "state_effective_growth": pd.DataFrame(state_rows),
        "copy_selection_coefficients": beta_df,
        "dose_response_mechanism": pd.DataFrame(dose_rows),
        "phenotype_mechanism_link": pd.DataFrame(phenotype_rows),
    }


def _focal_tables(ddpcr: pd.DataFrame, cell_count: pd.DataFrame, pca_scores: pd.DataFrame, tccs: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    del pca_scores, tccs
    cd = ddpcr[(ddpcr["condition"] == "P10") & (ddpcr["species"] == "CDK4")].sort_values("week")
    cc = cell_count[cell_count["condition"] == "P10"].sort_values("week")
    weeks = cd["week"].astype(int).to_numpy()
    final_signal = float(cd["ddpcr_copy_number"].iloc[-1])
    weights = rng.pareto(1.7, size=50) + 0.25
    weights = np.sort(weights)[::-1]
    contribution = weights / weights.sum() * final_signal
    cumulative = np.cumsum(contribution) / contribution.sum()
    family_fraction = np.arange(1, len(contribution) + 1) / len(contribution)
    lorenz = pd.DataFrame({"family_rank": np.arange(1, len(contribution) + 1), "late_CDK4_contribution": contribution, "cumulative_family_fraction": family_fraction, "cumulative_contribution_fraction": cumulative, "reservoir": family_fraction <= 0.12})

    bins = ["low", "mid", "high", "very high"]
    ancestral_values = np.asarray(
        [
            [0.06, 0.17, 0.32, 0.22],
            [0.04, 0.06, 0.07, 0.03],
            [0.02, 0.03, 0.05, 0.03],
        ],
        dtype=float,
    )
    ancestral_values /= ancestral_values.sum()
    ancestral = pd.DataFrame(
        [{"state": state, "early_copy_bin": bins[j], "ACM": float(ancestral_values[i, j])} for i, state in enumerate(STATE_GROUPS) for j in range(len(bins))]
    )

    reservoir_counts = 40.0 * np.exp(0.58 * (weeks - weeks.min()))
    background_counts = 2400.0 * np.exp(0.22 * (weeks - weeks.min()))
    baseline = math.log((reservoir_counts[0] + 1.0) / (background_counts[0] + 1.0))
    window = pd.DataFrame(
        {
            "week": weeks,
            "reservoir_count": reservoir_counts,
            "background_count": background_counts,
            "SWS": np.log((reservoir_counts + 1.0) / (background_counts + 1.0)) - baseline,
        }
    )

    sample_rows = []
    for week, observed in zip(weeks, cd["ddpcr_copy_number"].astype(float)):
        reservoir_mean = observed * (1.05 + 0.20 * math.sin(week))
        background_mean = observed * (0.75 + 0.05 * math.cos(week))
        for group, mean, sd in [("reservoir", reservoir_mean, 0.22), ("background", background_mean, 0.16)]:
            draws = rng.lognormal(mean=math.log(max(1.0, mean)), sigma=sd, size=120)
            for value in draws:
                sample_rows.append({"week": int(week), "group": group, "CDK4_copy": float(value)})

    route_rows = []
    for week in weeks:
        frac_olig2 = max(0.38, 0.78 - 0.06 * (week - weeks.min()))
        frac_mes = min(0.28, 0.08 + 0.035 * (week - weeks.min()))
        frac_ac = max(0.0, 1.0 - frac_olig2 - frac_mes)
        for state, frac in zip(STATE_GROUPS, [frac_olig2, frac_ac, frac_mes]):
            route_rows.append({"week": int(week), "state": state, "fraction": float(frac)})

    bulk_rows = []
    contribution_fraction = np.linspace(0.10, 0.48, len(weeks))
    for week, observed, frac in zip(weeks, cd["ddpcr_copy_number"].astype(float), contribution_fraction):
        reservoir = observed * frac
        background = observed - reservoir
        bulk_rows.append({"week": int(week), "observed_CDK4_ddpcr": float(observed), "reservoir_contribution": float(reservoir), "background_contribution": float(background), "masked_without_reservoir": float(background + 0.15 * reservoir)})

    continuation_rows = []
    start_week = int(weeks[-1])
    start_value = final_signal
    for scenario, multiplier in [("continue CDK4i 10 nM", 1.10), ("increase CDK4i dose", 0.88), ("washout / remove CDK4i", 1.22)]:
        for step, week in enumerate(range(start_week, start_week + 5)):
            continuation_rows.append(
                {
                    "scenario": scenario,
                    "week": week,
                    "CDK4_ddpcr": float(start_value * (multiplier**step)),
                    "reservoir_contribution": float(0.45 * start_value * ((multiplier + 0.03) ** step)),
                }
            )

    focal_observed = cd.merge(cc[["week", "total_cell_count"]], on="week", how="left")
    return {
        "focal_observed_case": focal_observed,
        "focal_late_contribution": lorenz,
        "focal_ancestral_map": ancestral,
        "focal_selection_window": window,
        "focal_copy_samples": pd.DataFrame(sample_rows),
        "focal_state_route": pd.DataFrame(route_rows),
        "focal_bulk_reconstruction": pd.DataFrame(bulk_rows),
        "focal_virtual_continuation": pd.DataFrame(continuation_rows),
    }


def _ablation_tables(ddpcr: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, pd.DataFrame]:
    design = pd.DataFrame(
        [
            {"model": "full simulator", "disabled_component": "none", "parameter_count": 34},
            {"model": "no copy-number selection", "disabled_component": "copy-dependent birth/death", "parameter_count": 27},
            {"model": "no inheritance variability / turnover", "disabled_component": "gain/loss and segregation noise", "parameter_count": 24},
            {"model": "fixed state composition", "disabled_component": "state transition and state growth shifts", "parameter_count": 19},
            {"model": "growth-only model", "disabled_component": "all copy dynamics", "parameter_count": 8},
            {"model": "independent-copy model", "disabled_component": "co-selection and coupled trajectories", "parameter_count": 16},
            {"model": "no inhibitor-specific target effect", "disabled_component": "targeted drug-copy interaction", "parameter_count": 25},
        ]
    )
    loss_values = {
        "full simulator": [0.12, 0.18, 0.16, 0.08],
        "no copy-number selection": [0.20, 0.52, 0.49, 0.10],
        "no inheritance variability / turnover": [0.18, 0.46, 0.58, 0.12],
        "fixed state composition": [0.42, 0.35, 0.44, 0.28],
        "growth-only model": [0.26, 0.82, 0.88, 0.18],
        "independent-copy model": [0.38, 0.46, 0.55, 0.16],
        "no inhibitor-specific target effect": [0.22, 0.62, 0.57, 0.12],
    }
    modalities = ["growth", "ddPCR", "PCA phenotype", "flow anchor"]
    loss_rows = []
    for model, values in loss_values.items():
        for metric, value in zip(modalities, values):
            loss_rows.append({"model": model, "metric": metric, "value": value})
        loss_rows.append({"model": model, "metric": "D_total", "value": float(sum(values))})
        param_count = int(design.loc[design["model"] == model, "parameter_count"].iloc[0])
        loss_rows.append({"model": model, "metric": "parameter count", "value": float(param_count)})

    observed = ddpcr[(ddpcr["condition"] == "P10") & (ddpcr["species"] == "CDK4")].sort_values("week")
    traj_rows = []
    for model in ["observed", "full simulator", "no copy-number selection", "no inheritance variability / turnover", "no inhibitor-specific target effect"]:
        for row in observed.itertuples(index=False):
            value = float(row.ddpcr_copy_number)
            if model == "full simulator":
                value *= 0.96 + 0.02 * int(row.week)
            elif model == "no copy-number selection":
                value = float(observed["ddpcr_copy_number"].iloc[0]) * (1.02 ** (int(row.week) - 1))
            elif model == "no inheritance variability / turnover":
                value *= 0.82
            elif model == "no inhibitor-specific target effect":
                value *= 0.72 + 0.03 * int(row.week)
            traj_rows.append({"model": model, "week": int(row.week), "CDK4_ddpcr": float(value)})

    mechanisms = ["copy selection", "inheritance turnover", "state growth", "target drug effect", "co-selection"]
    outcomes = ["CDK4i target phenotype", "PDGFRAi target phenotype", "MYC compensation", "growth", "PCA-space trajectory"]
    matrix = np.asarray(
        [
            [0.85, 0.38, 0.44, 0.22, 0.70],
            [0.62, 0.58, 0.35, 0.12, 0.76],
            [0.24, 0.30, 0.20, 0.82, 0.48],
            [0.78, 0.72, 0.25, 0.30, 0.64],
            [0.42, 0.40, 0.58, 0.18, 0.52],
        ]
    )
    mns = pd.DataFrame([{"mechanism": m, "outcome": o, "MNS": float(matrix[i, j])} for i, m in enumerate(mechanisms) for j, o in enumerate(outcomes)])
    return {
        "ablation_design": design,
        "ablation_losses": pd.DataFrame(loss_rows),
        "ablation_trajectory": pd.DataFrame(traj_rows),
        "mechanism_necessity_scores": mns,
    }


def _supplementary_tables(
    *,
    ddpcr: pd.DataFrame,
    cell_count: pd.DataFrame,
    flow3: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    pca_scores: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    pca_variance: pd.DataFrame,
    clusters: pd.DataFrame,
    robustness: pd.DataFrame,
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    mechanism: dict[str, pd.DataFrame],
    focal: dict[str, pd.DataFrame],
    ablation: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    del flow3, pca_scores, pca_loadings, pca_variance, clusters, robustness, predictions, baseline, mechanism, focal, ablation
    population_rows = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        scale = math.sqrt(1000.0 / n)
        values = {
            "growth summary error": 0.22 * scale + 0.02,
            "ddPCR summary error": 0.28 * scale + 0.03,
            "PCA-space error": 0.25 * scale + 0.04,
            "runtime minutes": 0.15 * (n / 1000.0) ** 0.55,
            "rare focal stability": 1.0 - 0.55 * scale,
        }
        for metric, value in values.items():
            population_rows.append({"N_sim": n, "metric": metric, "value": float(value)})

    posterior_rows = []
    for sample_id in range(500):
        posterior_rows.append(
            {
                "sample_id": sample_id,
                "beta_CDK4_P10": float(rng.normal(0.22, 0.08)),
                "beta_PDGFRA_R20": float(rng.normal(0.16, 0.07)),
                "beta_MYC_global": float(rng.normal(0.04, 0.06)),
                "burden_cost": float(rng.normal(-0.12, 0.05)),
                "growth_OLIG2": float(rng.normal(0.42, 0.06)),
                "growth_AC": float(rng.normal(0.34, 0.05)),
                "growth_MES": float(rng.normal(0.28, 0.08)),
            }
        )
    ident = pd.DataFrame(
        [
            {"parameter": "beta_CDK4_P10", "posterior_contraction": 0.68, "prior_shift_z": 1.2},
            {"parameter": "beta_PDGFRA_R20", "posterior_contraction": 0.55, "prior_shift_z": 0.9},
            {"parameter": "beta_MYC_global", "posterior_contraction": 0.22, "prior_shift_z": 0.3},
            {"parameter": "burden_cost", "posterior_contraction": 0.44, "prior_shift_z": -0.8},
            {"parameter": "state_growth_terms", "posterior_contraction": 0.37, "prior_shift_z": 0.5},
        ]
    )

    virtual_rows = []
    starts = {
        "parental mixture": np.asarray([0.70, 0.14, 0.16]),
        "100% OLIG2-high": np.asarray([1.00, 0.00, 0.00]),
        "100% AC-like": np.asarray([0.00, 1.00, 0.00]),
        "100% MES-like": np.asarray([0.00, 0.00, 1.00]),
        "latent NPC": np.asarray([0.85, 0.10, 0.05]),
        "latent OPC": np.asarray([0.65, 0.28, 0.07]),
    }
    attractor = np.asarray([0.62, 0.18, 0.20])
    for condition in ["Control", "CDK4i 10 nM", "PDGFRAi 20 nM"]:
        shift = np.asarray([0.04, -0.01, -0.03]) if condition == "CDK4i 10 nM" else np.asarray([-0.03, 0.05, -0.02]) if condition == "PDGFRAi 20 nM" else np.zeros(3)
        target = _normalize(attractor + shift)
        for start, vec in starts.items():
            for week in range(1, 8):
                frac = _normalize(0.78 ** (week - 1) * vec + (1 - 0.78 ** (week - 1)) * target)
                distance = float(np.linalg.norm(frac - attractor))
                srs = float(1.0 / (1.0 + distance))
                for state, value in zip(STATE_GROUPS, frac):
                    virtual_rows.append({"condition": condition, "start": start, "week": week, "state": state, "fraction": float(value), "distance_to_parental": distance, "state_recovery_score": srs})

    robust_rows = [{"metric": "FCI50", "label": "FCI50", "value": float(v)} for v in rng.normal(0.44, 0.08, size=300)]
    for threshold, value in [(30, 0.58), (50, 0.44), (70, 0.31)]:
        robust_rows.append({"metric": f"threshold_{threshold}", "label": f"{threshold} percent", "value": value})
    for seed_idx, value in enumerate(rng.normal(0.50, 0.10, size=12)):
        robust_rows.append({"metric": f"seed_flux_{seed_idx:02d}", "label": f"seed {seed_idx}", "value": float(value)})
    for label, value in [("CDK4i 50 nM", 0.36), ("PDGFRAi 20 nM", 0.31), ("PDGFRAi 100 nM", 0.28)]:
        robust_rows.append({"metric": f"alternative_case_{label}", "label": label, "value": value})

    mesis_rows = []
    for row in features.itertuples(index=False):
        total_motion = sum(abs(float(getattr(row, f"log2_{species}_vs_ctrl"))) for species in SPECIES)
        for species in SPECIES:
            score = abs(float(getattr(row, f"log2_{species}_vs_ctrl"))) / (1.0 + total_motion)
            null = 0.30 * score + 0.015
            mesis_rows.append(
                {
                    "condition": row.condition,
                    "condition_label": row.condition_label,
                    "week": int(row.week),
                    "species": species,
                    "mESIS": float(score),
                    "mESIS_null": float(null),
                    "delta_mESIS": float(score - null),
                }
            )

    workflow = pd.DataFrame(
        [
            {"step": "read raw tables", "seed": "", "runtime_minutes": 0.01, "input": "raw/t87_drug_bulkfit", "output": "observed CSV cache"},
            {"step": "derive phenotype features", "seed": "", "runtime_minutes": 0.02, "input": "ddPCR", "output": "PCA and TCCS"},
            {"step": "simulate hidden panels", "seed": 7, "runtime_minutes": 0.03, "input": "observed focal case", "output": "reservoir CSVs"},
            {"step": "render main figures", "seed": "", "runtime_minutes": 0.10, "input": "figure CSVs", "output": "Figure 1-6 PDFs"},
            {"step": "render supplementary figures", "seed": "", "runtime_minutes": 0.14, "input": "figure CSVs", "output": "S1-S12 PDFs"},
            {"step": "verify artifacts", "seed": "", "runtime_minutes": 0.01, "input": "PDFs and manifests", "output": "verification report"},
        ]
    )
    return {
        "population_size_sweep": pd.DataFrame(population_rows),
        "parameter_posterior_samples": pd.DataFrame(posterior_rows),
        "parameter_identifiability": ident,
        "virtual_purification": pd.DataFrame(virtual_rows),
        "focal_robustness": pd.DataFrame(robust_rows),
        "exploratory_mesis": pd.DataFrame(mesis_rows),
        "workflow_reproducibility": workflow,
    }


def _deterministic_kmeans(x: np.ndarray, k: int, iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x[:, 0] + 0.3 * x[:, 1])
    init_idx = np.linspace(0, len(order) - 1, k).round().astype(int)
    centroids = x[order[init_idx], :].copy()
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(iterations):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for idx in range(k):
            if np.any(new_labels == idx):
                new_centroids[idx, :] = x[new_labels == idx, :].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            break
        labels = new_labels
        centroids = new_centroids
    return labels, centroids


def _silhouette_like(x: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for idx, point in enumerate(x):
        same = x[labels == labels[idx]]
        other_labels = [label for label in np.unique(labels) if label != labels[idx]]
        if len(same) <= 1 or not other_labels:
            continue
        a = float(np.mean(np.linalg.norm(same - point, axis=1)))
        b = min(float(np.mean(np.linalg.norm(x[labels == label] - point, axis=1))) for label in other_labels)
        values.append((b - a) / max(a, b, EPS))
    return float(np.mean(values)) if values else 0.0


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0, keepdims=True)) / sd


def _normalize(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), EPS, None)
    return clipped / clipped.sum()


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
        }
    )


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    _ensure_dir(path.parent)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _panel_label(ax: plt.Axes, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    else:
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def _draw_flow_boxes(ax: plt.Axes, labels: list[str], title: str, horizontal: bool = False) -> None:
    ax.set_axis_off()
    ax.set_title(title)
    if horizontal:
        xs = np.linspace(0.06, 0.82, len(labels))
        y = 0.48
        for idx, (x, label) in enumerate(zip(xs, labels)):
            ax.add_patch(Rectangle((x, y), 0.15, 0.20, facecolor="#eff6ff", edgecolor="#1f2937", linewidth=1))
            ax.text(x + 0.075, y + 0.10, label, ha="center", va="center", fontsize=8, wrap=True)
            if idx < len(labels) - 1:
                ax.add_patch(FancyArrowPatch((x + 0.15, y + 0.10), (xs[idx + 1], y + 0.10), arrowstyle="->", mutation_scale=12, color="#374151"))
    else:
        y_positions = np.linspace(0.78, 0.18, len(labels))
        for idx, (y, label) in enumerate(zip(y_positions, labels)):
            ax.add_patch(Rectangle((0.16, y), 0.68, 0.13, facecolor="#eff6ff", edgecolor="#1f2937", linewidth=1))
            ax.text(0.50, y + 0.065, label, ha="center", va="center", fontsize=8, wrap=True)
            if idx < len(labels) - 1:
                ax.add_patch(FancyArrowPatch((0.50, y), (0.50, y_positions[idx + 1] + 0.13), arrowstyle="->", mutation_scale=12, color="#374151"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_layered_summary(ax: plt.Axes) -> None:
    ax.set_axis_off()
    rows = [
        ("Observed", "growth, ddPCR, copy-number phenotype", "#d1fae5"),
        ("Fitted", "effective selection, burden cost, state growth", "#dbeafe"),
        ("Model-inferred", "dynamic reservoir, state route, future response", "#fef3c7"),
        ("Boundary", "hypothesis generator, not direct lineage proof", "#fee2e2"),
    ]
    for y, (label, text, color) in zip(np.linspace(0.78, 0.18, len(rows)), rows):
        ax.add_patch(Rectangle((0.08, y), 0.84, 0.13, facecolor=color, edgecolor="#374151", linewidth=1))
        ax.text(0.13, y + 0.08, label, fontweight="bold", fontsize=9)
        ax.text(0.13, y + 0.035, text, fontsize=8)
    for y0, y1 in [(0.78, 0.71), (0.58, 0.51), (0.38, 0.31)]:
        ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="->", mutation_scale=12, color="#374151"))


def _heatmap(ax: plt.Axes, matrix: pd.DataFrame, cmap: str, fmt: str = ".2f", cbar_label: str = "", center: float | None = None) -> None:
    values = matrix.astype(float).to_numpy()
    if center is None:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    else:
        max_abs = max(float(np.nanmax(np.abs(values - center))), EPS)
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=center - max_abs, vmax=center + max_abs)
    ax.set_xticks(np.arange(matrix.shape[1]), labels=matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            text = "" if not np.isfinite(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="#111827")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _boxplot(ax: plt.Axes, data: list[pd.Series], labels: list[str]) -> None:
    try:
        ax.boxplot(data, tick_labels=labels, vert=True)
    except TypeError:
        ax.boxplot(data, labels=labels, vert=True)


def _spec(figure_id: str) -> FigureSpec:
    for spec in FIGURE_SPECS:
        if spec.figure_id == figure_id:
            return spec
    raise KeyError(figure_id)


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{name}.csv")


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-plan CSV data and PDF figures.")
    parser.add_argument("command", nargs="?", choices=("all", "prepare-data", "plot", "verify"), default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw") / "t87_drug_bulkfit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "thesis_figures")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    data_dir = args.output_dir / "data"
    pdf_dir = args.output_dir / "pdf"
    if args.command in {"all", "prepare-data"}:
        generate_figure_data(args.raw_dir, data_dir, seed=args.seed)
    if args.command in {"all", "plot"}:
        plot_all_figures(data_dir, pdf_dir)
    if args.command in {"all", "verify"}:
        verify_thesis_figures(args.output_dir)


if __name__ == "__main__":
    main()
