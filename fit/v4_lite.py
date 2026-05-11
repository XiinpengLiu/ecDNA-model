"""v4-lite-bulk summaries used only for initialization and proposal centers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, read_table, write_dataset_netcdf, write_json, write_table, write_text_pdf
from fit.observation import load_observation_params
from fit.raw import load_clean_tables


def fit_v4_lite_summary_posterior(
    empirical_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
    seed: int = 1,
    posterior_draws: int = 64,
) -> dict[str, Path]:
    """Build the method-specified ``v4-lite-bulk`` artifacts.

    The first argument is now the clean-data directory. The name is retained
    for CLI compatibility with older callers.
    """

    del seed, posterior_draws
    clean = load_clean_tables(empirical_dir)
    obs = load_observation_params(obs_params_path)
    out = ensure_dir(output_dir)

    ddpcr_traj = _ddpcr_trajectories(clean["ddpcr"], obs)
    count_traj = _cell_count_trajectories(clean["cell_count"], obs)
    copy_velocity = _copy_velocity(ddpcr_traj)
    growth_velocity = _growth_velocity(count_traj)
    flow3 = obs["flow3"]["target"]
    initializer = _hidden_4state_initializer(flow3)
    sampler = _initial_population_sampler(clean["ddpcr"], clean["cell_count"], flow3)
    prior_scales = _prior_scales(growth_velocity, copy_velocity)
    unavailable = {"qpcdr": "closed", "ectag": "closed", "flow4": "closed", "state_specific_copy": "closed"}

    write_table(ddpcr_traj, out / "BULK_LITE_ddpcr_trajectories.parquet")
    write_table(count_traj, out / "BULK_LITE_cell_count_trajectories.parquet")
    write_table(growth_velocity, out / "BULK_LITE_growth_velocity.parquet")
    write_table(copy_velocity, out / "BULK_LITE_copy_velocity.parquet")
    write_json(out / "BULK_LITE_flow3_steady.json", flow3)
    write_table(initializer, out / "BULK_LITE_hidden_4state_initializer.parquet")
    write_json(out / "BULK_LITE_initial_population_sampler.json", sampler)
    write_json(out / "BULK_LITE_to_FULL_prior_scales.json", prior_scales)
    write_json(out / "BULK_LITE_to_FULL_fit_mask.json", schemas.FIT_MASK)
    write_json(out / "BULK_LITE_unavailable_modalities.json", unavailable)
    write_dataset_netcdf(
        out / "BULK_LITE_final_fit.nc",
        {
            "ddpcr_log_mean": ddpcr_traj["log_mean"].astype(float).to_numpy(),
            "cell_count_log_mean": count_traj["log_count"].astype(float).to_numpy(),
            "growth_velocity": growth_velocity["r_center"].astype(float).to_numpy(),
            "copy_velocity": copy_velocity["v_center"].astype(float).to_numpy(),
        },
        attrs={"role": "v4-lite-bulk", "method_source": "markdown/fit_method.md", "fit_mask": str(schemas.FIT_MASK)},
    )
    write_text_pdf(
        out / "BULK_LITE_ppc_report.pdf",
        "v4-lite-bulk PPC Report",
        [
            "Bulk-lite only smooths ddPCR bulk trajectories, cell count trajectories, and flow3 steady composition.",
            "qPCDR, ecTAG, state-specific copy, zero fraction, and high-copy tail are not fitted.",
            f"ddPCR rows={len(ddpcr_traj)}; cell count rows={len(count_traj)}; flow3 target={flow3['fractions']}",
        ],
    )
    return {name: out / name for name in schemas.LITE_OUTPUTS}


def validate_lite_artifacts(lite_dir: str | Path) -> None:
    base = Path(lite_dir)
    missing = [name for name in schemas.LITE_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing bulk-lite artifacts: {', '.join(missing)}")
    mask = read_json(base / "BULK_LITE_to_FULL_fit_mask.json")
    for key, expected in schemas.FIT_MASK.items():
        if bool(mask.get(key)) != expected:
            raise ValueError(f"Bulk-lite fit mask mismatch for {key}")
    copy = read_table(base / "BULK_LITE_copy_velocity.parquet")
    growth = read_table(base / "BULK_LITE_growth_velocity.parquet")
    if "v_center" not in copy or "r_center" not in growth:
        raise ValueError("Bulk-lite velocity outputs must expose v_center and r_center")


def load_lite_artifacts(lite_dir: str | Path) -> dict:
    validate_lite_artifacts(lite_dir)
    base = Path(lite_dir)
    return {
        "ddpcr": read_table(base / "BULK_LITE_ddpcr_trajectories.parquet"),
        "cell_count": read_table(base / "BULK_LITE_cell_count_trajectories.parquet"),
        "growth_velocity": read_table(base / "BULK_LITE_growth_velocity.parquet"),
        "copy_velocity": read_table(base / "BULK_LITE_copy_velocity.parquet"),
        "flow3": read_json(base / "BULK_LITE_flow3_steady.json"),
        "initializer": read_table(base / "BULK_LITE_hidden_4state_initializer.parquet"),
        "sampler": read_json(base / "BULK_LITE_initial_population_sampler.json"),
        "prior_scales": read_json(base / "BULK_LITE_to_FULL_prior_scales.json"),
        "fit_mask": read_json(base / "BULK_LITE_to_FULL_fit_mask.json"),
    }


def _ddpcr_trajectories(ddpcr: pd.DataFrame, obs: dict) -> pd.DataFrame:
    rows = []
    sd_by_species = obs["ddpcr"]["log_sd_by_species"]
    for row in ddpcr.itertuples(index=False):
        value = max(1e-9, float(row.ddpcr_copy_number))
        rows.append(
            {
                "week": int(row.week),
                "condition": str(row.condition),
                "replicate": str(row.replicate),
                "species": str(row.species),
                "bulk_mean": value,
                "log_mean": float(np.log(value)),
                "log_sd": float(sd_by_species.get(str(row.species), obs["ddpcr"]["default_log_sd"])),
                "phase": schemas.phase_for_week(row.week),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "replicate", "species", "week"]).reset_index(drop=True)


def _cell_count_trajectories(cell_count: pd.DataFrame, obs: dict) -> pd.DataFrame:
    rows = []
    for row in cell_count.itertuples(index=False):
        value = max(0.0, float(row.total_cell_count))
        rows.append(
            {
                "week": int(row.week),
                "condition": str(row.condition),
                "replicate": str(row.replicate),
                "total_cell_count": value,
                "log_count": float(np.log(value + 1.0)),
                "log_sd": float(obs["cell_count"]["log_sd"]),
                "phase": schemas.phase_for_week(row.week),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "replicate", "week"]).reset_index(drop=True)


def _copy_velocity(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in traj.groupby(["condition", "replicate", "species"], dropna=False):
        ordered = group.sort_values("week")
        for current, nxt in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
            rows.append(
                {
                    "condition": str(key[0]),
                    "replicate": str(key[1]),
                    "species": str(key[2]),
                    "from_week": int(current.week),
                    "to_week": int(nxt.week),
                    "phase": schemas.phase_for_week(current.week),
                    "v_center": float(nxt.log_mean - current.log_mean),
                }
            )
    return pd.DataFrame(rows)


def _growth_velocity(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in traj.groupby(["condition", "replicate"], dropna=False):
        ordered = group.sort_values("week")
        for current, nxt in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
            rows.append(
                {
                    "condition": str(key[0]),
                    "replicate": str(key[1]),
                    "from_week": int(current.week),
                    "to_week": int(nxt.week),
                    "phase": schemas.phase_for_week(current.week),
                    "r_center": float(nxt.log_count - current.log_count),
                }
            )
    return pd.DataFrame(rows)


def _hidden_4state_initializer(flow3: dict) -> pd.DataFrame:
    g = flow3["fractions"]
    rows = []
    rho_values = (0.25, 0.5, 0.75)
    for rho in rho_values:
        olig2 = float(g["OLIG2-high"])
        rows.extend(
            [
                {"rho": rho, "state_gate": "NPC-like", "fraction": float(rho * olig2), "role": "prior-only split"},
                {"rho": rho, "state_gate": "OPC-like", "fraction": float((1.0 - rho) * olig2), "role": "prior-only split"},
                {"rho": rho, "state_gate": "AC-like", "fraction": float(g["AC"]), "role": "flow3 constrained"},
                {"rho": rho, "state_gate": "MES-like", "fraction": float(g["MES"]), "role": "flow3 constrained"},
            ]
        )
    return pd.DataFrame(rows)


def _initial_population_sampler(ddpcr: pd.DataFrame, cell_count: pd.DataFrame, flow3: dict) -> dict:
    first_week = int(min(cell_count["week"].min(), ddpcr["week"].min()))
    count_anchor = cell_count[cell_count["week"] == first_week]
    ddpcr_anchor = ddpcr[ddpcr["week"] == first_week]
    return {
        "schema_version": 1,
        "initial_week": first_week,
        "n_sim_cells_fit": 10000,
        "hidden_npc_opc_split_prior": "Beta(2,2)",
        "copy_initialization": "mean-matched ZINB prior; current data do not identify single-cell distribution shape",
        "flow3_target": flow3,
        "cell_count_anchor": count_anchor.to_dict(orient="records"),
        "ddpcr_bulk_anchor": ddpcr_anchor.to_dict(orient="records"),
        "disabled_modalities": ["qpcdr", "ectag", "flow4", "state_specific_copy"],
    }


def _prior_scales(growth: pd.DataFrame, copy: pd.DataFrame) -> dict:
    return {
        "schema_version": 1,
        "role": "proposal centers and prior scales, not experimental facts",
        "r_center_sd": float(max(0.05, growth["r_center"].astype(float).std(ddof=0) if len(growth) else 0.05)),
        "v_center_sd": float(max(0.05, copy["v_center"].astype(float).std(ddof=0) if len(copy) else 0.05)),
        "flow3_bias_sd": 0.05,
        "division_death_turnover_sd": 1.0,
        "gain_loss_turnover_sd": 1.0,
    }
