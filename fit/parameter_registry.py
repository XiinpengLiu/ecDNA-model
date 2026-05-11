"""Parameter registry and prior-gate stages from ``fit_method.md``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_table, write_text_pdf


REGISTRY: dict[str, dict] = {
    "net_growth_rate": {"role": "active_effective_control", "transform": "bounded_logit", "min": -3.0, "max": 3.0, "fitted_from": "cell_count"},
    "bulk_copy_velocity": {"role": "active_effective_control", "transform": "bounded_logit", "min": -1.5, "max": 1.5, "fitted_from": "ddpcr"},
    "flow3_projection_bias": {"role": "active_effective_control", "transform": "bounded_logit", "min": -0.25, "max": 0.25, "fitted_from": "flow3_steady"},
    "division_death_turnover": {"role": "prior_constrained_nuisance", "transform": "bounded_logit", "min": 0.0, "max": 8.0, "prior": "LogNormal(log(1),0.75)", "fitted_from": "none"},
    "ecDNA_gain_loss_turnover": {"role": "prior_constrained_nuisance", "transform": "bounded_logit", "min": 0.0, "max": 10.0, "prior": "LogNormal(log(1),0.75)", "fitted_from": "none"},
    "hidden_npc_opc_split": {"role": "prior_only", "transform": "logit", "prior": "Beta(2,2)", "fitted_from": "none"},
    "state_specific_copy_enrichment": {"role": "prior_constrained_nuisance", "transform": "bounded_logit", "min": -2.0, "max": 2.0, "prior": "Normal(0,1)", "fitted_from": "none"},
    "co_segregation_strength": {"role": "prior_constrained_nuisance", "transform": "bounded_logit", "min": -0.8, "max": 0.8, "prior": "Normal(0,0.25)", "fitted_from": "none"},
    "observation_noise": {"role": "fixed", "transform": "none", "fitted_from": "locked_observation_model", "fixed_reason": "obs_params_for_full.json is locked before full fitting"},
    "flow3_projection_matrix": {"role": "fixed", "transform": "none", "fitted_from": "method_definition", "fixed_reason": "A maps NPC+OPC, AC, MES and is not fitted"},
    "ddpcr_bulk_mean": {"role": "derived_only", "transform": "none", "fitted_from": "derived"},
    "single_cell_copy_distribution_shape": {"role": "prior_only", "transform": "ZINB_prior", "prior": "fixed hyperprior", "fitted_from": "none"},
}


def _progress(message: str) -> None:
    print(f"[fit] {message}", flush=True)


def build_parameter_registry(lite_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Resolve the method parameter registry and block definitions."""

    del lite_dir
    out = ensure_dir(output_dir)
    _write_registry_yaml(out / "PARAMETER_registry_resolved.yaml", REGISTRY)
    active_blocks = {
        "growth_block": {"parameters": ["net_growth_rate"], "scored_by": ["cell_count"]},
        "copy_MYC_block": {"parameters": ["bulk_copy_velocity:MYC"], "scored_by": ["ddpcr_MYC"]},
        "copy_CDK4_block": {"parameters": ["bulk_copy_velocity:CDK4"], "scored_by": ["ddpcr_CDK4"]},
        "copy_PDGFRA_block": {"parameters": ["bulk_copy_velocity:PDGFRA"], "scored_by": ["ddpcr_PDGFRA"]},
        "flow3_projection_block": {"parameters": ["flow3_projection_bias"], "scored_by": ["flow3_steady"]},
    }
    nuisance = {name: spec for name, spec in REGISTRY.items() if spec["role"] in {"prior_constrained_nuisance", "prior_only"}}
    hard_bounds = {name: {"min": spec.get("min"), "max": spec.get("max")} for name, spec in REGISTRY.items() if "min" in spec and "max" in spec}
    table = pd.DataFrame(
        [
            {
                "parameter": name,
                "role": spec["role"],
                "transform": spec.get("transform", ""),
                "fitted_from": spec.get("fitted_from", "none"),
                "prior": spec.get("prior", ""),
                "interpretation_status": _initial_status(spec["role"]),
            }
            for name, spec in REGISTRY.items()
        ]
    )
    write_json(out / "PARAMETER_active_blocks.json", active_blocks)
    write_json(out / "PARAMETER_nuisance_blocks.json", nuisance)
    write_json(out / "PARAMETER_hard_bounds.json", hard_bounds)
    write_table(table, out / "PARAMETER_interpretability_prior_table.csv")
    return {name: out / name for name in schemas.PARAMETER_REGISTRY_OUTPUTS}


def run_prior_predictive_gate(registry_dir: str | Path, lite_dir: str | Path, obs_params_path: str | Path, output_dir: str | Path, seed: int = 1, candidates: int = 2000) -> dict[str, Path]:
    """Check prior envelope feasibility before data fitting."""

    registry_path = Path(registry_dir) / "PARAMETER_registry_resolved.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing resolved parameter registry: {registry_path}")
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    lite = Path(lite_dir)
    prior_scales = read_json(lite / "BULK_LITE_to_FULL_prior_scales.json")
    sampler = read_json(lite / "BULK_LITE_initial_population_sampler.json")
    obs = read_json(obs_params_path)
    _progress(f"prior predictive gate start: candidates={int(candidates)}, seed={seed}")
    accepted, rejection = _sample_prior_gate(rng, prior_scales, sampler, obs, int(candidates), active_relaxation=1.0)
    accepted_region = accepted[accepted["accepted"]].copy()
    relaxed = False
    if len(accepted_region) < max(1, int(0.01 * candidates)):
        relaxed = True
        _progress(
            f"prior predictive gate relaxing active controls: accepted={len(accepted_region)}/{int(candidates)} "
            f"({len(accepted_region) / max(1, int(candidates)):.1%})"
        )
        accepted, rejection = _sample_prior_gate(rng, prior_scales, sampler, obs, int(candidates), active_relaxation=1.2)
        accepted_region = accepted[accepted["accepted"]].copy()
    if len(accepted_region) < max(1, int(0.01 * candidates)):
        report = out / "PRIOR_region_incompatible_report.md"
        report.write_text(
            "# Prior Region Incompatible\n\n"
            "Prior predictive accepted fraction remained below 1% after one 20% active-control bound relaxation.\n\n"
            "Nuisance biological bounds were not relaxed. Stop before full fitting.\n",
            encoding="utf-8",
        )
        write_table(accepted, out / "PRIOR_predictive_accepted_region.parquet")
        write_table(rejection, out / "PRIOR_predictive_rejection_reasons.csv")
        _progress(
            f"prior predictive gate failed: accepted={len(accepted_region)}/{int(candidates)} "
            f"({len(accepted_region) / max(1, int(candidates)):.1%}), report={report}"
        )
        raise RuntimeError(f"prior predictive accepted fraction <1%; wrote {report}")
    write_table(accepted_region, out / "PRIOR_predictive_accepted_region.parquet")
    write_table(rejection, out / "PRIOR_predictive_rejection_reasons.csv")
    write_text_pdf(
        out / "PRIOR_predictive_gate_report.pdf",
        "Prior Predictive Gate Report",
        [
            f"accepted_fraction={len(accepted_region) / max(1, candidates):.4f}",
            f"active_control_relaxed_once={relaxed}",
            "Gate checks explosion/extinction, copy jumps, flow3 steady projection, turnover cancellation, and hard bounds.",
            "Only active-control relaxation is allowed if the accepted fraction is too low.",
        ],
    )
    _progress(
        f"prior predictive gate done: accepted={len(accepted_region)}/{int(candidates)} "
        f"({len(accepted_region) / max(1, int(candidates)):.1%}), relaxed={relaxed}, output={out}"
    )
    return {name: out / name for name in schemas.PRIOR_GATE_OUTPUTS}


def _sample_prior_gate(
    rng: np.random.Generator,
    prior_scales: dict,
    sampler: dict,
    obs: dict,
    candidates: int,
    active_relaxation: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rejection_rows = []
    r_bound = 3.0 * float(active_relaxation)
    v_bound = 1.5 * float(active_relaxation)
    flow_bound = 0.25 * float(active_relaxation)
    flow_target = np.asarray(
        [obs["flow3"]["target"]["fractions"][group] for group in schemas.FLOW3_GROUPS],
        dtype=float,
    )
    first_counts = pd.DataFrame(sampler.get("cell_count_anchor", []))
    first_ddpcr = pd.DataFrame(sampler.get("ddpcr_bulk_anchor", []))
    count_anchor = float(first_counts["total_cell_count"].median()) if "total_cell_count" in first_counts else 1.0
    copy_anchor = float(first_ddpcr["ddpcr_copy_number"].median()) if "ddpcr_copy_number" in first_ddpcr else 1.0
    for particle_id in range(int(candidates)):
        r = float(rng.normal(0.0, prior_scales.get("r_center_sd", 0.1)))
        v = float(rng.normal(0.0, prior_scales.get("v_center_sd", 0.1)))
        flow_bias = float(rng.normal(0.0, prior_scales.get("flow3_bias_sd", 0.05)))
        tau_n = float(rng.lognormal(0.0, 0.75))
        tau_k = float(rng.lognormal(0.0, 0.75))
        rho = float(rng.beta(2, 2))
        reasons = []
        if abs(r) > r_bound:
            reasons.append("net_growth_bound")
        if abs(v) > v_bound:
            reasons.append("copy_velocity_bound")
        if abs(flow_bias) > flow_bound:
            reasons.append("flow3_projection_bias_bound")
        if tau_n > 8.0:
            reasons.append("division_death_turnover_bound")
        if tau_k > 10.0:
            reasons.append("gain_loss_turnover_bound")
        if tau_n > 6.0 and abs(r) < 0.05:
            reasons.append("division_death_cancellation")
        if tau_k > 8.0 and abs(v) < 0.05:
            reasons.append("gain_loss_cancellation")
        ten_week_count = count_anchor * float(np.exp(np.clip(r * 9.0, -50.0, 50.0)))
        if ten_week_count > max(1.0, count_anchor) * 1.0e4:
            reasons.append("cell_count_explosion")
        if ten_week_count < max(1.0, count_anchor) * 1.0e-4:
            reasons.append("cell_count_immediate_extinction")
        ten_week_copy = copy_anchor * float(np.exp(np.clip(v * 9.0, -50.0, 50.0)))
        if ten_week_copy > max(1.0, copy_anchor) * 100.0:
            reasons.append("bulk_copy_explosion")
        if ten_week_copy < max(1e-6, copy_anchor) * 0.01:
            reasons.append("bulk_copy_collapse")
        flow_raw = flow_target.copy()
        flow_raw[0] += flow_bias
        flow_raw[1:] -= flow_bias / 2.0
        flow_projected = schemas.normalize_probabilities(np.clip(flow_raw, 1e-6, None), name="prior gate flow3")
        if float(np.mean(np.abs(flow_projected - flow_target))) > 0.07:
            reasons.append("flow3_projection_not_steady")
        if max(abs(r), abs(v)) > 0.85 * max(r_bound, v_bound):
            reasons.append("extreme_active_control")
        accepted = not reasons
        row = {
            "particle_id": particle_id,
            "net_growth_rate": r,
            "bulk_copy_velocity": v,
            "flow3_projection_bias": flow_bias,
            "division_death_turnover": tau_n,
            "ecDNA_gain_loss_turnover": tau_k,
            "hidden_npc_opc_split": rho,
            "simulated_count_week10": ten_week_count,
            "simulated_bulk_copy_week10": ten_week_copy,
            "flow3_mean_abs_error": float(np.mean(np.abs(flow_projected - flow_target))),
            "D_prior": float((r / r_bound) ** 2 + (v / v_bound) ** 2 + (flow_bias / flow_bound) ** 2 + (np.log(tau_n) / 0.75) ** 2 + (np.log(tau_k) / 0.75) ** 2 + ((rho - 0.5) / np.sqrt(0.05)) ** 2),
            "accepted": accepted,
            "active_control_relaxation": float(active_relaxation),
        }
        rows.append(row)
        if reasons:
            for reason in reasons:
                rejection_rows.append({"particle_id": particle_id, "reason": reason, "active_control_relaxation": float(active_relaxation)})
    return pd.DataFrame(rows), pd.DataFrame(rejection_rows, columns=["particle_id", "reason", "active_control_relaxation"])


def _write_registry_yaml(path: Path, registry: dict[str, dict]) -> None:
    lines = ["parameters:"]
    for name, spec in registry.items():
        lines.append(f"  {name}:")
        for key, value in spec.items():
            lines.append(f"    {key}: {value!r}" if isinstance(value, str) else f"    {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _initial_status(role: str) -> str:
    return {
        "active_effective_control": "weakly-informed",
        "prior_constrained_nuisance": "prior-driven",
        "prior_only": "prior-driven",
        "derived_only": "derived-only",
        "fixed": "hard-fixed",
    }.get(role, "prior-driven")
