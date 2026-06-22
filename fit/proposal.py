"""SMC proposal schedule and candidate generation in log-parameter space.

Generation 0 perturbs around phi_0 with width sigma_0 (fit_method.md eq. between
25-34). Later generations apply the shrink-to-reference move
    phi^(g) = phi_0 + rho_g (phi^(g-1)_parent - phi_0) + sigma_g z
(fit_method.md eq. between 35-52).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

from .parameters import PARAMETER_SPECS


# Per-generation proposal constants (fit_method.md / GOAL.md schedule).
# sigma_0 = 0.40 (gen 0 width); rho_g and sigma_small/sigma_large for gen 1-3.
PROPOSAL_SCHEDULE: dict[int, dict[str, float]] = {
    0: {"anchor": 1, "small": 0, "large": 49, "sigma_large": 0.40, "rho": 0.0},
    1: {"anchor": 1, "small": 19, "large": 30, "sigma_small": 0.15, "sigma_large": 0.30, "rho": 0.70},
    2: {"anchor": 1, "small": 19, "large": 30, "sigma_small": 0.08, "sigma_large": 0.18, "rho": 0.45},
    3: {"anchor": 1, "small": 19, "large": 30, "sigma_small": 0.04, "sigma_large": 0.10, "rho": 0.20},
}


@dataclass(frozen=True)
class Candidate:
    generation: int
    candidate_id: int       # local index within its generation
    global_id: int          # monotonically increasing across generations
    proposal_type: str      # "anchor" | "small" | "large"
    parent_generation: int
    parent_candidate_id: int
    seed: int
    phi: np.ndarray


def _schedule_for_generation(generation: int) -> dict[str, float]:
    if generation in PROPOSAL_SCHEDULE:
        return PROPOSAL_SCHEDULE[generation]
    return PROPOSAL_SCHEDULE[max(PROPOSAL_SCHEDULE)]


def proposal_counts(n_per_generation: int, generation: int) -> tuple[int, int, int]:
    """Return (anchor_count, small_count, large_count) for a generation."""
    cfg.require(n_per_generation >= 1, "n_per_generation must be at least 1.")
    if generation == 0:
        return 1, 0, int(n_per_generation) - 1
    if int(n_per_generation) == 50:
        return 1, 19, 30
    non_anchor = int(n_per_generation) - 1
    small = int(round(non_anchor * 19.0 / 49.0))
    small = min(max(0, small), non_anchor)
    return 1, small, non_anchor - small


def candidate_seed(base_seed: int, global_id: int, proposal_type: str) -> int:
    """Anchors reuse the base seed; all other particles get a unique offset seed."""
    if proposal_type == "anchor":
        return int(base_seed)
    return int(base_seed + 1009 * (int(global_id) + 1))


def _parent_phi(parent: pd.Series) -> np.ndarray:
    return np.asarray([float(parent[f"phi_{spec.column_token}"]) for spec in PARAMETER_SPECS], dtype=float)


def generate_candidates(
    *,
    generation: int,
    global_start: int,
    n_per_generation: int,
    base_seed: int,
    rng: np.random.Generator,
    phi0: np.ndarray,
    previous_accepted: pd.DataFrame | None,
) -> list[Candidate]:
    """Produce the candidate list for one generation.

    Generation 0: every perturbation is phi_0 + sigma_0 z (no parent).
    Generation g>=1: phi_0 + rho_g (phi_parent - phi_0) + sigma_g z, with the
    parent drawn uniformly from the previous generation's accepted particles.
    """
    anchor_count, small_count, large_count = proposal_counts(n_per_generation, generation)
    schedule = _schedule_for_generation(generation)
    candidates: list[Candidate] = []

    def append(proposal_type: str, phi: np.ndarray, parent: pd.Series | None = None) -> None:
        local_id = len(candidates)
        global_id = int(global_start + local_id)
        candidates.append(
            Candidate(
                generation=int(generation),
                candidate_id=int(local_id),
                global_id=global_id,
                proposal_type=proposal_type,
                parent_generation=-1 if parent is None else int(parent["generation"]),
                parent_candidate_id=-1 if parent is None else int(parent["candidate_id"]),
                seed=candidate_seed(base_seed, global_id, proposal_type),
                phi=np.asarray(phi, dtype=float),
            )
        )

    for _ in range(anchor_count):
        append("anchor", phi0.copy())

    def proposal_phi(sigma: float) -> tuple[np.ndarray, pd.Series | None]:
        if generation == 0 or previous_accepted is None or previous_accepted.empty:
            # Generation 0 move: phi_0 + sigma z, z ~ N(0, I).
            return phi0 + float(sigma) * rng.normal(size=len(phi0)), None
        parent = previous_accepted.iloc[int(rng.integers(0, len(previous_accepted)))]
        rho = float(schedule["rho"])
        # Shrink-to-reference move: phi_0 + rho (phi_parent - phi_0) + sigma z.
        return phi0 + rho * (_parent_phi(parent) - phi0) + float(sigma) * rng.normal(size=len(phi0)), parent

    sigma_small = float(schedule.get("sigma_small", schedule["sigma_large"]))
    for _ in range(small_count):
        phi, parent = proposal_phi(sigma_small)
        append("small", phi, parent)
    for _ in range(large_count):
        phi, parent = proposal_phi(float(schedule["sigma_large"]))
        append("large", phi, parent)
    return candidates


def candidate_row(candidate: Candidate, phi0: np.ndarray) -> dict[str, Any]:
    """Flatten a candidate into a record dict (without the score columns)."""
    row: dict[str, Any] = {
        "generation": int(candidate.generation),
        "candidate_id": int(candidate.candidate_id),
        "global_id": int(candidate.global_id),
        "proposal_type": candidate.proposal_type,
        "parent_generation": int(candidate.parent_generation),
        "parent_candidate_id": int(candidate.parent_candidate_id),
        "seed": int(candidate.seed),
        "distance_to_config": float(np.sqrt(np.mean((candidate.phi - phi0) ** 2))),
    }
    for value, config_value, spec in zip(candidate.phi, phi0, PARAMETER_SPECS):
        theta = float(np.exp(float(value)))  # all fitted params use log transform
        row[spec.theta_column] = theta
        row[spec.log2_fold_column] = float((float(value) - float(config_value)) / np.log(2.0))
        row[spec.phi_column] = float(value)
    return row
