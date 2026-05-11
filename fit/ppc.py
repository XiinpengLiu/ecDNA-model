"""Posterior predictive check compatibility wrapper."""

from __future__ import annotations

from pathlib import Path

from fit.io_utils import ensure_dir, write_json


def run_full_ppc(full_dir: str | Path, lite_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    del full_dir, lite_dir
    out = ensure_dir(output_dir or ".")
    payload = {
        "particle_scope": "accepted_particles_only",
        "raw_like_channels": ["ddpcr", "cell_count", "flow3"],
        "disabled_modalities": ["qpcdr", "ectag", "flow4", "state_specific_copy"],
    }
    write_json(Path(out) / "full_ppc_report.json", payload)
    return payload
