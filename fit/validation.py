"""Consistency checks for implementation and produced artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit import schemas
from fit.final_report import validate_final_artifacts
from fit.io_utils import read_json
from fit.v4_lite import validate_lite_artifacts


def validate_full_artifacts(full_dir: str | Path) -> None:
    base = Path(full_dir)
    missing = [name for name in schemas.FULL_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing full artifacts: {', '.join(missing)}")
    weights = pd.read_parquet(base / "particle_weights.parquet")
    if "weight" not in weights or abs(float(weights["weight"].sum()) - 1.0) > 1e-6:
        raise ValueError("particle_weights.parquet must contain normalized posterior weights")
    scenarios = pd.read_parquet(base / "scenario_classes.parquet")
    if "scenario_class" not in scenarios:
        raise ValueError("scenario_classes.parquet must contain scenario_class")
    import zarr
    from zarr.storage import FSStore, ZipStore

    zarr_path = base / "FULL_particles_final.zarr"
    store = FSStore(str(zarr_path), mode="r") if zarr_path.is_dir() else ZipStore(str(zarr_path), mode="r")
    zarr_root = zarr.open_group(store=store, mode="r")
    if zarr_root.attrs.get("role") != "accepted conditional single-cell history ensemble":
        raise ValueError("FULL_particles_final.zarr must store the accepted history ensemble")
    for group_name in ("history", "weights", "events"):
        if group_name not in zarr_root:
            raise ValueError(f"FULL_particles_final.zarr missing {group_name} group")


def validate_method_contracts(observation_dir: str | Path, lite_dir: str | Path, full_dir: str | Path, final_dir: str | Path | None = None) -> dict:
    obs = read_json(Path(observation_dir) / "obs_params_for_full.json")
    if not obs.get("locked_for_full"):
        raise ValueError("Observation params are not locked for full reconstruction")
    if obs.get("ddpcr", {}).get("likelihood") != "lognormal_on_bulk_pooled_mean":
        raise ValueError("ddPCR must be scored as a pooled bulk mean")
    validate_lite_artifacts(lite_dir)
    validate_full_artifacts(full_dir)
    final_valid = False
    if final_dir is not None:
        validate_final_artifacts(final_dir)
        final_valid = True
    return {
        "observation_locked": True,
        "ddpcr_pooled_mean_only": True,
        "lite_artifacts_valid": True,
        "full_artifacts_valid": True,
        "final_artifacts_valid": final_valid,
    }
