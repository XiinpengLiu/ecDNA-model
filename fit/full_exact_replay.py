"""Compatibility entry point for final replay.

The current method performs replay inside ``run_full_reconstruction`` and writes
``FULL_replay_histories.zarr``. This module keeps the old public CLI/API name
without reopening qPCDR/ecTAG or modifying the full model.
"""

from __future__ import annotations

from pathlib import Path

from fit.io_utils import ensure_dir, write_json

EXACT_REPLAY_OUTPUTS: tuple[str, ...] = ("FULL_replay_histories.zarr", "FULL_exact_replay_manifest.json")


def run_full_exact_replay(
    full_dir: str | Path,
    lite_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 1,
    acceptance_quantile: float = 0.5,
) -> dict[str, Path]:
    del lite_dir, obs_params_path, seed, acceptance_quantile
    base = Path(full_dir)
    out = ensure_dir(base if output_dir is None else output_dir)
    replay = base / "FULL_replay_histories.zarr"
    if not replay.exists():
        raise FileNotFoundError("FULL_replay_histories.zarr must be produced by run_full_reconstruction")
    if out != base:
        import shutil

        target = out / "FULL_replay_histories.zarr"
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        if replay.is_dir():
            shutil.copytree(replay, target)
        else:
            shutil.copy2(replay, target)
        replay = target
    manifest = out / "FULL_exact_replay_manifest.json"
    write_json(
        manifest,
        {
            "method_source": "markdown/fit_method.md",
            "policy": "final replay already produced by bulk partial-observation SMC",
            "disabled_likelihoods": ["qpcdr", "ectag", "flow4", "state_specific_copy"],
        },
    )
    return {"FULL_replay_histories.zarr": replay, "FULL_exact_replay_manifest.json": manifest}
