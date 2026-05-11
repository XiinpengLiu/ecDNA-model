"""Raw-table PPC compatibility helpers for the bulk-only method."""

from __future__ import annotations

from pathlib import Path

from fit.io_utils import ensure_dir, write_json


def generate_full_raw_table_ppc(
    full_dir: str | Path,
    obs_params_path: str | Path,
    lite_dir: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 1,
) -> dict[str, Path]:
    del full_dir, obs_params_path, lite_dir, seed
    out = ensure_dir(Path(output_dir) if output_dir is not None else Path(full_dir))
    report = out / "raw_table_ppc_report.json"
    write_json(
        report,
        {
            "method_source": "markdown/fit_method.md",
            "history_source": "FULL_replay_histories.zarr",
            "summary_coverage_by_channel": {},
            "disabled_modalities": ["qpcdr", "ectag", "flow4", "state_specific_copy"],
            "policy": "raw-table PPC does not synthesize unavailable qPCDR/ecTAG tables in the bulk-only fit",
        },
    )
    return {"raw_table_ppc_report": report}
