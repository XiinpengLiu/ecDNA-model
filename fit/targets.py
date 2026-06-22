"""Load observed ddPCR targets and derive the simulator record-time grid.

The ABC fitting target is longitudinal bulk ddPCR copy number of MYC, CDK4 and
PDGFRA across treatment conditions and time points (fit_method.md paragraph 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

DDPCR_TARGET_TO_SPECIES = {"ecMyc": "MYC", "ecCDK4": "CDK4", "ecPDGFRA": "PDGFRA"}
DAY_START = 14.0           # experimental day 14 is simulation time 0
DAYS_PER_SIM_TIME = 3.5    # one simulation-time unit per 3.5 experimental days
FILTERED_DDPCR_FILENAME = "2026-05-04-ddPCR-T87-drug-treatment-days-28-35-42-filtered.csv"
DEFAULT_DATA_DIR = ROOT / "data"


def _sim_time_for_day(day: float) -> float:
    return (float(day) - DAY_START) / DAYS_PER_SIM_TIME


def _week_for_day(day: float) -> int:
    return int(round((float(day) - DAY_START) / 7.0)) + 1


def _conditions_tuple(conditions) -> tuple[str, ...]:
    values = tuple(str(token).strip() for token in conditions if str(token).strip())
    cfg.require(bool(values), "At least one condition is required.")
    return tuple(dict.fromkeys(values))


def _read_raw_anchor_targets(raw_dir: Path, conditions: tuple[str, ...]) -> pd.DataFrame:
    """Week-1 ddPCR anchor (experimental day 14) from the raw bulk table."""
    path = raw_dir / "ddpcr.csv"
    rows = pd.read_csv(path)
    rows = rows[rows["condition"].isin(conditions) & rows["species"].isin(cfg.SPECIES)].copy()
    # R500 has no day-14 anchor (treatment starts later); drop any spurious match.
    rows = rows[rows["condition"] != "R500"].copy()
    rows["day"] = DAY_START
    rows["week"] = 1
    rows["sim_time"] = 0.0
    rows["ddpcr_obs"] = rows["ddpcr_copy_number"].astype(float)
    rows["source"] = "raw_week1_anchor"
    return rows[["condition", "week", "day", "sim_time", "species", "ddpcr_obs", "source"]]


def _read_filtered_targets(data_dir: Path, conditions: tuple[str, ...]) -> pd.DataFrame:
    """Longitudinal filtered ddPCR (days 28/35/42) parsed from the CNV report."""
    path = data_dir / FILTERED_DDPCR_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=["condition", "week", "day", "sim_time", "species", "ddpcr_obs", "source"])

    raw = pd.read_csv(path)
    parsed = raw["Sample"].astype(str).str.extract(r"^d(?P<day>\d+)\s+(?P<condition>\S+)$")
    rows = raw.join(parsed)
    rows = rows[rows["condition"].isin(conditions) & rows["Target"].isin(DDPCR_TARGET_TO_SPECIES)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["condition", "week", "day", "sim_time", "species", "ddpcr_obs", "source"])
    rows["day"] = rows["day"].astype(float)
    rows["week"] = rows["day"].map(_week_for_day).astype(int)
    rows["sim_time"] = rows["day"].map(_sim_time_for_day).astype(float)
    rows["species"] = rows["Target"].map(DDPCR_TARGET_TO_SPECIES)
    rows["ddpcr_obs"] = rows["CNV"].astype(float)
    rows["source"] = "filtered_ddpcr"
    return rows[["condition", "week", "day", "sim_time", "species", "ddpcr_obs", "source"]]


def load_ddpcr_targets(
    raw_dir: Path,
    conditions,
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatenate week-1 anchor and longitudinal filtered ddPCR targets.

    Deduplicates on (condition, day, species) keeping the week-1 anchor when it
    overlaps a filtered measurement. Returns columns:
    ``condition, week, day, sim_time, species, ddpcr_obs, source``.
    """
    conditions = _conditions_tuple(conditions)
    unknown = [c for c in conditions if c not in cfg.T87_CONDITION_TREATMENTS]
    cfg.require(not unknown, f"Unsupported condition(s): {unknown}")

    anchor = _read_raw_anchor_targets(Path(raw_dir), conditions)
    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    filtered = _read_filtered_targets(resolved_data_dir, conditions)

    combined = pd.concat(
        [anchor.assign(_priority=0), filtered.assign(_priority=1)],
        ignore_index=True,
    )
    combined = (
        combined.sort_values(["_priority", "condition", "day", "species"])
        .drop_duplicates(["condition", "day", "species"], keep="first")
        .drop(columns="_priority")
    )
    return combined.sort_values(["condition", "sim_time", "species"]).reset_index(drop=True)


def record_times_from_targets(targets: pd.DataFrame) -> tuple[float, ...]:
    """Sorted unique simulation times at which the ddPCR observations sit."""
    return tuple(sorted(float(value) for value in targets["sim_time"].astype(float).unique()))
