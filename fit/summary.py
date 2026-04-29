"""Compatibility exports for summary helpers."""

from __future__ import annotations

import numpy as np

from fit.summary_types import SummaryBlock, SummaryCollection
from fit.v4_lite import summarize_dataset_v4_lite as summarize_dataset


def mean_summary_collection(summaries: tuple[SummaryCollection, ...]) -> SummaryCollection:
    if not summaries:
        return SummaryCollection({})
    reference = summaries[0]
    maps: dict[str, dict[str, float]] = {name: {} for name in reference.block_names()}
    for block_name in reference.block_names():
        values = np.stack([summary.blocks[block_name].values for summary in summaries], axis=0)
        for key, value in zip(reference.blocks[block_name].keys, np.mean(values, axis=0).tolist()):
            maps[block_name][key] = float(value)
    return SummaryCollection.from_block_maps(maps)


def summarize_simulation_runset(*_args, **_kwargs) -> tuple[SummaryCollection, ...]:
    return ()
