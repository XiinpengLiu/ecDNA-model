"""
Default stage orchestration for v4-lite fitting.

The public names in this module intentionally point to the week-level
v4-lite runner from ``fit.v4_lite``. This keeps the default fitting method
aligned with ``markdown/fit_method.md`` and avoids running the full core
simulator during first-round fitting.
"""

from __future__ import annotations

from fit.v4_lite import (
    V4_LITE_STAGE_SEQUENCE as STAGE_SEQUENCE,
    V4LiteFitResult as StagedFitResult,
    V4LiteFitRunner as StagedFitRunner,
    V4LiteOptimizationSettings as OptimizationSettings,
    V4LiteStageDefinition as StageDefinition,
    V4LiteStageFitResult as StageFitResult,
)


__all__ = (
    "STAGE_SEQUENCE",
    "StagedFitResult",
    "StagedFitRunner",
    "OptimizationSettings",
    "StageDefinition",
    "StageFitResult",
)
