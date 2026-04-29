"""Compatibility aliases for the default v4-lite stage runner."""

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
