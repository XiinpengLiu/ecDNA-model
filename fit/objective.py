"""Compatibility wrapper for the v4-lite objective."""

from fit.v4_lite import V4LiteBlockResult as BlockLikelihoodResult
from fit.v4_lite import V4LiteObjective as SyntheticLikelihoodObjective
from fit.v4_lite import V4LiteObjectiveArtifacts as SyntheticLikelihoodArtifacts
from fit.v4_lite import V4LiteObjectiveResult as SyntheticLikelihoodResult

INVALID_OBJECTIVE = 1e18

__all__ = (
    "BlockLikelihoodResult",
    "SyntheticLikelihoodArtifacts",
    "SyntheticLikelihoodObjective",
    "SyntheticLikelihoodResult",
    "INVALID_OBJECTIVE",
)
