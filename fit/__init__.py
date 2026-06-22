"""Config-centered local ABC-SMC fit for ddPCR-constrained parameters.

Implements the method described in ``markdown/fit_method.md``: sequential Monte
Carlo approximate Bayesian computation that perturbs ten positive parameters in
log space around the reference parameterization in ``config.py``, scoring each
candidate by the log2 ddPCR RMSE against observed bulk ddPCR trajectories.
"""

from __future__ import annotations

from .engine import FitConfig, run_local_abc_fit
from .io_utils import read_table, write_table
from .outputs import REQUIRED_OUTPUTS, validate_outputs
from .parameters import PARAMETER_SPECS, N_PARAMETERS, reference_phi, reference_params
from .proposal import PROPOSAL_SCHEDULE, generate_candidates
from .targets import load_ddpcr_targets, record_times_from_targets

__all__ = [
    "FitConfig",
    "N_PARAMETERS",
    "PARAMETER_SPECS",
    "PROPOSAL_SCHEDULE",
    "REQUIRED_OUTPUTS",
    "generate_candidates",
    "load_ddpcr_targets",
    "read_table",
    "record_times_from_targets",
    "reference_params",
    "reference_phi",
    "run_local_abc_fit",
    "validate_outputs",
    "write_table",
]
