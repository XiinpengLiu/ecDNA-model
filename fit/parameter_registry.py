"""Minimal parameter bundle utilities used by full calibration."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import config as cfg


@dataclass
class ParameterBundle:
    model: cfg.ModelParameters
    observation: cfg.ObservationParameters

    def deep_copy(self) -> "ParameterBundle":
        return ParameterBundle(copy.deepcopy(self.model), copy.deepcopy(self.observation))
