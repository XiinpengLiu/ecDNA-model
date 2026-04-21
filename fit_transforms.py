"""
Transforms between constrained parameter groups and unconstrained optimizer vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg
from fit_parameter_registry import (
    FitParameterSpec,
    ParameterBoundsError,
    ParameterBundle,
    TRANSFORM_COLUMN_SIMPLEX,
    TRANSFORM_IDENTITY,
    TRANSFORM_LOG,
    TRANSFORM_LOGIT,
    TRANSFORM_SIMPLEX,
    TRANSFORM_ZERO_SUM,
)


def _clamp_probabilities(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower + 1e-12, upper - 1e-12)


def _simplex_to_unconstrained(values: np.ndarray) -> np.ndarray:
    simplex = np.asarray(values, dtype=float).reshape(-1)
    simplex = np.clip(simplex, 1e-12, None)
    simplex = simplex / np.sum(simplex)
    reference = simplex[-1]
    return np.log(simplex[:-1] / reference)


def _simplex_from_unconstrained(values: np.ndarray) -> np.ndarray:
    unconstrained = np.asarray(values, dtype=float).reshape(-1)
    logits = np.concatenate([unconstrained, np.array([0.0], dtype=float)])
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _column_simplex_to_unconstrained(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float).reshape(cfg.N_STATES, cfg.N_STATES)
    pieces = [_simplex_to_unconstrained(matrix[:, column]) for column in range(cfg.N_STATES)]
    return np.concatenate(pieces, axis=0)


def _column_simplex_from_unconstrained(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    pieces: list[np.ndarray] = []
    width = cfg.N_STATES - 1
    for column in range(cfg.N_STATES):
        start = column * width
        stop = start + width
        pieces.append(_simplex_from_unconstrained(flat[start:stop]))
    return np.stack(pieces, axis=1)


def _zero_sum_to_unconstrained(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).reshape(cfg.N_STATES)
    centered = centered - np.mean(centered)
    return cfg.HELMERT_SUBMATRIX.T @ centered


def _zero_sum_from_unconstrained(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(cfg.LATENT_DIM)
    return cfg.HELMERT_SUBMATRIX @ flat


def _logit_to_unconstrained(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    clamped = _clamp_probabilities(values, lower, upper)
    scaled = (clamped - lower) / (upper - lower)
    return np.log(scaled / (1.0 - scaled))


def _logit_from_unconstrained(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    unconstrained = np.asarray(values, dtype=float)
    scaled = cfg.sigmoid(unconstrained)
    return lower + (upper - lower) * scaled


@dataclass(frozen=True)
class VectorSlice:
    spec_name: str
    start: int
    stop: int


class ParameterVectorAdapter:
    def __init__(self, specs: tuple[FitParameterSpec, ...], base_bundle: ParameterBundle):
        self.specs = specs
        self.base_bundle = base_bundle.deep_copy()
        slices: list[VectorSlice] = []
        offset = 0
        for spec in self.specs:
            slices.append(VectorSlice(spec_name=spec.name, start=offset, stop=offset + spec.unconstrained_size))
            offset += spec.unconstrained_size
        self.vector_slices = tuple(slices)
        self.dimension = offset

    def default_vector(self) -> np.ndarray:
        return self.pack_bundle(self.base_bundle)

    def pack_bundle(self, bundle: ParameterBundle) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for spec in self.specs:
            raw = spec.raw_values(bundle)
            pieces.append(self._to_unconstrained(spec, raw))
        if not pieces:
            return np.zeros(0, dtype=float)
        return np.concatenate(pieces, axis=0)

    def unpack_vector(self, vector: np.ndarray) -> ParameterBundle:
        flat = np.asarray(vector, dtype=float).reshape(-1)
        cfg.require(flat.size == self.dimension, f"Expected vector dimension {self.dimension}, got {flat.size}.")
        bundle = self.base_bundle.deep_copy()
        for spec, vector_slice in zip(self.specs, self.vector_slices):
            unconstrained = flat[vector_slice.start : vector_slice.stop]
            raw = self._from_unconstrained(spec, unconstrained)
            try:
                spec.apply(bundle, raw)
            except ParameterBoundsError:
                raise
        cfg.validate_model_parameters(bundle.model)
        cfg.validate_observation_parameters(bundle.observation)
        return bundle

    def raw_parameter_map(self, bundle: ParameterBundle) -> dict[str, np.ndarray]:
        return {spec.name: spec.raw_values(bundle).copy() for spec in self.specs}

    def proposal_scales(self) -> np.ndarray:
        scales: list[np.ndarray] = []
        for spec in self.specs:
            if spec.transform == TRANSFORM_LOG:
                scales.append(np.full(spec.unconstrained_size, 0.30, dtype=float))
            elif spec.transform in (TRANSFORM_LOGIT, TRANSFORM_SIMPLEX, TRANSFORM_COLUMN_SIMPLEX):
                scales.append(np.full(spec.unconstrained_size, 0.35, dtype=float))
            elif spec.transform == TRANSFORM_ZERO_SUM:
                scales.append(np.full(spec.unconstrained_size, 0.45, dtype=float))
            else:
                scales.append(np.full(spec.unconstrained_size, 0.40, dtype=float))
        if not scales:
            return np.zeros(0, dtype=float)
        return np.concatenate(scales, axis=0)

    @staticmethod
    def _to_unconstrained(spec: FitParameterSpec, raw: np.ndarray) -> np.ndarray:
        if spec.transform == TRANSFORM_IDENTITY:
            return np.asarray(raw, dtype=float).reshape(-1)
        if spec.transform == TRANSFORM_LOG:
            clipped = np.clip(np.asarray(raw, dtype=float).reshape(-1), spec.lower, None)
            return np.log(clipped)
        if spec.transform == TRANSFORM_LOGIT:
            return _logit_to_unconstrained(raw, spec.lower, spec.upper)
        if spec.transform == TRANSFORM_SIMPLEX:
            return _simplex_to_unconstrained(raw)
        if spec.transform == TRANSFORM_COLUMN_SIMPLEX:
            return _column_simplex_to_unconstrained(raw)
        if spec.transform == TRANSFORM_ZERO_SUM:
            return _zero_sum_to_unconstrained(raw)
        raise ValueError(f"Unsupported transform {spec.transform}.")

    @staticmethod
    def _from_unconstrained(spec: FitParameterSpec, unconstrained: np.ndarray) -> np.ndarray:
        if spec.transform == TRANSFORM_IDENTITY:
            return np.asarray(unconstrained, dtype=float).reshape(-1)
        if spec.transform == TRANSFORM_LOG:
            return np.exp(np.asarray(unconstrained, dtype=float).reshape(-1))
        if spec.transform == TRANSFORM_LOGIT:
            return _logit_from_unconstrained(unconstrained, spec.lower, spec.upper)
        if spec.transform == TRANSFORM_SIMPLEX:
            return _simplex_from_unconstrained(unconstrained)
        if spec.transform == TRANSFORM_COLUMN_SIMPLEX:
            return _column_simplex_from_unconstrained(unconstrained).reshape(-1)
        if spec.transform == TRANSFORM_ZERO_SUM:
            return _zero_sum_from_unconstrained(unconstrained)
        raise ValueError(f"Unsupported transform {spec.transform}.")
