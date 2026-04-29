"""Small summary containers shared by the fitting code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass(frozen=True)
class SummaryBlock:
    name: str
    keys: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        cfg.require(len(self.keys) == values.size, f"SummaryBlock {self.name} key/value length mismatch.")
        object.__setattr__(self, "values", values)

    def as_mapping(self) -> dict[str, float]:
        return {key: float(value) for key, value in zip(self.keys, self.values.tolist())}

    def align_to(self, reference: "SummaryBlock") -> "SummaryBlock":
        mapping = self.as_mapping()
        missing = [key for key in reference.keys if key not in mapping]
        cfg.require(not missing, f"Summary block {self.name} is missing keys: {missing[:5]}")
        return SummaryBlock(
            name=self.name,
            keys=reference.keys,
            values=np.asarray([mapping[key] for key in reference.keys], dtype=float),
        )


@dataclass(frozen=True)
class SummaryCollection:
    blocks: dict[str, SummaryBlock]

    def block_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.blocks))

    def align_to(self, reference: "SummaryCollection") -> "SummaryCollection":
        aligned: dict[str, SummaryBlock] = {}
        for block_name, reference_block in reference.blocks.items():
            cfg.require(block_name in self.blocks, f"Summary collection is missing block {block_name}.")
            aligned[block_name] = self.blocks[block_name].align_to(reference_block)
        return SummaryCollection(aligned)

    @classmethod
    def from_block_maps(cls, block_maps: dict[str, dict[str, float]]) -> "SummaryCollection":
        blocks: dict[str, SummaryBlock] = {}
        for block_name, mapping in block_maps.items():
            if not mapping:
                continue
            keys = tuple(sorted(mapping))
            blocks[block_name] = SummaryBlock(
                name=block_name,
                keys=keys,
                values=np.asarray([mapping[key] for key in keys], dtype=float),
            )
        return cls(blocks)
