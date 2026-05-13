from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocols import DatasetProtocol


@dataclass
class SubsetDataset:
    base: DatasetProtocol
    indices: np.ndarray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx, dtype=np.int64)
        return self.base.get_batch(self.indices[idx])


def train_val_split(
    dataset: DatasetProtocol, val_fraction: float, seed: int = 0
) -> tuple[SubsetDataset, SubsetDataset]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")

    n = len(dataset)
    if n < 2:
        raise ValueError("train_val_split requires at least 2 samples.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n - 1, n_val))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return SubsetDataset(base=dataset, indices=train_idx), SubsetDataset(
        base=dataset, indices=val_idx
    )
