from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from .protocols import DatasetProtocol


@dataclass
class DataLoader:
    dataset: DatasetProtocol
    batch_size: int
    shuffle: bool = True
    seed: int = 0
    _dataset_size: int = field(default=0, init=False, repr=False)
    _batch_offsets: NDArray[np.int64] | None = field(
        default=None, init=False, repr=False
    )
    _cached_cycle: int | None = field(default=None, init=False, repr=False)
    _cached_n: int | None = field(default=None, init=False, repr=False)
    _cached_perm: NDArray[np.int64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)
        self._dataset_size = int(len(self.dataset))
        self._batch_offsets = np.arange(self.batch_size, dtype=np.int64)

    def _dataset_len(self) -> int:
        return self._dataset_size

    def _permutation_for_cycle(self, *, n: int, cycle: int) -> NDArray[np.int64]:
        cycle = int(cycle)
        if (
            self._cached_perm is not None
            and self._cached_cycle == cycle
            and self._cached_n == n
        ):
            return self._cached_perm

        perm = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(self.seed + cycle)
        rng.shuffle(perm)

        self._cached_cycle = cycle
        self._cached_n = n
        self._cached_perm = perm
        return perm

    def _train_indices(self, step: int) -> NDArray[np.int64]:
        n = self._dataset_len()
        if n <= 0:
            raise ValueError("Training dataset must contain at least one sample.")
        if self._batch_offsets is None:
            raise RuntimeError("DataLoader batch offsets are not initialized.")

        start = int(step) * self.batch_size
        offsets = start + self._batch_offsets
        cycle = offsets // n
        position = offsets % n

        if not self.shuffle:
            return position

        first_cycle = int(cycle[0])
        if int(cycle[-1]) == first_cycle:
            perm = self._permutation_for_cycle(n=n, cycle=first_cycle)
            return perm[position]

        out = np.empty((self.batch_size,), dtype=np.int64)
        unique_cycle_values = np.unique(cycle)
        for c in unique_cycle_values:
            mask = cycle == c
            perm = self._permutation_for_cycle(n=n, cycle=int(c))
            out[mask] = perm[position[mask]]
        return out

    def train_batch(self, step: int) -> dict[str, Any]:
        batch = dict(self.dataset.get_batch(self._train_indices(step)))
        dataset_mask = batch.get("valid_mask")
        loader_mask = np.ones((self.batch_size,), dtype=np.float32)
        batch["valid_mask"] = self._combine_valid_masks(dataset_mask, loader_mask)
        return batch

    @staticmethod
    def _combine_valid_masks(dataset_mask: Any, loader_mask: np.ndarray) -> np.ndarray:
        if dataset_mask is None:
            return loader_mask.astype(np.float32)
        return np.asarray(dataset_mask, dtype=np.float32) * loader_mask.astype(
            np.float32
        )

    def iter_eval_batches(self) -> Iterator[dict[str, Any]]:
        n = self._dataset_len()
        if n <= 0:
            return
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = np.arange(start, end, dtype=np.int64)
            real = idx.shape[0]
            if real < self.batch_size:
                pad_idx = np.full((self.batch_size - real,), idx[-1], dtype=np.int64)
                full_idx = np.concatenate([idx, pad_idx], axis=0)
                valid_mask = np.concatenate(
                    [
                        np.ones((real,), dtype=np.float32),
                        np.zeros((self.batch_size - real,), dtype=np.float32),
                    ],
                    axis=0,
                )
            else:
                full_idx = idx
                valid_mask = np.ones((self.batch_size,), dtype=np.float32)
            batch = dict(self.dataset.get_batch(full_idx))
            batch["valid_mask"] = self._combine_valid_masks(
                batch.get("valid_mask"), valid_mask
            )
            yield batch
