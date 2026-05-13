from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .protocols import Batch, DatasetProtocol


def _validate_leaf_name(name: str, *, field_name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in name:
        raise ValueError(f"{field_name} must not contain '/': {name!r}.")
    return name


@dataclass
class MappedDataset:
    base: DatasetProtocol
    map_batch: Callable[[Batch], Batch]

    def __len__(self) -> int:
        return len(self.base)

    def get_batch(self, idx: np.ndarray) -> Batch:
        mapped = self.map_batch(
            dict(self.base.get_batch(np.asarray(idx, dtype=np.int64)))
        )
        if not isinstance(mapped, dict):
            raise TypeError("map_batch must return a batch dictionary.")
        return mapped


def pack_xy_as_tree(*, x_leaf: str, y_leaf: str) -> Callable[[Batch], Batch]:
    x_leaf = _validate_leaf_name(x_leaf, field_name="x_leaf")
    y_leaf = _validate_leaf_name(y_leaf, field_name="y_leaf")

    def _map(batch: Batch) -> Batch:
        out = dict(batch)
        out["x"] = {x_leaf: batch["x"]}
        out["y"] = {y_leaf: batch["y"]}
        return out

    return _map
