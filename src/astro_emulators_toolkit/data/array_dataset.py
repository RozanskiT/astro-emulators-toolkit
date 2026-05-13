from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocols import Batch


def _display_path(path: str) -> str:
    return path or "<root>"


def _validate_tree_key(key: Any, *, field_name: str, path: str) -> str:
    if not isinstance(key, str):
        raise ValueError(
            f"{field_name} keys must be strings at '{_display_path(path)}'."
        )
    if not key:
        raise ValueError(f"{field_name} keys must be non-empty strings.")
    if "/" in key:
        raise ValueError(f"{field_name} keys must not contain '/': {key!r}.")
    return key


def _to_numpy_array_tree(
    value: Any, *, field_name: str, path: str = ""
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a nested dict of arrays at '{_display_path(path)}'."
        )
    if not value:
        raise ValueError(f"{field_name} must contain at least one array leaf.")

    out: dict[str, Any] = {}
    for key, child in value.items():
        key = _validate_tree_key(key, field_name=field_name, path=path)
        next_path = f"{path}/{key}" if path else key
        if isinstance(child, dict):
            out[key] = _to_numpy_array_tree(
                child, field_name=field_name, path=next_path
            )
            continue
        if isinstance(child, (list, tuple)):
            raise ValueError(
                f"{field_name} leaf '{next_path}' must be a NumPy/JAX array, not {type(child).__name__}."
            )
        arr = np.asarray(child)
        if arr.ndim == 0:
            raise ValueError(
                f"{field_name} leaf '{next_path}' must be at least 1D with a leading batch axis."
            )
        if arr.dtype.kind == "O":
            raise ValueError(
                f"{field_name} leaf '{next_path}' must not be an object array."
            )
        out[key] = arr
    return out


def _leading_dim(value: dict[str, Any]) -> int:
    dims: list[int] = []
    for child in value.values():
        if isinstance(child, dict):
            dims.append(_leading_dim(child))
            continue
        dims.append(int(np.asarray(child).shape[0]))
    if not dims:
        raise ValueError("TreeArrayDataset trees must contain at least one array leaf.")
    if len(set(dims)) != 1:
        raise ValueError(
            "All TreeArrayDataset leaves must share the same first dimension."
        )
    return int(dims[0])


def _take_tree(value: Any, idx: np.ndarray) -> Any:
    if isinstance(value, dict):
        return {k: _take_tree(v, idx) for k, v in value.items()}
    return np.asarray(value)[idx]


def _to_numpy_batch_array(value: Any, *, field_name: str) -> np.ndarray:
    if isinstance(value, dict):
        raise ValueError(f"{field_name} must be a NumPy/JAX array, not a dict tree.")
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(f"{field_name} must be at least 1D with a leading batch axis.")
    if arr.dtype.kind == "O":
        raise ValueError(f"{field_name} must not be an object array.")
    return arr


@dataclass
class TreeArrayDataset:
    x: dict[str, Any]
    y: dict[str, Any]
    sample_weight: np.ndarray | None = None
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = _to_numpy_array_tree(self.x, field_name="TreeArrayDataset.x")
        self.y = _to_numpy_array_tree(self.y, field_name="TreeArrayDataset.y")
        x_size = _leading_dim(self.x)
        y_size = _leading_dim(self.y)
        if x_size != y_size:
            raise ValueError("x and y must have matching first dimension.")
        self._size = x_size

        if self.sample_weight is not None:
            self.sample_weight = np.asarray(self.sample_weight)
            if self.sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a flat 1D array.")
            if self.sample_weight.shape[0] != self._size:
                raise ValueError("sample_weight must match x/y first dimension.")

    def __len__(self) -> int:
        return self._size

    def get_batch(self, idx: np.ndarray) -> Batch:
        idx = np.asarray(idx)
        batch = {"x": _take_tree(self.x, idx), "y": _take_tree(self.y, idx)}
        if self.sample_weight is not None:
            batch["sample_weight"] = self.sample_weight[idx]
        return batch


@dataclass
class XYArrayDataset:
    x: Any
    y: Any
    sample_weight: np.ndarray | None = None
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = _to_numpy_batch_array(self.x, field_name="XYArrayDataset.x")
        self.y = _to_numpy_batch_array(self.y, field_name="XYArrayDataset.y")
        if int(self.x.shape[0]) != int(self.y.shape[0]):
            raise ValueError("x and y must have matching first dimension.")
        self._size = int(self.x.shape[0])

        if self.sample_weight is not None:
            self.sample_weight = np.asarray(self.sample_weight)
            if self.sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a flat 1D array.")
            if self.sample_weight.shape[0] != self._size:
                raise ValueError("sample_weight must match x/y first dimension.")

    def __len__(self) -> int:
        return self._size

    def get_batch(self, idx: np.ndarray) -> Batch:
        idx = np.asarray(idx)
        batch: Batch = {"x": self.x[idx], "y": self.y[idx]}
        if self.sample_weight is not None:
            batch["sample_weight"] = self.sample_weight[idx]
        return batch
