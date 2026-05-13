# src/astro_emulators_toolkit/data/npy_table.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..config.schema import NpyTableConfig
from .subset import SubsetDataset, train_val_split


def _resolve_cols(
    arr: np.ndarray,
    cols: tuple[int | str, ...],
    columns: tuple[str, ...] | None,
) -> list[int] | list[str]:
    # Structured array: field names exist
    if arr.dtype.names is not None:
        resolved: list[str] = []
        for c in cols:
            if not isinstance(c, str):
                raise ValueError(
                    "Structured .npy requires string column names in config."
                )
            resolved.append(c)
        return resolved

    # Plain 2D array: allow integer cols or column name mapping
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array or structured array, got shape={arr.shape}"
        )

    if all(isinstance(c, int) for c in cols):
        return [int(c) for c in cols]

    if columns is None:
        raise ValueError(
            "Config.columns must be provided if inputs/targets use string names."
        )
    name_to_idx = {name: i for i, name in enumerate(columns)}
    resolved_i: list[int] = []
    for c in cols:
        if isinstance(c, int):
            resolved_i.append(int(c))
        else:
            if c not in name_to_idx:
                raise KeyError(
                    f"Unknown column name '{c}'. Available={list(name_to_idx)[:10]}..."
                )
            resolved_i.append(name_to_idx[c])
    return resolved_i


@dataclass
class NpyTableDataset:
    """A lightweight dataset wrapper for a single .npy file (memmap-friendly)."""

    arr: np.ndarray
    input_cols: Sequence[int] | Sequence[str]
    target_cols: Sequence[int] | Sequence[str]
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.float32))

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    @classmethod
    def from_config(cls, cfg: NpyTableConfig) -> "NpyTableDataset":
        path = Path(cfg.path)
        if not path.exists():
            raise FileNotFoundError(path)
        mmap_mode: Literal["r"] | None = "r" if cfg.memmap else None
        arr = np.load(path, mmap_mode=mmap_mode)
        input_cols = _resolve_cols(arr, cfg.inputs, cfg.columns)
        target_cols = _resolve_cols(arr, cfg.targets, cfg.columns)
        return cls(
            arr=arr,
            input_cols=input_cols,
            target_cols=target_cols,
            dtype=np.dtype(cfg.dtype),
        )

    def _take(self, cols: Sequence[int] | Sequence[str], idx: np.ndarray) -> np.ndarray:
        if self.arr.dtype.names is not None:
            # structured
            stacked = np.stack([self.arr[name][idx] for name in cols], axis=1)
            return stacked.astype(self.dtype, copy=False)
        # 2D
        selected = self.arr[idx][:, list(cols)]
        return np.asarray(selected, dtype=self.dtype)

    def get_batch(self, idx: np.ndarray) -> dict[str, Any]:
        x = self._take(self.input_cols, idx)
        y = self._take(self.target_cols, idx)
        return {"x": x, "y": y}

    def train_val_split(
        self, val_fraction: float, seed: int = 0
    ) -> tuple[SubsetDataset, SubsetDataset]:
        return train_val_split(self, val_fraction=val_fraction, seed=seed)
