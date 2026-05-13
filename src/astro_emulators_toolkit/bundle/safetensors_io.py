# src/astro_emulators_toolkit/bundle/safetensors_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file as _st_load
from safetensors.numpy import save_file as _st_save

_EMPTY_DICT_SENTINEL = "__empty_dict__"


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not d and prefix:
        out[f"{prefix}/{_EMPTY_DICT_SENTINEL}"] = np.asarray([0], dtype=np.int8)
        return out
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def _unflatten_dict(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split("/")
        if parts[-1] == _EMPTY_DICT_SENTINEL:
            cur = root
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            continue
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return root


def save_weights(path: str | Path, *, params: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = {}
    flat.update(_flatten_dict(params, prefix="params"))
    _st_save(flat, str(path))


def load_weights(path: str | Path) -> dict[str, Any]:
    flat = _st_load(str(path))
    nested = _unflatten_dict(flat)
    params = nested.get("params", {})
    return params if isinstance(params, dict) else {}


def save_arrays(path: str | Path, arrays: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): np.asarray(v) for k, v in arrays.items()}
    _st_save(payload, str(path))


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    return dict(_st_load(str(path)))
