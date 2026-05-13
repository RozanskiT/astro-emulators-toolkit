from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def to_jax_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, dict):
        return {k: to_jax_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_jax_tree(v) for v in value)
    if isinstance(value, list):
        return [to_jax_tree(v) for v in value]
    return value


def to_numpy_tree(tree: Any) -> Any:
    host_tree = jax.device_get(tree)
    return jax.tree_util.tree_map(np.asarray, host_tree)


def logs_device_to_python(logs_device: dict[str, Any]) -> dict[str, Any]:
    logs_host = jax.device_get(logs_device)
    logs_py: dict[str, Any] = {}
    for k, v in logs_host.items():
        try:
            logs_py[k] = float(v)
        except Exception:
            logs_py[k] = v
    return logs_py
