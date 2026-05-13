from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    key = str(name).lower()
    if key == "relu":
        return jax.nn.relu
    if key == "tanh":
        return jnp.tanh
    if key == "gelu":
        return jax.nn.gelu
    if key in {"silu", "swish"}:
        return jax.nn.silu
    if key == "sigmoid":
        return jax.nn.sigmoid
    if key == "linear":
        return lambda x: x
    if key == "squared_relu":
        return lambda x: jnp.square(jax.nn.relu(x))
    raise ValueError(f"Unknown activation '{name}'.")
