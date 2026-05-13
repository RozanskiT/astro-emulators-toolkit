from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config.parsing import parse_bool


@dataclass(frozen=True)
class CannonConfig:
    include_bias: bool = True
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CannonConfig":
        allowed = {"include_bias", "dtype"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown Cannon params: {unknown}.")
        return cls(
            include_bias=parse_bool(
                d.get("include_bias", True), field_name="include_bias"
            ),
            dtype=str(d.get("dtype", "float32")),
        )


def cannon_feature_dim(in_dim: int, *, include_bias: bool = True) -> int:
    base = int(in_dim) + (int(in_dim) * (int(in_dim) + 1)) // 2
    return base + (1 if include_bias else 0)


def cannon_design_matrix(x: jnp.ndarray, *, include_bias: bool = True) -> jnp.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Cannon expects x.ndim == 2, got shape={x.shape}.")

    n_samples, in_dim = x.shape
    tri_i, tri_j = np.triu_indices(int(in_dim))
    quadratic = jnp.einsum("bi,bj->bij", x, x)[:, tri_i, tri_j]

    columns = [x, quadratic]
    if include_bias:
        columns.insert(0, jnp.ones((n_samples, 1), dtype=x.dtype))
    return jnp.concatenate(columns, axis=1)


class Cannon(nnx.Module):
    def __init__(self, *, in_dim: int, out_dim: int, cfg: CannonConfig, rngs: nnx.Rngs):
        del rngs
        self.include_bias = cfg.include_bias
        dtype = jnp.dtype(cfg.dtype)
        feat_dim = cannon_feature_dim(in_dim, include_bias=self.include_bias)
        self.coefficients = nnx.Param(jnp.zeros((feat_dim, out_dim), dtype=dtype))

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        phi = cannon_design_matrix(x, include_bias=self.include_bias)
        return phi @ self.coefficients[...]
