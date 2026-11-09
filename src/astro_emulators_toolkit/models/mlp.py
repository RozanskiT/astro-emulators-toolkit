# src/astro_emulators_toolkit/models/mlp.py
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import jax.numpy as jnp
from flax import nnx

from ..config.parsing import parse_bool
from .activations import get_activation
from .dtypes import parse_dtype_name, resolve_param_dtype


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "gelu"
    output_activation: str = "linear"
    use_bias: bool = True
    reference_width: int | None = None
    reference_depth: int | None = None
    # compute dtype
    dtype: str = "float32"
    # parameter storage dtype; None means "same as dtype"
    param_dtype: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLPConfig":
        allowed = {
            "hidden_sizes",
            "activation",
            "output_activation",
            "use_bias",
            "reference_width",
            "reference_depth",
            "dtype",
            "param_dtype",
        }
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown MLP params: {unknown}.")
        hs = d.get("hidden_sizes", (256, 256, 256))
        if isinstance(hs, list):
            hs = tuple(int(x) for x in hs)
        hidden_sizes = tuple(int(x) for x in hs)
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("MLP hidden_sizes entries must all be > 0.")
        activation = str(d.get("activation", "gelu"))
        get_activation(activation)
        output_activation = str(d.get("output_activation", "linear"))
        get_activation(output_activation)
        reference_width = _normalize_optional_positive_int(
            d.get("reference_width", None), field_name="reference_width"
        )
        reference_depth = _normalize_optional_positive_int(
            d.get("reference_depth", None), field_name="reference_depth"
        )
        dtype = parse_dtype_name(
            d.get("dtype", "float32"), field_name="dtype", owner="MLP"
        )
        param_dtype = resolve_param_dtype(d, dtype=dtype, owner="MLP")
        return cls(
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            use_bias=parse_bool(d.get("use_bias", True), field_name="use_bias"),
            reference_width=reference_width,
            reference_depth=reference_depth,
            dtype=dtype,
            param_dtype=param_dtype,
        )


class MLP(nnx.Module):
    __data__ = ("layers", "activation", "output_activation")

    def __init__(self, *, in_dim: int, out_dim: int, cfg: MLPConfig, rngs: nnx.Rngs):
        dtype = jnp.dtype(cfg.dtype)
        param_dtype = jnp.dtype(cfg.param_dtype or cfg.dtype)
        sizes = (in_dim,) + tuple(cfg.hidden_sizes) + (out_dim,)

        self.layers = nnx.List(
            [
                nnx.Linear(
                    sizes[i],
                    sizes[i + 1],
                    rngs=rngs,
                    use_bias=cfg.use_bias,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
                for i in range(len(sizes) - 1)
            ]
        )

        self.activation = cfg.activation
        self.output_activation = cfg.output_activation

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        act = get_activation(self.activation)
        for layer in self.layers[:-1]:
            x = act(layer(x))
        return get_activation(self.output_activation)(self.layers[-1](x))


def _normalize_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer or None.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return normalized
