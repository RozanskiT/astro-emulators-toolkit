from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ...config.parsing import parse_bool


def _positive_int(value: Any, *, field_name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0, got {out}.")
    return out


def _positive_finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a positive finite float.")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field_name} must be > 0 and finite, got {value}.")
    return out


def _uniform_initializer(bound: float) -> Callable[..., jax.Array]:
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

    return init


@dataclass(frozen=True)
class SirenConfig:
    hidden_sizes: tuple[int, ...] = (128, 128, 128)
    omega0_first: float = 30.0
    omega0_hidden: float = 1.0
    use_bias: bool = True
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SirenConfig":
        allowed = {
            "hidden_sizes",
            "omega0_first",
            "omega0_hidden",
            "use_bias",
            "dtype",
        }
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown Siren params: {unknown}.")

        raw_hidden = d.get("hidden_sizes", (128, 128, 128))
        if isinstance(raw_hidden, list):
            raw_hidden = tuple(raw_hidden)
        hidden_sizes = tuple(
            _positive_int(value, field_name=f"hidden_sizes[{i}]")
            for i, value in enumerate(raw_hidden)
        )
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")

        dtype = str(d.get("dtype", "float32"))
        try:
            jnp.dtype(dtype)
        except TypeError as exc:
            raise ValueError(
                f"Siren dtype must be a valid JAX dtype, got {dtype!r}."
            ) from exc

        return cls(
            hidden_sizes=hidden_sizes,
            omega0_first=_positive_finite_float(
                d.get("omega0_first", 30.0), field_name="omega0_first"
            ),
            omega0_hidden=_positive_finite_float(
                d.get("omega0_hidden", 1.0), field_name="omega0_hidden"
            ),
            use_bias=parse_bool(d.get("use_bias", True), field_name="use_bias"),
            dtype=dtype,
        )


class Siren(nnx.Module):
    __data__ = ("layers",)

    def __init__(self, *, in_dim: int, out_dim: int, cfg: SirenConfig, rngs: nnx.Rngs):
        dtype = jnp.dtype(cfg.dtype)
        sizes = (int(in_dim),) + tuple(cfg.hidden_sizes) + (int(out_dim),)
        if sizes[0] <= 0 or sizes[-1] <= 0:
            raise ValueError(f"SIREN dimensions must be positive, got {sizes}.")

        self.layers: nnx.List = nnx.List()
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            if i == 0:
                bound = 1.0 / fan_in
            else:
                omega = cfg.omega0_hidden
                bound = math.sqrt(6.0 / fan_in) / omega
            self.layers.append(
                nnx.Linear(
                    sizes[i],
                    sizes[i + 1],
                    rngs=rngs,
                    use_bias=cfg.use_bias,
                    dtype=dtype,
                    kernel_init=_uniform_initializer(float(bound)),
                    bias_init=jax.nn.initializers.zeros,
                )
            )

        self.omega0_first = float(cfg.omega0_first)
        self.omega0_hidden = float(cfg.omega0_hidden)

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        for i, layer in enumerate(self.layers[:-1]):
            omega = self.omega0_first if i == 0 else self.omega0_hidden
            x = jnp.sin(omega * layer(x))
        return self.layers[-1](x)
