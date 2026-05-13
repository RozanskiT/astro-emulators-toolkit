from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from flax import nnx

from ...config.parsing import parse_bool
from ...models.activations import get_activation


@dataclass(frozen=True)
class MLP2DRegressionConfig:
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "gelu"
    use_bias: bool = True
    dtype: str = "float32"
    channels: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLP2DRegressionConfig":
        allowed = {"hidden_sizes", "activation", "use_bias", "dtype", "channels"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown MLP2DRegression params: {unknown}.")
        hs = d.get("hidden_sizes", (256, 256, 256))
        if isinstance(hs, list):
            hs = tuple(int(x) for x in hs)
        return cls(
            hidden_sizes=tuple(int(x) for x in hs),
            activation=str(d.get("activation", "gelu")),
            use_bias=parse_bool(d.get("use_bias", True), field_name="use_bias"),
            dtype=str(d.get("dtype", "float32")),
            channels=int(d.get("channels", 2)),
        )


class MLP2DRegression(nnx.Module):
    __data__ = ("layers",)

    def __init__(
        self, *, in_dim: int, out_dim: int, cfg: MLP2DRegressionConfig, rngs: nnx.Rngs
    ):
        if cfg.channels <= 0:
            raise ValueError(f"channels must be > 0, got {cfg.channels}.")

        dtype = jnp.dtype(cfg.dtype)
        self.out_dim = int(out_dim)
        self.channels = int(cfg.channels)
        sizes = (in_dim,) + tuple(cfg.hidden_sizes) + (self.out_dim * self.channels,)

        self.layers = nnx.List(
            [
                nnx.Linear(
                    sizes[i],
                    sizes[i + 1],
                    rngs=rngs,
                    use_bias=cfg.use_bias,
                    dtype=dtype,
                )
                for i in range(len(sizes) - 1)
            ]
        )

        self.activation = cfg.activation

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        act = get_activation(self.activation)
        for layer in self.layers[:-1]:
            x = act(layer(x))
        x = self.layers[-1](x)
        return x.reshape((x.shape[0], self.out_dim, self.channels))
