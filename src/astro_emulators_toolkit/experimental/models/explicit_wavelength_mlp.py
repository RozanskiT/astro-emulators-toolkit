from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ...config.parsing import parse_bool
from ...models.activations import get_activation


def _logspace_frequencies(
    n: int, f_min: float, f_max: float, dtype: jnp.dtype
) -> jnp.ndarray:
    if n <= 0:
        raise ValueError(f"number of frequencies must be > 0, got {n}.")
    if f_min <= 0.0 or f_max <= 0.0:
        raise ValueError(
            f"frequency bounds must be > 0, got f_min={f_min}, f_max={f_max}."
        )
    if f_max < f_min:
        raise ValueError(
            f"frequency max must be >= min, got f_min={f_min}, f_max={f_max}."
        )
    return jnp.exp(jnp.linspace(jnp.log(f_min), jnp.log(f_max), n, dtype=dtype))


@dataclass(frozen=True)
class ExplicitWavelengthMLPConfig:
    wavelength_embedding_dim: int = 32
    wavelength_hidden_dim: int = 64
    parameter_hidden_dim: int = 128
    joint_hidden_dim: int = 128
    activation: str = "gelu"
    use_bias: bool = True
    dtype: str = "float32"
    channels: int = 1
    frequency_min: float = 0.1
    frequency_max: float = 100.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExplicitWavelengthMLPConfig":
        allowed = {
            "wavelength_embedding_dim",
            "wavelength_hidden_dim",
            "parameter_hidden_dim",
            "joint_hidden_dim",
            "activation",
            "use_bias",
            "dtype",
            "channels",
            "frequency_min",
            "frequency_max",
        }
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown ExplicitWavelengthMLP params: {unknown}.")
        return cls(
            wavelength_embedding_dim=int(d.get("wavelength_embedding_dim", 32)),
            wavelength_hidden_dim=int(d.get("wavelength_hidden_dim", 64)),
            parameter_hidden_dim=int(d.get("parameter_hidden_dim", 128)),
            joint_hidden_dim=int(d.get("joint_hidden_dim", 128)),
            activation=str(d.get("activation", "gelu")),
            use_bias=parse_bool(d.get("use_bias", True), field_name="use_bias"),
            dtype=str(d.get("dtype", "float32")),
            channels=int(d.get("channels", 1)),
            frequency_min=float(d.get("frequency_min", 0.1)),
            frequency_max=float(d.get("frequency_max", 100.0)),
        )


class ExplicitWavelengthMLP(nnx.Module):
    __data__ = (
        "wavelength_proj",
        "param_proj_1",
        "param_proj_2",
        "joint_proj_1",
        "joint_proj_2",
    )

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        cfg: ExplicitWavelengthMLPConfig,
        rngs: nnx.Rngs,
    ):
        if cfg.wavelength_embedding_dim <= 0 or cfg.wavelength_embedding_dim % 2 != 0:
            raise ValueError(
                "wavelength_embedding_dim must be a positive even integer."
            )
        if cfg.channels <= 0:
            raise ValueError(f"channels must be > 0, got {cfg.channels}.")
        if out_dim != 1:
            raise ValueError(
                "explicit_wavelength_mlp requires output init size 1 because wavelength is an explicit input axis."
            )

        self.channels = int(cfg.channels)
        dtype = jnp.dtype(cfg.dtype)
        n_freq = cfg.wavelength_embedding_dim // 2
        freq = _logspace_frequencies(
            n_freq, cfg.frequency_min, cfg.frequency_max, dtype
        )
        self.frequencies = tuple(float(x) for x in freq)
        self.activation = cfg.activation.lower()

        self.wavelength_proj = nnx.Linear(
            cfg.wavelength_embedding_dim,
            cfg.wavelength_hidden_dim,
            rngs=rngs,
            use_bias=cfg.use_bias,
            dtype=dtype,
        )
        self.param_proj_1 = nnx.Linear(
            in_dim,
            cfg.parameter_hidden_dim,
            rngs=rngs,
            use_bias=cfg.use_bias,
            dtype=dtype,
        )
        self.param_proj_2 = nnx.Linear(
            cfg.parameter_hidden_dim,
            cfg.parameter_hidden_dim,
            rngs=rngs,
            use_bias=cfg.use_bias,
            dtype=dtype,
        )
        self.joint_proj_1 = nnx.Linear(
            cfg.wavelength_hidden_dim + cfg.parameter_hidden_dim,
            cfg.joint_hidden_dim,
            rngs=rngs,
            use_bias=cfg.use_bias,
            dtype=dtype,
        )
        self.joint_proj_2 = nnx.Linear(
            cfg.joint_hidden_dim,
            self.channels,
            rngs=rngs,
            use_bias=cfg.use_bias,
            dtype=dtype,
        )

    def _act(self, x: jnp.ndarray) -> jnp.ndarray:
        return get_activation(self.activation)(x)

    def _predict_scalar_wavelength(
        self, wavelength_scalar: jnp.ndarray, params: jnp.ndarray
    ) -> jnp.ndarray:
        frequencies = jnp.asarray(self.frequencies, dtype=wavelength_scalar.dtype)
        phases = wavelength_scalar * frequencies
        w_emb = jnp.concatenate((jnp.sin(phases), jnp.cos(phases)), axis=-1)
        w_feat = self.wavelength_proj(w_emb)

        p_feat = self._act(self.param_proj_1(params))
        p_feat = self.param_proj_2(p_feat)

        joint = jnp.concatenate((w_feat, p_feat), axis=-1)
        hidden = self._act(self.joint_proj_1(joint))
        out = self.joint_proj_2(hidden)
        if self.channels == 1:
            return out[0]
        return out

    def _predict_sample(
        self, wavelengths: jnp.ndarray, params: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.vmap(self._predict_scalar_wavelength, in_axes=(0, None))(
            wavelengths, params
        )

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        params, wavelengths = x
        params = jnp.asarray(params)
        wavelengths = jnp.asarray(wavelengths)
        return jax.vmap(self._predict_sample, in_axes=(0, 0))(wavelengths, params)
