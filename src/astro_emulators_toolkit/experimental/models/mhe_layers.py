"""Multiresolution hash interpolation, adapted from exojax_emulator.mhe_layers."""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from .mhe_config import MHEProjectionConfig, finite_float


def activation(name, x):
    return {
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "silu": jax.nn.silu,
        "tanh": jnp.tanh,
        "linear": lambda x: x,
    }[name](x)


def linear(in_dim, out_dim, *, rngs, bias, dtype, initializer, sigma=1.0, zero=False):
    init = (
        nnx.initializers.zeros_init()
        if zero
        else {
            "lecun_normal": nnx.initializers.lecun_normal,
            "glorot_uniform": nnx.initializers.glorot_uniform,
            "he_normal": nnx.initializers.he_normal,
            "zeros": nnx.initializers.zeros_init,
        }[initializer]()
    )
    if sigma != 1.0:
        base = init

        def init(key, shape, dtype):
            return sigma * base(key, shape, dtype)

    return nnx.Linear(
        in_dim,
        out_dim,
        use_bias=bias,
        kernel_init=init,
        param_dtype=jnp.dtype(dtype),
        dtype=jnp.dtype(dtype),
        rngs=rngs,
    )


def rms_norm(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


class MHEHashEncoding(nnx.Module):
    """Float64 wavelength coordinates -> interpolated features from one packed table.

    Names, layout, addressing, and default initialization match the original.
    """

    def __init__(self, *, rngs, min_period, max_period, cfg, dtype="float32"):
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("MHE requires JAX_ENABLE_X64=1")
        finite_float("min_period", min_period, positive=True)
        finite_float("max_period", max_period, positive=True)
        if min_period > max_period:
            raise ValueError("min_period must be <= max_period")
        self.dtype = jnp.dtype(dtype)
        self.levels, self.features_per_level = cfg.levels, cfg.features_per_level
        self.wavelength_range = cfg.wavelength_range
        lo, hi = self.wavelength_range
        self.cell_widths = tuple(
            np.geomspace(max_period / 2, min_period / 2, cfg.levels)
        )
        vertices = [math.ceil((hi - lo) / width) + 1 for width in self.cell_widths]
        if max(vertices) >= 2**31 - 1:
            raise ValueError("MHE range/min_period exceeds int32 grid indexing")
        self.table_sizes = tuple(
            1 << min((n - 1).bit_length(), cfg.log2_hashmap_size) for n in vertices
        )
        if sum(self.table_sizes) >= 2**31:
            raise ValueError("Packed MHE tables exceed int32 indexing")
        self.offsets = tuple(int(v) for v in np.cumsum((0, *self.table_sizes[:-1])))
        self.table = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (sum(self.table_sizes), cfg.features_per_level),
                dtype=self.dtype,
                minval=-cfg.table_init_scale,
                maxval=cfg.table_init_scale,
            )
        )

    def hash_features(self, wavelengths):
        wave = jnp.asarray(wavelengths)
        if wave.dtype != jnp.float64:
            raise TypeError("MHE wavelengths must be float64")
        position = (wave[..., None] - self.wavelength_range[0]) / jnp.asarray(
            self.cell_widths, dtype=jnp.float64
        )
        lower = jnp.floor(position)
        fraction = (position - lower).astype(self.dtype)[..., None]
        lower = lower.astype(jnp.int32)
        mask = jnp.asarray(self.table_sizes, dtype=jnp.int32) - 1
        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)
        # Bundle readers may supply NumPy weights; gather must remain traceable
        # inside the wavelength vmap, including eager predict after loading.
        table = jnp.asarray(self.table[...])
        left = table[(lower & mask) + offsets]
        right = table[((lower + 1) & mask) + offsets]
        return (left + fraction * (right - left)).reshape(
            *wave.shape, self.levels * self.features_per_level
        )

    def __call__(self, wavelengths):
        return self.hash_features(wavelengths)


class MHEWavelengthEncoding(MHEHashEncoding):
    def __init__(self, dim, *, rngs, min_period, max_period, cfg, dtype="float32"):
        cfg = MHEProjectionConfig.from_dict(cfg)
        super().__init__(
            rngs=rngs,
            min_period=min_period,
            max_period=max_period,
            cfg=cfg,
            dtype=dtype,
        )
        hidden = cfg.projection_hidden_dim or dim
        options = dict(
            rngs=rngs,
            bias=cfg.projection_bias,
            dtype=dtype,
            initializer=cfg.projection_kernel_init,
            sigma=cfg.projection_sigma,
        )
        self.linear_in = linear(cfg.levels * cfg.features_per_level, hidden, **options)
        self.linear_out = linear(hidden, dim, **options)
        self.projection_activation = cfg.projection_activation
        self.normalize_output = cfg.normalize_output
        self.output_rms_epsilon = cfg.output_rms_epsilon
        self.output_rms = cfg.output_rms
        if cfg.normalize_output and cfg.output_rms is None:
            # Preserve the original deterministic, coordinate-only normalization.
            probe = jnp.asarray(
                np.random.default_rng(0).uniform(
                    cfg.wavelength_range[0], cfg.wavelength_range[1], 4096
                ),
                dtype=jnp.float64,
            )
            periods = jnp.logspace(
                jnp.log10(min_period), jnp.log10(max_period), dim, dtype=jnp.float64
            )
            reference = jnp.sin(2 * jnp.pi * probe[:, None] / periods).astype(
                self.dtype
            )
            self.output_rms = float(jnp.sqrt(jnp.mean(reference**2)))

    def project(self, wavelengths):
        return self.linear_out(
            activation(
                self.projection_activation,
                self.linear_in(self.hash_features(wavelengths)),
            )
        )

    def __call__(self, wavelengths):
        features = self.project(wavelengths)
        if self.normalize_output:
            features = features * (
                self.output_rms
                * jax.lax.rsqrt(
                    jnp.mean(features**2, axis=-1, keepdims=True)
                    + self.output_rms_epsilon
                )
            )
        return features
