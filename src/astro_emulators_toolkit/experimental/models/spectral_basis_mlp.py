"""A coefficient MLP and a learned spectral basis for wavelength-explicit prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from flax import nnx

from ...models.activations import get_activation
from ...models.mlp import MLP, MLPConfig
from .mhe import _broadcast_spectral_inputs, _labels, _waves
from .mhe_config import JSONConfig, finite_float, positive_int, widths


@dataclass(frozen=True)
class SpectralBasisMLPConfig(JSONConfig):
    """Segments are (first coordinate, last coordinate, number of uniform knots)."""

    grid_segments: tuple[tuple[float, float, int], ...]
    rank: int = 128
    hidden_sizes: tuple[int, ...] = (256, 256, 256, 256)
    channels: int = 1
    activation: str = "silu"
    log10_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        positive_int("rank", self.rank)
        positive_int("channels", self.channels)
        object.__setattr__(
            self, "hidden_sizes", widths("hidden_sizes", self.hidden_sizes)
        )
        get_activation(self.activation)
        if self.log10_bounds is not None:
            lo, hi = (finite_float("log10 bound", v) for v in self.log10_bounds)
            if lo >= hi:
                raise ValueError("log10_bounds must be increasing")
            object.__setattr__(self, "log10_bounds", (lo, hi))
        segments: list[tuple[float, float, int]] = []
        for lo, hi, n in self.grid_segments:
            lo, hi = finite_float("segment start", lo), finite_float("segment end", hi)
            n = positive_int("segment size", n)
            if n < 2 or lo >= hi or (segments and segments[-1][1] >= lo):
                raise ValueError(
                    "Segments need at least two knots and strictly increasing, disjoint bounds"
                )
            segments.append((lo, hi, n))
        if not segments or self.rank > self.channels * sum(s[2] for s in segments):
            raise ValueError("grid_segments must provide at least rank spectral values")
        object.__setattr__(self, "grid_segments", tuple(segments))


class SpectralBasisMLP(nnx.Module):
    """Evaluate a low-rank spectrum at shared or row-specific wavelength queries."""

    def __init__(self, *, in_dim, out_dim, cfg: SpectralBasisMLPConfig, rngs):
        if not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "SpectralBasisMLP requires JAX_ENABLE_X64=1 for wavelength coordinates"
            )
        if out_dim != 1:
            raise ValueError("The wavelength axis is supplied at evaluation time")
        self.in_dim, self.config = in_dim, cfg
        self.head = MLP(
            in_dim=in_dim,
            out_dim=cfg.rank,
            cfg=MLPConfig(hidden_sizes=cfg.hidden_sizes, activation=cfg.activation),
            rngs=rngs,
        )
        self.knots = sum(n for _, _, n in cfg.grid_segments)
        size = self.knots * cfg.channels
        self.mean = nnx.Param(jnp.zeros(size, dtype=jnp.float32))
        self.basis = nnx.Param(
            jax.random.normal(rngs.params(), (cfg.rank, size), dtype=jnp.float32)
            / math.sqrt(cfg.rank)
        )

    def prepare_context(self, labels):
        return self.head(_labels(labels, self.in_dim, jnp.float32))

    def _decode(self, context):
        native = (context @ self.basis[...] + self.mean[...]).reshape(
            len(context), self.knots, self.config.channels
        )
        return native[..., 0] if self.config.channels == 1 else native

    def predict_native(self, labels, *, linear_intensity=False):
        """Evaluate all native knots in segment order."""
        native = self._decode(self.prepare_context(labels))
        bounds = self.config.log10_bounds
        if linear_intensity:
            if bounds is None:
                raise ValueError("Physical intensity requires log10_bounds")
            return jnp.power(jnp.float32(10), native)
        if bounds is not None:
            return (native - bounds[0]) / (bounds[1] - bounds[0])
        return native

    def predict_queries(self, context, wavelengths):
        native = self._decode(context)
        bounds = self.config.log10_bounds
        if bounds is None:
            return _interpolate(native, wavelengths, self.config.grid_segments)
        intensity = jnp.power(jnp.float32(10), native)
        y = _interpolate(intensity, wavelengths, self.config.grid_segments)
        return (jnp.log10(y) - bounds[0]) / (bounds[1] - bounds[0])

    def predict_one(self, context, wavelengths):
        w = _waves(wavelengths, len(context), one=True)
        return self.predict_queries(context, w[:, None])[:, 0]

    def __call__(self, x, *, train=False, rngs=None):
        labels, wavelengths = _broadcast_spectral_inputs(*x)
        return self.predict_queries(self.prepare_context(labels), wavelengths)


def _interpolate(native, wavelengths, segments):
    """Linear interpolation on disjoint uniform grids, with shared or per-row queries."""
    w = jnp.asarray(wavelengths, dtype=jnp.float64)
    if w.ndim != 1 and (w.ndim != 2 or w.shape[0] != len(native)):
        raise ValueError("queries must have shape (queries,) or (batch, queries)")
    starts = jnp.asarray([s[0] for s in segments], dtype=jnp.float64)
    ends = jnp.asarray([s[1] for s in segments], dtype=jnp.float64)
    lengths = jnp.asarray([s[2] for s in segments], dtype=jnp.int32)
    chip = jnp.searchsorted(ends[:-1], w, side="left", method="compare_all")
    lo = starts[chip]
    valid = (w >= lo) & (w <= ends[chip])
    u = (jnp.where(valid, w, lo) - lo) * ((lengths - 1) / (ends - starts))[chip]
    left = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, lengths[chip] - 2)
    index = (jnp.cumsum(lengths) - lengths)[chip] + left
    f = (u - left).astype(jnp.float32)
    if w.ndim == 1:
        a, b = native[:, index], native[:, index + 1]
    else:
        rows = jnp.arange(len(native))[:, None]
        a, b = native[rows, index], native[rows, index + 1]
    if native.ndim == 3:
        valid, f = valid[..., None], f[..., None]
    return jnp.where(valid, (1 - f) * a + f * b, jnp.nan)


def make_native_apply(emu, *, linear_intensity=False):
    """Freeze the native-grid mapping for an enclosing JAX calculation.

    Accepts canonical labels (batch, parameter_dim). Returns an array in
    segment order, with the same output convention as the canonical leaf.
    Set linear_intensity=True to exponentiate a log-intensity decoder directly.
    """
    if emu.cfg.model.name != "experimental/spectral_basis_mlp" or emu.graphdef is None:
        raise ValueError(
            "Load or initialize an experimental/spectral_basis_mlp emulator first"
        )
    state = nnx.merge_state(emu.params, emu.model_state)
    native = nnx.call((emu.graphdef, state)).core_model.predict_native

    def apply(labels):
        value, _ = native(labels, linear_intensity=linear_intensity)
        return value

    return apply
