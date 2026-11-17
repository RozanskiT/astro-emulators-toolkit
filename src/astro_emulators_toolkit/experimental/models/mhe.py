"""MHE networks with a shared parameter trunk and independent wavelength queries.

Adapted from exojax_emulator.mhe_models. Trained parameter names and numerical
operations are retained for reference comparisons.
"""

from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx

from .mhe_config import MHETransformerPayneConfig, MHEMLPConfig, positive_int
from .mhe_layers import (
    MHEHashEncoding,
    MHEWavelengthEncoding,
    activation,
    linear,
    rms_norm,
)


def _validate_dims(in_dim, out_dim):
    positive_int("in_dim", in_dim)
    if out_dim != 1:
        raise ValueError(
            "MHE models require output init size 1; wavelength is an explicit input axis."
        )


def _labels(labels, in_dim, dtype):
    labels = jnp.asarray(labels, dtype=dtype)
    if labels.ndim != 2 or labels.shape[1] != in_dim:
        raise ValueError(f"parameters must have shape (batch, {in_dim})")
    return labels


def _waves(waves, batch, *, one=False):
    # Match TransformerPayne's float64 coordinate path. Callers must preserve
    # precision before this point; promoting an already rounded array cannot restore it.
    waves = jnp.asarray(waves, dtype=jnp.float64)
    if one:
        if waves.ndim == 0:
            waves = jnp.broadcast_to(waves, (batch,))
        if waves.shape != (batch,):
            raise ValueError(
                "predict_one expects a scalar or one wavelength per atmosphere"
            )
    else:
        if waves.ndim == 1:
            waves = jnp.broadcast_to(waves, (batch, waves.shape[0]))
        if waves.ndim != 2 or waves.shape[0] != batch:
            raise ValueError("queries must have shape (queries,) or (batch, queries)")
    return waves


class TransformerContext(NamedTuple):
    """Differentiable per-block K/V without a wavelength axis."""

    keys_values: tuple


class CrossAttention(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.heads, self.head_dim, self.alpha_att = (
            (cfg.dim // cfg.dim_head),
            cfg.dim_head,
            cfg.alpha_att,
        )
        options = dict(
            rngs=rngs,
            bias=cfg.bias_attention,
            dtype=cfg.dtype,
            initializer=cfg.kernel_init,
            sigma=cfg.sigma,
        )
        self.q = linear(cfg.dim, cfg.dim, zero=cfg.init_att_q == "zero", **options)
        self.k = linear(cfg.dim, cfg.dim, **options)
        self.v = linear(cfg.dim, cfg.dim, **options)
        self.out = linear(cfg.dim, cfg.dim, zero=cfg.init_att_o == "zero", **options)

    def prepare_context(self, tokens):
        shape = (*tokens.shape[:-1], self.heads, self.head_dim)
        return self.k(tokens).reshape(shape), self.v(tokens).reshape(shape)

    def __call__(self, queries, context):
        k, v = context
        q = self.q(queries).reshape(*queries.shape[:-1], self.heads, self.head_dim)
        scores = jnp.einsum("bqhd,bthd->bhqt", q, k) / self.head_dim
        if self.alpha_att != 1.0:
            scores = scores * self.alpha_att
        values = jnp.einsum("bhqt,bthd->bqhd", jax.nn.softmax(scores, axis=-1), v)
        return self.out(values.reshape(queries.shape))


class Block(nnx.Module):
    def __init__(self, cfg, width, *, rngs):
        self.attention = CrossAttention(cfg, rngs=rngs)
        options = dict(
            rngs=rngs,
            bias=cfg.bias("feed_forward"),
            dtype=cfg.dtype,
            initializer=cfg.kernel_init,
            sigma=cfg.sigma,
        )
        self.ff_in = linear(cfg.dim, width, **options)
        self.ff_out = linear(width, cfg.dim, zero=cfg.ff_init == "zero", **options)
        self.activation, self.epsilon, self.residual_scale = (
            cfg.activation,
            cfg.rms_epsilon,
            cfg.residual_scale,
        )

    def __call__(self, h, context):
        update = self.attention(rms_norm(h, self.epsilon), context)
        h = h + (update if self.residual_scale == 1.0 else self.residual_scale * update)
        update = self.ff_out(
            activation(self.activation, self.ff_in(rms_norm(h, self.epsilon)))
        )
        return h + (
            update if self.residual_scale == 1.0 else self.residual_scale * update
        )


class MHETransformerPayne(nnx.Module):
    """Shared atmospheric tokens/KV, with independent wavelength predictions.

    Inputs: labels (B,D), float64 wavelength queries (Q,) or (B,Q).
    Outputs: (B,Q) or (B,Q,C). Rebuild prepared context if labels or weights change.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        cfg: MHETransformerPayneConfig,
        rngs: nnx.Rngs,
    ):
        _validate_dims(in_dim, out_dim)
        self.in_dim = in_dim
        self.config = cfg
        self.dim, self.parameter_tokens = cfg.dim, cfg.no_tokens
        options = dict(
            rngs=rngs, dtype=cfg.dtype, initializer=cfg.kernel_init, sigma=cfg.sigma
        )
        width = cfg.parameter_hidden_dim or cfg.dim
        self.label_in = linear(
            in_dim, width, bias=cfg.bias("parameter_embedding"), **options
        )
        self.label_out = linear(
            width,
            cfg.no_tokens * cfg.dim,
            bias=cfg.bias("parameter_embedding"),
            zero=cfg.emb_init == "zero",
            **options,
        )
        widths = (
            cfg.ff_hidden_sizes or (cfg.dim * cfg.dim_ff_multiplier,) * cfg.no_layers
        )
        self.blocks = nnx.List([Block(cfg, width, rngs=rngs) for width in widths])
        width = cfg.head_hidden_dim or cfg.dim
        self.head_in = linear(cfg.dim, width, bias=cfg.bias("output_head"), **options)
        self.head_out = linear(
            width,
            cfg.channels,
            bias=cfg.bias("output_head"),
            zero=cfg.head_init == "zero",
            **options,
        )
        self.wavelength_encoder = MHEWavelengthEncoding(
            cfg.dim,
            rngs=rngs,
            min_period=cfg.min_period,
            max_period=cfg.max_period,
            cfg=cfg.mhe,
            dtype=cfg.dtype,
        )

    def encode_wavelength(self, wavelengths):
        return self.wavelength_encoder(wavelengths)

    def prepare_context(self, labels):
        labels = _labels(labels, self.in_dim, self.config.dtype)
        tokens = self.label_out(
            activation(self.config.activation, self.label_in(labels))
        )
        tokens = tokens.reshape(labels.shape[0], self.parameter_tokens, self.dim)
        if self.config.alpha_emb != 1.0:
            tokens = self.config.alpha_emb * tokens
        tokens = rms_norm(tokens, self.config.rms_epsilon)
        return TransformerContext(
            tuple(block.attention.prepare_context(tokens) for block in self.blocks)
        )

    def predict_one(self, context, wavelengths):
        """One wavelength per atmosphere, with the trunk already evaluated."""
        if len(context.keys_values) != len(self.blocks):
            raise ValueError("Context must contain K/V for every transformer block")
        wave = _waves(wavelengths, context.keys_values[0][0].shape[0], one=True)
        h = self.encode_wavelength(wave[:, None])
        for block, kv in zip(self.blocks, context.keys_values):
            h = block(h, kv)
        h = self.head_out(
            activation(
                self.config.activation,
                self.head_in(rms_norm(h, self.config.rms_epsilon)),
            )
        )
        h = activation(self.config.output_activation, h)[:, 0, :]
        return h[:, 0] if self.config.channels == 1 else h

    def predict_queries(self, context, wavelengths):
        if len(context.keys_values) != len(self.blocks):
            raise ValueError("Context must contain K/V for every transformer block")
        waves = _waves(wavelengths, context.keys_values[0][0].shape[0])
        return jax.vmap(
            lambda wave: self.predict_one(context, wave), in_axes=1, out_axes=1
        )(waves)

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        labels, wavelengths = x
        return self.predict_queries(self.prepare_context(labels), wavelengths)


class MHEMLP(nnx.Module):
    """Shared parameter MLP; MHE features enter each vmapped query layer.

    GELU is the default; alternative activations are explicit ablations.
    Context and query shape conventions match MHETransformerPayne.
    """

    def __init__(self, *, in_dim: int, out_dim: int, cfg: MHEMLPConfig, rngs: nnx.Rngs):
        _validate_dims(in_dim, out_dim)
        self.in_dim = in_dim
        self.config = cfg
        assert cfg.mhe is not None and cfg.parameter_hidden_sizes is not None
        self.wavelength_encoder = MHEHashEncoding(
            rngs=rngs,
            min_period=cfg.min_period,
            max_period=cfg.max_period,
            cfg=cfg.mhe,
            dtype=cfg.dtype,
        )
        features = cfg.mhe.levels * cfg.mhe.features_per_level
        options = dict(
            rngs=rngs, dtype=cfg.dtype, initializer=cfg.kernel_init, sigma=cfg.sigma
        )
        sizes = (in_dim, *cfg.parameter_hidden_sizes)
        self.parameter_layers = nnx.List(
            [
                linear(a, b, bias=cfg.bias_parameter_embedding, **options)
                for a, b in zip(sizes[:-1], sizes[1:])
            ]
        )
        sizes = (cfg.parameter_hidden_sizes[-1], *cfg.hidden_sizes)
        self.layers = nnx.List(
            [
                linear(a + features, b, bias=cfg.bias_decoder, **options)
                for a, b in zip(sizes[:-1], sizes[1:])
            ]
            + [
                linear(
                    cfg.hidden_sizes[-1],
                    cfg.channels,
                    bias=cfg.bias_output_head,
                    **options,
                )
            ]
        )

    def encode_parameters(self, labels):
        h = _labels(labels, self.in_dim, self.config.dtype)
        for layer in self.parameter_layers:
            h = activation(self.config.activation, layer(h))
        return h

    def prepare_context(self, labels):
        return self.encode_parameters(labels)

    def predict_one(self, context, wavelengths):
        wave = _waves(wavelengths, context.shape[0], one=True)
        features, h = self.wavelength_encoder(wave[:, None]), context[:, None, :]
        for layer in self.layers[:-1]:
            h = activation(
                self.config.activation, layer(jnp.concatenate((h, features), axis=-1))
            )
        h = activation(self.config.output_activation, self.layers[-1](h))[:, 0, :]
        return h[:, 0] if self.config.channels == 1 else h

    def predict_queries(self, context, wavelengths):
        waves = _waves(wavelengths, context.shape[0])
        return jax.vmap(
            lambda wave: self.predict_one(context, wave), in_axes=1, out_axes=1
        )(waves)

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        labels, wavelengths = x
        return self.predict_queries(self.prepare_context(labels), wavelengths)
