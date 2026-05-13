from __future__ import annotations

import math
from numbers import Integral, Real
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ..config.parsing import parse_bool


def derive_transformer_payne_channel_semantics(
    output_names: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    semantics: list[dict[str, str]] = []
    for name in output_names:
        dataset_key = (
            name[9:] if name.startswith("log_flux_") and len(name) > 9 else name
        )
        semantics.append({"name": str(name), "dataset_key": dataset_key})
    return tuple(semantics)


def _act(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    from .activations import get_activation

    return get_activation(name)


_ALLOWED_TRANSFORMER_INIT_NAMES = {"si", "zero"}


def _normalize_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return normalized


def _normalize_finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite float.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _validate_transformer_activation(value: Any, *, field_name: str) -> str:
    activation = str(value)
    _act(activation)
    return activation


def _validate_transformer_init_name(value: Any, *, field_name: str) -> str:
    name = str(value).lower()
    if name not in _ALLOWED_TRANSFORMER_INIT_NAMES:
        raise ValueError(
            f"{field_name} must be one of {sorted(_ALLOWED_TRANSFORMER_INIT_NAMES)}, got {value!r}."
        )
    return name


def _parse_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return parse_bool(value, field_name=field_name)


def _truncated_normal(
    key: jax.Array, shape: tuple[int, ...], stddev: float, dtype: jnp.dtype
) -> jax.Array:
    # Match variance used in the original low-level implementation.
    truncation_correction = jnp.asarray(0.87962566103423978, dtype=dtype)
    x = jax.random.truncated_normal(
        key, lower=-2.0, upper=2.0, shape=shape, dtype=dtype
    )
    return x * (jnp.asarray(stddev, dtype=dtype) / truncation_correction)


def _rms_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean2 + eps)


def _frequency_encoding(
    wavelengths: jnp.ndarray,
    *,
    min_period: float,
    max_period: float,
    dim: int,
    wavelength_dtype: jnp.dtype,
    output_dtype: jnp.dtype,
) -> jnp.ndarray:
    wavelengths = jnp.asarray(wavelengths, dtype=wavelength_dtype)
    periods = jnp.logspace(
        jnp.log10(min_period), jnp.log10(max_period), num=dim, dtype=wavelength_dtype
    )
    phase = (
        jnp.asarray(2.0 * jnp.pi, dtype=wavelength_dtype) * wavelengths[..., None]
    ) / periods[None, None, :]
    return jnp.sin(phase).astype(output_dtype)


class _LinearEinsum(nnx.Module):
    __data__ = ("w", "b")

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        rngs: nnx.Rngs,
        sigma: float,
        use_bias: bool,
        dtype: jnp.dtype,
        zero_init: bool = False,
        name: str = "",
    ):
        del name
        if zero_init:
            w = jnp.zeros((in_dim, out_dim), dtype=dtype)
        else:
            stddev = float(sigma) * (in_dim**-0.5)
            w = _truncated_normal(
                rngs.params(), (in_dim, out_dim), stddev=stddev, dtype=dtype
            )
        self.w = nnx.Param(w)
        self.b = nnx.Param(jnp.zeros((out_dim,), dtype=dtype)) if use_bias else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.einsum("...i,ij->...j", x, self.w[...])
        if self.b is not None:
            y = y + self.b[...]
        return y


class _ParameterEmbedding(nnx.Module):
    __data__ = ("w0", "w1", "b0", "b1")

    def __init__(
        self,
        *,
        in_dim: int,
        dim: int,
        no_tokens: int,
        activation: str,
        use_bias: bool,
        emb_init: str,
        sigma: float,
        alpha_emb: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype,
    ):
        self.dim = int(dim)
        self.no_tokens = int(no_tokens)
        self.alpha_emb = float(alpha_emb)
        self.activation = activation

        stddev0 = sigma * (in_dim**-0.5)
        stddev1 = sigma * (dim**-0.5)

        w0 = _truncated_normal(rngs.params(), (in_dim, dim), stddev0, dtype)
        if emb_init == "zero":
            w1 = jnp.zeros((dim, no_tokens * dim), dtype=dtype)
        else:
            w1 = _truncated_normal(
                rngs.params(), (dim, no_tokens * dim), stddev1, dtype
            )

        self.w0 = nnx.Param(w0)
        self.w1 = nnx.Param(w1)
        self.b0 = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else None
        self.b1 = (
            nnx.Param(jnp.zeros((no_tokens * dim,), dtype=dtype)) if use_bias else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = _act(self.activation)
        p = jnp.einsum("bi,ij->bj", x, self.w0[...])
        if self.b0 is not None:
            p = p + self.b0[...]
        p = act(p)

        p = jnp.einsum("bi,ij->bj", p, self.w1[...])
        if self.b1 is not None:
            p = p + self.b1[...]

        p = p.reshape((x.shape[0], self.no_tokens, self.dim))
        return self.alpha_emb * p


class _FeedForward(nnx.Module):
    __data__ = ("in_proj", "out_proj")

    def __init__(
        self,
        *,
        dim: int,
        dim_ff_multiplier: int,
        activation: str,
        use_bias: bool,
        ff_init: str,
        sigma: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype,
    ):
        hidden_dim = int(dim * dim_ff_multiplier)
        self.activation = activation
        self.in_proj = _LinearEinsum(
            in_dim=dim,
            out_dim=hidden_dim,
            rngs=rngs,
            sigma=sigma,
            use_bias=use_bias,
            dtype=dtype,
        )
        self.out_proj = _LinearEinsum(
            in_dim=hidden_dim,
            out_dim=dim,
            rngs=rngs,
            sigma=sigma,
            use_bias=use_bias,
            dtype=dtype,
            zero_init=(ff_init == "zero"),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.out_proj(_act(self.activation)(self.in_proj(x)))


class _CrossAttention(nnx.Module):
    __data__ = ("wq", "wk", "wv", "wo", "bq", "bk", "bv", "bo")

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int,
        use_bias: bool,
        init_att_q: str,
        init_att_o: str,
        sigma: float,
        alpha_att: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype,
    ):
        if dim % dim_head != 0:
            raise ValueError(
                f"dim must be divisible by dim_head, got dim={dim}, dim_head={dim_head}."
            )
        self.num_heads = dim // dim_head
        self.dim_head = int(dim_head)
        self.alpha_att = float(alpha_att)

        stddev = sigma * (dim**-0.5)

        def _zero_init(_key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
            return jnp.zeros(shape, dtype=dtype)

        q_init = (
            _zero_init
            if init_att_q == "zero"
            else lambda key, shape: _truncated_normal(key, shape, stddev, dtype)
        )
        o_init = (
            _zero_init
            if init_att_o == "zero"
            else lambda key, shape: _truncated_normal(key, shape, stddev, dtype)
        )

        self.wq = nnx.Param(q_init(rngs.params(), (dim, self.num_heads, self.dim_head)))
        self.wk = nnx.Param(
            _truncated_normal(
                rngs.params(), (dim, self.num_heads, self.dim_head), stddev, dtype
            )
        )
        self.wv = nnx.Param(
            _truncated_normal(
                rngs.params(), (dim, self.num_heads, self.dim_head), stddev, dtype
            )
        )
        self.wo = nnx.Param(o_init(rngs.params(), (self.num_heads, self.dim_head, dim)))

        self.bq = (
            nnx.Param(jnp.zeros((self.num_heads, self.dim_head), dtype=dtype))
            if use_bias
            else None
        )
        self.bk = (
            nnx.Param(jnp.zeros((self.num_heads, self.dim_head), dtype=dtype))
            if use_bias
            else None
        )
        self.bv = (
            nnx.Param(jnp.zeros((self.num_heads, self.dim_head), dtype=dtype))
            if use_bias
            else None
        )
        self.bo = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else None

    def __call__(self, q_in: jnp.ndarray, kv_in: jnp.ndarray) -> jnp.ndarray:
        q = jnp.einsum("bli,ihd->blhd", q_in, self.wq[...])
        k = jnp.einsum("bti,ihd->bthd", kv_in, self.wk[...])
        v = jnp.einsum("bti,ihd->bthd", kv_in, self.wv[...])

        if self.bq is not None and self.bk is not None and self.bv is not None:
            q = q + self.bq[...]
            k = k + self.bk[...]
            v = v + self.bv[...]

        # Non-standard scaling O(1/dim_head) is intentional and follows from Linge (2025; 2404.05728)
        scale = jnp.asarray(self.dim_head**-0.5, dtype=q.dtype)
        attn_scores = jnp.einsum(
            "blhd,bthd->bhlt", q * scale, k * scale * self.alpha_att
        )
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        context = jnp.einsum("bhlt,bthd->blhd", attn_weights, v)

        out = jnp.einsum("blhd,hdm->blm", context, self.wo[...])
        if self.bo is not None:
            out = out + self.bo[...]
        return out


class _PredictionHead(nnx.Module):
    __data__ = ("proj0", "proj1", "activation", "output_activation")

    def __init__(
        self,
        *,
        dim: int,
        out_dim: int,
        activation: str,
        output_activation: str,
        use_bias: bool,
        head_init: str,
        sigma: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype,
    ):
        self.proj0 = _LinearEinsum(
            in_dim=dim,
            out_dim=dim,
            rngs=rngs,
            sigma=sigma,
            use_bias=use_bias,
            dtype=dtype,
        )
        self.proj1 = _LinearEinsum(
            in_dim=dim,
            out_dim=out_dim,
            rngs=rngs,
            sigma=sigma,
            use_bias=use_bias,
            dtype=dtype,
            zero_init=(head_init == "zero"),
        )
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _act(self.activation)(self.proj0(x))
        return _act(self.output_activation)(self.proj1(x))


@dataclass(frozen=True)
class TransformerPayneConfig:
    dim: int = 128
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 8
    dim_head: int = 32
    channels: int = 1
    min_period: float = 1e-6
    max_period: float = 10.0
    bias_dense: bool = False
    bias_parameter_embedding: bool | None = None
    bias_feed_forward: bool | None = None
    bias_output_head: bool | None = None
    bias_attention: bool = False
    activation: str = "gelu"
    output_activation: str = "linear"
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    sigma: float = 1.0
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    reference_depth: int | None = None
    reference_width: int | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bias_dense", parse_bool(self.bias_dense, field_name="bias_dense")
        )
        object.__setattr__(
            self,
            "bias_parameter_embedding",
            _parse_optional_bool(
                self.bias_parameter_embedding, field_name="bias_parameter_embedding"
            ),
        )
        object.__setattr__(
            self,
            "bias_feed_forward",
            _parse_optional_bool(
                self.bias_feed_forward, field_name="bias_feed_forward"
            ),
        )
        object.__setattr__(
            self,
            "bias_output_head",
            _parse_optional_bool(self.bias_output_head, field_name="bias_output_head"),
        )
        object.__setattr__(
            self,
            "bias_attention",
            parse_bool(self.bias_attention, field_name="bias_attention"),
        )
        object.__setattr__(
            self, "dim", _normalize_positive_int(self.dim, field_name="dim")
        )
        object.__setattr__(
            self,
            "dim_ff_multiplier",
            _normalize_positive_int(
                self.dim_ff_multiplier, field_name="dim_ff_multiplier"
            ),
        )
        object.__setattr__(
            self,
            "no_tokens",
            _normalize_positive_int(self.no_tokens, field_name="no_tokens"),
        )
        object.__setattr__(
            self,
            "no_layers",
            _normalize_positive_int(self.no_layers, field_name="no_layers"),
        )
        object.__setattr__(
            self,
            "dim_head",
            _normalize_positive_int(self.dim_head, field_name="dim_head"),
        )
        object.__setattr__(
            self,
            "channels",
            _normalize_positive_int(self.channels, field_name="channels"),
        )

        min_period = _normalize_finite_float(self.min_period, field_name="min_period")
        max_period = _normalize_finite_float(self.max_period, field_name="max_period")
        if min_period <= 0.0:
            raise ValueError("min_period must be > 0.")
        if max_period < min_period:
            raise ValueError("max_period must be >= min_period.")
        object.__setattr__(self, "min_period", min_period)
        object.__setattr__(self, "max_period", max_period)

        sigma = _normalize_finite_float(self.sigma, field_name="sigma")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0.")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(
            self,
            "alpha_emb",
            _normalize_finite_float(self.alpha_emb, field_name="alpha_emb"),
        )
        object.__setattr__(
            self,
            "alpha_att",
            _normalize_finite_float(self.alpha_att, field_name="alpha_att"),
        )

        if self.reference_depth is not None:
            object.__setattr__(
                self,
                "reference_depth",
                _normalize_positive_int(
                    self.reference_depth, field_name="reference_depth"
                ),
            )
        if self.reference_width is not None:
            object.__setattr__(
                self,
                "reference_width",
                _normalize_positive_int(
                    self.reference_width, field_name="reference_width"
                ),
            )

        object.__setattr__(
            self,
            "activation",
            _validate_transformer_activation(self.activation, field_name="activation"),
        )
        object.__setattr__(
            self,
            "output_activation",
            _validate_transformer_activation(
                self.output_activation, field_name="output_activation"
            ),
        )
        for field_name in (
            "init_att_q",
            "init_att_o",
            "emb_init",
            "ff_init",
            "head_init",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_transformer_init_name(
                    getattr(self, field_name), field_name=field_name
                ),
            )

        dtype = str(self.dtype)
        try:
            jdtype = jnp.dtype(dtype)
        except TypeError as exc:
            raise ValueError(
                f"transformer_payne dtype must be a valid JAX dtype, got {dtype!r}."
            ) from exc
        if jdtype not in (jnp.float32, jnp.float64):
            raise ValueError(
                "transformer_payne cfg.dtype must be 'float32' or 'float64'."
            )
        object.__setattr__(self, "dtype", dtype)

    @property
    def use_parameter_embedding_bias(self) -> bool:
        if self.bias_parameter_embedding is not None:
            return self.bias_parameter_embedding
        return self.bias_dense

    @property
    def use_feed_forward_bias(self) -> bool:
        if self.bias_feed_forward is not None:
            return self.bias_feed_forward
        return self.bias_dense

    @property
    def use_output_head_bias(self) -> bool:
        if self.bias_output_head is not None:
            return self.bias_output_head
        return self.bias_dense

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformerPayneConfig":
        allowed = {
            "dim",
            "dim_ff_multiplier",
            "no_tokens",
            "no_layers",
            "dim_head",
            "channels",
            "min_period",
            "max_period",
            "bias_dense",
            "bias_parameter_embedding",
            "bias_feed_forward",
            "bias_output_head",
            "bias_attention",
            "activation",
            "output_activation",
            "init_att_q",
            "init_att_o",
            "emb_init",
            "ff_init",
            "head_init",
            "sigma",
            "alpha_emb",
            "alpha_att",
            "reference_depth",
            "reference_width",
            "dtype",
        }
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown TransformerPayne params: {unknown}.")
        return cls(
            dim=int(d.get("dim", 128)),
            dim_ff_multiplier=int(d.get("dim_ff_multiplier", 4)),
            no_tokens=int(d.get("no_tokens", 16)),
            no_layers=int(d.get("no_layers", 8)),
            dim_head=int(d.get("dim_head", 32)),
            channels=int(d.get("channels", 1)),
            min_period=float(d.get("min_period", 1e-6)),
            max_period=float(d.get("max_period", 10.0)),
            bias_dense=parse_bool(d.get("bias_dense", False), field_name="bias_dense"),
            bias_parameter_embedding=_parse_optional_bool(
                d.get("bias_parameter_embedding", None),
                field_name="bias_parameter_embedding",
            ),
            bias_feed_forward=_parse_optional_bool(
                d.get("bias_feed_forward", None), field_name="bias_feed_forward"
            ),
            bias_output_head=_parse_optional_bool(
                d.get("bias_output_head", None), field_name="bias_output_head"
            ),
            bias_attention=parse_bool(
                d.get("bias_attention", False), field_name="bias_attention"
            ),
            activation=str(d.get("activation", "gelu")),
            output_activation=str(d.get("output_activation", "linear")),
            init_att_q=str(d.get("init_att_q", "si")),
            init_att_o=str(d.get("init_att_o", "si")),
            emb_init=str(d.get("emb_init", "si")),
            ff_init=str(d.get("ff_init", "si")),
            head_init=str(d.get("head_init", "si")),
            sigma=float(d.get("sigma", 1.0)),
            alpha_emb=float(d.get("alpha_emb", 1.0)),
            alpha_att=float(d.get("alpha_att", 1.0)),
            reference_depth=d.get("reference_depth", None),
            reference_width=d.get("reference_width", None),
            dtype=str(d.get("dtype", "float32")),
        )


class TransformerPayne(nnx.Module):
    __data__ = (
        "param_embedding",
        "attn_layers",
        "ff_layers",
        "head",
        "residual_scaling",
    )

    def __init__(
        self, *, in_dim: int, out_dim: int, cfg: TransformerPayneConfig, rngs: nnx.Rngs
    ):
        if out_dim != 1:
            raise ValueError(
                "transformer_payne requires output init size 1 because wavelength is an explicit input axis."
            )
        if cfg.channels <= 0:
            raise ValueError(f"channels must be > 0, got {cfg.channels}.")
        if (
            cfg.min_period <= 0.0
            or cfg.max_period <= 0.0
            or cfg.max_period < cfg.min_period
        ):
            raise ValueError(
                f"invalid period bounds: min_period={cfg.min_period}, max_period={cfg.max_period}."
            )

        allowed_init = {"si", "zero"}
        if cfg.emb_init not in allowed_init:
            raise ValueError(
                f"emb_init must be one of {sorted(allowed_init)}, got {cfg.emb_init!r}."
            )
        if cfg.ff_init not in allowed_init:
            raise ValueError(
                f"ff_init must be one of {sorted(allowed_init)}, got {cfg.ff_init!r}."
            )
        if cfg.head_init not in allowed_init:
            raise ValueError(
                f"head_init must be one of {sorted(allowed_init)}, got {cfg.head_init!r}."
            )
        if cfg.init_att_q not in allowed_init:
            raise ValueError(
                f"init_att_q must be one of {sorted(allowed_init)}, got {cfg.init_att_q!r}."
            )
        if cfg.init_att_o not in allowed_init:
            raise ValueError(
                f"init_att_o must be one of {sorted(allowed_init)}, got {cfg.init_att_o!r}."
            )

        dtype = jnp.dtype(cfg.dtype)
        if dtype not in (jnp.float32, jnp.float64):
            raise ValueError(
                "transformer_payne cfg.dtype must be 'float32' or 'float64'."
            )
        if not bool(jax.config.read("jax_enable_x64")):
            raise ValueError(
                "transformer_payne requires JAX_ENABLE_X64=1 because wavelengths are always handled in float64 "
                "up to the embedding cast. Model activations and targets may still use cfg.dtype='float32'."
            )
        self.channels = int(cfg.channels)
        self.dim = int(cfg.dim)
        self.min_period = float(cfg.min_period)
        self.max_period = float(cfg.max_period)
        self.dtype = dtype
        self.wavelength_dtype = jnp.float64

        self.param_embedding = _ParameterEmbedding(
            in_dim=in_dim,
            dim=cfg.dim,
            no_tokens=cfg.no_tokens,
            activation=cfg.activation,
            use_bias=cfg.use_parameter_embedding_bias,
            emb_init=cfg.emb_init,
            sigma=cfg.sigma,
            alpha_emb=cfg.alpha_emb,
            rngs=rngs,
            dtype=dtype,
        )

        self.attn_layers = nnx.List(
            [
                _CrossAttention(
                    dim=cfg.dim,
                    dim_head=cfg.dim_head,
                    use_bias=cfg.bias_attention,
                    init_att_q=cfg.init_att_q,
                    init_att_o=cfg.init_att_o,
                    sigma=cfg.sigma,
                    alpha_att=cfg.alpha_att,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(cfg.no_layers)
            ]
        )
        self.ff_layers = nnx.List(
            [
                _FeedForward(
                    dim=cfg.dim,
                    dim_ff_multiplier=cfg.dim_ff_multiplier,
                    activation=cfg.activation,
                    use_bias=cfg.use_feed_forward_bias,
                    ff_init=cfg.ff_init,
                    sigma=cfg.sigma,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(cfg.no_layers)
            ]
        )

        self.head = _PredictionHead(
            dim=cfg.dim,
            out_dim=self.channels,
            activation=cfg.activation,
            output_activation=cfg.output_activation,
            use_bias=cfg.use_output_head_bias,
            head_init=cfg.head_init,
            sigma=cfg.sigma,
            rngs=rngs,
            dtype=dtype,
        )

        if cfg.reference_depth is None:
            self.residual_scaling = 1.0
        else:
            self.residual_scaling = (cfg.no_layers / float(cfg.reference_depth)) ** (
                -1.0
            )

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        atmospheric_parameters, wavelengths = x
        atmospheric_parameters = jnp.asarray(atmospheric_parameters, dtype=self.dtype)
        wavelengths = jnp.asarray(wavelengths, dtype=self.wavelength_dtype)

        if wavelengths.ndim == 1:
            wavelengths = jnp.broadcast_to(
                wavelengths[None, :],
                (atmospheric_parameters.shape[0], wavelengths.shape[0]),
            )

        enc_w = _frequency_encoding(
            wavelengths,
            min_period=self.min_period,
            max_period=self.max_period,
            dim=self.dim,
            wavelength_dtype=self.wavelength_dtype,
            output_dtype=self.dtype,
        )

        enc_p = _rms_norm(self.param_embedding(atmospheric_parameters))
        h = enc_w

        for attn, ff in zip(self.attn_layers, self.ff_layers):
            h = h + self.residual_scaling * attn(_rms_norm(h), enc_p)
            h = h + self.residual_scaling * ff(_rms_norm(h))

        h = _rms_norm(h)
        pred = self.head(h)

        if self.channels == 1:
            return pred[..., 0]
        return pred
