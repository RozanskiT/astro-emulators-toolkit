"""Primitive JSON/YAML configuration for the experimental MHE families.

Adapted from exojax_emulator.mhe_config. Wavelength ranges and periods are
in the coordinates supplied by the caller; no physical scaling is implicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real


ACTIVATIONS = ("gelu", "relu", "silu", "tanh", "linear")
INITIALIZERS = ("lecun_normal", "glorot_uniform", "he_normal", "zeros")


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def finite_float(name, value, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def choice(name, value, options):
    if value not in options:
        raise ValueError(f"{name} must be one of {options}")
    return value


def boolean(name, value, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def widths(name, value):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    return tuple(positive_int(name, x) for x in value)


@dataclass(frozen=True)
class JSONConfig:
    @classmethod
    def from_dict(cls, values):
        if type(values) is cls:
            return values
        if not isinstance(values, Mapping):
            raise TypeError(f"{cls.__name__} requires a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class MHEConfig(JSONConfig):
    levels: int = 16
    features_per_level: int = 2
    log2_hashmap_size: int = 15
    wavelength_range: tuple[float, float] = (0.0, 1.0)
    table_init_scale: float = 1e-4

    def __post_init__(self):
        for name in ("levels", "features_per_level", "log2_hashmap_size"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))
        if self.levels < 2 or self.log2_hashmap_size > 30:
            raise ValueError("Require levels >= 2 and log2_hashmap_size <= 30")
        bounds = self.wavelength_range
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("wavelength_range must have two increasing bounds")
        lo, hi = (finite_float("wavelength_range", x) for x in bounds)
        if lo >= hi:
            raise ValueError("wavelength_range must have two increasing bounds")
        object.__setattr__(self, "wavelength_range", (lo, hi))
        object.__setattr__(
            self,
            "table_init_scale",
            finite_float("table_init_scale", self.table_init_scale, positive=True),
        )


@dataclass(frozen=True)
class MHEProjectionConfig(MHEConfig):
    normalize_output: bool = False
    projection_hidden_dim: int | None = None
    projection_activation: str = "gelu"
    projection_bias: bool = True
    projection_kernel_init: str = "lecun_normal"
    projection_sigma: float = 1.0
    output_rms: float | None = None
    output_rms_epsilon: float = 1e-16

    def __post_init__(self):
        super().__post_init__()
        for name in ("normalize_output", "projection_bias"):
            boolean(name, getattr(self, name))
        if self.projection_hidden_dim is not None:
            positive_int("projection_hidden_dim", self.projection_hidden_dim)
        choice("projection_activation", self.projection_activation, ACTIVATIONS)
        choice("projection_kernel_init", self.projection_kernel_init, INITIALIZERS)
        for name in ("projection_sigma", "output_rms_epsilon"):
            finite_float(name, getattr(self, name), positive=True)
        if self.output_rms is not None:
            finite_float("output_rms", self.output_rms, positive=True)
            if not self.normalize_output:
                raise ValueError("output_rms requires normalize_output=True")


def _validate_common(config, encoding_cls):
    positive_int("channels", config.channels)
    for name in ("activation", "output_activation"):
        choice(name, getattr(config, name), ACTIVATIONS)
    choice("kernel_init", config.kernel_init, INITIALIZERS)
    choice("dtype", config.dtype, ("float32", "float64"))
    finite_float("sigma", config.sigma, positive=True)
    finite_float("min_period", config.min_period, positive=True)
    finite_float("max_period", config.max_period, positive=True)
    if config.min_period > config.max_period:
        raise ValueError("min_period must be <= max_period")
    object.__setattr__(
        config, "mhe", encoding_cls.from_dict({} if config.mhe is None else config.mhe)
    )


@dataclass(frozen=True)
class MHETransformerPayneConfig(JSONConfig):
    """Transformer dimensions use the same names as stable TransformerPayne."""

    channels: int = 1
    dim: int = 192
    dim_head: int = 32
    no_tokens: int = 8
    no_layers: int = 6
    dim_ff_multiplier: int = 4
    parameter_hidden_dim: int | None = None
    ff_hidden_sizes: tuple[int, ...] | None = None
    head_hidden_dim: int | None = None
    min_period: float = 1e-5
    max_period: float = 10.0
    mhe: MHEProjectionConfig | None = None
    activation: str = "gelu"
    output_activation: str = "linear"
    bias_dense: bool = False
    bias_parameter_embedding: bool | None = None
    bias_feed_forward: bool | None = None
    bias_output_head: bool | None = None
    bias_attention: bool = False
    kernel_init: str = "lecun_normal"
    sigma: float = 1.0
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    residual_scale: float = 1.0
    rms_epsilon: float = 1e-6
    dtype: str = "float32"

    def __post_init__(self):
        _validate_common(self, MHEProjectionConfig)
        for name in ("dim", "dim_head", "no_tokens", "no_layers", "dim_ff_multiplier"):
            positive_int(name, getattr(self, name))
        if self.dim % self.dim_head:
            raise ValueError("dim must be divisible by dim_head")
        for name in ("parameter_hidden_dim", "head_hidden_dim"):
            if getattr(self, name) is not None:
                positive_int(name, getattr(self, name))
        if self.ff_hidden_sizes is not None:
            object.__setattr__(
                self, "ff_hidden_sizes", widths("ff_hidden_sizes", self.ff_hidden_sizes)
            )
            if len(self.ff_hidden_sizes) != self.no_layers:
                raise ValueError("ff_hidden_sizes must give one width per block")
        for name in ("bias_dense", "bias_attention"):
            boolean(name, getattr(self, name))
        for name in (
            "bias_parameter_embedding",
            "bias_feed_forward",
            "bias_output_head",
        ):
            boolean(name, getattr(self, name), optional=True)
        for name in ("init_att_q", "init_att_o", "emb_init", "ff_init", "head_init"):
            choice(name, getattr(self, name), ("si", "zero"))
        for name in ("alpha_emb", "alpha_att", "residual_scale"):
            finite_float(name, getattr(self, name))
        finite_float("rms_epsilon", self.rms_epsilon, positive=True)

    def bias(self, component):
        value = getattr(self, f"bias_{component}")
        return self.bias_dense if value is None else value


@dataclass(frozen=True)
class MHEMLPConfig(JSONConfig):
    channels: int = 1
    hidden_sizes: tuple[int, ...] = (128, 128, 128)
    parameter_hidden_sizes: tuple[int, ...] | None = None
    min_period: float = 1e-6
    max_period: float = 10.0
    mhe: MHEConfig | None = None
    activation: str = "gelu"
    output_activation: str = "linear"
    bias_parameter_embedding: bool = True
    bias_decoder: bool = True
    bias_output_head: bool = True
    kernel_init: str = "glorot_uniform"
    sigma: float = 1.0
    dtype: str = "float32"

    def __post_init__(self):
        _validate_common(self, MHEConfig)
        object.__setattr__(
            self, "hidden_sizes", widths("hidden_sizes", self.hidden_sizes)
        )
        values = (
            self.hidden_sizes
            if self.parameter_hidden_sizes is None
            else self.parameter_hidden_sizes
        )
        object.__setattr__(
            self, "parameter_hidden_sizes", widths("parameter_hidden_sizes", values)
        )
        for name in ("bias_parameter_embedding", "bias_decoder", "bias_output_head"):
            boolean(name, getattr(self, name))
