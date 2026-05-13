from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..data.protocols import DeviceBatchTransform
from ..io_trees import get_leaf_by_path, set_leaf_by_path


def _validate_strictly_increasing(*, name: str, wave: jax.Array) -> None:
    diffs = np.diff(np.asarray(wave, dtype=np.float64))
    if diffs.size == 0 or np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly monotonic increasing.")


def _validate_interval(*, min_w: float, max_w: float, name: str) -> None:
    if not np.isfinite(min_w) or not np.isfinite(max_w):
        raise ValueError(f"{name} must be finite.")
    if min_w >= max_w:
        raise ValueError(
            f"{name} must satisfy min_w < max_w, got min_w={min_w}, max_w={max_w}."
        )


def _resolve_wavelength_dtype(wavelength_dtype: Any, *, context_name: str) -> jnp.dtype:
    dtype = jnp.dtype(wavelength_dtype)
    if dtype == jnp.float64 and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            f"{context_name} requires JAX_ENABLE_X64=1 for float64 wavelength handling."
        )
    return cast(jnp.dtype, dtype)


def _sample_random_wave(
    rng: jax.Array,
    batch_size: int,
    n_wavelength: int,
    *,
    min_w: float,
    max_w: float,
    dtype: jnp.dtype,
) -> jax.Array:
    wave = jax.random.uniform(
        rng, shape=(batch_size, n_wavelength), minval=min_w, maxval=max_w, dtype=dtype
    )
    return jnp.sort(wave, axis=1)


def _require_train_rng(*, rng: jax.Array | None, context_name: str) -> jax.Array:
    if rng is None:
        raise ValueError(f"{context_name} requires rng when train=True.")
    return rng


def _interp_single_channel(
    source_wave: jax.Array, y: jax.Array, query_wave: jax.Array
) -> jax.Array:
    return jax.vmap(jnp.interp, in_axes=(0, None, 0), out_axes=0)(
        query_wave, source_wave, y
    )


def _interp_multi_channel(
    source_wave: jax.Array, y: jax.Array, query_wave: jax.Array
) -> jax.Array:
    def _interp_sample(sample_wave: jax.Array, sample_y: jax.Array) -> jax.Array:
        y_chan_first = jnp.swapaxes(sample_y, 0, 1)
        interp_chan_first = jax.vmap(
            lambda fp: jnp.interp(sample_wave, source_wave, fp), in_axes=0, out_axes=0
        )(y_chan_first)
        return jnp.swapaxes(interp_chan_first, 0, 1)

    return jax.vmap(_interp_sample, in_axes=(0, 0), out_axes=0)(query_wave, y)


def _interpolate(
    source_wave: jax.Array, y: jax.Array, query_wave: jax.Array
) -> jax.Array:
    if y.ndim == 2:
        return _interp_single_channel(source_wave, y, query_wave)
    return _interp_multi_channel(source_wave, y, query_wave)


def _relative_role_path(role_path: str, *, section_name: str) -> str:
    prefix = f"{section_name}/"
    if not role_path.startswith(prefix):
        raise ValueError(
            f"Role path '{role_path}' does not belong to '{section_name}'."
        )
    return role_path.removeprefix(prefix)


def _extract_role_leaf(
    tree: dict[str, Any], role_path: str, *, section_name: str, field_name: str
) -> Any:
    try:
        return get_leaf_by_path(
            tree, _relative_role_path(role_path, section_name=section_name)
        )
    except KeyError as exc:
        raise ValueError(
            f"Canonical {field_name} is missing required leaf '{role_path}'."
        ) from exc


def _inject_query_wavelengths(
    x_payload: Any,
    query_wave: jax.Array,
    *,
    parameter_role_path: str | None,
    wavelength_role_path: str | None,
) -> Any:
    if not isinstance(x_payload, dict):
        return (x_payload, query_wave)
    if parameter_role_path is None or wavelength_role_path is None:
        raise ValueError(
            "Canonical transformer batch transform requires parameter_role_path and wavelength_role_path."
        )
    _extract_role_leaf(
        x_payload, parameter_role_path, section_name="inputs", field_name="input"
    )
    out = deepcopy(x_payload)
    set_leaf_by_path(
        out,
        _relative_role_path(wavelength_role_path, section_name="inputs"),
        query_wave,
    )
    return out


def _extract_target_array(y_payload: Any, *, output_role_path: str | None) -> jax.Array:
    if not isinstance(y_payload, dict):
        return jnp.asarray(y_payload)
    if output_role_path is None:
        raise ValueError(
            "Canonical transformer batch transform requires output_role_path for dict targets."
        )
    return jnp.asarray(
        _extract_role_leaf(
            y_payload, output_role_path, section_name="outputs", field_name="target"
        )
    )


def _wrap_target_output(
    value: jax.Array, *, output_role_path: str | None, use_dict_output: bool
) -> Any:
    if not use_dict_output:
        return value
    if output_role_path is None:
        raise ValueError(
            "Canonical transformer batch transform requires output_role_path for dict outputs."
        )
    out: dict[str, Any] = {}
    set_leaf_by_path(
        out, _relative_role_path(output_role_path, section_name="outputs"), value
    )
    return out


@dataclass(frozen=True, eq=False)
class TransformerPayneFluxDeviceBatchTransform:
    wavelength_grid: Any
    n_wavelength: int
    eval_wavelength_grid: Any | None = None
    min_w: float | None = None
    max_w: float | None = None
    allow_extrapolation: bool = False
    parameter_role_path: str | None = None
    wavelength_role_path: str | None = None
    output_role_path: str | None = None
    dtype: Any = np.float32
    wavelength_dtype: Any | None = None
    _jdtype: jnp.dtype = field(init=False, repr=False)
    _wave_dtype: jnp.dtype = field(init=False, repr=False)
    _source_wave: jax.Array = field(init=False, repr=False)
    _eval_grid: jax.Array = field(init=False, repr=False)
    _min_w: float = field(init=False, repr=False)
    _max_w: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_wavelength = int(self.n_wavelength)
        if n_wavelength <= 0:
            raise ValueError("n_wavelength must be > 0")
        object.__setattr__(self, "n_wavelength", n_wavelength)

        jdtype = jnp.dtype(self.dtype)
        wave_dtype = _resolve_wavelength_dtype(
            jdtype if self.wavelength_dtype is None else self.wavelength_dtype,
            context_name=type(self).__name__,
        )
        source_wave = jnp.asarray(self.wavelength_grid, dtype=wave_dtype)
        if source_wave.ndim != 1:
            raise ValueError(
                f"wavelength_grid must be 1D, got shape={tuple(source_wave.shape)}"
            )
        _validate_strictly_increasing(name="wavelength_grid", wave=source_wave)

        min_wv = float(source_wave[0]) if self.min_w is None else float(self.min_w)
        max_wv = float(source_wave[-1]) if self.max_w is None else float(self.max_w)
        _validate_interval(min_w=min_wv, max_w=max_wv, name="flux sampling interval")
        support_min = float(source_wave[0])
        support_max = float(source_wave[-1])
        if not self.allow_extrapolation and (
            min_wv < support_min or max_wv > support_max
        ):
            raise ValueError(
                "flux sampling interval must lie inside wavelength_grid support; "
                "set allow_extrapolation=True to override."
            )

        if self.eval_wavelength_grid is not None:
            eval_grid = jnp.asarray(self.eval_wavelength_grid, dtype=wave_dtype)
        elif source_wave.shape[0] == n_wavelength:
            eval_grid = source_wave
        else:
            raise ValueError(
                "Evaluation wavelength grid must be provided when len(wavelength_grid) != n_wavelength."
            )

        if eval_grid.ndim != 1:
            raise ValueError(
                f"eval_wavelength_grid must be 1D, got shape={tuple(eval_grid.shape)}"
            )
        if eval_grid.shape[0] != n_wavelength:
            raise ValueError(
                f"eval_wavelength_grid length must equal n_wavelength={n_wavelength}, got {eval_grid.shape[0]}."
            )
        _validate_strictly_increasing(name="eval_wavelength_grid", wave=eval_grid)
        if float(eval_grid[0]) < min_wv or float(eval_grid[-1]) > max_wv:
            raise ValueError("eval_wavelength_grid must lie inside [min_w, max_w].")

        object.__setattr__(self, "_jdtype", jdtype)
        object.__setattr__(self, "_wave_dtype", wave_dtype)
        object.__setattr__(self, "_source_wave", source_wave)
        object.__setattr__(self, "_eval_grid", eval_grid)
        object.__setattr__(self, "_min_w", min_wv)
        object.__setattr__(self, "_max_w", max_wv)

    def for_init(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self(batch, train=False, rng=None)

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        rng: jax.Array | None = None,
        train: bool,
    ) -> dict[str, Any]:
        out = dict(batch)
        source_y = _extract_target_array(
            batch["y"], output_role_path=self.output_role_path
        )
        batch_size = int(source_y.shape[0])
        if train:
            query_wave = _sample_random_wave(
                _require_train_rng(rng=rng, context_name=type(self).__name__),
                batch_size,
                self.n_wavelength,
                min_w=self._min_w,
                max_w=self._max_w,
                dtype=self._wave_dtype,
            )
        else:
            query_wave = jnp.broadcast_to(
                self._eval_grid[None, :], (batch_size, self.n_wavelength)
            )
        out["x"] = _inject_query_wavelengths(
            batch["x"],
            query_wave,
            parameter_role_path=self.parameter_role_path,
            wavelength_role_path=self.wavelength_role_path,
        )
        interpolated = _interpolate(
            self._source_wave,
            jnp.asarray(source_y, dtype=self._wave_dtype),
            query_wave,
        ).astype(self._jdtype)
        out["y"] = _wrap_target_output(
            interpolated,
            output_role_path=self.output_role_path,
            use_dict_output=isinstance(batch["y"], dict),
        )
        return out


@dataclass(frozen=True, eq=False)
class TransformerPayneIntensityDeviceBatchTransform:
    common_waves: Mapping[str, Any]
    n_wavelength: int
    eval_wavelength_grid: Any
    output_order: tuple[str, ...] = ("lines", "continuum")
    channels: int | None = None
    expected_output_names: tuple[str, ...] | None = None
    expected_channel_dataset_keys: tuple[str, ...] | None = None
    min_w: float | None = None
    max_w: float | None = None
    allow_extrapolation: bool = False
    allow_eval_outside_overlap: bool = False
    parameter_role_path: str | None = None
    wavelength_role_path: str | None = None
    output_role_path: str | None = None
    dtype: Any = np.float32
    wavelength_dtype: Any | None = None
    _jdtype: jnp.dtype = field(init=False, repr=False)
    _wave_dtype: jnp.dtype = field(init=False, repr=False)
    _source_waves: dict[str, jax.Array] = field(init=False, repr=False)
    _eval_grid: jax.Array = field(init=False, repr=False)
    _min_w: float = field(init=False, repr=False)
    _max_w: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_wavelength = int(self.n_wavelength)
        if n_wavelength <= 0:
            raise ValueError("n_wavelength must be > 0")
        object.__setattr__(self, "n_wavelength", n_wavelength)

        output_order = tuple(self.output_order)
        if not output_order:
            raise ValueError("output_order must be non-empty.")
        object.__setattr__(self, "output_order", output_order)

        jdtype = jnp.dtype(self.dtype)
        wave_dtype = _resolve_wavelength_dtype(
            jdtype if self.wavelength_dtype is None else self.wavelength_dtype,
            context_name=type(self).__name__,
        )

        source_key_set = set(self.common_waves)
        output_order_set = set(output_order)
        missing = [name for name in output_order if name not in self.common_waves]
        if missing:
            raise ValueError(
                f"common_waves is missing required keys from output_order: {missing}."
            )
        extra = sorted(source_key_set - output_order_set)
        if extra:
            raise ValueError(
                f"common_waves contains keys not present in output_order: {extra}."
            )

        source_waves = {
            name: jnp.asarray(self.common_waves[name], dtype=wave_dtype)
            for name in output_order
        }
        for name, wave in source_waves.items():
            if wave.ndim != 1:
                raise ValueError(
                    f"common_waves['{name}'] must be 1D, got shape={tuple(wave.shape)}"
                )
            _validate_strictly_increasing(name=f"common_waves['{name}']", wave=wave)
        if self.channels is not None and len(output_order) != int(self.channels):
            raise ValueError(
                f"output_order length ({len(output_order)}) must equal channels ({int(self.channels)})."
            )
        if (
            self.expected_output_names is not None
            and len(self.expected_output_names) > 1
            and len(self.expected_output_names) != len(output_order)
        ):
            raise ValueError(
                "Per-channel runtime naming requires len(output_order) == "
                f"len(expected_output_names) ({len(self.expected_output_names)})."
            )
        if self.expected_channel_dataset_keys is not None:
            if len(self.expected_channel_dataset_keys) != len(output_order):
                raise ValueError(
                    "expected_channel_dataset_keys length must match output_order length "
                    f"({len(output_order)}), got {len(self.expected_channel_dataset_keys)}."
                )
            if tuple(output_order) != tuple(self.expected_channel_dataset_keys):
                raise ValueError(
                    "output_order must match runtime channel dataset-key contract exactly. "
                    f"Expected {tuple(self.expected_channel_dataset_keys)}, got {tuple(output_order)}."
                )

        eval_grid = jnp.asarray(self.eval_wavelength_grid, dtype=wave_dtype)
        if eval_grid.ndim != 1:
            raise ValueError(
                f"eval_wavelength_grid must be 1D, got shape={tuple(eval_grid.shape)}"
            )
        if eval_grid.shape[0] != n_wavelength:
            raise ValueError(
                f"eval_wavelength_grid length must equal n_wavelength={n_wavelength}, got {eval_grid.shape[0]}."
            )
        _validate_strictly_increasing(name="eval_wavelength_grid", wave=eval_grid)

        overlap_min = float(max(float(w[0]) for w in source_waves.values()))
        overlap_max = float(min(float(w[-1]) for w in source_waves.values()))
        _validate_interval(
            min_w=overlap_min, max_w=overlap_max, name="channel overlap interval"
        )

        min_wv = overlap_min if self.min_w is None else float(self.min_w)
        max_wv = overlap_max if self.max_w is None else float(self.max_w)
        _validate_interval(
            min_w=min_wv, max_w=max_wv, name="intensity sampling interval"
        )
        if not self.allow_extrapolation and (
            min_wv < overlap_min or max_wv > overlap_max
        ):
            raise ValueError(
                "intensity sampling interval must lie inside channel overlap interval; "
                "set allow_extrapolation=True to override."
            )

        if not self.allow_eval_outside_overlap and (
            float(eval_grid[0]) < overlap_min or float(eval_grid[-1]) > overlap_max
        ):
            raise ValueError(
                "eval_wavelength_grid must lie inside the channel overlap interval by default; "
                "set allow_eval_outside_overlap=True to override."
            )

        object.__setattr__(self, "_jdtype", jdtype)
        object.__setattr__(self, "_wave_dtype", wave_dtype)
        object.__setattr__(self, "_source_waves", source_waves)
        object.__setattr__(self, "_eval_grid", eval_grid)
        object.__setattr__(self, "_min_w", min_wv)
        object.__setattr__(self, "_max_w", max_wv)

    def _extract_channel_arrays(self, y_payload: Any) -> dict[str, jax.Array]:
        if not isinstance(y_payload, dict):
            raise ValueError(
                "Intensity batch transform requires batch['y'] as dict keyed by output_order."
            )
        missing = [name for name in self.output_order if name not in y_payload]
        if missing:
            raise ValueError(
                f"Intensity batch transform missing required y channels: {missing}."
            )
        extra = sorted(set(y_payload) - set(self.output_order))
        if extra:
            raise ValueError(
                f"Intensity batch transform got unexpected y channels: {extra}."
            )
        return {
            name: jnp.asarray(y_payload[name], dtype=self._wave_dtype)
            for name in self.output_order
        }

    def for_init(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self(batch, train=False, rng=None)

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        rng: jax.Array | None = None,
        train: bool,
    ) -> dict[str, Any]:
        out = dict(batch)
        channel_arrays = self._extract_channel_arrays(batch["y"])
        batch_size = int(next(iter(channel_arrays.values())).shape[0])
        if train:
            query_wave = _sample_random_wave(
                _require_train_rng(rng=rng, context_name=type(self).__name__),
                batch_size,
                self.n_wavelength,
                min_w=self._min_w,
                max_w=self._max_w,
                dtype=self._wave_dtype,
            )
        else:
            query_wave = jnp.broadcast_to(
                self._eval_grid[None, :], (batch_size, self.n_wavelength)
            )

        interpolated = []
        for name in self.output_order:
            chan = channel_arrays[name]
            if chan.ndim != 2:
                raise ValueError(
                    f"Intensity y channel '{name}' must have shape (batch, n_source_wave), got {chan.shape}."
                )
            interpolated.append(
                _interp_single_channel(self._source_waves[name], chan, query_wave)
            )

        out["x"] = _inject_query_wavelengths(
            batch["x"],
            query_wave,
            parameter_role_path=self.parameter_role_path,
            wavelength_role_path=self.wavelength_role_path,
        )
        stacked = jnp.stack(interpolated, axis=-1).astype(self._jdtype)
        out["y"] = _wrap_target_output(
            stacked,
            output_role_path=self.output_role_path,
            use_dict_output=isinstance(batch["y"], dict)
            and self.output_role_path is not None,
        )
        return out


def make_flux_batch_transform(
    *,
    wavelength_grid: np.ndarray,
    n_wavelength: int,
    eval_wavelength_grid: np.ndarray | None = None,
    min_w: float | None = None,
    max_w: float | None = None,
    allow_extrapolation: bool = False,
    parameter_role_path: str | None = None,
    wavelength_role_path: str | None = None,
    output_role_path: str | None = None,
    dtype=np.float32,
    wavelength_dtype: Any | None = None,
) -> DeviceBatchTransform:
    return TransformerPayneFluxDeviceBatchTransform(
        wavelength_grid=wavelength_grid,
        n_wavelength=n_wavelength,
        eval_wavelength_grid=eval_wavelength_grid,
        min_w=min_w,
        max_w=max_w,
        allow_extrapolation=allow_extrapolation,
        parameter_role_path=parameter_role_path,
        wavelength_role_path=wavelength_role_path,
        output_role_path=output_role_path,
        dtype=dtype,
        wavelength_dtype=wavelength_dtype,
    )


def make_intensity_batch_transform(
    *,
    common_waves: Mapping[str, np.ndarray],
    n_wavelength: int,
    eval_wavelength_grid: np.ndarray,
    output_order: tuple[str, ...] = ("lines", "continuum"),
    channels: int | None = None,
    expected_output_names: tuple[str, ...] | None = None,
    expected_channel_dataset_keys: tuple[str, ...] | None = None,
    min_w: float | None = None,
    max_w: float | None = None,
    allow_extrapolation: bool = False,
    allow_eval_outside_overlap: bool = False,
    parameter_role_path: str | None = None,
    wavelength_role_path: str | None = None,
    output_role_path: str | None = None,
    dtype=np.float32,
    wavelength_dtype: Any | None = None,
) -> DeviceBatchTransform:
    return TransformerPayneIntensityDeviceBatchTransform(
        common_waves=common_waves,
        n_wavelength=n_wavelength,
        eval_wavelength_grid=eval_wavelength_grid,
        output_order=output_order,
        channels=channels,
        expected_output_names=expected_output_names,
        expected_channel_dataset_keys=expected_channel_dataset_keys,
        min_w=min_w,
        max_w=max_w,
        allow_extrapolation=allow_extrapolation,
        allow_eval_outside_overlap=allow_eval_outside_overlap,
        parameter_role_path=parameter_role_path,
        wavelength_role_path=wavelength_role_path,
        output_role_path=output_role_path,
        dtype=dtype,
        wavelength_dtype=wavelength_dtype,
    )
