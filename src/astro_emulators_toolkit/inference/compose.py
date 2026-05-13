from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve


__all__ = [
    "add_distance_modulus",
    "downgrade_spectral_resolution",
]


def _add_offset(value: Any, offset: jax.Array) -> Any:
    if isinstance(value, dict):
        return {key: _add_offset(child, offset) for key, child in value.items()}
    return value + offset


def add_distance_modulus(
    apply_abs_mag_fn: Callable[[Any], Any],
) -> Callable[[Any, jax.Array], Any]:
    @jax.jit
    def apply_obs(x, mu):
        return _add_offset(apply_abs_mag_fn(x), mu)

    return apply_obs


def _positive_float(value: float, *, field_name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value.")
    return out


def _uniform_log_wavelength_step(log_wavelength: Any) -> float:
    log_wavelength_array = np.asarray(log_wavelength, dtype=np.float64)
    if log_wavelength_array.ndim != 1:
        raise ValueError("log_wavelength must be a 1D array.")
    if log_wavelength_array.shape[0] < 2:
        raise ValueError("log_wavelength must contain at least two samples.")
    if not np.all(np.isfinite(log_wavelength_array)):
        raise ValueError("log_wavelength must contain only finite values.")

    n = log_wavelength_array.shape[0]
    delta_log = float((log_wavelength_array[-1] - log_wavelength_array[0]) / (n - 1))
    if not np.isfinite(delta_log) or delta_log <= 0.0:
        raise ValueError("log_wavelength must be strictly increasing.")

    diffs = np.diff(log_wavelength_array)
    step_tolerance = max(abs(delta_log) * 5e-2, 1e-7)
    if not np.allclose(diffs, delta_log, rtol=5e-2, atol=step_tolerance):
        raise ValueError("log_wavelength must be uniformly spaced.")
    return delta_log


def _spectral_resolution_kernel(
    log_wavelength: Any,
    *,
    resolution: float,
    window_size: float,
) -> jax.Array:
    delta_log = _uniform_log_wavelength_step(log_wavelength)
    resolution_value = _positive_float(resolution, field_name="resolution")
    window_size_value = _positive_float(window_size, field_name="window_size")

    sigma_log = 1.0 / (
        2.0 * np.sqrt(2.0 * np.log(2.0)) * resolution_value * np.log(10.0)
    )
    half_width = max(1, int(window_size_value * sigma_log / delta_log + 1.0))
    offsets = jnp.arange(-half_width, half_width + 1, dtype=jnp.float32)
    x = offsets * jnp.asarray(delta_log, dtype=jnp.float32)
    sigma = jnp.asarray(sigma_log, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * jnp.square(x / sigma))
    return kernel / jnp.sum(kernel)


def _validate_axis(axis: int) -> int:
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError("axis must be -1 or -2.")
    if axis not in (-1, -2):
        raise ValueError("axis must be -1 or -2.")
    return axis


def _smooth_axis(
    flux: Any,
    kernel: jax.Array,
    *,
    axis: int,
) -> jax.Array:
    axis = _validate_axis(axis)
    flux_array = jnp.asarray(flux)
    if flux_array.ndim < 1:
        raise ValueError("flux must have at least one dimension.")
    if axis == -2 and flux_array.ndim < 2:
        raise ValueError("axis -2 requires flux with at least two dimensions.")
    target_axis = flux_array.ndim + axis

    dtype = jnp.result_type(flux_array, jnp.float32)
    flux_array = flux_array.astype(dtype)
    kernel = jnp.asarray(kernel, dtype=dtype)

    def smooth_one(spectrum: jax.Array) -> jax.Array:
        return convolve(spectrum, kernel, mode="same")

    flux_array = jnp.moveaxis(flux_array, target_axis, -1)
    if flux_array.ndim == 1:
        return smooth_one(flux_array)

    flat_flux = jnp.reshape(flux_array, (-1, flux_array.shape[-1]))
    flat_smoothed = jax.vmap(smooth_one)(flat_flux)
    smoothed = jnp.reshape(flat_smoothed, flux_array.shape)
    return jnp.moveaxis(smoothed, -1, target_axis)


def _split_output_path(output_path: str) -> tuple[str, ...]:
    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string.")
    parts = tuple(output_path.split("/"))
    if not parts or any(part == "" for part in parts):
        raise ValueError("output_path must be a non-empty slash-delimited path.")
    return parts


def _get_leaf(tree: dict[str, Any], parts: tuple[str, ...], output_path: str) -> Any:
    node: Any = tree
    for part in parts:
        if not isinstance(node, dict):
            raise KeyError(f"output_path '{output_path}' descends into a leaf.")
        if part not in node:
            raise KeyError(f"output_path '{output_path}' is missing segment '{part}'.")
        node = node[part]
    if isinstance(node, dict):
        raise KeyError(f"output_path '{output_path}' refers to a subtree.")
    return node


def _replace_leaf(
    tree: dict[str, Any],
    parts: tuple[str, ...],
    output_path: str,
    value: Any,
) -> dict[str, Any]:
    out = dict(tree)
    input_node: Any = tree
    output_node = out

    for part in parts[:-1]:
        child = input_node[part]
        if not isinstance(child, dict):
            raise KeyError(f"output_path '{output_path}' descends into a leaf.")
        child_out = dict(child)
        output_node[part] = child_out
        input_node = child
        output_node = child_out

    output_node[parts[-1]] = value
    return out


def _validate_axis_tree_leaf(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value not in (-1, -2):
        location = path or "<root>"
        raise ValueError(f"axis_tree leaf '{location}' must be None, -1, or -2.")
    return value


def _smooth_with_axis_tree(
    tree: dict[str, Any],
    axis_tree: dict[str, Any],
    kernel: jax.Array,
    *,
    path: str = "",
) -> dict[str, Any]:
    tree_keys = set(tree)
    axis_tree_keys = set(axis_tree)
    if tree_keys != axis_tree_keys:
        missing = sorted(tree_keys - axis_tree_keys)
        extra = sorted(axis_tree_keys - tree_keys)
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"extra keys: {extra}")
        location = path or "<root>"
        raise ValueError(f"axis_tree mismatch at '{location}': {', '.join(details)}")

    out: dict[str, Any] = {}
    for key, value in tree.items():
        axis_value = axis_tree[key]
        child_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            if not isinstance(axis_value, dict):
                raise ValueError(f"axis_tree must contain a subtree at '{child_path}'.")
            out[key] = _smooth_with_axis_tree(
                value,
                axis_value,
                kernel,
                path=child_path,
            )
            continue
        if isinstance(axis_value, dict):
            raise ValueError(f"axis_tree has a subtree where '{child_path}' is a leaf.")

        axis = _validate_axis_tree_leaf(axis_value, child_path)
        out[key] = (
            value
            if axis is None
            else _smooth_axis(
                value,
                kernel,
                axis=axis,
            )
        )
    return out


def _smooth_leaves_named(
    tree: dict[str, Any],
    leaf_name: str,
    kernel: jax.Array,
    *,
    axis: int,
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    count = 0
    for key, value in tree.items():
        if isinstance(value, dict):
            child, child_count = _smooth_leaves_named(
                value,
                leaf_name,
                kernel,
                axis=axis,
            )
            out[key] = child
            count += child_count
            continue
        if key == leaf_name:
            out[key] = _smooth_axis(
                value,
                kernel,
                axis=axis,
            )
            count += 1
        else:
            out[key] = value
    return out, count


def downgrade_spectral_resolution(
    apply_high_resolution_fn: Callable[..., Any],
    log_wavelength: Any,
    resolution: float,
    *,
    output_path: str = "flux",
    window_size: float = 4.0,
    axis: int = -1,
    axis_tree: dict[str, Any] | None = None,
    jit: bool = True,
) -> Callable[..., Any]:
    """Wrap an apply function with spectral-resolution postprocessing.

    ``log_wavelength`` must be the uniform log10 wavelength grid for the output
    flux leaf. A slash-delimited ``output_path`` selects one exact leaf. A bare
    name, such as ``"flux"``, smooths all leaves with that name. For mixed
    output trees, pass ``axis_tree`` with leaves set to ``None``, ``-1``, or
    ``-2``.
    """
    kernel = _spectral_resolution_kernel(
        log_wavelength,
        resolution=resolution,
        window_size=window_size,
    )
    axis = _validate_axis(axis)
    if axis_tree is not None and not isinstance(axis_tree, dict):
        raise ValueError("axis_tree must be a nested dict tree.")
    output_parts = _split_output_path(output_path)
    exact_parts = output_parts if "/" in output_path else None
    leaf_name = output_parts[0] if exact_parts is None else output_path

    def apply_low_resolution(*args: Any, **kwargs: Any) -> Any:
        y = apply_high_resolution_fn(*args, **kwargs)
        if not isinstance(y, dict):
            raise ValueError(
                "downgrade_spectral_resolution expects the wrapped apply function "
                "to return a dict tree."
            )
        if axis_tree is not None:
            return _smooth_with_axis_tree(y, axis_tree, kernel)
        if exact_parts is not None:
            flux = _get_leaf(y, exact_parts, output_path)
            smoothed = _smooth_axis(
                flux,
                kernel,
                axis=axis,
            )
            return _replace_leaf(y, exact_parts, output_path, smoothed)

        smoothed_tree, count = _smooth_leaves_named(
            y,
            leaf_name,
            kernel,
            axis=axis,
        )
        if count == 0:
            raise KeyError(f"output tree does not contain a leaf named '{leaf_name}'.")
        return smoothed_tree

    if jit:
        return jax.jit(apply_low_resolution)
    return apply_low_resolution
