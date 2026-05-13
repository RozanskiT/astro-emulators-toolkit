from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _infer_batch_size_from_tree(value: Any, *, field_name: str) -> int:
    if isinstance(value, dict):
        for child in value.values():
            return _infer_batch_size_from_tree(child, field_name=field_name)
        raise ValueError(f"{field_name} must contain at least one array leaf.")
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        raise ValueError(
            f"{field_name} leaves must be at least 1D with a leading batch axis."
        )
    return int(arr.shape[0])


def _normalize_optional_batch_vector(
    name: str, values, *, batch_size: int | None = None, dtype=None
):
    if values is None:
        return None

    arr = jnp.asarray(values, dtype=dtype)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = jnp.squeeze(arr, axis=1)

    if arr.ndim != 1:
        raise ValueError(f"{name} must have shape (N,) or (N, 1), got {arr.shape}.")

    if batch_size is not None and int(arr.shape[0]) != int(batch_size):
        raise ValueError(
            f"{name} first dimension must match batch_size={batch_size}, got {arr.shape}."
        )
    return arr


def sample_weight_from_batch(batch: dict[str, Any], *, batch_size: int | None = None):
    return _normalize_optional_batch_vector(
        "sample_weight",
        batch.get("sample_weight"),
        batch_size=batch_size,
        dtype=jnp.float32,
    )


def valid_mask_from_batch(batch: dict[str, Any], *, batch_size: int | None = None):
    if "valid_mask" in batch:
        mask = _normalize_optional_batch_vector(
            "valid_mask",
            batch["valid_mask"],
            batch_size=batch_size,
            dtype=jnp.float32,
        )
        return mask
    if batch_size is None:
        y = batch.get("y")
        batch_size = _infer_batch_size_from_tree(y, field_name="batch['y']")
    return jnp.ones((int(batch_size),), dtype=jnp.float32)


def reduce_over_non_batch_axes(values):
    values = jnp.asarray(values)
    if values.ndim <= 1:
        return values
    return jnp.mean(values, axis=tuple(range(1, values.ndim)))


def masked_mean(values, valid_mask):
    values = jnp.asarray(values)
    mask = jnp.asarray(valid_mask, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return jnp.sum(values * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-12)


def weighted_masked_mean(values, sample_weight=None, valid_mask=None):
    values = jnp.asarray(values)
    eff = effective_sample_weight(
        sample_weight=sample_weight, valid_mask=valid_mask, batch_size=values.shape[0]
    )
    if eff is None:
        return jnp.mean(values, axis=0)
    weight = eff.astype(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight[..., None]
    return jnp.sum(values * weight, axis=0) / jnp.maximum(
        jnp.sum(weight, axis=0), 1e-12
    )


def effective_sample_weight(
    *, sample_weight=None, valid_mask=None, batch_size: int | None = None
):
    if sample_weight is None and valid_mask is None:
        return None
    if sample_weight is None:
        return jnp.asarray(valid_mask, dtype=jnp.float32)
    sw = jnp.asarray(sample_weight, dtype=jnp.float32)
    if valid_mask is None:
        return sw
    return sw * jnp.asarray(valid_mask, dtype=jnp.float32)
