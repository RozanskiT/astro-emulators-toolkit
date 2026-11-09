"""Dtype parsing shared by model-family configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


def parse_dtype_name(value: Any, *, field_name: str, owner: str) -> str:
    """Return ``value`` as a dtype name, raising if JAX does not know it."""
    name = str(value)
    try:
        jnp.dtype(name)
    except TypeError as exc:
        raise ValueError(
            f"{owner} {field_name} must be a valid JAX dtype, got {name!r}."
        ) from exc
    return name


def resolve_param_dtype(params: Mapping[str, Any], *, dtype: str, owner: str) -> str:
    """Parameter storage dtype: an explicit ``param_dtype``, else ``dtype``.

    ``dtype`` is the compute dtype. Keeping parameters in a wider dtype than
    the compute dtype (e.g. ``param_dtype="float32"`` with
    ``dtype="bfloat16"``) is the usual mixed-precision setup; it is also how
    toolkit 0.1.x built these families for every non-float32 ``dtype``.
    """
    raw = params.get("param_dtype")
    if raw is None:
        return dtype
    return parse_dtype_name(raw, field_name="param_dtype", owner=owner)
