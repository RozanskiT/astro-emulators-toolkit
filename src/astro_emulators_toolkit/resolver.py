from __future__ import annotations
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config.schema import RootConfig
from .models import (
    ModelRegistryEntry,
    build_model,
    get_stable_model_entry,
    get_stable_model_registry,
)
from .tasks import build_task, get_stable_task_registry

_EXPERIMENTAL_PREFIX = "experimental/"


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def canonicalize_model_name(name: str) -> str:
    return _normalize(name)


def canonicalize_task_name(name: str) -> str:
    return _normalize(name)


def _split_experimental_name(name: str, *, kind: str) -> str:
    if not name.startswith(_EXPERIMENTAL_PREFIX):
        return name
    suffix = name.removeprefix(_EXPERIMENTAL_PREFIX)
    if not suffix:
        raise KeyError(f"Unknown {kind} '{name}'.")
    return suffix


def get_stable_model_entry_from_name(name: str) -> ModelRegistryEntry | None:
    canonical = canonicalize_model_name(name)
    if canonical.startswith(_EXPERIMENTAL_PREFIX):
        return None
    return get_stable_model_entry(canonical)


def get_model_entry_from_name(name: str) -> ModelRegistryEntry | None:
    canonical = canonicalize_model_name(name)
    if canonical.startswith(_EXPERIMENTAL_PREFIX):
        from .experimental.models import get_experimental_model_entry

        suffix = _split_experimental_name(canonical, kind="model")
        try:
            return get_experimental_model_entry(suffix)
        except KeyError:
            return None
    try:
        return get_stable_model_entry(canonical)
    except KeyError:
        return None


def derive_model_family_id(name: str) -> str | None:
    entry = get_model_entry_from_name(name)
    if entry is None:
        return None
    return str(entry.family_id) if entry.family_id is not None else None


def _normalize_init_hints(init_hints: Mapping[str, Any] | None) -> dict[str, Any]:
    if init_hints is None:
        return {}
    return {str(k): v for k, v in dict(init_hints).items()}


def _merge_model_init_hints(
    *,
    cfg: RootConfig,
    init_hints: Mapping[str, Any] | None,
    derived_hints: dict[str, Any],
) -> dict[str, Any]:
    merged = _normalize_init_hints(cfg.model.init_hints)
    merged.update(_normalize_init_hints(init_hints))
    merged.update(derived_hints)
    return merged


def resolve_model_init_context(
    cfg: RootConfig,
    *,
    spec: dict[str, Any],
    inputs: Any | None = None,
    outputs: Any | None = None,
    init_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_name = canonicalize_model_name(cfg.model.name)
    model_entry = get_model_entry_from_name(model_name)

    if model_entry is not None and model_entry.runtime is not None:
        return model_entry.runtime.resolve_init_context(
            cfg=cfg,
            spec=spec,
            inputs=inputs,
            outputs=outputs,
            init_hints=init_hints,
        )

    return _merge_model_init_hints(cfg=cfg, init_hints=init_hints, derived_hints={})


def build_model_from_name(
    name: str,
    params: dict[str, Any],
    *,
    init_context: dict[str, Any],
    rngs,
    cfg=None,
    spec: dict[str, Any] | None = None,
):
    canonical = canonicalize_model_name(name)
    if canonical.startswith(_EXPERIMENTAL_PREFIX):
        from .experimental.models import build_experimental_model

        return build_experimental_model(
            _split_experimental_name(canonical, kind="model"),
            params,
            init_context=init_context,
            rngs=rngs,
            cfg=cfg,
            spec=spec,
        )
    return build_model(
        canonical, params, init_context=init_context, rngs=rngs, cfg=cfg, spec=spec
    )


def build_task_from_name(name: str, params: dict[str, Any]):
    canonical = canonicalize_task_name(name)
    if canonical.startswith(_EXPERIMENTAL_PREFIX):
        from .experimental.tasks import build_experimental_task

        return build_experimental_task(
            _split_experimental_name(canonical, kind="task"), params
        )
    return build_task(canonical, params)


def validate_model_io_compatibility(
    model_name: str,
    model_params: dict[str, Any],
    *,
    init_context: Mapping[str, Any] | None = None,
) -> None:
    channels = int(model_params.get("channels", 1))
    if channels <= 0:
        raise ValueError(f"Model '{model_name}' requires channels > 0, got {channels}.")
    for key, value in _normalize_init_hints(init_context).items():
        if isinstance(value, (int, np.integer)) and int(value) <= 0:
            raise ValueError(
                f"Model '{model_name}' requires init hint '{key}' > 0, got {value}."
            )


def get_supported_stable_model_names() -> tuple[str, ...]:
    return tuple(get_stable_model_registry())


def get_supported_stable_task_names() -> tuple[str, ...]:
    return tuple(get_stable_task_registry())


def get_supported_experimental_model_names() -> tuple[str, ...]:
    from .experimental.models import get_experimental_model_registry

    return tuple(f"experimental/{name}" for name in get_experimental_model_registry())


def get_supported_experimental_task_names() -> tuple[str, ...]:
    from .experimental.tasks import get_experimental_task_registry

    return tuple(f"experimental/{name}" for name in get_experimental_task_registry())
