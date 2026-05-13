from __future__ import annotations

from pathlib import Path
from typing import Any

from ..spec import to_json_compatible
from .safetensors_io import load_arrays, save_arrays
from .versions import SPEC_VERSION
from ..config.schema import RootConfig
from ..io_trees import (
    flatten_minmax_trees,
    unflatten_minmax_trees,
    validate_minmax_values,
)
from ..models.runtime_adapters import RuntimeAdapter
from ..resolver import get_model_entry_from_name
from .integrity import validate_bundle_relpath


def _get_runtime_adapter(cfg: RootConfig) -> RuntimeAdapter | None:
    entry = get_model_entry_from_name(cfg.model.name)
    if entry is None:
        return None
    return entry.runtime


def _load_minmax_from_storage(
    storage: dict[str, Any],
    dirpath: Path,
    *,
    field_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    filename = validate_bundle_relpath(
        storage.get("filename"),
        field_name=f"{field_name}.storage.filename",
    )
    path = dirpath / filename
    if not path.exists():
        raise FileNotFoundError(f"{field_name} sidecar file not found: {path}")
    return unflatten_minmax_trees(load_arrays(path))


def canonicalize_reference_scaling_block(
    spec_payload: dict[str, Any],
    cfg,
    dirpath: Path,
    *,
    block_name: str,
    filename: str,
    field_name: str,
    validator_name: str,
    model_init: dict[str, Any],
) -> dict[str, Any]:
    runtime = _get_runtime_adapter(cfg)
    ref = spec_payload.get(block_name)
    if ref is None:
        normalized = dict(spec_payload)
        normalized["spec_version"] = SPEC_VERSION
        normalized.pop(block_name, None)
        return normalized

    if not isinstance(ref, dict):
        raise ValueError(f"{field_name} must be a dictionary when provided.")

    if isinstance(ref.get("min_tree"), dict) and isinstance(ref.get("max_tree"), dict):
        source_tree = {"min_tree": ref["min_tree"], "max_tree": ref["max_tree"]}
    elif isinstance(ref.get("storage"), dict):
        min_tree, max_tree = _load_minmax_from_storage(
            ref["storage"],
            dirpath,
            field_name=block_name,
        )
        source_tree = {"min_tree": min_tree, "max_tree": max_tree}
    else:
        raise ValueError(
            f"Unsupported {block_name} payload; expected min_tree/max_tree or storage descriptor."
        )
    validate_minmax_values(
        source_tree["min_tree"],
        source_tree["max_tree"],
        field_name=field_name,
        require_positive_span=True,
    )
    if runtime is not None:
        getattr(runtime, validator_name)(
            cfg=cfg,
            spec=spec_payload,
            scaling=source_tree,
            model_init=model_init,
        )
    arrays = flatten_minmax_trees(source_tree["min_tree"], source_tree["max_tree"])
    save_arrays(dirpath / filename, arrays)

    normalized = dict(spec_payload)
    normalized["spec_version"] = SPEC_VERSION
    normalized[block_name] = {
        "kind": ref["kind"],
        "applies_to": ref["applies_to"],
        "source_space": ref["source_space"],
        "target_space": ref["target_space"],
        "min_tree": to_json_compatible(source_tree["min_tree"]),
        "max_tree": to_json_compatible(source_tree["max_tree"]),
        "storage": {
            "format": "safetensors_v1",
            "filename": filename,
            "layout": "split_minmax_tree_v1",
        },
    }
    return normalized


def canonicalize_input_domain(
    spec_payload: dict[str, Any],
    cfg,
    dirpath: Path,
    *,
    model_init: dict[str, Any],
) -> dict[str, Any]:
    runtime = _get_runtime_adapter(cfg)
    domain = spec_payload.get("input_domain")
    if domain is not None and not isinstance(domain, dict):
        raise ValueError("spec['input_domain'] must be a dictionary when provided.")
    if domain is None:
        normalized = dict(spec_payload)
        normalized.pop("input_domain", None)
        return normalized

    if isinstance(domain.get("min_tree"), dict) and isinstance(
        domain.get("max_tree"),
        dict,
    ):
        source_tree = {"min_tree": domain["min_tree"], "max_tree": domain["max_tree"]}
    elif isinstance(domain.get("storage"), dict):
        min_tree, max_tree = _load_minmax_from_storage(
            domain["storage"],
            dirpath,
            field_name="input_domain",
        )
        source_tree = {"min_tree": min_tree, "max_tree": max_tree}
    else:
        raise ValueError(
            "Unsupported input_domain payload; expected min_tree/max_tree or storage descriptor."
        )
    validate_minmax_values(
        source_tree["min_tree"],
        source_tree["max_tree"],
        field_name="spec['input_domain']",
        require_positive_span=False,
    )
    if runtime is not None:
        runtime.validate_input_domain(
            cfg=cfg,
            spec=spec_payload,
            domain=source_tree,
            model_init=model_init,
        )

    filename = "input_domain.safetensors"
    save_arrays(
        dirpath / filename,
        flatten_minmax_trees(source_tree["min_tree"], source_tree["max_tree"]),
    )

    normalized = dict(spec_payload)
    normalized["spec_version"] = SPEC_VERSION
    normalized["input_domain"] = {
        "kind": domain["kind"],
        "value_space": domain["value_space"],
        "min_tree": to_json_compatible(source_tree["min_tree"]),
        "max_tree": to_json_compatible(source_tree["max_tree"]),
        "storage": {
            "format": "safetensors_v1",
            "filename": filename,
            "layout": "split_minmax_tree_v1",
        },
    }
    return normalized


def hydrate_input_domain_sidecar(
    spec_payload: dict[str, Any],
    bundle_dir: Path,
    cfg,
    *,
    model_init: dict[str, Any],
) -> dict[str, Any]:
    domain = spec_payload.get("input_domain")
    if not isinstance(domain, dict):
        return spec_payload
    storage = domain.get("storage")
    if not isinstance(storage, dict):
        return spec_payload

    filename = validate_bundle_relpath(
        storage.get("filename"),
        field_name="spec['input_domain']['storage']['filename']",
    )
    path = bundle_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"input_domain sidecar file not found: {path}")

    try:
        payload = load_arrays(path)
    except Exception as exc:  # pragma: no cover - defensive for decoder/backend errors
        raise ValueError(f"Failed to decode input_domain sidecar: {path}") from exc
    min_tree, max_tree = unflatten_minmax_trees(payload)
    tree = {"min_tree": min_tree, "max_tree": max_tree}
    validate_minmax_values(
        tree["min_tree"],
        tree["max_tree"],
        field_name="spec['input_domain']",
        require_positive_span=False,
    )

    runtime = _get_runtime_adapter(cfg)
    if runtime is not None:
        runtime.validate_input_domain(
            cfg=cfg,
            spec=spec_payload,
            domain=tree,
            model_init=model_init,
        )

    hydrated = dict(spec_payload)
    hydrated_domain = dict(domain)
    hydrated_domain["min_tree"] = to_json_compatible(min_tree)
    hydrated_domain["max_tree"] = to_json_compatible(max_tree)
    hydrated["input_domain"] = hydrated_domain
    return hydrated


def hydrate_reference_scaling_block(
    spec_payload: dict[str, Any],
    bundle_dir: Path,
    cfg,
    *,
    block_name: str,
    field_name: str,
    validator_name: str,
    model_init: dict[str, Any],
) -> dict[str, Any]:
    ref = spec_payload.get(block_name)
    if not isinstance(ref, dict):
        return spec_payload
    storage = ref.get("storage")
    if not isinstance(storage, dict):
        return spec_payload

    filename = validate_bundle_relpath(
        storage.get("filename"),
        field_name=f"{field_name}['storage']['filename']",
    )
    path = bundle_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{block_name} sidecar file not found: {path}")

    try:
        payload = load_arrays(path)
    except Exception as exc:  # pragma: no cover - defensive for decoder/backend errors
        raise ValueError(f"Failed to decode {block_name} sidecar: {path}") from exc

    min_tree, max_tree = unflatten_minmax_trees(payload)
    tree = {"min_tree": min_tree, "max_tree": max_tree}
    validate_minmax_values(
        tree["min_tree"],
        tree["max_tree"],
        field_name=field_name,
        require_positive_span=True,
    )
    runtime = _get_runtime_adapter(cfg)
    if runtime is not None:
        getattr(runtime, validator_name)(
            cfg=cfg,
            spec=spec_payload,
            scaling=tree,
            model_init=model_init,
        )
    hydrated = dict(spec_payload)
    hydrated_ref = dict(ref)
    hydrated_ref["min_tree"] = to_json_compatible(min_tree)
    hydrated_ref["max_tree"] = to_json_compatible(max_tree)
    hydrated[block_name] = hydrated_ref
    return hydrated


__all__ = [
    "canonicalize_input_domain",
    "canonicalize_reference_scaling_block",
    "hydrate_input_domain_sidecar",
    "hydrate_reference_scaling_block",
]
