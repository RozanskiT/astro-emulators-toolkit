from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from .config.schema import IOTreeSpec, MinMaxTreeSpec, RootConfig
from .io_trees import validate_minmax_values
from .resolver import get_model_entry_from_name


_IOTREE_KEYS = {
    "structure_tree",
    "channel_names_tree",
    "leaf_units_tree",
    "channel_units_tree",
    "leaf_meanings_tree",
    "channel_meanings_tree",
}
_SCALING_KEYS = {
    "kind",
    "applies_to",
    "source_space",
    "target_space",
    "min_tree",
    "max_tree",
    "storage",
}
_DOMAIN_KEYS = {"kind", "value_space", "min_tree", "max_tree", "storage"}
_SPEC_KEYS = {
    "spec_version",
    "inputs",
    "outputs",
    "reference_scaling_inputs",
    "reference_scaling_outputs",
    "input_domain",
}
_LEGACY_SPEC_KEYS = {"x", "y", "x_names", "y_names"}
_SCALING_BLOCK_DEFAULTS = {
    "reference_scaling_inputs": {
        "kind": "affine_minmax_v1",
        "applies_to": "inputs",
        "source_space": "physical_input_dict_tree_v1",
        "target_space": "canonical_input_dict_tree_v1",
    },
    "reference_scaling_outputs": {
        "kind": "affine_minmax_v1",
        "applies_to": "outputs",
        "source_space": "canonical_output_dict_tree_v1",
        "target_space": "physical_output_dict_tree_v1",
    },
}
_DOMAIN_BLOCK_DEFAULTS = {
    "input_domain": {
        "kind": "box_v1",
        "value_space": "physical_input_dict_tree_v1",
    },
}
SPEC_VERSION = 1


def _serialize_iotree_spec(value: IOTreeSpec) -> dict[str, Any]:
    return cast(dict[str, Any], to_json_compatible(asdict(value)))


def _serialize_minmax_tree_spec(value: MinMaxTreeSpec) -> dict[str, Any]:
    return cast(dict[str, Any], to_json_compatible(asdict(value)))


def _normalize_literal_metadata(
    block: dict[str, Any],
    *,
    field_name: str,
    key: str,
    expected: str,
) -> None:
    value = block.get(key)
    if value is None:
        block[key] = expected
        return
    if value != expected:
        raise ValueError(f"{field_name}.{key} must be '{expected}'.")


def _normalize_scaling_metadata(
    block: dict[str, Any],
    *,
    field_name: str,
    block_name: str,
) -> None:
    defaults = _SCALING_BLOCK_DEFAULTS[block_name]
    for key, expected in defaults.items():
        _normalize_literal_metadata(
            block,
            field_name=field_name,
            key=key,
            expected=expected,
        )


def _normalize_domain_metadata(
    block: dict[str, Any],
    *,
    field_name: str,
    block_name: str,
) -> None:
    defaults = _DOMAIN_BLOCK_DEFAULTS[block_name]
    for key, expected in defaults.items():
        _normalize_literal_metadata(
            block,
            field_name=field_name,
            key=key,
            expected=expected,
        )


def _normalize_iotree_section(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary when provided.")
    unknown_keys = sorted(set(value) - _IOTREE_KEYS)
    if unknown_keys:
        raise ValueError(f"{field_name} contains unknown keys: {unknown_keys}.")
    if "structure_tree" not in value:
        raise ValueError(f"{field_name} must include 'structure_tree'.")
    return _serialize_iotree_spec(IOTreeSpec(**deepcopy(value)))


def _normalize_minmax_section(
    value: Any,
    *,
    field_name: str,
    block_name: str,
    allowed_keys: set[str],
    require_positive_span: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary when provided.")
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"{field_name} contains unknown keys: {unknown_keys}.")
    block = deepcopy(value)
    has_minmax = "min_tree" in block or "max_tree" in block
    if has_minmax:
        if "min_tree" not in block or "max_tree" not in block:
            raise ValueError(
                f"{field_name} must provide both 'min_tree' and 'max_tree' when either is present."
            )
        normalized = _serialize_minmax_tree_spec(
            MinMaxTreeSpec(min_tree=block["min_tree"], max_tree=block["max_tree"])
        )
        validate_minmax_values(
            normalized["min_tree"],
            normalized["max_tree"],
            field_name=field_name,
            require_positive_span=require_positive_span,
        )
        block["min_tree"] = normalized["min_tree"]
        block["max_tree"] = normalized["max_tree"]
    elif "storage" not in block:
        raise ValueError(
            f"{field_name} must provide inline min/max trees or a storage descriptor."
        )
    if block_name in _SCALING_BLOCK_DEFAULTS:
        _normalize_scaling_metadata(
            block,
            field_name=field_name,
            block_name=block_name,
        )
    elif block_name in _DOMAIN_BLOCK_DEFAULTS:
        _normalize_domain_metadata(
            block,
            field_name=field_name,
            block_name=block_name,
        )
    return cast(dict[str, Any], to_json_compatible(block))


def _validate_public_spec_top_level(spec: dict[str, Any]) -> None:
    legacy_keys = sorted(set(spec) & _LEGACY_SPEC_KEYS)
    if legacy_keys:
        raise ValueError(f"Legacy spec keys are not supported: {legacy_keys}.")
    unknown_keys = sorted(set(spec) - _SPEC_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown spec keys: {unknown_keys}.")
    if "spec_version" not in spec or not isinstance(spec["spec_version"], int):
        raise ValueError("spec['spec_version'] must be present and be an int.")
    if int(spec["spec_version"]) != SPEC_VERSION:
        raise ValueError(
            f"Unsupported spec_version={spec['spec_version']} (expected {SPEC_VERSION})."
        )


def make_minimal_spec_from_cfg(cfg: RootConfig) -> dict[str, Any]:
    spec: dict[str, Any] = {"spec_version": SPEC_VERSION}
    if cfg.io.inputs is not None:
        spec["inputs"] = _serialize_iotree_spec(cfg.io.inputs)
    if cfg.io.outputs is not None:
        spec["outputs"] = _serialize_iotree_spec(cfg.io.outputs)
    if cfg.io.reference_scaling_inputs is not None:
        spec["reference_scaling_inputs"] = _serialize_minmax_tree_spec(
            cfg.io.reference_scaling_inputs
        )
    if cfg.io.reference_scaling_outputs is not None:
        spec["reference_scaling_outputs"] = _serialize_minmax_tree_spec(
            cfg.io.reference_scaling_outputs
        )
    if cfg.io.input_domain is not None:
        spec["input_domain"] = _serialize_minmax_tree_spec(cfg.io.input_domain)
    return spec


def materialize_effective_spec(
    cfg: RootConfig, spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = make_minimal_spec_from_cfg(cfg)
    if spec is not None:
        if not isinstance(spec, dict):
            raise ValueError("spec must be a dictionary.")
        payload.update(deepcopy(spec))
    payload.setdefault("spec_version", SPEC_VERSION)
    _validate_public_spec_top_level(payload)

    entry = get_model_entry_from_name(cfg.model.name)
    if entry is not None and entry.runtime is not None:
        payload = entry.runtime.materialize_spec(cfg=cfg, spec=payload)

    materialized: dict[str, Any] = {"spec_version": int(payload["spec_version"])}
    for field_name in ("inputs", "outputs"):
        normalized = _normalize_iotree_section(
            payload.get(field_name), field_name=f"spec['{field_name}']"
        )
        if normalized is not None:
            materialized[field_name] = normalized
    reference_scaling_inputs = _normalize_minmax_section(
        payload.get("reference_scaling_inputs"),
        field_name="spec['reference_scaling_inputs']",
        block_name="reference_scaling_inputs",
        allowed_keys=_SCALING_KEYS,
        require_positive_span=True,
    )
    if reference_scaling_inputs is not None:
        materialized["reference_scaling_inputs"] = reference_scaling_inputs
    reference_scaling_outputs = _normalize_minmax_section(
        payload.get("reference_scaling_outputs"),
        field_name="spec['reference_scaling_outputs']",
        block_name="reference_scaling_outputs",
        allowed_keys=_SCALING_KEYS,
        require_positive_span=True,
    )
    if reference_scaling_outputs is not None:
        materialized["reference_scaling_outputs"] = reference_scaling_outputs
    input_domain = _normalize_minmax_section(
        payload.get("input_domain"),
        field_name="spec['input_domain']",
        block_name="input_domain",
        allowed_keys=_DOMAIN_KEYS,
        require_positive_span=False,
    )
    if input_domain is not None:
        materialized["input_domain"] = input_domain

    validate_spec(materialized, cfg)
    if entry is not None and entry.runtime is not None:
        entry.runtime.validate_io_spec(cfg=cfg, spec=materialized)
    return materialized


def validate_spec(spec: dict[str, Any], cfg: RootConfig) -> None:
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dictionary.")
    _validate_public_spec_top_level(spec)

    for field_name in ("inputs", "outputs"):
        _normalize_iotree_section(
            spec.get(field_name), field_name=f"spec['{field_name}']"
        )
    _normalize_minmax_section(
        spec.get("reference_scaling_inputs"),
        field_name="spec['reference_scaling_inputs']",
        block_name="reference_scaling_inputs",
        allowed_keys=_SCALING_KEYS,
        require_positive_span=True,
    )
    _normalize_minmax_section(
        spec.get("reference_scaling_outputs"),
        field_name="spec['reference_scaling_outputs']",
        block_name="reference_scaling_outputs",
        allowed_keys=_SCALING_KEYS,
        require_positive_span=True,
    )
    _normalize_minmax_section(
        spec.get("input_domain"),
        field_name="spec['input_domain']",
        block_name="input_domain",
        allowed_keys=_DOMAIN_KEYS,
        require_positive_span=False,
    )

    entry = get_model_entry_from_name(cfg.model.name)
    if entry is not None and entry.runtime is not None:
        entry.runtime.validate_io_spec(cfg=cfg, spec=spec)


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_json_compatible(value.item())
        return [to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_sequence_for_display(value: list[Any], *, max_items: int) -> str | None:
    if not value:
        return "[]"
    arr = np.asarray(value)
    if arr.dtype != object and arr.size > max_items:
        return f"array(shape={arr.shape}, dtype={arr.dtype})"
    if len(value) > max_items and all(
        not isinstance(v, (dict, list, tuple)) for v in value
    ):
        return f"array(shape=({len(value)},), dtype=object)"
    return None


def format_spec_for_display(
    spec: dict[str, Any], *, max_sequence_items: int = 8, indent: int = 2
) -> str:
    def _fmt(value: Any, level: int) -> list[str]:
        pad = " " * (indent * level)
        if isinstance(value, dict):
            lines: list[str] = []
            for key, subval in value.items():
                if isinstance(subval, (dict, list, tuple)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(_fmt(subval, level + 1))
                else:
                    lines.append(f"{pad}{key}: {subval}")
            return lines

        if isinstance(value, (list, tuple)):
            seq = list(value)
            summarized = _summarize_sequence_for_display(
                seq, max_items=max_sequence_items
            )
            if summarized is not None:
                return [f"{pad}{summarized}"]

            lines = []
            for item in seq:
                if isinstance(item, (dict, list, tuple)):
                    lines.append(f"{pad}-")
                    lines.extend(_fmt(item, level + 1))
                else:
                    lines.append(f"{pad}- {item}")
            return lines

        return [f"{pad}{value}"]

    return "\n".join(_fmt(spec, 0))
