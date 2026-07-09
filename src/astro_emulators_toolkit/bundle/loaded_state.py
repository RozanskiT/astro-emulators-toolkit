"""Validation of loaded bundle weight payloads against an initialized model.

Loading a bundle re-initializes the model from its saved config and then
checks the loaded parameter/model-state trees leaf-by-leaf (structure, shape,
dtype) against the freshly initialized ones, so a bundle can never silently
load into a mismatched architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import PytreeDict
from ..config.schema import RootConfig
from ..resolver import get_model_entry_from_name
from .versions import SUPPORTED_WEIGHTS_LAYOUTS


def validate_bundle_weights_payload(
    metadata: dict[str, Any], params_pure: dict[str, Any]
) -> None:
    layout = metadata.get("weights_layout")
    if layout not in SUPPORTED_WEIGHTS_LAYOUTS:
        raise ValueError(
            f"Unsupported bundle weights_layout '{layout}' "
            f"(supported: {sorted(SUPPORTED_WEIGHTS_LAYOUTS)})."
        )
    required_keys = ("params", "model_state")
    missing = [key for key in required_keys if key not in params_pure]
    if missing:
        raise ValueError(
            f"weights_layout '{layout}' is missing required weights keys: {missing}."
        )


def _display_tree_path(path: str) -> str:
    return path or "<root>"


def _normalize_tree_keys_for_bundle_validation(
    tree: Any, *, tree_name: str, path: str = ""
) -> Any:
    if not isinstance(tree, dict):
        return tree
    normalized: dict[str, Any] = {}
    for raw_key, value in tree.items():
        key = str(raw_key)
        if not key:
            raise ValueError(
                f"{tree_name} contains an empty key at '{_display_tree_path(path)}'."
            )
        if "/" in key:
            raise ValueError(
                f"{tree_name} contains key {raw_key!r} with '/' at '{_display_tree_path(path)}'."
            )
        if key in normalized:
            raise ValueError(
                f"{tree_name} key collision after string normalization at '{_display_tree_path(path)}': {raw_key!r}."
            )
        next_path = f"{path}/{key}" if path else key
        normalized[key] = _normalize_tree_keys_for_bundle_validation(
            value, tree_name=tree_name, path=next_path
        )
    return normalized


def _validate_loaded_tree_matches_initialized(
    *,
    initialized_tree: Any,
    loaded_tree: Any,
    tree_name: str,
    path: str = "",
) -> None:
    if isinstance(initialized_tree, dict):
        if not isinstance(loaded_tree, dict):
            raise ValueError(
                f"Loaded {tree_name} tree has a leaf at '{_display_tree_path(path)}' "
                f"where the initialized {tree_name} tree has a subtree."
            )
        initialized_keys = set(initialized_tree)
        loaded_keys = set(loaded_tree)
        if initialized_keys != loaded_keys:
            missing = sorted(initialized_keys - loaded_keys)
            extra = sorted(loaded_keys - initialized_keys)
            detail: list[str] = []
            if missing:
                detail.append(f"missing keys={missing}")
            if extra:
                detail.append(f"extra keys={extra}")
            raise ValueError(
                f"Loaded {tree_name} tree structure mismatch at '{_display_tree_path(path)}': {', '.join(detail)}."
            )
        for key in sorted(initialized_keys):
            next_path = f"{path}/{key}" if path else key
            _validate_loaded_tree_matches_initialized(
                initialized_tree=initialized_tree[key],
                loaded_tree=loaded_tree[key],
                tree_name=tree_name,
                path=next_path,
            )
        return
    if isinstance(loaded_tree, dict):
        raise ValueError(
            f"Loaded {tree_name} tree has a subtree at '{_display_tree_path(path)}' "
            f"where the initialized {tree_name} tree has a leaf."
        )

    initialized_arr = np.asarray(initialized_tree)
    loaded_arr = np.asarray(loaded_tree)
    initialized_shape = tuple(int(dim) for dim in initialized_arr.shape)
    loaded_shape = tuple(int(dim) for dim in loaded_arr.shape)
    if initialized_shape != loaded_shape:
        raise ValueError(
            f"Loaded {tree_name} leaf '{_display_tree_path(path)}' has shape "
            f"{loaded_shape}, expected {initialized_shape}."
        )
    initialized_dtype = np.dtype(initialized_arr.dtype)
    loaded_dtype = np.dtype(loaded_arr.dtype)
    if initialized_dtype != loaded_dtype:
        raise ValueError(
            f"Loaded {tree_name} leaf '{_display_tree_path(path)}' has dtype "
            f"{loaded_dtype}, expected {initialized_dtype}."
        )


def validate_loaded_runtime_trees_match_initialization(
    *,
    initialized_params: PytreeDict,
    loaded_params: PytreeDict,
    initialized_model_state: PytreeDict,
    loaded_model_state: PytreeDict,
) -> None:
    normalized_initialized_params = _normalize_tree_keys_for_bundle_validation(
        initialized_params, tree_name="initialized parameter tree"
    )
    normalized_loaded_params = _normalize_tree_keys_for_bundle_validation(
        loaded_params, tree_name="loaded parameter tree"
    )
    _validate_loaded_tree_matches_initialized(
        initialized_tree=normalized_initialized_params,
        loaded_tree=normalized_loaded_params,
        tree_name="parameter",
    )

    normalized_initialized_model_state = _normalize_tree_keys_for_bundle_validation(
        initialized_model_state, tree_name="initialized model_state tree"
    )
    normalized_loaded_model_state = _normalize_tree_keys_for_bundle_validation(
        loaded_model_state, tree_name="loaded model_state tree"
    )
    _validate_loaded_tree_matches_initialized(
        initialized_tree=normalized_initialized_model_state,
        loaded_tree=normalized_loaded_model_state,
        tree_name="model_state",
    )


def validate_runtime_contract_metadata(
    metadata: dict[str, Any], *, cfg: RootConfig, model_init: dict[str, Any]
) -> None:
    runtime_contract = metadata.get("runtime_contract")
    if runtime_contract is None:
        return
    if not isinstance(runtime_contract, dict):
        raise ValueError(
            "Bundle metadata field 'runtime_contract' must be a dictionary when present."
        )
    new_required_keys = ("surface", "role_paths", "affine_leaf_specs")
    missing_new = [key for key in new_required_keys if key not in runtime_contract]
    if missing_new:
        raise ValueError(
            f"Bundle metadata field 'runtime_contract' is missing required keys: {missing_new}."
        )
    if not isinstance(runtime_contract.get("role_paths"), dict):
        raise ValueError(
            "Bundle metadata field 'runtime_contract.role_paths' must be a dictionary."
        )
    if "affine_leaf_specs" in runtime_contract and not isinstance(
        runtime_contract.get("affine_leaf_specs"), dict
    ):
        raise ValueError(
            "Bundle metadata field 'runtime_contract.affine_leaf_specs' must be a dictionary."
        )

    entry = get_model_entry_from_name(cfg.model.name)
    if entry is None or entry.runtime is None:
        return
    expected = entry.runtime.describe_runtime(
        cfg=cfg, spec=metadata["spec"], model_init=model_init
    )
    if runtime_contract != expected:
        raise ValueError(
            "Bundle metadata runtime_contract does not match the resolved model family runtime contract."
        )


__all__ = [
    "validate_bundle_weights_payload",
    "validate_loaded_runtime_trees_match_initialization",
    "validate_runtime_contract_metadata",
]
