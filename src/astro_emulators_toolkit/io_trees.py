from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ._typing import PytreeDict


FlatNumericTree: TypeAlias = dict[str, np.ndarray]


def _is_dict_tree(value: Any) -> bool:
    return isinstance(value, dict)


def _display_path(path: str) -> str:
    return path or "<root>"


def _validate_tree_key(key: Any, *, field_name: str, parent_path: str) -> str:
    if not isinstance(key, str):
        raise ValueError(
            f"{field_name} keys must be strings at '{_display_path(parent_path)}'."
        )
    if not key:
        raise ValueError(f"{field_name} keys must be non-empty strings.")
    if "/" in key:
        raise ValueError(f"{field_name} keys must not contain '/': {key!r}.")
    return key


def _is_scalar_metadata_value(value: Any) -> bool:
    return value is None or isinstance(
        value, (str, bool, int, float, complex, np.generic)
    )


def _is_array_serializable_leaf(value: Any) -> bool:
    if isinstance(value, (np.ndarray, jax.Array)):
        return True
    if isinstance(value, (list, tuple)):
        return all(
            _is_array_serializable_leaf(item) or _is_scalar_array_element(item)
            for item in value
        )
    return False


def _is_scalar_array_element(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, np.generic))


def _is_metadata_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return all(_is_scalar_metadata_value(item) for item in value)


def _default_metadata_leaf_validator(value: Any, *, allow_sequences: bool) -> None:
    if _is_scalar_metadata_value(value):
        return
    if allow_sequences and _is_metadata_sequence(value):
        return
    seq_note = " or a flat metadata sequence" if allow_sequences else ""
    raise ValueError(f"metadata leaves must be scalar metadata values{seq_note}.")


def _split_leaf_path(path: str) -> list[str]:
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not path:
        raise ValueError("path must be a non-empty slash-delimited string.")
    parts = path.split("/")
    if any(part == "" for part in parts):
        raise ValueError(f"path must not contain empty segments: {path!r}.")
    if any("/" in part for part in parts):
        raise ValueError(f"path segments must not contain '/': {path!r}.")
    return parts


def iter_leaf_paths(tree: dict[str, Any], *, prefix: str = ""):
    if not isinstance(tree, dict):
        raise ValueError("tree must be a nested dict.")
    for key, value in tree.items():
        key = _validate_tree_key(key, field_name="tree", parent_path=prefix)
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from iter_leaf_paths(value, prefix=path)
        else:
            yield path, value


def get_leaf_by_path(tree: dict[str, Any], path: str) -> Any:
    node: Any = tree
    traversed: list[str] = []
    for part in _split_leaf_path(path):
        traversed_path = "/".join(traversed)
        if not isinstance(node, dict):
            raise KeyError(
                f"path '{path}' descends into a leaf at '{_display_path(traversed_path)}'."
            )
        if part not in node:
            raise KeyError(f"path '{path}' is missing segment '{part}'.")
        node = node[part]
        traversed.append(part)
    if isinstance(node, dict):
        raise KeyError(f"path '{path}' refers to a subtree, not a leaf.")
    return node


def set_leaf_by_path(tree: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    if not isinstance(tree, dict):
        raise ValueError("tree must be a nested dict.")
    parts = _split_leaf_path(path)
    node = tree
    for part in parts[:-1]:
        existing = node.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            node[part] = child
            node = child
            continue
        if not isinstance(existing, dict):
            raise KeyError(f"path '{path}' descends into a leaf at segment '{part}'.")
        node = existing
    node[parts[-1]] = value
    return tree


def validate_structure_tree(
    tree: dict[str, Any], *, field_name: str = "structure_tree"
) -> None:
    if not isinstance(tree, dict):
        raise ValueError(f"{field_name} must be a nested dict.")
    for path, value in iter_leaf_paths(tree):
        if isinstance(value, dict):
            raise ValueError(f"{field_name} leaf '{path}' must not be a dict.")
        if (
            _is_scalar_metadata_value(value)
            or _is_metadata_sequence(value)
            or _is_array_serializable_leaf(value)
        ):
            continue
        raise ValueError(
            f"{field_name} leaf '{path}' must be an array-like value or scalar metadata, got {type(value).__name__}."
        )


def validate_metadata_tree_like(
    metadata_tree: dict[str, Any],
    structure_tree: dict[str, Any],
    *,
    field_name: str = "metadata_tree",
    allow_sequences: bool = True,
    leaf_validator: Callable[[Any, str], None] | None = None,
) -> None:
    validate_structure_tree(structure_tree, field_name="structure_tree")
    if not isinstance(metadata_tree, dict):
        raise ValueError(f"{field_name} must be a nested dict.")

    def _walk(
        structure_node: dict[str, Any], metadata_node: dict[str, Any], path: str
    ) -> None:
        struct_keys = []
        for key in structure_node:
            struct_keys.append(
                _validate_tree_key(key, field_name="structure_tree", parent_path=path)
            )
        metadata_keys = []
        for key in metadata_node:
            metadata_keys.append(
                _validate_tree_key(key, field_name=field_name, parent_path=path)
            )

        missing = sorted(set(struct_keys) - set(metadata_keys))
        extra = sorted(set(metadata_keys) - set(struct_keys))
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing keys={missing}")
            if extra:
                detail.append(f"extra keys={extra}")
            raise ValueError(
                f"{field_name} structure mismatch at '{_display_path(path)}': {', '.join(detail)}"
            )

        for key in struct_keys:
            next_path = f"{path}/{key}" if path else key
            structure_value = structure_node[key]
            metadata_value = metadata_node[key]
            if isinstance(structure_value, dict):
                if not isinstance(metadata_value, dict):
                    raise ValueError(
                        f"{field_name} must be a dict at '{next_path}' to match structure_tree."
                    )
                _walk(structure_value, metadata_value, next_path)
                continue
            if isinstance(metadata_value, dict):
                raise ValueError(f"{field_name} leaf '{next_path}' must not be a dict.")
            if leaf_validator is not None:
                leaf_validator(metadata_value, next_path)
            else:
                try:
                    _default_metadata_leaf_validator(
                        metadata_value, allow_sequences=allow_sequences
                    )
                except ValueError as exc:
                    raise ValueError(f"{field_name} leaf '{next_path}' {exc}") from exc

    _walk(structure_tree, metadata_tree, "")


def validate_same_structure(
    reference_tree,
    other_tree,
    *,
    name_reference: str = "reference_tree",
    name_other: str = "other_tree",
):
    if isinstance(reference_tree, dict):
        if not isinstance(other_tree, dict):
            raise ValueError(
                f"{name_other} must be a dict at this node to match {name_reference}."
            )
        ref_keys = set(reference_tree)
        other_keys = set(other_tree)
        if ref_keys != other_keys:
            missing = sorted(ref_keys - other_keys)
            extra = sorted(other_keys - ref_keys)
            detail = []
            if missing:
                detail.append(f"missing keys={missing}")
            if extra:
                detail.append(f"extra keys={extra}")
            raise ValueError(
                f"{name_other} structure mismatch against {name_reference}: {', '.join(detail)}"
            )
        for key in reference_tree:
            validate_same_structure(
                reference_tree[key],
                other_tree[key],
                name_reference=f"{name_reference}.{key}",
                name_other=f"{name_other}.{key}",
            )
        return
    if isinstance(other_tree, dict):
        raise ValueError(f"{name_other} has a dict where {name_reference} has a leaf.")


def validate_minmax_values(
    min_tree: dict[str, Any],
    max_tree: dict[str, Any],
    *,
    field_name: str,
    require_positive_span: bool = False,
) -> None:
    validate_same_structure(
        min_tree,
        max_tree,
        name_reference=f"{field_name}.min_tree",
        name_other=f"{field_name}.max_tree",
    )
    for path, lo in iter_leaf_paths(min_tree):
        hi = get_leaf_by_path(max_tree, path)
        lo_arr = np.asarray(lo)
        hi_arr = np.asarray(hi)
        if lo_arr.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"{field_name} leaf '{path}' must be numeric.")
        if hi_arr.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"{field_name} leaf '{path}' must be numeric.")
        if not np.all(np.isfinite(lo_arr)) or not np.all(np.isfinite(hi_arr)):
            raise ValueError(f"{field_name} leaf '{path}' must be finite.")
        if require_positive_span:
            if not np.all(hi_arr > lo_arr):
                raise ValueError(f"{field_name} leaf '{path}' must have max > min.")
            continue
        if not np.all(hi_arr >= lo_arr):
            raise ValueError(f"{field_name} leaf '{path}' must have max >= min.")


def validate_channel_names_tree(channel_names_tree):
    validate_structure_tree(channel_names_tree, field_name="channel_names_tree")
    for path, value in iter_leaf_paths(channel_names_tree):
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"channel_names_tree leaf '{path}' must be None or list[str]."
            )


def _is_shared_broadcast_shape(shape: tuple[int, ...]) -> bool:
    return all(int(dim) == 1 for dim in shape)


def _is_shared_or_last_axis_broadcast_shape(
    shape: tuple[int, ...], *, last_axis: int
) -> bool:
    if _is_shared_broadcast_shape(shape):
        return True
    if not shape:
        return False
    return int(shape[-1]) == int(last_axis) and all(int(dim) == 1 for dim in shape[:-1])


def validate_semantic_broadcast_leaf_shape(
    value: Any,
    *,
    mode: str,
    field_name: str,
    path: str,
    last_axis: int | None = None,
) -> None:
    shape = tuple(int(dim) for dim in np.asarray(value).shape)
    if mode == "scalar_only":
        if _is_shared_broadcast_shape(shape):
            return
        raise ValueError(
            f"{field_name} leaf '{path}' must use a shared form with only singleton dimensions, got shape {shape}."
        )
    if mode == "scalar_or_last_axis":
        if last_axis is None:
            raise ValueError(
                f"{field_name} leaf '{path}' requires last_axis for mode 'scalar_or_last_axis'."
            )
        if _is_shared_or_last_axis_broadcast_shape(shape, last_axis=int(last_axis)):
            return
        raise ValueError(
            f"{field_name} leaf '{path}' must use a shared singleton shape or end in ({int(last_axis)},) "
            f"with only singleton leading dimensions, got shape {shape}."
        )
    raise ValueError(f"{field_name} leaf '{path}' has unsupported mode '{mode}'.")


def normalize_tree(value_tree, min_tree, max_tree):
    validate_same_structure(
        value_tree, min_tree, name_reference="value_tree", name_other="min_tree"
    )
    validate_same_structure(
        value_tree, max_tree, name_reference="value_tree", name_other="max_tree"
    )
    validate_minmax_values(
        min_tree,
        max_tree,
        field_name="normalization bounds",
        require_positive_span=True,
    )
    return jax.tree.map(
        lambda x, lo, hi: (
            (jnp.asarray(x) - jnp.asarray(lo)) / (jnp.asarray(hi) - jnp.asarray(lo))
        ),
        value_tree,
        min_tree,
        max_tree,
        is_leaf=lambda x: isinstance(x, dict) is False,
    )


def denormalize_tree(value_tree, min_tree, max_tree):
    validate_same_structure(
        value_tree, min_tree, name_reference="value_tree", name_other="min_tree"
    )
    validate_same_structure(
        value_tree, max_tree, name_reference="value_tree", name_other="max_tree"
    )
    validate_minmax_values(
        min_tree,
        max_tree,
        field_name="denormalization bounds",
        require_positive_span=False,
    )
    return jax.tree.map(
        lambda x, lo, hi: (
            jnp.asarray(lo) + jnp.asarray(x) * (jnp.asarray(hi) - jnp.asarray(lo))
        ),
        value_tree,
        min_tree,
        max_tree,
        is_leaf=lambda x: isinstance(x, dict) is False,
    )


def flatten_numeric_tree(tree: PytreeDict, *, prefix: str = "") -> FlatNumericTree:
    if not isinstance(tree, dict):
        raise ValueError("Numeric tree must be a nested dict.")
    out: FlatNumericTree = {}
    for path, value in iter_leaf_paths(tree, prefix=prefix):
        out[path] = np.asarray(value)
    return out


def unflatten_numeric_tree(flat: Mapping[str, Any]) -> PytreeDict:
    tree: PytreeDict = {}
    for key, value in flat.items():
        set_leaf_by_path(tree, str(key), np.asarray(value))
    return tree


def flatten_minmax_trees(min_tree: PytreeDict, max_tree: PytreeDict) -> FlatNumericTree:
    validate_same_structure(
        min_tree, max_tree, name_reference="min_tree", name_other="max_tree"
    )
    out: FlatNumericTree = {}
    out.update({f"min/{k}": v for k, v in flatten_numeric_tree(min_tree).items()})
    out.update({f"max/{k}": v for k, v in flatten_numeric_tree(max_tree).items()})
    return out


def unflatten_minmax_trees(
    arrays: Mapping[str, Any],
) -> tuple[PytreeDict, PytreeDict]:
    min_flat: FlatNumericTree = {}
    max_flat: FlatNumericTree = {}
    for key, value in arrays.items():
        s = str(key)
        if s.startswith("min/"):
            min_flat[s.removeprefix("min/")] = np.asarray(value)
        elif s.startswith("max/"):
            max_flat[s.removeprefix("max/")] = np.asarray(value)
    min_tree = unflatten_numeric_tree(min_flat)
    max_tree = unflatten_numeric_tree(max_flat)
    validate_same_structure(
        min_tree, max_tree, name_reference="min_tree", name_other="max_tree"
    )
    return min_tree, max_tree
