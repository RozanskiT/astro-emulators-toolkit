from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..io_trees import (
    flatten_numeric_tree,
    normalize_tree,
    unflatten_numeric_tree,
    validate_same_structure,
)
from .extras import canonicalize_bundle_extras
from .integrity import validate_bundle_relpath, write_bundle_integrity_manifest
from .readme import render_bundle_readme
from .safetensors_io import load_arrays, save_arrays
from .versions import BUNDLE_FORMAT_VERSION


FINGERPRINT_EVALUATION_DIRNAME = "fingerprint_evaluation"
FINGERPRINT_EVALUATION_KIND = "canonical_inputs_outputs_v1"
FINGERPRINT_INPUTS_FILENAME = f"{FINGERPRINT_EVALUATION_DIRNAME}/inputs.safetensors"
FINGERPRINT_OUTPUTS_FILENAME = f"{FINGERPRINT_EVALUATION_DIRNAME}/outputs.safetensors"
FINGERPRINT_LAYOUT = "numeric_dict_tree_v1"
FINGERPRINT_INPUT_SPACE = "canonical_input_dict_trees_v1"
FINGERPRINT_OUTPUT_SPACE = "canonical_output_dict_trees_v1"
DEFAULT_RELEASE_STATUS = "released"
DEFAULT_FINGERPRINT_RTOL = 1e-5
DEFAULT_FINGERPRINT_ATOL = 1e-7


def _validate_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _validate_nonnegative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.floating, np.integer)
    ):
        raise ValueError(f"{field_name} must be a non-negative float.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be a non-negative float.")
    return normalized


def _validate_fingerprint_descriptor(
    block: Any,
    *,
    field_name: str,
    expected_space: str,
) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    if block.get("format") != "safetensors_v1":
        raise ValueError(f"{field_name}.format must be 'safetensors_v1'.")
    filename = _validate_string(
        block.get("filename"), field_name=f"{field_name}.filename"
    )
    filename = validate_bundle_relpath(filename, field_name=f"{field_name}.filename")
    filename_parts = Path(filename).parts
    if not filename_parts or filename_parts[0] != FINGERPRINT_EVALUATION_DIRNAME:
        raise ValueError(
            f"{field_name}.filename must live under '{FINGERPRINT_EVALUATION_DIRNAME}/'."
        )
    if block.get("layout") != FINGERPRINT_LAYOUT:
        raise ValueError(f"{field_name}.layout must be '{FINGERPRINT_LAYOUT}'.")
    if block.get("space") != expected_space:
        raise ValueError(f"{field_name}.space must be '{expected_space}'.")
    return {
        "format": "safetensors_v1",
        "filename": filename,
        "layout": FINGERPRINT_LAYOUT,
        "space": expected_space,
    }


def validate_fingerprint_evaluation_metadata(fingerprint: Any) -> None:
    if fingerprint is None:
        return
    if not isinstance(fingerprint, dict):
        raise ValueError(
            "Bundle metadata field 'fingerprint_evaluation' must be a dictionary."
        )
    if fingerprint.get("kind") != FINGERPRINT_EVALUATION_KIND:
        raise ValueError(
            "Bundle metadata field 'fingerprint_evaluation.kind' must be "
            f"'{FINGERPRINT_EVALUATION_KIND}'."
        )
    _validate_string(
        fingerprint.get("selection_strategy"),
        field_name="Bundle metadata field 'fingerprint_evaluation.selection_strategy'",
    )
    _validate_nonnegative_float(
        fingerprint.get("rtol"),
        field_name="Bundle metadata field 'fingerprint_evaluation.rtol'",
    )
    _validate_nonnegative_float(
        fingerprint.get("atol"),
        field_name="Bundle metadata field 'fingerprint_evaluation.atol'",
    )
    _validate_fingerprint_descriptor(
        fingerprint.get("inputs"),
        field_name="Bundle metadata field 'fingerprint_evaluation.inputs'",
        expected_space=FINGERPRINT_INPUT_SPACE,
    )
    _validate_fingerprint_descriptor(
        fingerprint.get("outputs"),
        field_name="Bundle metadata field 'fingerprint_evaluation.outputs'",
        expected_space=FINGERPRINT_OUTPUT_SPACE,
    )


def load_fingerprint_evaluation_artifacts(
    bundle_dir: str | Path,
    fingerprint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_fingerprint_evaluation_metadata(fingerprint)
    root = Path(bundle_dir)

    def _load_tree(block_name: str) -> dict[str, Any]:
        descriptor = fingerprint[block_name]
        filename = validate_bundle_relpath(
            descriptor["filename"],
            field_name=f"fingerprint_evaluation.{block_name}.filename",
        )
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"fingerprint_evaluation sidecar file not found: {path}"
            )
        try:
            arrays = load_arrays(path)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive for decoder/backend errors
            raise ValueError(
                f"Failed to decode fingerprint_evaluation sidecar: {path}"
            ) from exc
        return unflatten_numeric_tree(arrays)

    return _load_tree("inputs"), _load_tree("outputs")


def load_bundle_fingerprint_evaluation(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    fingerprint = metadata.get("fingerprint_evaluation")
    if fingerprint is None:
        raise ValueError("Bundle metadata is missing 'fingerprint_evaluation'.")
    inputs, outputs = load_fingerprint_evaluation_artifacts(root, fingerprint)
    return {
        "metadata": fingerprint,
        "inputs": inputs,
        "outputs": outputs,
    }


def _midpoint_tree(
    min_tree: dict[str, Any], max_tree: dict[str, Any]
) -> dict[str, Any]:
    validate_same_structure(
        min_tree, max_tree, name_reference="min_tree", name_other="max_tree"
    )
    midpoint_flat: dict[str, np.ndarray] = {}
    min_flat = flatten_numeric_tree(min_tree)
    max_flat = flatten_numeric_tree(max_tree)
    for path, min_arr in min_flat.items():
        max_arr = max_flat[path]
        midpoint_flat[path] = (min_arr + max_arr) * np.asarray(
            0.5, dtype=np.result_type(min_arr, max_arr, np.float32)
        )
    return unflatten_numeric_tree(midpoint_flat)


def _ensure_batch_axis_tree(tree: dict[str, Any]) -> dict[str, Any]:
    batched_flat: dict[str, np.ndarray] = {}
    for path, value in flatten_numeric_tree(tree).items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            batched_flat[path] = arr.reshape(1, 1)
            continue
        if arr.ndim == 1:
            batched_flat[path] = np.expand_dims(arr, axis=0)
            continue
        if int(arr.shape[0]) != 1:
            raise ValueError(
                f"fingerprint_inputs leaf '{path}' must describe exactly one "
                f"example; got shape {tuple(int(dim) for dim in arr.shape)}. "
                "Use an explicit leading batch axis of size 1 or provide a single unbatched example."
            )
        batched_flat[path] = arr
    return unflatten_numeric_tree(batched_flat)


def _expand_shared_input_broadcasts_for_runtime(
    tree: dict[str, Any],
    *,
    emu,
) -> dict[str, Any]:
    metadata = emu.bundle_metadata or {}
    runtime_contract = metadata.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        return tree
    affine_leaf_specs = runtime_contract.get("affine_leaf_specs")
    if not isinstance(affine_leaf_specs, dict):
        return tree

    expanded_flat: dict[str, Any] = {}
    for path, value in flatten_numeric_tree(tree).items():
        arr = np.asarray(value)
        leaf_spec = affine_leaf_specs.get(f"inputs/{path}")
        if (
            isinstance(leaf_spec, dict)
            and leaf_spec.get("mode") == "scalar_or_last_axis"
        ):
            last_axis = leaf_spec.get("last_axis")
            if isinstance(last_axis, (int, np.integer)) and not isinstance(
                last_axis, bool
            ):
                if arr.ndim == 0 or all(int(dim) == 1 for dim in arr.shape):
                    arr = np.full(
                        (1, int(last_axis)), arr.reshape(-1)[0], dtype=arr.dtype
                    )
        expanded_flat[path] = arr
    return unflatten_numeric_tree(expanded_flat)


def _release_identity(name: str, version: str, *, status: str) -> dict[str, str]:
    return {
        "name": _validate_string(name, field_name="release_name"),
        "version": _validate_string(version, field_name="release_version"),
        "status": _validate_string(status, field_name="release_status"),
    }


def _fingerprint_descriptor(*, filename: str, space: str) -> dict[str, str]:
    return {
        "format": "safetensors_v1",
        "filename": filename,
        "layout": FINGERPRINT_LAYOUT,
        "space": space,
    }


def _extract_minmax_trees(
    block: dict[str, Any] | None, *, block_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if block is None:
        raise ValueError(
            f"Cannot synthesize fingerprint inputs without explicit {block_name} metadata."
        )
    min_tree = block.get("min_tree")
    max_tree = block.get("max_tree")
    if not isinstance(min_tree, dict) or not isinstance(max_tree, dict):
        raise ValueError(
            f"{block_name} must provide inline 'min_tree' and 'max_tree' entries for automatic fingerprint inputs."
        )
    return min_tree, max_tree


def _extract_input_domain_bounds(
    block: dict[str, Any] | None, *, block_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _extract_minmax_trees(block, block_name=block_name)


def _resolve_release_fingerprint_inputs(
    emu,
    *,
    fingerprint_inputs: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    inputs_block = emu.input_spec
    structure_tree = (
        None if inputs_block is None else inputs_block.get("structure_tree")
    )
    if not isinstance(structure_tree, dict):
        raise ValueError(
            "Bundle spec is missing 'inputs.structure_tree'; cannot prepare release fingerprint."
        )

    if fingerprint_inputs is not None:
        if not isinstance(fingerprint_inputs, dict):
            raise ValueError("fingerprint_inputs must be a canonical input dict tree.")
        canonical_inputs = _ensure_batch_axis_tree(fingerprint_inputs)
        validate_same_structure(
            structure_tree,
            canonical_inputs,
            name_reference="spec.inputs.structure_tree",
            name_other="fingerprint_inputs",
        )
        flatten_numeric_tree(canonical_inputs)
        return canonical_inputs, "provided_canonical_inputs_v1"

    input_domain = emu.input_domain
    reference_scaling_inputs = emu.reference_scaling_inputs
    if input_domain is None and reference_scaling_inputs is None:
        raise ValueError(
            "Cannot synthesize fingerprint inputs without explicit input_domain or "
            "reference_scaling_inputs metadata."
        )

    source_name = "input_domain"
    source_block = input_domain
    if source_block is None:
        source_name = "reference_scaling_inputs"
        source_block = reference_scaling_inputs
        input_min, input_max = _extract_minmax_trees(
            source_block, block_name=source_name
        )
    else:
        input_min, input_max = _extract_input_domain_bounds(
            source_block, block_name=source_name
        )
    domain_midpoint = _expand_shared_input_broadcasts_for_runtime(
        _ensure_batch_axis_tree(_midpoint_tree(input_min, input_max)),
        emu=emu,
    )

    ref = reference_scaling_inputs
    if isinstance(ref, dict):
        ref_min_tree, ref_max_tree = _extract_minmax_trees(
            ref, block_name="reference_scaling_inputs"
        )
        canonical_inputs = _expand_shared_input_broadcasts_for_runtime(
            normalize_tree(domain_midpoint, ref_min_tree, ref_max_tree),
            emu=emu,
        )
        validate_same_structure(
            structure_tree,
            canonical_inputs,
            name_reference="spec.inputs.structure_tree",
            name_other="fingerprint_inputs",
        )
        return canonical_inputs, (
            "midpoint_from_input_domain_then_reference_scaling_inputs_v1"
            if source_name == "input_domain"
            else "midpoint_from_reference_scaling_inputs_v1"
        )

    validate_same_structure(
        structure_tree,
        domain_midpoint,
        name_reference="spec.inputs.structure_tree",
        name_other="fingerprint_inputs",
    )
    return domain_midpoint, "midpoint_from_input_domain_v1"


def _copy_bundle_tree(source_dir: Path, target_dir: Path) -> Path:
    if source_dir.resolve() == target_dir.resolve():
        return target_dir
    if target_dir.exists():
        entries = {path.name for path in target_dir.iterdir()}
        if entries.difference({".gitignore"}):
            raise FileExistsError(
                f"Release bundle destination already exists and is not empty: {target_dir}"
            )
        for path in source_dir.iterdir():
            destination = target_dir / path.name
            if path.is_dir():
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)
        return target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def prepare_bundle_release(
    bundle_dir: str | Path,
    *,
    path: str | Path | None = None,
    release_name: str,
    release_version: str,
    fingerprint_inputs: dict[str, Any] | None = None,
    release_status: str = DEFAULT_RELEASE_STATUS,
    rtol: float = DEFAULT_FINGERPRINT_RTOL,
    atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> Path:
    from ..emulator import Emulator

    source_root = Path(bundle_dir)
    root = source_root if path is None else _copy_bundle_tree(source_root, Path(path))
    emu = Emulator.from_bundle(root)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    canonical_inputs, selection_strategy = _resolve_release_fingerprint_inputs(
        emu,
        fingerprint_inputs=fingerprint_inputs,
    )
    outputs = emu.predict(canonical_inputs)

    save_arrays(
        root / FINGERPRINT_INPUTS_FILENAME, flatten_numeric_tree(canonical_inputs)
    )
    save_arrays(root / FINGERPRINT_OUTPUTS_FILENAME, flatten_numeric_tree(outputs))

    metadata["release"] = _release_identity(
        release_name,
        release_version,
        status=release_status,
    )
    metadata["bundle_format_version"] = BUNDLE_FORMAT_VERSION
    metadata["fingerprint_evaluation"] = {
        "kind": FINGERPRINT_EVALUATION_KIND,
        "selection_strategy": selection_strategy,
        "rtol": _validate_nonnegative_float(rtol, field_name="rtol"),
        "atol": _validate_nonnegative_float(atol, field_name="atol"),
        "inputs": _fingerprint_descriptor(
            filename=FINGERPRINT_INPUTS_FILENAME,
            space=FINGERPRINT_INPUT_SPACE,
        ),
        "outputs": _fingerprint_descriptor(
            filename=FINGERPRINT_OUTPUTS_FILENAME,
            space=FINGERPRINT_OUTPUT_SPACE,
        ),
    }
    extras = canonicalize_bundle_extras(emu.bundle_extras, dirpath=root)
    if extras is not None:
        metadata["extras"] = extras
    else:
        metadata.pop("extras", None)

    readme_text = render_bundle_readme(emu.cfg, metadata, metadata.get("fit_method"))
    (root / "README.txt").write_text(readme_text)

    metadata.pop("bundle_id", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    manifest = write_bundle_integrity_manifest(
        root,
        metadata_without_bundle_id=dict(metadata),
    )
    metadata["bundle_id"] = manifest["bundle_id"]

    verify_bundle_fingerprint_evaluation(root)
    return root


def verify_bundle_fingerprint_evaluation(bundle_dir: str | Path) -> dict[str, Any]:
    from ..emulator import Emulator

    payload = load_bundle_fingerprint_evaluation(bundle_dir)
    fingerprint = payload["metadata"]
    inputs = payload["inputs"]
    expected_outputs = payload["outputs"]
    emu = Emulator.from_bundle(bundle_dir)

    input_structure_tree = (emu.input_spec or {}).get("structure_tree")
    output_structure_tree = (emu.output_spec or {}).get("structure_tree")
    if not isinstance(input_structure_tree, dict) or not isinstance(
        output_structure_tree, dict
    ):
        raise ValueError(
            "Bundle spec is missing input/output structure trees required for fingerprint evaluation."
        )

    validate_same_structure(
        input_structure_tree,
        inputs,
        name_reference="spec.inputs.structure_tree",
        name_other="fingerprint_evaluation.inputs",
    )
    validate_same_structure(
        output_structure_tree,
        expected_outputs,
        name_reference="spec.outputs.structure_tree",
        name_other="fingerprint_evaluation.outputs",
    )

    actual_outputs = emu.predict(inputs)
    validate_same_structure(
        expected_outputs,
        actual_outputs,
        name_reference="fingerprint_evaluation.outputs",
        name_other="predicted_outputs",
    )

    expected_flat = flatten_numeric_tree(expected_outputs)
    actual_flat = flatten_numeric_tree(actual_outputs)
    rtol = float(fingerprint["rtol"])
    atol = float(fingerprint["atol"])
    max_abs_error = 0.0
    checked_paths: list[str] = []

    for path in sorted(expected_flat):
        expected = np.asarray(expected_flat[path])
        actual = np.asarray(actual_flat[path])
        max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - expected))))
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError(
                "Bundle fingerprint evaluation failed for output leaf "
                f"'{path}' with rtol={rtol} and atol={atol}."
            )
        checked_paths.append(path)

    return {
        "checked_output_paths": checked_paths,
        "max_abs_error": max_abs_error,
        "rtol": rtol,
        "atol": atol,
    }


__all__ = [
    "DEFAULT_FINGERPRINT_ATOL",
    "DEFAULT_FINGERPRINT_RTOL",
    "DEFAULT_RELEASE_STATUS",
    "FINGERPRINT_EVALUATION_DIRNAME",
    "FINGERPRINT_EVALUATION_KIND",
    "load_bundle_fingerprint_evaluation",
    "load_fingerprint_evaluation_artifacts",
    "prepare_bundle_release",
    "validate_fingerprint_evaluation_metadata",
    "verify_bundle_fingerprint_evaluation",
]
