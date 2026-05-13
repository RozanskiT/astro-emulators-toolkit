from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, TypeGuard, cast

from .release import (
    load_fingerprint_evaluation_artifacts,
    validate_fingerprint_evaluation_metadata,
)
from .versions import (
    BUNDLE_FORMAT_VERSION,
    CONFIG_SCHEMA_VERSION,
    SPEC_VERSION,
    WEIGHTS_LAYOUT,
)
from ..io_trees import validate_same_structure
from ..models import get_stable_model_entry
from ..resolver import get_model_entry_from_name
from ..spec import to_json_compatible
from ..tasks import get_stable_task_registry


def _is_dataclass_instance(value: Any) -> TypeGuard[object]:
    return is_dataclass(value) and not isinstance(value, type)


def _to_plain_mapping(value: Any) -> Any:
    if _is_dataclass_instance(value):
        return _to_plain_mapping(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(k): _to_plain_mapping(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_mapping(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_mapping(v) for v in value]
    return value


def _resolve_model_config(cfg) -> dict[str, Any]:
    model_name = str(cfg.model.name).lower()
    model_params = dict(cfg.model.params)
    try:
        entry = get_stable_model_entry(model_name)
    except KeyError:
        if not model_name.startswith("experimental/"):
            return {"name": model_name, "params": to_json_compatible(model_params)}
        from ..experimental.models import get_experimental_model_registry

        exp_name = model_name.removeprefix("experimental/")
        registry = get_experimental_model_registry()
        if exp_name not in registry:
            return {"name": model_name, "params": to_json_compatible(model_params)}
        entry = registry[exp_name]

    resolved = entry.config_cls.from_dict(model_params)
    return {
        "name": model_name,
        "params": to_json_compatible(_to_plain_mapping(resolved)),
    }


def _resolve_task_config(cfg) -> dict[str, Any]:
    task_name = str(cfg.task.name).lower()
    task_params = dict(cfg.task.params)
    stable_registry = get_stable_task_registry()
    if task_name in stable_registry:
        cfg_cls, _task_cls = stable_registry[task_name]
        resolved = cfg_cls.from_dict(task_params)
        return {
            "name": task_name,
            "params": to_json_compatible(_to_plain_mapping(resolved)),
        }

    exp_name = task_name.removeprefix("experimental/")
    if task_name.startswith("experimental/"):
        from ..experimental.tasks import get_experimental_task_registry

        experimental_registry = get_experimental_task_registry()
        if exp_name in experimental_registry:
            cfg_cls, _task_cls = experimental_registry[exp_name]
            resolved = cfg_cls.from_dict(task_params)
            return {
                "name": task_name,
                "params": to_json_compatible(_to_plain_mapping(resolved)),
            }

    return {"name": task_name, "params": to_json_compatible(task_params)}


def _resolve_solver_config(cfg) -> dict[str, Any]:
    from ..training.solvers import (
        ClosedFormLinearSolverConfig,
        default_solver_for_model,
    )

    requested = str(cfg.solver.name).lower()
    model_name = str(cfg.model.name).lower()
    task_name = str(cfg.task.name).lower()
    task_params = dict(cfg.task.params)
    resolved_name = (
        default_solver_for_model(
            model_name,
            task_name=task_name,
            task_params=task_params,
        )
        if requested == "auto"
        else requested
    )

    if resolved_name == "closed_form_linear":
        resolved = ClosedFormLinearSolverConfig.from_mapping(dict(cfg.solver.params))
        params = to_json_compatible(_to_plain_mapping(resolved))
    else:
        params = to_json_compatible(dict(cfg.solver.params))
    return {"name": resolved_name, "params": params}


def build_resolved_bundle_config(cfg) -> dict[str, Any]:
    return {
        "model": _resolve_model_config(cfg),
        "task": _resolve_task_config(cfg),
        "solver": _resolve_solver_config(cfg),
    }


def get_bundle_model_init(metadata: dict[str, Any], *, cfg) -> dict[str, Any]:
    model_init = metadata.get("model_init")
    if isinstance(model_init, dict):
        hints = model_init.get("hints")
        if isinstance(hints, dict):
            return {str(k): v for k, v in hints.items()}
    return {str(k): v for k, v in dict(cfg.model.init_hints).items()}


def apply_resolved_bundle_metadata(cfg, metadata: dict[str, Any]):
    resolved = metadata.get("resolved")
    if not isinstance(resolved, dict):
        return cfg

    model_block = resolved.get("model")
    task_block = resolved.get("task")
    solver_block = resolved.get("solver")

    model = cfg.model
    if isinstance(model_block, dict):
        model_name = str(model_block.get("name", model.name))
        model_params = model_block.get("params", model.params)
        if isinstance(model_params, dict):
            model = model.__class__(
                name=model_name,
                params=dict(model_params),
                init_hints=dict(model.init_hints),
            )

    task = cfg.task
    if isinstance(task_block, dict):
        task_name = str(task_block.get("name", task.name))
        task_params = task_block.get("params", task.params)
        if isinstance(task_params, dict):
            task = task.__class__(name=task_name, params=dict(task_params))

    solver = cfg.solver
    if isinstance(solver_block, dict):
        solver_name = str(solver_block.get("name", solver.name))
        solver_params = solver_block.get("params", solver.params)
        if isinstance(solver_params, dict):
            solver = solver.__class__(name=solver_name, params=dict(solver_params))

    return cfg.with_updates(model=model, task=task, solver=solver)


def build_representation_contract(
    cfg, *, spec: dict[str, Any], model_init: dict[str, Any]
) -> dict[str, Any]:
    entry = get_model_entry_from_name(cfg.model.name)
    if entry is not None and entry.runtime is not None:
        runtime_contract = entry.runtime.describe_runtime(
            cfg=cfg,
            spec=spec,
            model_init=model_init,
        )
    else:
        runtime_contract = None
    return {
        "model_init": {
            "representation": "model-local init hints only",
            "hints": to_json_compatible(dict(model_init)),
        },
        "runtime_contract": runtime_contract,
    }


def _validate_release_metadata(release: Any) -> None:
    if release is None:
        return
    if not isinstance(release, dict):
        raise ValueError(
            "Bundle metadata field 'release' must be null or a dictionary."
        )
    required_keys = ("name", "version")
    missing = [key for key in required_keys if key not in release]
    if missing:
        raise ValueError(
            f"Bundle metadata field 'release' is missing required keys: {missing}."
        )
    for key in ("name", "version", "status"):
        value = release.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Bundle metadata field 'release.{key}' must be a string when provided."
            )


def _validate_solver_metadata(solver_metadata: Any) -> None:
    if solver_metadata is None:
        return
    if not isinstance(solver_metadata, dict):
        raise ValueError(
            "Bundle metadata field 'solver_metadata' must be null or a dictionary."
        )
    name = solver_metadata.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(
            "Bundle metadata field 'solver_metadata.name' must be a string when provided."
        )
    for key in ("params", "diagnostics", "design_matrix"):
        value = solver_metadata.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                f"Bundle metadata field 'solver_metadata.{key}' must be a dictionary when provided."
            )


def validate_bundle_header(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Bundle metadata must be a dictionary.")

    bundle_format_version = metadata.get("bundle_format_version")
    if not isinstance(bundle_format_version, int):
        raise ValueError(
            "Bundle metadata field 'bundle_format_version' must be present and be an int."
        )
    if bundle_format_version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported bundle_format_version={bundle_format_version} (expected {BUNDLE_FORMAT_VERSION})."
        )

    _validate_release_metadata(metadata.get("release"))

    config_schema_version = metadata.get("config_schema_version")
    if not isinstance(config_schema_version, int):
        raise ValueError(
            "Bundle metadata field 'config_schema_version' must be present and be an int."
        )
    if config_schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config_schema_version={config_schema_version} (expected {CONFIG_SCHEMA_VERSION})."
        )

    spec = metadata.get("spec")
    if spec is None:
        raise ValueError(
            "Bundle metadata is missing required 'spec' entry. Re-save the bundle with current toolkit version."
        )
    if not isinstance(spec, dict):
        raise ValueError("Bundle metadata field 'spec' must be a dictionary.")
    spec_version = spec.get("spec_version")
    if not isinstance(spec_version, int):
        raise ValueError(
            "Bundle metadata field 'spec.spec_version' must be present and be an int."
        )
    if spec_version != SPEC_VERSION:
        raise ValueError(
            f"Unsupported spec_version={spec_version} (expected {SPEC_VERSION})."
        )

    weights_layout = metadata.get("weights_layout")
    if not isinstance(weights_layout, str):
        raise ValueError(
            "Bundle metadata field 'weights_layout' must be present and be a string."
        )
    if weights_layout != WEIGHTS_LAYOUT:
        raise ValueError(f"Unsupported bundle weights_layout '{weights_layout}'.")

    model_family_id = metadata.get("model_family_id")
    if not isinstance(model_family_id, str) or not model_family_id:
        raise ValueError(
            "Bundle metadata field 'model_family_id' must be a non-empty string."
        )

    _validate_solver_metadata(metadata.get("solver_metadata"))
    validate_fingerprint_evaluation_metadata(metadata.get("fingerprint_evaluation"))


def validate_fingerprint_evaluation_payload(
    bundle_dir,
    metadata: dict[str, Any],
) -> None:
    fingerprint = metadata.get("fingerprint_evaluation")
    if fingerprint is None:
        return

    spec = metadata.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("Bundle metadata field 'spec' must be a dictionary.")
    inputs_block = spec.get("inputs")
    outputs_block = spec.get("outputs")
    if not isinstance(inputs_block, dict) or not isinstance(outputs_block, dict):
        raise ValueError(
            "Release bundles with fingerprint_evaluation require both 'spec.inputs' and 'spec.outputs'."
        )
    input_structure_tree = inputs_block.get("structure_tree")
    output_structure_tree = outputs_block.get("structure_tree")
    if not isinstance(input_structure_tree, dict) or not isinstance(
        output_structure_tree, dict
    ):
        raise ValueError(
            "Release bundles with fingerprint_evaluation require input/output structure trees."
        )

    fingerprint_inputs, fingerprint_outputs = load_fingerprint_evaluation_artifacts(
        bundle_dir,
        fingerprint,
    )
    validate_same_structure(
        input_structure_tree,
        fingerprint_inputs,
        name_reference="spec.inputs.structure_tree",
        name_other="fingerprint_evaluation.inputs",
    )
    validate_same_structure(
        output_structure_tree,
        fingerprint_outputs,
        name_reference="spec.outputs.structure_tree",
        name_other="fingerprint_evaluation.outputs",
    )


def validate_user_spec_for_bundle_save(spec: dict[str, Any] | None) -> None:
    if spec is None:
        return
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dictionary.")

    forbidden_keys = sorted(key for key in ("spec_version",) if key in spec)
    if forbidden_keys:
        raise ValueError(
            "save_bundle(spec=...) must not set internal compatibility keys "
            f"{forbidden_keys}; they are stamped automatically by the toolkit."
        )


__all__ = [
    "apply_resolved_bundle_metadata",
    "build_representation_contract",
    "build_resolved_bundle_config",
    "get_bundle_model_init",
    "validate_bundle_header",
    "validate_fingerprint_evaluation_payload",
    "validate_user_spec_for_bundle_save",
]
