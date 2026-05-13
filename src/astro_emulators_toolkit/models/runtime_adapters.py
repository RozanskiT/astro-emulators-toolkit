from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ..data.protocols import DeviceBatchTransform
from .._typing import AffineLeafSpecs
from ..config.schema import IOTreeSpec
from ..io_trees import (
    get_leaf_by_path,
    iter_leaf_paths,
    set_leaf_by_path,
    validate_semantic_broadcast_leaf_shape,
)
from .canonical_wrappers import (
    CanonicalArrayModelWrapper,
    CanonicalTransformerModelWrapper,
)
from .transformer_payne import derive_transformer_payne_channel_semantics


class RuntimeAdapter(Protocol):
    """Stable-family extension seam used by resolver/spec/runtime/bundle dispatch."""

    def resolve_init_context(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        inputs: Any | None = None,
        outputs: Any | None = None,
        init_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def resolve_constructor_dims(
        self, *, cfg, init_context: dict[str, Any]
    ) -> tuple[int, int]: ...
    def wrap_model(self, *, cfg, spec: dict[str, Any], core_model: Any) -> Any: ...
    def validate_io_spec(self, *, cfg, spec: dict[str, Any]) -> None: ...
    def materialize_spec(self, *, cfg, spec: dict[str, Any]) -> dict[str, Any]: ...
    def derive_role_paths(self, *, cfg, spec: dict[str, Any]) -> dict[str, str]: ...
    def affine_leaf_specs(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> AffineLeafSpecs: ...
    def validate_reference_scaling_inputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None: ...
    def validate_reference_scaling_outputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None: ...
    def validate_input_domain(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        domain: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None: ...
    def describe_runtime(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, Any]: ...
    def make_device_batch_transform(
        self, *, cfg, spec: dict[str, Any], **kwargs: Any
    ) -> DeviceBatchTransform: ...


def _validate_affine_leaf(
    value: Any, leaf_spec: dict[str, Any], *, field_name: str, path: str
) -> None:
    mode = str(leaf_spec.get("mode", ""))
    last_axis = leaf_spec.get("last_axis")
    validate_semantic_broadcast_leaf_shape(
        value,
        mode=mode,
        field_name=field_name,
        path=path,
        last_axis=None if last_axis is None else int(last_axis),
    )


def _validate_affine_tree(
    *, tree: dict[str, Any], expected_specs: AffineLeafSpecs, field_name: str
) -> None:
    actual_paths = {path for path, _ in iter_leaf_paths(tree)}
    expected_paths = set(expected_specs)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing:
        raise ValueError(f"{field_name} is missing required leaves: {missing}.")
    if extra:
        raise ValueError(f"{field_name} contains unexpected leaves: {extra}.")
    for path, value in iter_leaf_paths(tree):
        _validate_affine_leaf(
            value, expected_specs[path], field_name=field_name, path=path
        )


def _normalize_init_hints(init_hints: Mapping[str, Any] | None) -> dict[str, Any]:
    if init_hints is None:
        return {}
    return {str(k): v for k, v in dict(init_hints).items()}


def _merge_init_hints(
    *, cfg, init_hints: Mapping[str, Any] | None, derived_hints: dict[str, Any]
) -> dict[str, Any]:
    merged = _normalize_init_hints(cfg.model.init_hints)
    merged.update(_normalize_init_hints(init_hints))
    merged.update(derived_hints)
    return merged


def _serialize_iotree(section: dict[str, Any]) -> dict[str, Any]:
    return asdict(IOTreeSpec(**section))


def _merge_iotree_defaults(
    existing: dict[str, Any] | None,
    *,
    default_structure_tree: dict[str, Any],
    default_channel_names_tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = {} if existing is None else deepcopy(existing)
    block.setdefault("structure_tree", deepcopy(default_structure_tree))
    if "channel_names_tree" not in block and default_channel_names_tree is not None:
        block["channel_names_tree"] = deepcopy(default_channel_names_tree)
    return _serialize_iotree(block)


def _section_leaf_paths(spec: dict[str, Any], section_name: str) -> list[str]:
    section = spec.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"spec['{section_name}'] must be a dictionary.")
    structure_tree = section.get("structure_tree")
    if not isinstance(structure_tree, dict):
        raise ValueError(
            f"spec['{section_name}']['structure_tree'] must be a dictionary."
        )
    return [path for path, _ in iter_leaf_paths(structure_tree)]


def _full_section_paths(spec: dict[str, Any], section_name: str) -> list[str]:
    return [
        f"{section_name}/{path}" for path in _section_leaf_paths(spec, section_name)
    ]


def _relative_role_path(full_path: str, *, section_name: str) -> str:
    prefix = f"{section_name}/"
    if not full_path.startswith(prefix):
        raise ValueError(
            f"Role path '{full_path}' does not belong to section '{section_name}'."
        )
    return full_path.removeprefix(prefix)


def _metadata_leaf_for_role(
    spec: dict[str, Any], *, section_name: str, role_path: str, field_name: str
) -> Any:
    section = spec.get(section_name, {})
    metadata_tree = section.get(field_name)
    if metadata_tree is None:
        return None
    if not isinstance(metadata_tree, dict):
        raise ValueError(
            f"spec['{section_name}']['{field_name}'] must be a dictionary when provided."
        )
    return get_leaf_by_path(
        metadata_tree, _relative_role_path(role_path, section_name=section_name)
    )


def _final_token(path: str) -> str:
    return path.split("/")[-1]


def _select_wavelength_leaf(input_leaf_paths: list[str]) -> str:
    candidates = [
        path
        for path in input_leaf_paths
        if _final_token(path) in {"wavelength", "wavelengths"}
    ]
    if len(candidates) != 1:
        raise ValueError(
            "transformer_payne requires exactly one input leaf whose final key is 'wavelength' or 'wavelengths'."
        )
    return candidates[0]


def _input_affine_leaf_specs(
    role_paths: dict[str, str], affine_specs: AffineLeafSpecs
) -> AffineLeafSpecs:
    return {
        path: affine_specs[path]
        for key, path in role_paths.items()
        if key.endswith("input_leaf")
        or key in {"parameter_leaf", "wavelength_leaf", "input_leaf"}
    }


def _section_affine_leaf_specs(
    affine_specs: AffineLeafSpecs, *, section_name: str
) -> AffineLeafSpecs:
    prefix = f"{section_name}/"
    return {
        _relative_role_path(path, section_name=section_name): leaf_spec
        for path, leaf_spec in affine_specs.items()
        if path.startswith(prefix)
    }


def _batched_last_axis_size(value: Any, *, field_name: str) -> int:
    import numpy as np

    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(
            f"{field_name} must include an explicit leading batch axis and at least one non-batch axis, "
            f"got shape {arr.shape}. For a single scalar example, use shape (1, 1)."
        )
    if arr.ndim == 1:
        raise ValueError(
            f"{field_name} must include an explicit leading batch axis and at least one non-batch axis, "
            f"got shape {arr.shape}. For a single example, use shape (1, {int(arr.shape[0])}) instead of {arr.shape}."
        )
    return int(arr.shape[-1])


def _extract_role_leaf(
    example_tree: Any, *, role_path: str, section_name: str, field_name: str
) -> Any:
    if not isinstance(example_tree, dict):
        raise ValueError(
            f"{field_name} must be a canonical dict tree for stable model initialization."
        )
    prefix = f"{section_name}/"
    if not role_path.startswith(prefix):
        raise ValueError(
            f"Role path '{role_path}' does not belong to '{section_name}'."
        )
    return get_leaf_by_path(example_tree, role_path.removeprefix(prefix))


def _extract_single_array_leaf(value: Any, *, field_name: str) -> Any:
    if isinstance(value, dict):
        leaves = list(iter_leaf_paths(value))
        if len(leaves) != 1:
            raise ValueError(
                f"{field_name} must contain exactly one leaf, found {len(leaves)}."
            )
        return leaves[0][1]
    return value


def _require_positive_int(model_name: str, hints: dict[str, Any], key: str) -> int:
    if key not in hints:
        raise ValueError(
            f"Model '{model_name}' requires init hint '{key}'. Provide init_example=..., call initialize(...), "
            "fit(...) on a non-empty dataset, or set model.init_hints."
        )
    value = int(hints[key])
    if value <= 0:
        raise ValueError(
            f"Model '{model_name}' requires init hint '{key}' > 0, got {value}."
        )
    hints[key] = value
    return value


def _require_model_init_hint(model_init: dict[str, Any], key: str) -> int:
    if key not in model_init:
        raise ValueError(f"Runtime contract requires model init hint '{key}'.")
    value = int(model_init[key])
    if value <= 0:
        raise ValueError(
            f"Runtime contract requires model init hint '{key}' > 0, got {value}."
        )
    return value


@dataclass(frozen=True)
class ArrayRuntimeAdapter:
    family_name: str = "array_family"
    default_input_leaf_key: str = "parameters"
    default_output_leaf_key: str = "predictions"

    def resolve_init_context(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        inputs: Any | None = None,
        outputs: Any | None = None,
        init_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        derived_hints: dict[str, Any] = {}
        if inputs is not None:
            input_leaf = (
                _extract_role_leaf(
                    inputs,
                    role_path=role_paths["input_leaf"],
                    section_name="inputs",
                    field_name="init inputs",
                )
                if isinstance(inputs, dict)
                else _extract_single_array_leaf(inputs, field_name="init inputs")
            )
            derived_hints["input_last_axis"] = _batched_last_axis_size(
                input_leaf, field_name="init inputs"
            )
        if outputs is not None:
            output_leaf = (
                _extract_role_leaf(
                    outputs,
                    role_path=role_paths["output_leaf"],
                    section_name="outputs",
                    field_name="init outputs",
                )
                if isinstance(outputs, dict)
                else _extract_single_array_leaf(outputs, field_name="init outputs")
            )
            derived_hints["output_last_axis"] = _batched_last_axis_size(
                output_leaf, field_name="init outputs"
            )
        resolved = _merge_init_hints(
            cfg=cfg, init_hints=init_hints, derived_hints=derived_hints
        )
        model_name = str(cfg.model.name).lower()
        _require_positive_int(model_name, resolved, "input_last_axis")
        _require_positive_int(model_name, resolved, "output_last_axis")
        return resolved

    def resolve_constructor_dims(
        self, *, cfg, init_context: dict[str, Any]
    ) -> tuple[int, int]:
        del cfg
        return (
            _require_model_init_hint(init_context, "input_last_axis"),
            _require_model_init_hint(init_context, "output_last_axis"),
        )

    def wrap_model(self, *, cfg, spec: dict[str, Any], core_model: Any) -> Any:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        return CanonicalArrayModelWrapper(
            core_model=core_model,
            input_role_path=role_paths["input_leaf"],
            output_role_path=role_paths["output_leaf"],
        )

    def validate_io_spec(self, *, cfg, spec: dict[str, Any]) -> None:
        del cfg
        input_paths = _section_leaf_paths(spec, "inputs")
        output_paths = _section_leaf_paths(spec, "outputs")
        if len(input_paths) != 1:
            raise ValueError(
                f"{self.family_name} requires exactly one input leaf, found {len(input_paths)}."
            )
        if len(output_paths) != 1:
            raise ValueError(
                f"{self.family_name} requires exactly one output leaf, found {len(output_paths)}."
            )

    def materialize_spec(self, *, cfg, spec: dict[str, Any]) -> dict[str, Any]:
        del cfg
        out = deepcopy(spec)
        out["inputs"] = _merge_iotree_defaults(
            out.get("inputs"),
            default_structure_tree={self.default_input_leaf_key: None},
        )
        out["outputs"] = _merge_iotree_defaults(
            out.get("outputs"),
            default_structure_tree={self.default_output_leaf_key: None},
        )
        return out

    def derive_role_paths(self, *, cfg, spec: dict[str, Any]) -> dict[str, str]:
        self.validate_io_spec(cfg=cfg, spec=spec)
        return {
            "input_leaf": _full_section_paths(spec, "inputs")[0],
            "output_leaf": _full_section_paths(spec, "outputs")[0],
        }

    def affine_leaf_specs(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        input_last_axis = _require_model_init_hint(model_init, "input_last_axis")
        output_last_axis = _require_model_init_hint(model_init, "output_last_axis")
        return {
            role_paths["input_leaf"]: {
                "mode": "scalar_or_last_axis",
                "last_axis": input_last_axis,
            },
            role_paths["output_leaf"]: {
                "mode": "scalar_or_last_axis",
                "last_axis": output_last_axis,
            },
        }

    def validate_reference_scaling_inputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(scaling, dict)
            or "min_tree" not in scaling
            or "max_tree" not in scaling
        ):
            raise ValueError(
                "reference_scaling_inputs must provide 'min_tree' and 'max_tree'."
            )
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="inputs",
        )
        _validate_affine_tree(
            tree=scaling["min_tree"],
            expected_specs=specs,
            field_name="reference_scaling_inputs.min_tree",
        )
        _validate_affine_tree(
            tree=scaling["max_tree"],
            expected_specs=specs,
            field_name="reference_scaling_inputs.max_tree",
        )

    def validate_reference_scaling_outputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(scaling, dict)
            or "min_tree" not in scaling
            or "max_tree" not in scaling
        ):
            raise ValueError(
                "reference_scaling_outputs must provide 'min_tree' and 'max_tree'."
            )
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="outputs",
        )
        _validate_affine_tree(
            tree=scaling["min_tree"],
            expected_specs=specs,
            field_name="reference_scaling_outputs.min_tree",
        )
        _validate_affine_tree(
            tree=scaling["max_tree"],
            expected_specs=specs,
            field_name="reference_scaling_outputs.max_tree",
        )

    def validate_input_domain(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        domain: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(domain, dict)
            or "min_tree" not in domain
            or "max_tree" not in domain
        ):
            raise ValueError("input_domain must provide 'min_tree' and 'max_tree'.")
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="inputs",
        )
        _validate_affine_tree(
            tree=domain["min_tree"],
            expected_specs=specs,
            field_name="input_domain.min_tree",
        )
        _validate_affine_tree(
            tree=domain["max_tree"],
            expected_specs=specs,
            field_name="input_domain.max_tree",
        )

    def describe_runtime(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "surface": "canonical_dict_trees_v1",
            "role_paths": self.derive_role_paths(cfg=cfg, spec=spec),
            "affine_leaf_specs": self.affine_leaf_specs(
                cfg=cfg, spec=spec, model_init=model_init
            ),
        }

    def make_device_batch_transform(
        self, *, cfg, spec: dict[str, Any], **kwargs: Any
    ) -> DeviceBatchTransform:
        del cfg, spec, kwargs
        raise NotImplementedError(
            "Device batch transform helper is not defined for this model family."
        )


@dataclass(frozen=True)
class MLPRuntimeAdapter(ArrayRuntimeAdapter):
    family_name: str = "mlp"
    default_input_leaf_key: str = "parameters"
    default_output_leaf_key: str = "predictions"


@dataclass(frozen=True)
class CannonRuntimeAdapter(ArrayRuntimeAdapter):
    family_name: str = "cannon"
    default_input_leaf_key: str = "parameters"
    default_output_leaf_key: str = "predictions"


@dataclass(frozen=True)
class TransformerPayneRuntimeAdapter:
    def resolve_init_context(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        inputs: Any | None = None,
        outputs: Any | None = None,
        init_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del outputs
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        derived_hints: dict[str, Any] = {}
        if inputs is not None:
            parameter_leaf = (
                _extract_role_leaf(
                    inputs,
                    role_path=role_paths["parameter_leaf"],
                    section_name="inputs",
                    field_name="init inputs",
                )
                if isinstance(inputs, dict)
                else (
                    inputs[0]
                    if isinstance(inputs, (tuple, list)) and len(inputs) >= 1
                    else _extract_single_array_leaf(inputs, field_name="init inputs")
                )
            )
            derived_hints["parameter_dim"] = _batched_last_axis_size(
                parameter_leaf, field_name="init inputs"
            )
        resolved = _merge_init_hints(
            cfg=cfg, init_hints=init_hints, derived_hints=derived_hints
        )
        _require_positive_int(str(cfg.model.name).lower(), resolved, "parameter_dim")
        return resolved

    def resolve_constructor_dims(
        self, *, cfg, init_context: dict[str, Any]
    ) -> tuple[int, int]:
        del cfg
        return _require_model_init_hint(init_context, "parameter_dim"), 1

    def wrap_model(self, *, cfg, spec: dict[str, Any], core_model: Any) -> Any:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        return CanonicalTransformerModelWrapper(
            core_model=core_model,
            parameter_role_path=role_paths["parameter_leaf"],
            wavelength_role_path=role_paths["wavelength_leaf"],
            output_role_path=role_paths["output_leaf"],
        )

    def validate_io_spec(self, *, cfg, spec: dict[str, Any]) -> None:
        input_paths = _section_leaf_paths(spec, "inputs")
        output_paths = _section_leaf_paths(spec, "outputs")
        if len(input_paths) != 2:
            raise ValueError(
                f"transformer_payne requires exactly two input leaves, found {len(input_paths)}."
            )
        if len(output_paths) != 1:
            raise ValueError(
                f"transformer_payne requires exactly one output leaf, found {len(output_paths)}."
            )

        wavelength_leaf = _select_wavelength_leaf(input_paths)
        parameter_leaf = next(path for path in input_paths if path != wavelength_leaf)
        del parameter_leaf  # derived to confirm resolution is unique

        channels = int(dict(cfg.model.params).get("channels", 1))
        output_leaf = output_paths[0]
        output_channel_names = _metadata_leaf_for_role(
            spec,
            section_name="outputs",
            role_path=f"outputs/{output_leaf}",
            field_name="channel_names_tree",
        )
        if channels == 1:
            if output_channel_names not in (None, []):
                raise ValueError(
                    "transformer_payne single-channel outputs must not declare multiple channel names."
                )
            return
        if output_channel_names is None:
            raise ValueError(
                "transformer_payne multi-channel outputs require output channel names metadata."
            )
        if len(output_channel_names) != channels:
            raise ValueError(
                "transformer_payne output channel_names_tree length "
                f"({len(output_channel_names)}) must equal channels ({channels})."
            )

    def materialize_spec(self, *, cfg, spec: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(spec)
        channels = int(dict(cfg.model.params).get("channels", 1))
        out["inputs"] = _merge_iotree_defaults(
            out.get("inputs"),
            default_structure_tree={"parameters": None, "wavelengths": None},
        )
        out["outputs"] = _merge_iotree_defaults(
            out.get("outputs"),
            default_structure_tree={"flux": None},
        )
        if channels > 1 and out["outputs"].get("channel_names_tree") is None:
            output_leaf_paths = _section_leaf_paths(out, "outputs")
            if len(output_leaf_paths) == 1:
                output_channel_names_tree: dict[str, Any] = {}
                set_leaf_by_path(
                    output_channel_names_tree,
                    output_leaf_paths[0],
                    [f"channel_{i}" for i in range(channels)],
                )
                out["outputs"]["channel_names_tree"] = output_channel_names_tree
        return out

    def derive_role_paths(self, *, cfg, spec: dict[str, Any]) -> dict[str, str]:
        self.validate_io_spec(cfg=cfg, spec=spec)
        input_leaf_paths = _section_leaf_paths(spec, "inputs")
        wavelength_leaf = _select_wavelength_leaf(input_leaf_paths)
        parameter_leaf = next(
            path for path in input_leaf_paths if path != wavelength_leaf
        )
        output_leaf = _section_leaf_paths(spec, "outputs")[0]
        return {
            "parameter_leaf": f"inputs/{parameter_leaf}",
            "wavelength_leaf": f"inputs/{wavelength_leaf}",
            "output_leaf": f"outputs/{output_leaf}",
        }

    def affine_leaf_specs(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        channels = int(dict(cfg.model.params).get("channels", 1))
        output_spec = (
            {"mode": "scalar_only"}
            if channels == 1
            else {"mode": "scalar_or_last_axis", "last_axis": channels}
        )
        parameter_dim = _require_model_init_hint(model_init, "parameter_dim")
        return {
            role_paths["parameter_leaf"]: {
                "mode": "scalar_or_last_axis",
                "last_axis": parameter_dim,
            },
            role_paths["wavelength_leaf"]: {"mode": "scalar_only"},
            role_paths["output_leaf"]: output_spec,
        }

    def validate_reference_scaling_inputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(scaling, dict)
            or "min_tree" not in scaling
            or "max_tree" not in scaling
        ):
            raise ValueError(
                "reference_scaling_inputs must provide 'min_tree' and 'max_tree'."
            )
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="inputs",
        )
        _validate_affine_tree(
            tree=scaling["min_tree"],
            expected_specs=specs,
            field_name="reference_scaling_inputs.min_tree",
        )
        _validate_affine_tree(
            tree=scaling["max_tree"],
            expected_specs=specs,
            field_name="reference_scaling_inputs.max_tree",
        )

    def validate_reference_scaling_outputs(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        scaling: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(scaling, dict)
            or "min_tree" not in scaling
            or "max_tree" not in scaling
        ):
            raise ValueError(
                "reference_scaling_outputs must provide 'min_tree' and 'max_tree'."
            )
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="outputs",
        )
        _validate_affine_tree(
            tree=scaling["min_tree"],
            expected_specs=specs,
            field_name="reference_scaling_outputs.min_tree",
        )
        _validate_affine_tree(
            tree=scaling["max_tree"],
            expected_specs=specs,
            field_name="reference_scaling_outputs.max_tree",
        )

    def validate_input_domain(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        domain: dict[str, Any],
        model_init: dict[str, Any],
    ) -> None:
        if (
            not isinstance(domain, dict)
            or "min_tree" not in domain
            or "max_tree" not in domain
        ):
            raise ValueError("input_domain must provide 'min_tree' and 'max_tree'.")
        specs = _section_affine_leaf_specs(
            self.affine_leaf_specs(cfg=cfg, spec=spec, model_init=model_init),
            section_name="inputs",
        )
        _validate_affine_tree(
            tree=domain["min_tree"],
            expected_specs=specs,
            field_name="input_domain.min_tree",
        )
        _validate_affine_tree(
            tree=domain["max_tree"],
            expected_specs=specs,
            field_name="input_domain.max_tree",
        )

    def make_device_batch_transform(
        self, *, cfg, spec: dict[str, Any], **kwargs: Any
    ) -> DeviceBatchTransform:
        from .transformer_payne_batch import (
            make_flux_batch_transform,
            make_intensity_batch_transform,
        )

        channels = int(dict(cfg.model.params).get("channels", 1))
        mode = str(kwargs.pop("mode", "flux")).lower()
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        kwargs.setdefault("parameter_role_path", role_paths["parameter_leaf"])
        kwargs.setdefault("wavelength_role_path", role_paths["wavelength_leaf"])
        kwargs.setdefault("output_role_path", role_paths["output_leaf"])
        kwargs.setdefault("wavelength_dtype", "float64")
        if mode == "intensity" or channels > 1:
            output_channel_names = _metadata_leaf_for_role(
                spec,
                section_name="outputs",
                role_path=role_paths["output_leaf"],
                field_name="channel_names_tree",
            )
            runtime_names = tuple(output_channel_names or ())
            channel_semantics = derive_transformer_payne_channel_semantics(
                runtime_names
            )
            kwargs.setdefault("channels", channels)
            kwargs.setdefault("expected_output_names", runtime_names)
            if runtime_names and all(
                name.startswith("log_flux_") for name in runtime_names
            ):
                kwargs.setdefault(
                    "expected_channel_dataset_keys",
                    tuple(item["dataset_key"] for item in channel_semantics),
                )
            return make_intensity_batch_transform(**kwargs)
        return make_flux_batch_transform(**kwargs)

    def describe_runtime(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, Any]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        runtime: dict[str, Any] = {
            "surface": "canonical_dict_trees_v1",
            "role_paths": role_paths,
            "affine_leaf_specs": self.affine_leaf_specs(
                cfg=cfg, spec=spec, model_init=model_init
            ),
        }
        channels = int(dict(cfg.model.params).get("channels", 1))
        if channels > 1:
            output_channel_names = _metadata_leaf_for_role(
                spec,
                section_name="outputs",
                role_path=role_paths["output_leaf"],
                field_name="channel_names_tree",
            )
            runtime["transformer_payne_channels"] = list(
                derive_transformer_payne_channel_semantics(
                    tuple(output_channel_names or ())
                )
            )
        return runtime
