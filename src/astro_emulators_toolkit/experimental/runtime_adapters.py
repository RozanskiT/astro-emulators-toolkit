from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..data.protocols import DeviceBatchTransform
from .._typing import AffineLeafSpecs
import numpy as np

from ..models.canonical_wrappers import (
    CanonicalArrayModelWrapper,
    CanonicalTransformerModelWrapper,
)
from ..models.runtime_adapters import (
    ArrayRuntimeAdapter,
    _batched_last_axis_size,
    _extract_role_leaf,
    _extract_single_array_leaf,
    _merge_init_hints,
    _merge_iotree_defaults,
    _metadata_leaf_for_role,
    _require_model_init_hint,
    _require_positive_int,
    _section_affine_leaf_specs,
    _section_leaf_paths,
    _select_wavelength_leaf,
    _validate_affine_tree,
)


@dataclass(frozen=True)
class ExplicitWavelengthMLPRuntimeAdapter:
    family_name: str = "experimental/explicit_wavelength_mlp"
    default_input_leaf_key: str = "parameters"
    default_wavelength_leaf_key: str = "wavelengths"
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
                f"{self.family_name} requires exactly two input leaves, found {len(input_paths)}."
            )
        if len(output_paths) != 1:
            raise ValueError(
                f"{self.family_name} requires exactly one output leaf, found {len(output_paths)}."
            )

        wavelength_leaf = _select_wavelength_leaf(input_paths)
        parameter_leaves = [path for path in input_paths if path != wavelength_leaf]
        if len(parameter_leaves) != 1:
            raise ValueError(
                f"{self.family_name} requires exactly one non-wavelength parameter leaf."
            )

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
                    f"{self.family_name} single-channel outputs must not declare channel names."
                )
            return
        if output_channel_names is None:
            raise ValueError(
                f"{self.family_name} multi-channel outputs require output channel names metadata."
            )
        if len(output_channel_names) != channels:
            raise ValueError(
                f"{self.family_name} output channel_names_tree length ({len(output_channel_names)}) "
                f"must equal channels ({channels})."
            )

    def materialize_spec(self, *, cfg, spec: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(spec)
        channels = int(dict(cfg.model.params).get("channels", 1))
        output_channel_names_tree = None
        if channels > 1:
            output_channel_names_tree = {
                self.default_output_leaf_key: [f"channel_{i}" for i in range(channels)]
            }
        out["inputs"] = _merge_iotree_defaults(
            out.get("inputs"),
            default_structure_tree={
                self.default_input_leaf_key: None,
                self.default_wavelength_leaf_key: None,
            },
        )
        out["outputs"] = _merge_iotree_defaults(
            out.get("outputs"),
            default_structure_tree={self.default_output_leaf_key: None},
            default_channel_names_tree=output_channel_names_tree,
        )
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
    ) -> AffineLeafSpecs:
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
        from ..models.transformer_payne_batch import make_flux_batch_transform

        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        kwargs.setdefault("parameter_role_path", role_paths["parameter_leaf"])
        kwargs.setdefault("wavelength_role_path", role_paths["wavelength_leaf"])
        kwargs.setdefault("output_role_path", role_paths["output_leaf"])
        return make_flux_batch_transform(**kwargs)

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


@dataclass(frozen=True)
class MLP2DRegressionRuntimeAdapter(ArrayRuntimeAdapter):
    family_name: str = "experimental/mlp_2d_regression"
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
        derived_hints: dict[str, Any] = {
            "output_last_axis": int(dict(cfg.model.params).get("channels", 2)),
        }
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
            arr = np.asarray(output_leaf)
            channels = int(dict(cfg.model.params).get("channels", 2))
            if arr.ndim == 1:
                if channels != 1:
                    raise ValueError(
                        "experimental/mlp_2d_regression output examples must include a channel axis when channels > 1."
                    )
                derived_hints["output_sequence_length"] = 1
            elif arr.ndim == 2:
                if channels != 1:
                    raise ValueError(
                        "experimental/mlp_2d_regression output examples must have "
                        "shape (batch, sequence, channels) when channels > 1."
                    )
                derived_hints["output_sequence_length"] = int(arr.shape[-1])
            else:
                if int(arr.shape[-1]) != channels:
                    raise ValueError(
                        "experimental/mlp_2d_regression expected last output axis "
                        f"to equal channels={channels}, got shape {arr.shape}."
                    )
                derived_hints["output_sequence_length"] = int(arr.shape[-2])
        resolved = _merge_init_hints(
            cfg=cfg, init_hints=init_hints, derived_hints=derived_hints
        )
        model_name = str(cfg.model.name).lower()
        _require_positive_int(model_name, resolved, "input_last_axis")
        _require_positive_int(model_name, resolved, "output_last_axis")
        _require_positive_int(model_name, resolved, "output_sequence_length")
        return resolved

    def resolve_constructor_dims(
        self, *, cfg, init_context: dict[str, Any]
    ) -> tuple[int, int]:
        del cfg
        return (
            _require_model_init_hint(init_context, "input_last_axis"),
            _require_model_init_hint(init_context, "output_sequence_length"),
        )

    def wrap_model(self, *, cfg, spec: dict[str, Any], core_model: Any) -> Any:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        return CanonicalArrayModelWrapper(
            core_model=core_model,
            input_role_path=role_paths["input_leaf"],
            output_role_path=role_paths["output_leaf"],
        )

    def validate_io_spec(self, *, cfg, spec: dict[str, Any]) -> None:
        super().validate_io_spec(cfg=cfg, spec=spec)
        channels = int(dict(cfg.model.params).get("channels", 2))
        output_leaf = _section_leaf_paths(spec, "outputs")[0]
        output_channel_names = _metadata_leaf_for_role(
            spec,
            section_name="outputs",
            role_path=f"outputs/{output_leaf}",
            field_name="channel_names_tree",
        )
        if output_channel_names is None:
            if channels > 1:
                raise ValueError(
                    f"{self.family_name} multi-channel outputs require output channel names metadata."
                )
            return
        if len(output_channel_names) != channels:
            raise ValueError(
                f"{self.family_name} output channel_names_tree length ({len(output_channel_names)}) "
                f"must equal channels ({channels})."
            )

    def materialize_spec(self, *, cfg, spec: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(spec)
        channels = int(dict(cfg.model.params).get("channels", 2))
        output_channel_names_tree = None
        if channels > 1:
            output_channel_names_tree = {
                self.default_output_leaf_key: [f"channel_{i}" for i in range(channels)]
            }
        out["inputs"] = _merge_iotree_defaults(
            out.get("inputs"),
            default_structure_tree={self.default_input_leaf_key: None},
        )
        out["outputs"] = _merge_iotree_defaults(
            out.get("outputs"),
            default_structure_tree={self.default_output_leaf_key: None},
            default_channel_names_tree=output_channel_names_tree,
        )
        return out


@dataclass(frozen=True)
class SirenRuntimeAdapter(ArrayRuntimeAdapter):
    family_name: str = "experimental/siren"
    default_input_leaf_key: str = "parameters"
    default_output_leaf_key: str = "predictions"
