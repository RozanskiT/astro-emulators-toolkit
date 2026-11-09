from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.canonical_wrappers import (
    CanonicalArrayModelWrapper,
)
from ..io_trees import set_leaf_by_path
from ..models.runtime_adapters import (
    ArrayRuntimeAdapter,
    TwoInputSpectralRuntimeAdapter,
    _batched_last_axis_size,
    _extract_role_leaf,
    _extract_single_array_leaf,
    _merge_init_hints,
    _merge_iotree_defaults,
    _metadata_leaf_for_role,
    _require_model_init_hint,
    _require_positive_int,
    _section_leaf_paths,
)


@dataclass(frozen=True)
class ExplicitWavelengthMLPRuntimeAdapter(TwoInputSpectralRuntimeAdapter):
    """Explicit-wavelength MLP shares the two-input spectral runtime contract.

    Everything (init-context resolution, canonical wrapping, spec
    materialization, role paths, affine leaf specs, flux device transform)
    is inherited from :class:`TwoInputSpectralRuntimeAdapter`; only the
    family identity and default output leaf key differ.
    """

    family_name: str = "experimental/explicit_wavelength_mlp"
    default_output_leaf_key: str = "predictions"


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
        out["inputs"] = _merge_iotree_defaults(
            out.get("inputs"),
            default_structure_tree={self.default_input_leaf_key: None},
        )
        out["outputs"] = _merge_iotree_defaults(
            out.get("outputs"),
            default_structure_tree={self.default_output_leaf_key: None},
        )
        # Resolve the actual output leaf path before injecting default
        # channel names (see ExplicitWavelengthMLPRuntimeAdapter).
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


@dataclass(frozen=True)
class SirenRuntimeAdapter(ArrayRuntimeAdapter):
    family_name: str = "experimental/siren"
    default_input_leaf_key: str = "parameters"
    default_output_leaf_key: str = "predictions"
