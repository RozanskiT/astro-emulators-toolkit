from __future__ import annotations

from typing import Any

from flax import nnx

from ..io_trees import get_leaf_by_path, set_leaf_by_path


def _relative_role_path(role_path: str, *, section_name: str) -> str:
    prefix = f"{section_name}/"
    if not role_path.startswith(prefix):
        raise ValueError(
            f"Role path '{role_path}' does not belong to '{section_name}'."
        )
    return role_path.removeprefix(prefix)


def _extract_leaf(
    tree: dict[str, Any], role_path: str, *, section_name: str, field_name: str
) -> Any:
    try:
        return get_leaf_by_path(
            tree, _relative_role_path(role_path, section_name=section_name)
        )
    except KeyError as exc:
        raise ValueError(
            f"Canonical {field_name} is missing required leaf '{role_path}'."
        ) from exc


def _wrap_output(role_path: str, value: Any, *, section_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    set_leaf_by_path(
        out, _relative_role_path(role_path, section_name=section_name), value
    )
    return out


class CanonicalArrayModelWrapper(nnx.Module):
    def __init__(
        self, *, core_model: nnx.Module, input_role_path: str, output_role_path: str
    ):
        self.core_model = core_model
        self.input_role_path = str(input_role_path)
        self.output_role_path = str(output_role_path)

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        if isinstance(x, dict):
            arr = _extract_leaf(
                x, self.input_role_path, section_name="inputs", field_name="input"
            )
            pred = self.core_model(arr, train=train, rngs=rngs)
            return _wrap_output(self.output_role_path, pred, section_name="outputs")
        return self.core_model(x, train=train, rngs=rngs)


class CanonicalTransformerModelWrapper(nnx.Module):
    def __init__(
        self,
        *,
        core_model: nnx.Module,
        parameter_role_path: str,
        wavelength_role_path: str,
        output_role_path: str,
    ):
        self.core_model = core_model
        self.parameter_role_path = str(parameter_role_path)
        self.wavelength_role_path = str(wavelength_role_path)
        self.output_role_path = str(output_role_path)

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        if isinstance(x, dict):
            params = _extract_leaf(
                x, self.parameter_role_path, section_name="inputs", field_name="input"
            )
            wavelengths = _extract_leaf(
                x, self.wavelength_role_path, section_name="inputs", field_name="input"
            )
            pred = self.core_model((params, wavelengths), train=train, rngs=rngs)
            return _wrap_output(self.output_role_path, pred, section_name="outputs")
        return self.core_model(x, train=train, rngs=rngs)
