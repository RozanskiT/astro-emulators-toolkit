from __future__ import annotations

from typing import Any

from ...models import ModelRegistryEntry
from .explicit_wavelength_mlp import ExplicitWavelengthMLP, ExplicitWavelengthMLPConfig
from .mlp_2d_regression import MLP2DRegression, MLP2DRegressionConfig
from .siren import Siren, SirenConfig
from ..runtime_adapters import (
    ExplicitWavelengthMLPRuntimeAdapter,
    MLP2DRegressionRuntimeAdapter,
    SirenRuntimeAdapter,
)

_EXPERIMENTAL_MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "explicit_wavelength_mlp": ModelRegistryEntry(
        public_name="experimental/explicit_wavelength_mlp",
        family_id="experimental_explicit_wavelength_mlp_v1",
        config_cls=ExplicitWavelengthMLPConfig,
        model_cls=ExplicitWavelengthMLP,
        runtime=ExplicitWavelengthMLPRuntimeAdapter(),
    ),
    "mlp_2d_regression": ModelRegistryEntry(
        public_name="experimental/mlp_2d_regression",
        family_id="experimental_mlp_2d_regression_v1",
        config_cls=MLP2DRegressionConfig,
        model_cls=MLP2DRegression,
        runtime=MLP2DRegressionRuntimeAdapter(),
    ),
    "siren": ModelRegistryEntry(
        public_name="experimental/siren",
        family_id="experimental_siren_v1",
        config_cls=SirenConfig,
        model_cls=Siren,
        runtime=SirenRuntimeAdapter(),
    ),
}


def get_experimental_model_registry() -> dict[str, ModelRegistryEntry]:
    return dict(_EXPERIMENTAL_MODEL_REGISTRY)


def get_experimental_model_entry(name: str) -> ModelRegistryEntry:
    key = str(name).lower()
    if key not in _EXPERIMENTAL_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown experimental model '{name}'. Available={list(_EXPERIMENTAL_MODEL_REGISTRY)}"
        )
    return _EXPERIMENTAL_MODEL_REGISTRY[key]


def build_experimental_model(
    name: str,
    params: dict[str, Any],
    *,
    init_context: dict[str, object],
    rngs,
    cfg=None,
    spec: dict[str, Any] | None = None,
):
    entry = get_experimental_model_entry(name)
    if entry.runtime is None:
        raise RuntimeError(
            f"Experimental model '{name}' must define a runtime adapter."
        )
    model_cfg = entry.config_cls.from_dict(params)
    in_dim, out_dim = entry.runtime.resolve_constructor_dims(
        cfg=cfg, init_context=init_context
    )
    core_model = entry.model_cls(
        in_dim=in_dim, out_dim=out_dim, cfg=model_cfg, rngs=rngs
    )
    if spec is None or cfg is None or entry.runtime is None:
        return core_model
    return entry.runtime.wrap_model(cfg=cfg, spec=spec, core_model=core_model)
