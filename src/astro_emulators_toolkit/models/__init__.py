from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._typing import SupportsFromDict
from .cannon import Cannon, CannonConfig
from .mlp import MLP, MLPConfig
from .runtime_adapters import (
    CannonRuntimeAdapter,
    MLPRuntimeAdapter,
    RuntimeAdapter,
    TransformerPayneRuntimeAdapter,
)
from .transformer_payne import TransformerPayne, TransformerPayneConfig


@dataclass(frozen=True)
class ModelRegistryEntry:
    config_cls: SupportsFromDict
    model_cls: Any
    public_name: str | None = None
    family_id: str | None = None
    runtime: RuntimeAdapter | None = None


_STABLE_MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "mlp": ModelRegistryEntry(
        public_name="mlp",
        family_id="mlp_v1",
        config_cls=MLPConfig,
        model_cls=MLP,
        runtime=MLPRuntimeAdapter(),
    ),
    "transformer_payne": ModelRegistryEntry(
        public_name="transformer_payne",
        family_id="transformer_payne_v1",
        config_cls=TransformerPayneConfig,
        model_cls=TransformerPayne,
        runtime=TransformerPayneRuntimeAdapter(),
    ),
    "cannon": ModelRegistryEntry(
        public_name="cannon",
        family_id="cannon_v1",
        config_cls=CannonConfig,
        model_cls=Cannon,
        runtime=CannonRuntimeAdapter(),
    ),
}


def get_stable_model_registry() -> dict[str, ModelRegistryEntry]:
    return dict(_STABLE_MODEL_REGISTRY)


def get_stable_model_entry(name: str) -> ModelRegistryEntry:
    key = str(name).lower()
    if key not in _STABLE_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available={list(_STABLE_MODEL_REGISTRY)}"
        )
    return _STABLE_MODEL_REGISTRY[key]


def build_model(
    name: str,
    params: dict[str, Any],
    *,
    init_context: dict[str, Any],
    rngs,
    cfg=None,
    spec: dict[str, Any] | None = None,
):
    entry = get_stable_model_entry(name)
    if entry.runtime is None:
        raise RuntimeError(f"Stable model '{name}' must define a runtime adapter.")
    model_cfg = entry.config_cls.from_dict(params)
    in_dim, out_dim = entry.runtime.resolve_constructor_dims(
        cfg=cfg, init_context=init_context
    )
    core_model = entry.model_cls(
        in_dim=in_dim, out_dim=out_dim, cfg=model_cfg, rngs=rngs
    )
    if spec is None or cfg is None:
        return core_model
    return entry.runtime.wrap_model(cfg=cfg, spec=spec, core_model=core_model)
