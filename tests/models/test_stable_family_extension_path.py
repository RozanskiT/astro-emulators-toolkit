from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

import astro_emulators_toolkit.models as stable_models
from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOSpec, ModelSpec, RootConfig, TrainConfig
from astro_emulators_toolkit.data.protocols import FunctionalDeviceBatchTransform
from astro_emulators_toolkit.io_trees import get_leaf_by_path, set_leaf_by_path
from astro_emulators_toolkit.models import ModelRegistryEntry
from astro_emulators_toolkit.models.runtime_adapters import ArrayRuntimeAdapter


def _role_leaf(tree: dict[str, Any], role_path: str, *, section_name: str) -> Any:
    prefix = f"{section_name}/"
    if not role_path.startswith(prefix):
        raise ValueError(
            f"Role path '{role_path}' does not belong to '{section_name}'."
        )
    return get_leaf_by_path(tree, role_path.removeprefix(prefix))


def _batched_last_axis_size(value: Any) -> int:
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(
            "Stable family init examples must include an explicit leading batch axis and at least one non-batch axis."
        )
    if arr.ndim == 1:
        raise ValueError(
            f"Stable family init examples must use shape (1, {int(arr.shape[0])}) for a single example, got {arr.shape}."
        )
    return int(arr.shape[-1])


def _merge_hints(
    cfg: RootConfig, init_hints: dict[str, Any] | None, derived_hints: dict[str, Any]
) -> dict[str, Any]:
    merged = {str(k): v for k, v in dict(cfg.model.init_hints).items()}
    if init_hints is not None:
        merged.update({str(k): v for k, v in dict(init_hints).items()})
    merged.update(derived_hints)
    return merged


def _require_positive_int(model_name: str, hints: dict[str, Any], key: str) -> int:
    if key not in hints:
        raise ValueError(f"Model '{model_name}' requires init hint '{key}'.")
    value = int(hints[key])
    if value <= 0:
        raise ValueError(
            f"Model '{model_name}' requires init hint '{key}' > 0, got {value}."
        )
    hints[key] = value
    return value


@dataclass(frozen=True)
class PosteriorToyConfig:
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PosteriorToyConfig":
        allowed = {"dtype"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown posterior_toy params: {unknown}.")
        return cls(dtype=str(payload.get("dtype", "float32")))


class PosteriorToyModel(nnx.Module):
    def __init__(
        self, *, in_dim: int, out_dim: int, cfg: PosteriorToyConfig, rngs: nnx.Rngs
    ):
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs, dtype=jnp.dtype(cfg.dtype))

    def __call__(self, x, *, train: bool = False, rngs: nnx.Rngs | None = None):
        del train, rngs
        return self.proj(x)


@dataclass(frozen=True)
class PosteriorToyRuntimeAdapter(ArrayRuntimeAdapter):
    default_input_leaf_key: str = "context_vector"
    default_output_leaf_key: str = "posterior_logits"

    def resolve_init_context(
        self,
        *,
        cfg,
        spec: dict[str, Any],
        inputs: Any | None = None,
        outputs: Any | None = None,
        init_hints=None,
    ) -> dict[str, Any]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        derived: dict[str, Any] = {}
        if inputs is not None:
            derived["context_dim"] = _batched_last_axis_size(
                _role_leaf(inputs, role_paths["input_leaf"], section_name="inputs")
            )
        if outputs is not None:
            derived["posterior_dim"] = _batched_last_axis_size(
                _role_leaf(outputs, role_paths["output_leaf"], section_name="outputs")
            )
        resolved = _merge_hints(cfg, init_hints, derived)
        model_name = str(cfg.model.name).lower()
        _require_positive_int(model_name, resolved, "context_dim")
        _require_positive_int(model_name, resolved, "posterior_dim")
        return resolved

    def resolve_constructor_dims(
        self, *, cfg, init_context: dict[str, Any]
    ) -> tuple[int, int]:
        del cfg
        return (
            _require_positive_int("posterior_toy", init_context, "context_dim"),
            _require_positive_int("posterior_toy", init_context, "posterior_dim"),
        )

    def affine_leaf_specs(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)
        return {
            role_paths["input_leaf"]: {
                "mode": "scalar_or_last_axis",
                "last_axis": _require_positive_int(
                    "posterior_toy", model_init, "context_dim"
                ),
            },
            role_paths["output_leaf"]: {
                "mode": "scalar_or_last_axis",
                "last_axis": _require_positive_int(
                    "posterior_toy", model_init, "posterior_dim"
                ),
            },
        }

    def describe_runtime(
        self, *, cfg, spec: dict[str, Any], model_init: dict[str, Any]
    ) -> dict[str, Any]:
        runtime = super().describe_runtime(cfg=cfg, spec=spec, model_init=model_init)
        runtime["family_extension"] = {
            "init_hint_keys": ["context_dim", "posterior_dim"]
        }
        return runtime

    def make_device_batch_transform(self, *, cfg, spec: dict[str, Any], **kwargs: Any):
        del kwargs
        role_paths = self.derive_role_paths(cfg=cfg, spec=spec)

        def _transform(
            batch: dict[str, Any],
            *,
            train: bool = False,
            rng=None,
        ) -> dict[str, Any]:
            del train, rng
            x_tree: dict[str, Any] = {}
            y_tree: dict[str, Any] = {}
            set_leaf_by_path(
                x_tree, role_paths["input_leaf"].removeprefix("inputs/"), batch["x"]
            )
            set_leaf_by_path(
                y_tree, role_paths["output_leaf"].removeprefix("outputs/"), batch["y"]
            )
            return {"x": x_tree, "y": y_tree}

        return FunctionalDeviceBatchTransform(_transform)


def _register_posterior_toy_family(monkeypatch) -> None:
    monkeypatch.setitem(
        stable_models._STABLE_MODEL_REGISTRY,
        "posterior_toy",
        ModelRegistryEntry(
            config_cls=PosteriorToyConfig,
            model_cls=PosteriorToyModel,
            public_name="posterior_toy",
            family_id="posterior_toy_v1",
            runtime=PosteriorToyRuntimeAdapter(),
        ),
    )


def test_new_stable_family_can_stay_family_local(monkeypatch, tmp_path):
    _register_posterior_toy_family(monkeypatch)

    cfg = RootConfig(
        model=ModelSpec(name="posterior_toy", params={"dtype": "float32"}),
        training=TrainConfig(
            workdir=str(tmp_path / "run"), checkpoint_interval_steps=0
        ),
        io=IOSpec(),
    )

    init_example = {
        "inputs": {"context_vector": np.zeros((4, 3), dtype=np.float32)},
        "outputs": {"posterior_logits": np.zeros((4, 2), dtype=np.float32)},
    }
    emu = Emulator.from_config(cfg, init_example=init_example)

    assert emu.model_family_id == "posterior_toy_v1"
    assert emu.model_init == {"context_dim": 3, "posterior_dim": 2}
    assert emu.spec["inputs"]["structure_tree"] == {"context_vector": None}
    assert emu.spec["outputs"]["structure_tree"] == {"posterior_logits": None}

    device_batch_transform = emu.make_device_batch_transform()
    batch = device_batch_transform(
        {
            "x": np.ones((2, 3), dtype=np.float32),
            "y": np.zeros((2, 2), dtype=np.float32),
        },
        train=False,
        rng=None,
    )
    np.testing.assert_allclose(
        batch["x"]["context_vector"], np.ones((2, 3), dtype=np.float32)
    )
    np.testing.assert_allclose(
        batch["y"]["posterior_logits"], np.zeros((2, 2), dtype=np.float32)
    )
    init_batch = device_batch_transform.for_init(
        {
            "x": np.ones((1, 3), dtype=np.float32),
            "y": np.zeros((1, 2), dtype=np.float32),
        }
    )
    np.testing.assert_allclose(
        init_batch["x"]["context_vector"], np.ones((1, 3), dtype=np.float32)
    )

    pred = emu.predict({"context_vector": np.ones((5, 3), dtype=np.float32)})
    assert pred["posterior_logits"].shape == (5, 2)

    out = emu.save_bundle(tmp_path / "posterior_bundle")
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["model_family_id"] == "posterior_toy_v1"
    assert metadata["runtime_contract"]["role_paths"] == {
        "input_leaf": "inputs/context_vector",
        "output_leaf": "outputs/posterior_logits",
    }
    assert metadata["runtime_contract"]["family_extension"]["init_hint_keys"] == [
        "context_dim",
        "posterior_dim",
    ]

    loaded = Emulator.from_bundle(out)
    loaded_pred = loaded.predict({"context_vector": np.ones((5, 3), dtype=np.float32)})
    assert loaded_pred["posterior_logits"].shape == (5, 2)
    assert loaded.bundle_metadata["runtime_contract"]["family_extension"][
        "init_hint_keys"
    ] == ["context_dim", "posterior_dim"]
