from __future__ import annotations

import json
import re

import jax
import numpy as np
import pytest

from astro_emulators_toolkit.bundle.bundle import BUNDLE_INTEGRITY_FILENAME
from astro_emulators_toolkit.bundle.extras import hydrate_bundle_extras
from astro_emulators_toolkit.bundle.metadata import validate_bundle_header
from astro_emulators_toolkit.bundle.versions import (
    BUNDLE_FORMAT_VERSION,
    CONFIG_SCHEMA_VERSION,
    SPEC_VERSION,
    WEIGHTS_LAYOUT,
)
from astro_emulators_toolkit.bundle.safetensors_io import load_weights, save_weights
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    SolverConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.bundle.integrity import write_bundle_integrity_manifest
from astro_emulators_toolkit.models.mlp import MLPConfig

jax.config.update("jax_enable_x64", True)


def test_validate_bundle_header_is_schema_level_for_model_family_id():
    validate_bundle_header(
        {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "release": None,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "spec": {"spec_version": SPEC_VERSION},
            "weights_layout": WEIGHTS_LAYOUT,
            "model_family_id": "future_family_v1",
        }
    )


def _mlp_cfg(
    *,
    input_last_axis: int,
    output_last_axis: int,
    hidden_sizes: list[int] | None = None,
    training: TrainConfig | None = None,
) -> RootConfig:
    return RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            "mlp",
            {} if hidden_sizes is None else {"hidden_sizes": hidden_sizes},
            init_hints={
                "input_last_axis": input_last_axis,
                "output_last_axis": output_last_axis,
            },
        ),
        training=TrainConfig() if training is None else training,
    )


def _cannon_cfg(*, input_last_axis: int, output_last_axis: int) -> RootConfig:
    return RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            "cannon",
            {"include_bias": True},
            init_hints={
                "input_last_axis": input_last_axis,
                "output_last_axis": output_last_axis,
            },
        ),
    )


def _mutate_first_array(value):
    if isinstance(value, dict):
        for key, child in value.items():
            mutated_child, changed = _mutate_first_array(child)
            if changed:
                updated = dict(value)
                updated[key] = mutated_child
                return updated, True
        return value, False

    arr = np.asarray(value)
    if arr.size == 0:
        return value, False

    mutated = arr.copy()
    mutated.reshape(-1)[0] += np.asarray(1e-6, dtype=mutated.dtype)
    return mutated, True


def _first_leaf_path(tree: dict[str, object], *, prefix: str = "") -> str | None:
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            child_path = _first_leaf_path(value, prefix=path)
            if child_path is not None:
                return child_path
            continue
        return path
    return None


def _mutate_leaf_value_at_path(
    tree: dict[str, object], path: str, leaf_fn
) -> dict[str, object]:
    parts = path.split("/")
    updated = dict(tree)
    node = updated
    for part in parts[:-1]:
        child = node[part]
        assert isinstance(child, dict)
        copied_child = dict(child)
        node[part] = copied_child
        node = copied_child
    node[parts[-1]] = leaf_fn(node[parts[-1]])
    return updated


def _delete_leaf_at_path(tree: dict[str, object], path: str) -> dict[str, object]:
    parts = path.split("/")
    updated = dict(tree)
    node = updated
    for part in parts[:-1]:
        child = node[part]
        assert isinstance(child, dict)
        copied_child = dict(child)
        node[part] = copied_child
        node = copied_child
    node.pop(parts[-1])
    return updated


def _reshape_leaf(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.stack([arr, arr]).astype(arr.dtype, copy=False)
    padding = np.zeros((1, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, padding], axis=0)


def _cast_leaf_dtype(value, dtype) -> np.ndarray:
    return np.asarray(value).astype(dtype)


def _add_unexpected_leaf(tree: dict[str, object], path: str) -> dict[str, object]:
    parts = path.split("/")
    updated = dict(tree)
    node = updated
    for part in parts[:-1]:
        child = node[part]
        assert isinstance(child, dict)
        copied_child = dict(child)
        node[part] = copied_child
        node = copied_child
    leaf_value = np.asarray(node[parts[-1]])
    extra_key = "unexpected_leaf"
    while extra_key in node:
        extra_key = f"{extra_key}_x"
    node[extra_key] = leaf_value.copy()
    return updated


def _refresh_bundle_integrity(bundle_dir):
    write_bundle_integrity_manifest(bundle_dir)


def test_bundle_metadata_contains_resolved_model_task_solver_configs(tmp_path):
    cfg = _mlp_cfg(
        input_last_axis=2, output_last_axis=1, hidden_sizes=[32, 16]
    ).with_updates(solver=SolverConfig(name="auto", params={}))
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_resolved")

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["resolved"]["model"]["name"] == "mlp"
    assert metadata["resolved"]["model"]["params"]["hidden_sizes"] == [32, 16]
    assert metadata["resolved"]["task"]["name"] == "regression"
    assert metadata["resolved"]["solver"]["name"] == "gradient"


def test_bundle_serialization_canonicalizes_registry_names(tmp_path):
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            " MLP ",
            {},
            init_hints={"input_last_axis": 1, "output_last_axis": 1},
        ),
        task=TaskSpec(name=" Regression ", params={}),
        solver=SolverConfig(name=" AUTO ", params={}),
        optim=OptimConfig(name=" ADAMW ", schedule=" COSINE "),
    )

    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_canonical_names")

    config_payload = json.loads((out / "config.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())

    assert config_payload["model"]["name"] == "mlp"
    assert config_payload["task"]["name"] == "regression"
    assert config_payload["solver"]["name"] == "auto"
    assert config_payload["optim"]["name"] == "adamw"
    assert config_payload["optim"]["schedule"] == "cosine"
    assert metadata["resolved"]["model"]["name"] == "mlp"
    assert metadata["resolved"]["task"]["name"] == "regression"
    assert metadata["resolved"]["solver"]["name"] == "gradient"


def test_bundle_load_prefers_resolved_model_config_when_present(tmp_path):
    cfg = _mlp_cfg(input_last_axis=2, output_last_axis=1, hidden_sizes=[19, 11])
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_prefers_resolved")

    config_path = out / "config.json"
    payload = json.loads(config_path.read_text())
    payload["model"]["params"] = {"hidden_sizes": [3, 3, 3]}
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.model.params["hidden_sizes"] == [19, 11]


def test_bundle_load_prefers_resolved_transformer_channels_when_config_drifted(
    tmp_path,
):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                channel_names_tree={"flux": ("log_flux_lines", "log_flux_continuum")},
            ),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 2,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dim_ff_multiplier": 2,
            },
            init_hints={"parameter_dim": 4},
        ),
    )
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle_prefers_resolved_transformer"
    )

    config_path = out / "config.json"
    payload = json.loads(config_path.read_text())
    payload["model"]["params"]["channels"] = 1
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.model.params["channels"] == 2
    assert loaded.spec["outputs"]["channel_names_tree"]["flux"] == [
        "log_flux_lines",
        "log_flux_continuum",
    ]


def test_bundle_load_prefers_resolved_task_config_when_present(tmp_path):
    cfg = _mlp_cfg(
        input_last_axis=2, output_last_axis=1, hidden_sizes=[8]
    ).with_updates(
        task=TaskSpec(name="regression", params={"loss": "mae", "metrics": ["mse"]})
    )
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle_prefers_resolved_task"
    )

    config_path = out / "config.json"
    payload = json.loads(config_path.read_text())
    payload["task"]["params"] = {"loss": "mse", "metrics": ["mae"]}
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.task.params["loss"] == "mae"
    assert loaded.cfg.task.params["metrics"] == ["mse"]


def test_bundle_load_prefers_resolved_solver_config_when_present(tmp_path):
    cfg = _cannon_cfg(input_last_axis=2, output_last_axis=1).with_updates(
        solver=SolverConfig(name="auto", params={})
    )
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle_prefers_resolved_solver"
    )

    config_path = out / "config.json"
    payload = json.loads(config_path.read_text())
    payload["solver"] = {"name": "gradient", "params": {}}
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.solver.name == "closed_form_linear"
    assert loaded.cfg.solver.params["ridge"] > 0.0


def test_bundle_defaults_do_not_drift_when_code_defaults_change(monkeypatch, tmp_path):
    cfg = _mlp_cfg(input_last_axis=2, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_no_drift")

    original = MLPConfig.from_dict

    def _patched_from_dict(d):
        merged = dict(d)
        if "hidden_sizes" not in merged:
            merged["hidden_sizes"] = (13,)
        return original(merged)

    monkeypatch.setattr(
        MLPConfig, "from_dict", classmethod(lambda cls, d: _patched_from_dict(d))
    )

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.model.params["hidden_sizes"] == [256, 256, 256]


def test_bundle_metadata_contains_bundle_version_header(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_weights_layout")

    metadata = json.loads((out / "metadata.json").read_text())
    manifest = json.loads((out / BUNDLE_INTEGRITY_FILENAME).read_text())
    assert metadata["bundle_format_version"] == BUNDLE_FORMAT_VERSION
    assert "bundle_id" not in metadata
    assert metadata["release"] is None
    assert metadata["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert metadata["spec"]["spec_version"] == SPEC_VERSION
    assert metadata["weights_layout"] == WEIGHTS_LAYOUT
    assert metadata["model_family_id"] == "mlp_v1"
    assert manifest["bundle_id"].startswith("sha256:")
    assert {entry["path"] for entry in manifest["tree"]} == {
        "README.txt",
        "config.json",
        "metadata.json",
        "weights/weights.safetensors",
    }

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_metadata is not None
    assert loaded.bundle_metadata["bundle_id"] == manifest["bundle_id"]


def test_bundle_config_is_sanitized_for_portability(tmp_path):
    cfg = _mlp_cfg(
        input_last_axis=1,
        output_last_axis=1,
        training=TrainConfig(workdir=str((tmp_path / "absolute_run").resolve())),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_portable")

    payload = json.loads((out / "config.json").read_text())
    assert payload["training"]["workdir"] == "./runs/from_bundle"

    loaded = Emulator.from_bundle(out)
    assert loaded.cfg.training.workdir == "./runs/from_bundle"


def test_bundle_config_schema_version_is_stamped_from_library_constant(tmp_path):
    cfg = RootConfig(
        schema_version=CONFIG_SCHEMA_VERSION + 99,
        io=IOSpec(),
        model=ModelSpec(
            "mlp",
            {},
            init_hints={"input_last_axis": 1, "output_last_axis": 1},
        ),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_schema_sanitized")

    config_payload = json.loads((out / "config.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())

    assert config_payload["schema_version"] == CONFIG_SCHEMA_VERSION
    assert metadata["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert Emulator.from_bundle(out).cfg.schema_version == CONFIG_SCHEMA_VERSION


def test_bundle_extras_roundtrip_without_polluting_spec(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    extras = {
        "companion_recipe": {
            "kind": "hf_repo_file_v1",
            "path": "predict_physical.py",
        },
        "wavelength_angstrom": [5000.0, 5001.0],
    }
    out = emu.save_bundle(tmp_path / "bundle_with_extras", extras=extras)

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_metadata is not None
    assert loaded.bundle_metadata["extras"] == extras
    assert loaded.bundle_extras == extras
    assert "extras" not in loaded.spec


def test_bundle_extras_path_dict_does_not_collide_with_sidecar_descriptor(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    extras = {
        "scientific_metadata": {"path": "some scientific path string"},
        "notes": "kept inline",
    }

    out = emu.save_bundle(tmp_path / "bundle_with_plain_path_metadata", extras=extras)

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["extras"] == extras

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_extras == extras


def test_long_numeric_bundle_extras_are_externalized_to_sidecars(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    wavelength = np.linspace(5000.0, 5010.0, 64).tolist()
    extras = {
        "notes": "shared wavelength grid",
        "wavelength_angstrom": wavelength,
    }

    out = emu.save_bundle(tmp_path / "bundle_with_extra_sidecar", extras=extras)

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["extras"]["notes"] == "shared wavelength grid"
    assert metadata["extras"]["wavelength_angstrom"] == {
        "__aet_sidecar__": {
            "path": "extras/wavelength_angstrom.safetensors",
            "format": "safetensors_v1",
            "layout": "single_array_v1",
        }
    }
    assert (out / "extras" / "wavelength_angstrom.safetensors").exists()

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_extras == extras


def test_long_numeric_bundle_extras_detect_sanitized_sidecar_filename_collision(
    tmp_path,
):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    extras = {
        "a/b": np.linspace(0.0, 1.0, 64).tolist(),
        "a_b": np.linspace(1.0, 2.0, 64).tolist(),
    }

    with pytest.raises(
        ValueError, match="extras sidecar filename collision after sanitization"
    ):
        emu.save_bundle(
            tmp_path / "bundle_with_colliding_extras_sidecars", extras=extras
        )


def test_from_bundle_loads_legacy_v1_extras_sidecar_descriptor(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    wavelength = np.linspace(5000.0, 5010.0, 64).tolist()
    extras = {
        "notes": "shared wavelength grid",
        "wavelength_angstrom": wavelength,
    }

    out = emu.save_bundle(tmp_path / "bundle_with_legacy_extra_sidecar", extras=extras)
    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["bundle_format_version"] = 1
    metadata["extras"]["wavelength_angstrom"] = {
        "path": "extras/wavelength_angstrom.safetensors"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_extras == extras


def test_direct_extras_sidecar_loader_rejects_path_traversal(tmp_path):
    extras = {
        "wavelength_angstrom": {
            "__aet_sidecar__": {
                "path": "extras/../wavelength_angstrom.safetensors",
                "format": "safetensors_v1",
                "layout": "single_array_v1",
            }
        }
    }

    with pytest.raises(ValueError, match="relative POSIX path"):
        hydrate_bundle_extras(
            extras,
            bundle_dir=tmp_path,
            bundle_format_version=BUNDLE_FORMAT_VERSION,
        )


def test_from_bundle_rejects_unknown_weights_layout(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_bad_layout")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_layout"] = "unknown_layout_v99"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Unsupported bundle weights_layout"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_future_bundle_format(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_future_format")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["bundle_format_version"] = BUNDLE_FORMAT_VERSION + 1
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Unsupported bundle_format_version"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_integrity_mismatch_after_metadata_tamper(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_tampered_metadata")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["release"] = {
        "name": "tampered-bundle",
        "version": "0.1.0",
        "status": "released",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Bundle integrity verification failed"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_integrity_mismatch_after_weights_tamper(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_tampered_weights")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    mutated_payload, changed = _mutate_first_array(payload)
    assert changed
    save_weights(weights_path, params=mutated_payload)

    with pytest.raises(ValueError, match="Bundle integrity verification failed"):
        Emulator.from_bundle(out)


def test_from_bundle_ignores_extra_files_outside_integrity_manifest(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_with_os_junk")

    manifest = json.loads((out / BUNDLE_INTEGRITY_FILENAME).read_text())
    expected_paths = {entry["path"] for entry in manifest["tree"]}

    (out / ".DS_Store").write_bytes(b"macos finder metadata")
    (out / "weights" / "Thumbs.db").write_bytes(b"windows explorer metadata")
    (out / "extras" / "desktop.ini").parent.mkdir(exist_ok=True)
    (out / "extras" / "desktop.ini").write_bytes(b"windows desktop metadata")
    (out / "__MACOSX").mkdir()
    (out / "__MACOSX" / "._weights.safetensors").write_bytes(b"appledouble sidecar")
    (out / "model-card").mkdir()
    (out / "model-card" / "README.md").write_text("Extra hosting-side documentation\n")

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_metadata is not None
    assert loaded.bundle_metadata["bundle_id"] == manifest["bundle_id"]
    assert {entry["path"] for entry in manifest["tree"]} == expected_paths


def test_from_bundle_rejects_missing_required_keys_for_weights_layout(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_missing_model_state")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    payload.pop("model_state", None)
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="missing required weights keys"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_parameter_shape_mismatch_after_integrity_refresh(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_bad_param_shape")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    path = _first_leaf_path(payload["params"])
    assert path is not None
    payload["params"] = _mutate_leaf_value_at_path(
        payload["params"], path, _reshape_leaf
    )
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(
        ValueError,
        match=rf"Loaded parameter leaf '{re.escape(path)}' has shape",
    ):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_parameter_dtype_mismatch_after_integrity_refresh(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_bad_param_dtype")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    path = _first_leaf_path(payload["params"])
    assert path is not None
    payload["params"] = _mutate_leaf_value_at_path(
        payload["params"],
        path,
        lambda value: _cast_leaf_dtype(value, np.float64),
    )
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(
        ValueError,
        match=rf"Loaded parameter leaf '{re.escape(path)}' has dtype float64, expected float32",
    ):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_missing_parameter_leaf_after_integrity_refresh(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_missing_param_leaf")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    path = _first_leaf_path(payload["params"])
    assert path is not None
    parent_path, leaf_name = path.rsplit("/", 1)
    payload["params"] = _delete_leaf_at_path(payload["params"], path)
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(
        ValueError,
        match=rf"Loaded parameter tree structure mismatch at '{re.escape(parent_path)}': missing keys=\['{re.escape(leaf_name)}'\]",
    ):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_unexpected_parameter_leaf_after_integrity_refresh(
    tmp_path,
):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle_unexpected_param_leaf"
    )

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    path = _first_leaf_path(payload["params"])
    assert path is not None
    parent_path = path.rsplit("/", 1)[0]
    payload["params"] = _add_unexpected_leaf(payload["params"], path)
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(
        ValueError,
        match=rf"Loaded parameter tree structure mismatch at '{re.escape(parent_path)}': extra keys=\['unexpected_leaf'\]",
    ):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_unexpected_model_state_leaf_after_integrity_refresh(
    tmp_path,
):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_unexpected_state")

    weights_path = out / "weights" / "weights.safetensors"
    payload = load_weights(weights_path)
    payload["model_state"] = {
        "running_stats": {"mean": np.zeros((1,), dtype=np.float32)}
    }
    save_weights(weights_path, params=payload)
    _refresh_bundle_integrity(out)

    with pytest.raises(
        ValueError,
        match=r"Loaded model_state tree structure mismatch at '<root>': extra keys=\['running_stats'\]",
    ):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_model_family_mismatch(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_family_mismatch")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_family_id"] = "cannon_v1"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="does not match resolved model family"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_unknown_model_family(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_unknown_family")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_family_id"] = "unknown_family_v99"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="does not match resolved model family"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_malformed_runtime_contract(tmp_path):
    cfg = _mlp_cfg(input_last_axis=1, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_bad_runtime_contract")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["runtime_contract"] = {"surface": "canonical_dict_trees_v1"}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="runtime_contract' is missing required keys"):
        Emulator.from_bundle(out)
