from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from astro_emulators_toolkit.bundle.bundle import BUNDLE_INTEGRITY_FILENAME, Bundle
from astro_emulators_toolkit.bundle.versions import (
    BUNDLE_FORMAT_VERSION,
    CONFIG_SCHEMA_VERSION,
    WEIGHTS_LAYOUT,
)
from astro_emulators_toolkit.config import IOSpec, ModelSpec, RootConfig
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.spec import SPEC_VERSION


def _array_cfg() -> RootConfig:
    return RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_bundle_load_uses_canonical_filenames(tmp_path):
    cfg = RootConfig()
    params = {"dense": {"kernel": np.ones((1, 1), dtype=np.float32)}}

    bundle_dir = tmp_path / "bundle"
    Bundle(cfg=cfg, params_pure=params, metadata={"v": 1}).save(bundle_dir)

    assert (bundle_dir / "config.json").exists()
    assert (bundle_dir / "weights" / "weights.safetensors").exists()
    assert (bundle_dir / "README.txt").exists()
    assert (bundle_dir / "metadata.json").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_bundle_roundtrip_load_matches_saved_payload(tmp_path):
    cfg = RootConfig()
    params = {"dense": {"kernel": np.ones((1, 1), dtype=np.float32)}}
    metadata = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "weights_layout": WEIGHTS_LAYOUT,
    }

    bundle_dir = tmp_path / "bundle"
    Bundle(
        cfg=cfg,
        params_pure={"params": params, "model_state": {}},
        metadata=metadata,
    ).save(bundle_dir)
    loaded = Bundle.load(bundle_dir)

    assert loaded.cfg.io == cfg.io
    assert np.array_equal(
        loaded.params_pure["params"]["dense"]["kernel"], params["dense"]["kernel"]
    )
    assert loaded.metadata == metadata


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_bundle_load_requires_metadata_json(tmp_path):
    cfg = RootConfig()
    params = {"dense": {"kernel": np.ones((1, 1), dtype=np.float32)}}

    bundle_dir = tmp_path / "bundle"
    Bundle(cfg=cfg, params_pure=params, metadata={"v": 1}).save(bundle_dir)
    (bundle_dir / "metadata.json").unlink()

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        Bundle.load(bundle_dir)


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_bundle_load_requires_readme_txt(tmp_path):
    cfg = RootConfig()
    params = {"dense": {"kernel": np.ones((1, 1), dtype=np.float32)}}

    bundle_dir = tmp_path / "bundle"
    Bundle(cfg=cfg, params_pure=params, metadata={"v": 1}).save(bundle_dir)
    (bundle_dir / "README.txt").unlink()

    with pytest.raises(FileNotFoundError, match="README.txt"):
        Bundle.load(bundle_dir)


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_from_bundle_does_not_require_dataset_file_when_io_spec_present(tmp_path):
    cfg = _array_cfg()

    emu = Emulator.from_config(cfg)
    bundle_dir = tmp_path / "portable_bundle"
    emu.save_bundle(bundle_dir)

    loaded = Emulator.from_bundle(bundle_dir)
    pred = loaded.predict({"parameters": np.zeros((3, 2), dtype=np.float32)})
    assert pred["predictions"].shape == (3, 1)

    metadata = loaded.bundle_metadata or {}
    assert "model_init" in metadata
    assert metadata["model_init"]["hints"]["input_last_axis"] == 2
    assert metadata["model_init"]["hints"]["output_last_axis"] == 1
    assert metadata["runtime_contract"]["surface"] == "canonical_dict_trees_v1"
    assert "spec" in metadata
    assert metadata["spec"]["spec_version"] == SPEC_VERSION
    assert metadata["bundle_format_version"] == BUNDLE_FORMAT_VERSION
    assert metadata["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert metadata["release"] is None
    assert (bundle_dir / BUNDLE_INTEGRITY_FILENAME).exists()
    assert metadata["bundle_id"].startswith("sha256:")


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="safetensors not installed"
)
def test_bundle_load_requires_weights_in_weights_subdir(tmp_path):
    cfg = _array_cfg()
    emu = Emulator.from_config(cfg)
    bundle_dir = tmp_path / "bundle"
    emu.save_bundle(bundle_dir)

    weights_dir_file = bundle_dir / "weights" / "weights.safetensors"
    legacy = bundle_dir / "weights.safetensors"
    weights_dir_file.replace(legacy)

    with pytest.raises(FileNotFoundError, match="weights/weights.safetensors"):
        Bundle.load(bundle_dir)
