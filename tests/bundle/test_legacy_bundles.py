"""Bundles stamped config schema v1 keep loading, with their original meaning.

Two toolkit versions stamped v1: 0.1.0, and the 0.2.0 pre-release 0.2.0.dev0
(which already had v2 meaning). The fixtures under fixtures/v0_1_0/ and
fixtures/v0_2_0_dev0/ were written by that code itself (see
fixtures/make_fixtures.py), together with the predictions it made.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle.integrity import verify_bundle_integrity
from astro_emulators_toolkit.config.migrations import LegacyConfigWarning
from astro_emulators_toolkit.config.schema import CONFIG_SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures" / "v0_1_0"
DEV_FIXTURES = Path(__file__).parent / "fixtures" / "v0_2_0_dev0"
REFERENCE_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "assets"
    / "reference_bundle_release"
)


def _probe() -> dict[str, np.ndarray]:
    return {"parameters": np.load(FIXTURES / "probe_inputs.npy")}


def _predict(emu: Emulator) -> np.ndarray:
    return np.asarray(emu.predict(_probe())["predictions"], dtype=np.float32)


def _load_with_legacy_warnings(path: Path) -> tuple[Emulator, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emu = Emulator.from_bundle(path)
    messages = [
        str(w.message) for w in caught if issubclass(w.category, LegacyConfigWarning)
    ]
    return emu, messages


def test_v010_bundle_with_ignored_settings_loads_with_one_warning():
    bundle = FIXTURES / "mlp_ignored_settings"

    emu, messages = _load_with_legacy_warnings(bundle)

    assert len(messages) == 1
    (message,) = messages
    assert "config schema v1" in message
    assert "scale_embedding_lr=0.5" in message
    assert "decay_steps=3" in message
    # Reported by both config.json and metadata["resolved"]; listed once.
    assert message.count("solver.params ['foo']") == 1

    assert emu.cfg.schema_version == CONFIG_SCHEMA_VERSION
    assert emu.cfg.optim.scale_embedding_lr == 1.0
    assert emu.cfg.optim.decay_steps == 0
    assert dict(emu.cfg.solver.params) == {}
    np.testing.assert_allclose(
        _predict(emu),
        np.load(FIXTURES / "mlp_ignored_settings.predictions.npy"),
        rtol=1e-5,
        atol=1e-6,
    )

    # The upgrade is in memory only.
    on_disk = json.loads((bundle / "config.json").read_text())
    assert on_disk["schema_version"] == 1
    assert on_disk["optim"]["scale_embedding_lr"] == 0.5
    verify_bundle_integrity(bundle)


def test_v010_bfloat16_bundle_keeps_float32_parameter_storage():
    emu, messages = _load_with_legacy_warnings(
        FIXTURES / "mlp_bfloat16_auto_solver_params"
    )

    # 'auto' + params in config.json is valid; the resolved 'gradient'
    # solver is what carried the ignored params.
    assert len(messages) == 1
    assert "solver.params ['foo']" in messages[0]

    assert emu.cfg.model.params["dtype"] == "bfloat16"
    assert emu.cfg.model.params["param_dtype"] == "float32"
    leaf_dtypes = {np.dtype(leaf.dtype) for leaf in jax.tree.leaves(emu.params)}
    assert leaf_dtypes == {np.dtype("float32")}
    # bfloat16 compute: allow for platform differences at bf16 precision.
    np.testing.assert_allclose(
        _predict(emu),
        np.load(FIXTURES / "mlp_bfloat16_auto_solver_params.predictions.npy"),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "name", ["mlp_ignored_settings", "mlp_bfloat16_auto_solver_params"]
)
def test_resaving_a_v010_bundle_writes_a_strict_current_schema_bundle(tmp_path, name):
    emu, _ = _load_with_legacy_warnings(FIXTURES / name)
    out = tmp_path / "resaved"
    emu.save_bundle(out)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyConfigWarning)
        reloaded = Emulator.from_bundle(out)

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["config_schema_version"] == CONFIG_SCHEMA_VERSION
    np.testing.assert_array_equal(_predict(reloaded), _predict(emu))


def test_v020dev0_bundle_stamped_v1_keeps_its_bfloat16_parameter_storage():
    # 0.2.0.dev0 stamped v1 but already stored parameters in ``dtype``; the
    # provenance toolkit version and the saved weights both say so.
    emu, messages = _load_with_legacy_warnings(DEV_FIXTURES / "mlp_bfloat16")

    assert messages == []
    assert emu.bundle_metadata["config_schema_version"] == 1
    assert emu.cfg.model.params["dtype"] == "bfloat16"
    leaf_dtypes = {np.dtype(leaf.dtype) for leaf in jax.tree.leaves(emu.params)}
    assert leaf_dtypes == {jax.numpy.dtype("bfloat16")}
    probe = {"parameters": np.load(DEV_FIXTURES / "probe_inputs.npy")}
    np.testing.assert_allclose(
        np.asarray(emu.predict(probe)["predictions"], dtype=np.float32),
        np.load(DEV_FIXTURES / "mlp_bfloat16.predictions.npy"),
        rtol=1e-2,
        atol=1e-2,
    )


def test_v1_reference_bundle_loads_without_legacy_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyConfigWarning)
        emu = Emulator.from_bundle(REFERENCE_BUNDLE)
    assert emu.bundle_metadata["config_schema_version"] == 1
