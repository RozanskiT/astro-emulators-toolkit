from __future__ import annotations

import json

import jax
import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    RootConfig,
    SolverConfig,
)
from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset

jax.config.update("jax_enable_x64", True)


def test_bundle_summary_uses_effective_spec_channel_names(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 2,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 3},
        ),
    )
    emu = Emulator.from_config(cfg)

    out = emu.save_bundle(tmp_path / "bundle_summary_channels")
    metadata = json.loads((out / "metadata.json").read_text())

    assert metadata["model_init"]["representation"] == "model-local init hints only"
    assert metadata["spec"]["outputs"]["channel_names_tree"]["flux"] == [
        "channel_0",
        "channel_1",
    ]

    readme = (out / "README.txt").read_text()
    assert "channel_names_tree:" in readme
    assert "- channel_0" in readme
    assert "- channel_1" in readme


def test_runtime_metadata_matches_spec_channel_names(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 2,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 3},
        ),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_runtime_channels")

    loaded = Emulator.from_bundle(out)
    summary = loaded.describe_bundle()

    assert "role_paths=" in summary
    assert "inputs/parameters" in summary
    assert "outputs/flux" in summary
    assert "input_metadata=none" in summary
    assert "output_metadata=names" in summary
    assert loaded.output_channel_names_tree == {"flux": ["channel_0", "channel_1"]}
    assert loaded.spec["outputs"]["channel_names_tree"]["flux"] == [
        "channel_0",
        "channel_1",
    ]


def test_transformer_payne_custom_output_leaf_gets_default_channel_names(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
            outputs=IOTreeSpec(structure_tree={"spectra": {"intensity": None}}),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 2,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 3},
        ),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(tmp_path / "bundle_runtime_custom_output_leaf")
    metadata = json.loads((out / "metadata.json").read_text())

    assert (
        metadata["runtime_contract"]["role_paths"]["output_leaf"]
        == "outputs/spectra/intensity"
    )
    assert metadata["spec"]["outputs"]["channel_names_tree"] == {
        "spectra": {"intensity": ["channel_0", "channel_1"]},
    }

    loaded = Emulator.from_bundle(out)
    assert loaded.output_channel_names_tree == {
        "spectra": {"intensity": ["channel_0", "channel_1"]},
    }


def test_cannon_bundle_summary_includes_solver_metadata(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    y = rng.normal(size=(32, 2)).astype(np.float32)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 1e-3}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    emu.fit(TreeArrayDataset(x={"parameters": x}, y={"predictions": y}))
    out = emu.save_bundle(tmp_path / "bundle_cannon_summary")

    loaded = Emulator.from_bundle(out)
    summary = loaded.describe_bundle()

    assert "solver_params=" in summary
    assert "regularize_intercept" in summary
    assert "solver_diagnostics=" in summary
    assert "condition_number" in summary
