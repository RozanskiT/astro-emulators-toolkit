from __future__ import annotations

import json

import jax
import pytest

from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, ModelSpec, RootConfig
from astro_emulators_toolkit.emulator import Emulator

jax.config.update("jax_enable_x64", True)


def test_bundle_runtime_contract_uses_role_paths_and_leaf_specs(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"stellar_labels": None},
                channel_names_tree={"stellar_labels": ["teff", "logg"]},
            ),
            outputs=IOTreeSpec(structure_tree={"spectra": {"flux": None}}),
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
    )
    out = Emulator.from_config(cfg).save_bundle(tmp_path / "bundle_family")
    runtime = json.loads((out / "metadata.json").read_text())["runtime_contract"]

    assert runtime["surface"] == "canonical_dict_trees_v1"
    assert runtime["role_paths"] == {
        "input_leaf": "inputs/stellar_labels",
        "output_leaf": "outputs/spectra/flux",
    }
    assert "input_channel_names_tree" not in runtime
    assert "output_channel_names_tree" not in runtime
    assert (
        runtime["affine_leaf_specs"]["outputs/spectra/flux"]["mode"]
        == "scalar_or_last_axis"
    )


def test_array_family_materializes_default_inputs_outputs_when_omitted():
    cfg = RootConfig(io=IOSpec())
    emu = Emulator.from_config(cfg)

    assert emu.spec["inputs"]["structure_tree"] == {"parameters": None}
    assert emu.spec["outputs"]["structure_tree"] == {"predictions": None}
    assert emu.input_channel_names_tree is None
    assert emu.output_channel_names_tree is None


def test_transformer_runtime_contract_uses_wavelength_role_resolution_and_derived_channel_semantics():
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={
                    "stellar_labels": None,
                    "observation": {"wavelengths": None},
                },
                channel_names_tree={
                    "stellar_labels": ["teff", "logg", "feh"],
                    "observation": {"wavelengths": None},
                },
            ),
            outputs=IOTreeSpec(
                structure_tree={"spectra": {"flux": None}},
                channel_names_tree={
                    "spectra": {"flux": ["log_flux_lines", "log_flux_continuum"]}
                },
            ),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={"channels": 2},
            init_hints={"parameter_dim": 3},
        ),
    )
    emu = Emulator.from_config(cfg)
    runtime = (
        emu.bundle_metadata["runtime_contract"]
        if emu.bundle_metadata is not None
        else emu._stable_model_entry.runtime.describe_runtime(
            cfg=emu.cfg, spec=emu.spec, model_init=emu.model_init
        )
    )

    assert emu.output_channel_names_tree == {
        "spectra": {"flux": ["log_flux_lines", "log_flux_continuum"]}
    }
    assert runtime["role_paths"] == {
        "parameter_leaf": "inputs/stellar_labels",
        "wavelength_leaf": "inputs/observation/wavelengths",
        "output_leaf": "outputs/spectra/flux",
    }
    assert runtime["transformer_payne_channels"] == [
        {"name": "log_flux_lines", "dataset_key": "lines"},
        {"name": "log_flux_continuum", "dataset_key": "continuum"},
    ]


def test_transformer_payne_requires_resolvable_wavelength_leaf():
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"stellar_labels": None, "grid": None}),
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                channel_names_tree={"flux": ["log_flux_lines", "log_flux_continuum"]},
            ),
        ),
        model=ModelSpec(name="transformer_payne", params={"channels": 2}),
    )
    with pytest.raises(ValueError, match="final key is 'wavelength' or 'wavelengths'"):
        Emulator.from_config(cfg)


def test_experimental_runtime_adapters_materialize_canonical_specs_without_resolver_leaks():
    explicit_cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp",
            params={"channels": 1},
            init_hints={"parameter_dim": 3},
        ),
    )
    explicit = Emulator.from_config(explicit_cfg)

    assert explicit.model_family_id == "experimental_explicit_wavelength_mlp_v1"
    assert explicit.spec["inputs"]["structure_tree"] == {
        "parameters": None,
        "wavelengths": None,
    }
    assert explicit.spec["outputs"]["structure_tree"] == {"predictions": None}

    mlp2d_cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="experimental/mlp_2d_regression",
            params={"channels": 2},
            init_hints={"input_last_axis": 4, "output_sequence_length": 6},
        ),
    )
    mlp2d = Emulator.from_config(mlp2d_cfg)

    assert mlp2d.model_family_id == "experimental_mlp_2d_regression_v1"
    assert mlp2d.spec["inputs"]["structure_tree"] == {"parameters": None}
    assert mlp2d.spec["outputs"]["structure_tree"] == {"predictions": None}
    assert mlp2d.spec["outputs"]["channel_names_tree"] == {
        "predictions": ["channel_0", "channel_1"]
    }
