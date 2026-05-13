from __future__ import annotations

import jax
import pytest

from astro_emulators_toolkit.bundle.safetensors_io import load_arrays, save_arrays
from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, ModelSpec, RootConfig
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.io_trees import flatten_minmax_trees

jax.config.update("jax_enable_x64", True)


def _array_cfg(*, input_last_axis: int, output_last_axis: int) -> RootConfig:
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={
                "input_last_axis": input_last_axis,
                "output_last_axis": output_last_axis,
            },
        ),
    )


def test_reference_scaling_roundtrip_uses_minmax_trees(tmp_path):
    cfg = _array_cfg(input_last_axis=2, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle",
        spec={
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"parameters": [0.0, 0.0]},
                "max_tree": {"parameters": [1.0, 1.0]},
            },
        },
    )
    loaded = Emulator.from_bundle(out)
    ref = loaded.spec["reference_scaling_inputs"]
    assert ref["source_space"] == "physical_input_dict_tree_v1"
    assert ref["target_space"] == "canonical_input_dict_tree_v1"
    assert "min_tree" in ref and "max_tree" in ref


def test_sidecar_keys_use_split_minmax_layout(tmp_path):
    cfg = _array_cfg(input_last_axis=2, output_last_axis=1)
    out = Emulator.from_config(cfg).save_bundle(
        tmp_path / "bundle",
        spec={
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"parameters": [0.0, 0.0]},
                "max_tree": {"parameters": [1.0, 1.0]},
            },
        },
    )
    arrays = load_arrays(out / "reference_scaling_inputs.safetensors")
    assert "min/parameters" in arrays
    assert "max/parameters" in arrays


def test_transformer_flux_scaling_accepts_scalar_output_and_vector_params(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="transformer_payne", params={}, init_hints={"parameter_dim": 3}
        ),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(
        tmp_path / "tp",
        spec={
            "inputs": {"structure_tree": {"parameters": None, "wavelengths": None}},
            "outputs": {"structure_tree": {"flux": None}},
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"parameters": [0.0, 0.0, 0.0], "wavelengths": 0.0},
                "max_tree": {"parameters": [1.0, 1.0, 1.0], "wavelengths": 1.0},
            },
            "reference_scaling_outputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {"flux": 0.0},
                "max_tree": {"flux": 1.0},
            },
        },
    )
    assert (out / "reference_scaling_inputs.safetensors").exists()
    assert (out / "reference_scaling_outputs.safetensors").exists()


def test_wrong_param_length_rejected(tmp_path):
    cfg = _array_cfg(input_last_axis=2, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    with pytest.raises(ValueError, match="singleton leading dimensions"):
        emu.save_bundle(
            tmp_path / "bad",
            spec={
                "inputs": {"structure_tree": {"parameters": None}},
                "outputs": {"structure_tree": {"flux": None}},
                "reference_scaling_inputs": {
                    "kind": "affine_minmax_v1",
                    "applies_to": "inputs",
                    "source_space": "physical_input_dict_tree_v1",
                    "target_space": "canonical_input_dict_tree_v1",
                    "min_tree": {"parameters": [0.0, 0.0, 0.0]},
                    "max_tree": {"parameters": [1.0, 1.0, 1.0]},
                },
            },
        )


def test_reference_scaling_rejects_zero_width_span(tmp_path):
    cfg = _array_cfg(input_last_axis=2, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    with pytest.raises(
        ValueError,
        match=r"spec\['reference_scaling_inputs'\] leaf 'parameters' must have max > min",
    ):
        emu.save_bundle(
            tmp_path / "bad_zero_width",
            spec={
                "reference_scaling_inputs": {
                    "kind": "affine_minmax_v1",
                    "applies_to": "inputs",
                    "source_space": "physical_input_dict_tree_v1",
                    "target_space": "canonical_input_dict_tree_v1",
                    "min_tree": {"parameters": [0.0, 0.0]},
                    "max_tree": {"parameters": [1.0, 0.0]},
                },
            },
        )


def test_reference_scaling_storage_descriptor_rejects_zero_width_span(tmp_path):
    cfg = _array_cfg(input_last_axis=2, output_last_axis=1)
    emu = Emulator.from_config(cfg)
    bundle_dir = tmp_path / "bundle_with_sidecar"
    bundle_dir.mkdir()
    save_arrays(
        bundle_dir / "reference_scaling_inputs.safetensors",
        flatten_minmax_trees(
            {"parameters": [0.0, 0.0]},
            {"parameters": [1.0, 0.0]},
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"spec\['reference_scaling_inputs'\] leaf 'parameters' must have max > min",
    ):
        emu.save_bundle(
            bundle_dir,
            spec={
                "reference_scaling_inputs": {
                    "kind": "affine_minmax_v1",
                    "applies_to": "inputs",
                    "source_space": "physical_input_dict_tree_v1",
                    "target_space": "canonical_input_dict_tree_v1",
                    "storage": {
                        "format": "safetensors_v1",
                        "filename": "reference_scaling_inputs.safetensors",
                        "layout": "split_minmax_tree_v1",
                    },
                },
            },
        )
