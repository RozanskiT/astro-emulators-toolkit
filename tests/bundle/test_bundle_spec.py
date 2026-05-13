from __future__ import annotations
from copy import deepcopy

import pytest

from astro_emulators_toolkit.bundle.bundle import Bundle
from astro_emulators_toolkit.bundle.minmax_sidecars import (
    hydrate_reference_scaling_block,
)
from astro_emulators_toolkit.bundle.versions import (
    BUNDLE_FORMAT_VERSION,
    CONFIG_SCHEMA_VERSION,
    WEIGHTS_LAYOUT,
)
from astro_emulators_toolkit.bundle.safetensors_io import save_arrays
from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, RootConfig
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.bundle.integrity import write_bundle_integrity_manifest
from astro_emulators_toolkit.spec import SPEC_VERSION


def _cfg(*, x_dim: int = 2, y_dim: int = 3) -> RootConfig:
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"stellar": {"labels": None}}),
            outputs=IOTreeSpec(structure_tree={"spectra": {"flux": None}}),
        ),
        model=RootConfig().model.__class__(
            name="mlp",
            params={},
            init_hints={"input_last_axis": x_dim, "output_last_axis": y_dim},
        ),
    )


def _spec(
    *,
    reference_scaling_inputs=None,
    reference_scaling_outputs=None,
    input_domain=None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "inputs": {
            "structure_tree": {"stellar": {"labels": None}},
            "channel_names_tree": {"stellar": {"labels": ["teff", "logg"]}},
            "leaf_units_tree": {"stellar": {"labels": "dex"}},
            "channel_units_tree": {"stellar": {"labels": ["K", "dex"]}},
            "leaf_meanings_tree": {"stellar": {"labels": "stellar labels"}},
            "channel_meanings_tree": {
                "stellar": {"labels": ["effective temperature", "surface gravity"]}
            },
        },
        "outputs": {
            "structure_tree": {"spectra": {"flux": None}},
            "channel_names_tree": {"spectra": {"flux": ["blue", "green", "red"]}},
            "leaf_units_tree": {"spectra": {"flux": "normalized"}},
            "channel_units_tree": {"spectra": {"flux": ["", "", ""]}},
            "leaf_meanings_tree": {"spectra": {"flux": "normalized flux"}},
            "channel_meanings_tree": {
                "spectra": {"flux": ["blue band", "green band", "red band"]}
            },
        },
    }
    if reference_scaling_inputs is not None:
        payload["reference_scaling_inputs"] = reference_scaling_inputs
    if reference_scaling_outputs is not None:
        payload["reference_scaling_outputs"] = reference_scaling_outputs
    if input_domain is not None:
        payload["input_domain"] = input_domain
    return payload


def _refresh_bundle_integrity(bundle_dir):
    write_bundle_integrity_manifest(bundle_dir)


def test_bundle_roundtrip_preserves_current_tree_spec(tmp_path):
    emu = Emulator.from_config(_cfg())
    spec = _spec(
        reference_scaling_inputs={
            "kind": "affine_minmax_v1",
            "applies_to": "inputs",
            "source_space": "physical_input_dict_tree_v1",
            "target_space": "canonical_input_dict_tree_v1",
            "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
            "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
        },
        reference_scaling_outputs={
            "kind": "affine_minmax_v1",
            "applies_to": "outputs",
            "source_space": "canonical_output_dict_tree_v1",
            "target_space": "physical_output_dict_tree_v1",
            "min_tree": {"spectra": {"flux": [4.0, 5.0, 6.0]}},
            "max_tree": {"spectra": {"flux": [7.0, 8.0, 9.0]}},
        },
        input_domain={
            "kind": "box_v1",
            "value_space": "physical_input_dict_tree_v1",
            "min_tree": {"stellar": {"labels": [3500.0, 0.0]}},
            "max_tree": {"stellar": {"labels": [8000.0, 5.0]}},
        },
    )

    bundle_dir = tmp_path / "bundle"
    emu.save_bundle(bundle_dir, spec=spec)

    loaded = Emulator.from_bundle(bundle_dir)
    assert loaded.spec["inputs"]["channel_names_tree"]["stellar"]["labels"] == [
        "teff",
        "logg",
    ]
    assert (
        loaded.spec["outputs"]["leaf_meanings_tree"]["spectra"]["flux"]
        == "normalized flux"
    )
    assert (
        loaded.spec["reference_scaling_outputs"]["min_tree"]["spectra"]["flux"][0]
        == 4.0
    )
    assert loaded.spec["input_domain"]["max_tree"]["stellar"]["labels"][1] == 5.0


def test_bundle_accessors_expose_common_metadata_blocks(tmp_path):
    emu = Emulator.from_config(_cfg())
    extras = {"wavelength_angstrom": [5000.0, 5001.0, 5002.0]}
    spec = _spec(
        reference_scaling_inputs={
            "kind": "affine_minmax_v1",
            "applies_to": "inputs",
            "source_space": "physical_input_dict_tree_v1",
            "target_space": "canonical_input_dict_tree_v1",
            "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
            "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
        },
        reference_scaling_outputs={
            "kind": "affine_minmax_v1",
            "applies_to": "outputs",
            "source_space": "canonical_output_dict_tree_v1",
            "target_space": "physical_output_dict_tree_v1",
            "min_tree": {"spectra": {"flux": [4.0, 5.0, 6.0]}},
            "max_tree": {"spectra": {"flux": [7.0, 8.0, 9.0]}},
        },
        input_domain={
            "kind": "box_v1",
            "value_space": "physical_input_dict_tree_v1",
            "min_tree": {"stellar": {"labels": [3500.0, 0.0]}},
            "max_tree": {"stellar": {"labels": [8000.0, 5.0]}},
        },
    )

    out = emu.save_bundle(tmp_path / "bundle_accessors", spec=spec, extras=extras)
    loaded = Emulator.from_bundle(out)

    assert loaded.reference_scaling_inputs == loaded.spec["reference_scaling_inputs"]
    assert loaded.reference_scaling_outputs == loaded.spec["reference_scaling_outputs"]
    assert loaded.input_domain == loaded.spec["input_domain"]
    assert loaded.bundle_extras == extras
    assert loaded.input_spec == loaded.spec["inputs"]
    assert loaded.output_spec == loaded.spec["outputs"]
    assert loaded.input_channel_names_tree == {"stellar": {"labels": ["teff", "logg"]}}
    assert loaded.output_channel_names_tree == {
        "spectra": {"flux": ["blue", "green", "red"]}
    }
    assert loaded.input_spec["channel_units_tree"]["stellar"]["labels"] == ["K", "dex"]
    assert loaded.output_spec["leaf_units_tree"]["spectra"]["flux"] == "normalized"
    assert (
        loaded.output_spec["channel_meanings_tree"]["spectra"]["flux"][2] == "red band"
    )


def test_bundle_without_spec_is_rejected(tmp_path):
    cfg = _cfg(x_dim=2, y_dim=1)
    emu = Emulator.from_config(cfg)

    bundle_dir = tmp_path / "bundle_no_spec"
    Bundle(
        cfg=cfg,
        params_pure={"params": emu.params, "model_state": emu.model_state},
        metadata={
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "release": None,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "weights_layout": WEIGHTS_LAYOUT,
            "model_family_id": "mlp_v1",
        },
    ).save(bundle_dir)
    _refresh_bundle_integrity(bundle_dir)

    with pytest.raises(ValueError, match="missing required 'spec'"):
        Emulator.from_bundle(bundle_dir)


def test_save_bundle_materializes_reference_scaling_sidecars_when_present(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_ref_sidecar",
        spec=_spec(
            reference_scaling_inputs={
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
                "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
            },
            reference_scaling_outputs={
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {"spectra": {"flux": [4.0, 5.0, 6.0]}},
                "max_tree": {"spectra": {"flux": [7.0, 8.0, 9.0]}},
            },
        ),
    )
    assert (out / "reference_scaling_inputs.safetensors").exists()
    assert (out / "reference_scaling_outputs.safetensors").exists()

    loaded = Emulator.from_bundle(out)
    assert (
        loaded.spec["reference_scaling_inputs"]["storage"]["filename"]
        == "reference_scaling_inputs.safetensors"
    )
    assert (
        loaded.spec["reference_scaling_outputs"]["storage"]["filename"]
        == "reference_scaling_outputs.safetensors"
    )
    assert (
        loaded.spec["reference_scaling_inputs"]["min_tree"]["stellar"]["labels"][0]
        == 0.0
    )


def test_save_bundle_omits_reference_scaling_when_absent(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(tmp_path / "bundle_identity_ref", spec=_spec())
    metadata = Bundle.load(out).metadata
    loaded = Emulator.from_bundle(out)

    assert "reference_scaling_inputs" not in metadata["spec"]
    assert "reference_scaling_outputs" not in metadata["spec"]
    assert not (out / "reference_scaling_inputs.safetensors").exists()
    assert not (out / "reference_scaling_outputs.safetensors").exists()
    assert loaded.reference_scaling_inputs is None
    assert loaded.reference_scaling_outputs is None


def test_save_bundle_allows_input_only_reference_scaling_inputs(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_input_only_ref",
        spec=_spec(
            reference_scaling_inputs={
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
                "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
            }
        ),
    )

    loaded = Emulator.from_bundle(out)
    assert (out / "reference_scaling_inputs.safetensors").exists()
    assert not (out / "reference_scaling_outputs.safetensors").exists()
    assert loaded.reference_scaling_inputs is not None
    assert loaded.reference_scaling_outputs is None


def test_save_bundle_allows_output_only_reference_scaling_outputs(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_output_only_ref",
        spec=_spec(
            reference_scaling_outputs={
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {"spectra": {"flux": [4.0, 5.0, 6.0]}},
                "max_tree": {"spectra": {"flux": [7.0, 8.0, 9.0]}},
            }
        ),
    )

    loaded = Emulator.from_bundle(out)
    assert not (out / "reference_scaling_inputs.safetensors").exists()
    assert (out / "reference_scaling_outputs.safetensors").exists()
    assert loaded.reference_scaling_inputs is None
    assert loaded.reference_scaling_outputs is not None


def test_save_bundle_rejects_manual_spec_version_override(tmp_path):
    emu = Emulator.from_config(_cfg())

    with pytest.raises(ValueError, match="must not set internal compatibility keys"):
        emu.save_bundle(
            tmp_path / "bundle_manual_spec_version",
            spec={
                "spec_version": SPEC_VERSION,
                "inputs": {"structure_tree": {"stellar": {"labels": None}}},
                "outputs": {"structure_tree": {"spectra": {"flux": None}}},
            },
        )


@pytest.mark.parametrize(
    ("min_flux", "max_flux"),
    [
        ([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]),
        ([[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]),
        ([[[0.0, 1.0, 2.0]]], [[[3.0, 4.0, 5.0]]]),
    ],
)
def test_reference_scaling_accepts_shared_or_last_axis_broadcast_forms(
    tmp_path, min_flux, max_flux
):
    emu = Emulator.from_config(_cfg(x_dim=2, y_dim=3))
    out = emu.save_bundle(
        tmp_path / "bundle_ref_ok",
        spec=_spec(
            reference_scaling_outputs={
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {"spectra": {"flux": min_flux}},
                "max_tree": {"spectra": {"flux": max_flux}},
            }
        ),
    )
    assert (
        Emulator.from_bundle(out).spec["reference_scaling_outputs"]["storage"]["layout"]
        == "split_minmax_tree_v1"
    )


@pytest.mark.parametrize(
    ("min_flux", "max_flux"),
    [
        ([[0.0], [1.0]], [[2.0], [3.0]]),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]),
    ],
)
def test_reference_scaling_rejects_non_singleton_leading_dims(
    tmp_path, min_flux, max_flux
):
    emu = Emulator.from_config(_cfg(x_dim=2, y_dim=3))
    with pytest.raises(ValueError, match="singleton leading dimensions"):
        emu.save_bundle(
            tmp_path / "bundle_ref_bad",
            spec=_spec(
                reference_scaling_outputs={
                    "kind": "affine_minmax_v1",
                    "applies_to": "outputs",
                    "source_space": "canonical_output_dict_tree_v1",
                    "target_space": "physical_output_dict_tree_v1",
                    "min_tree": {"spectra": {"flux": min_flux}},
                    "max_tree": {"spectra": {"flux": max_flux}},
                }
            ),
        )


def test_from_bundle_fails_when_reference_scaling_sidecar_missing_required_key(
    tmp_path,
):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_missing_ref_key",
        spec=_spec(
            reference_scaling_inputs={
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
                "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
            }
        ),
    )
    save_arrays(
        out / "reference_scaling_inputs.safetensors",
        {
            "min/stellar/other": 0.0,
            "max/stellar/other": 1.0,
        },
    )
    _refresh_bundle_integrity(out)
    with pytest.raises(ValueError, match="missing required leaves"):
        Emulator.from_bundle(out)


def test_from_bundle_fails_when_input_domain_sidecar_missing_required_key(tmp_path):
    emu = Emulator.from_config(_cfg(x_dim=2, y_dim=1))
    out = emu.save_bundle(
        tmp_path / "bundle_missing_domain_key",
        spec=_spec(
            input_domain={
                "kind": "box_v1",
                "value_space": "physical_input_dict_tree_v1",
                "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
                "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
            }
        ),
    )
    save_arrays(
        out / "input_domain.safetensors",
        {
            "min/inputs/stellar/other": [0.0, 1.0],
            "max/inputs/stellar/other": [2.0, 3.0],
        },
    )
    _refresh_bundle_integrity(out)
    with pytest.raises(ValueError, match="missing required leaves"):
        Emulator.from_bundle(out)


def test_from_bundle_rejects_legacy_named_reference_scaling_sidecar(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_legacy_ref_scaling",
        spec=_spec(
            reference_scaling_inputs={
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"stellar": {"labels": [0.0, 1.0]}},
                "max_tree": {"stellar": {"labels": [2.0, 3.0]}},
            }
        ),
    )
    save_arrays(
        out / "reference_scaling_inputs.safetensors",
        {
            "min/x0": 0.0,
            "max/x0": 1.0,
            "min/x1": 10.0,
            "max/x1": 20.0,
        },
    )
    _refresh_bundle_integrity(out)
    with pytest.raises(ValueError, match="missing required leaves"):
        Emulator.from_bundle(out)


def test_direct_reference_scaling_sidecar_loader_rejects_path_traversal(tmp_path):
    cfg = _cfg()
    spec = _spec(
        reference_scaling_inputs={
            "kind": "affine_minmax_v1",
            "applies_to": "inputs",
            "source_space": "physical_input_dict_tree_v1",
            "target_space": "canonical_input_dict_tree_v1",
            "storage": {
                "format": "safetensors_v1",
                "filename": "../reference_scaling_inputs.safetensors",
                "layout": "split_minmax_tree_v1",
            },
        }
    )

    with pytest.raises(ValueError, match="relative POSIX path"):
        hydrate_reference_scaling_block(
            spec,
            tmp_path,
            cfg,
            block_name="reference_scaling_inputs",
            field_name="spec['reference_scaling_inputs']",
            validator_name="validate_reference_scaling_inputs",
            model_init={"input_last_axis": 2, "output_last_axis": 3},
        )


def test_custom_spec_payload_is_not_mutated_by_save_bundle(tmp_path):
    emu = Emulator.from_config(_cfg())
    spec = _spec()
    original = deepcopy(spec)
    emu.save_bundle(tmp_path / "bundle_mutation_guard", spec=spec)
    assert spec == original
