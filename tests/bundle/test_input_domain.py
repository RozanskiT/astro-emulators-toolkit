from __future__ import annotations

import json

import jax
import pytest

from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, ModelSpec, RootConfig
from astro_emulators_toolkit.emulator import Emulator

jax.config.update("jax_enable_x64", True)


def _array_cfg(*, x_dim: int = 2, y_dim: int = 1) -> RootConfig:
    return RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": x_dim, "output_last_axis": y_dim},
        ),
    )


def _transformer_cfg() -> RootConfig:
    return RootConfig(
        io=IOSpec(
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 1,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
            },
            init_hints={"parameter_dim": 3},
        ),
    )


def _input_domain_spec(min_value, max_value) -> dict[str, object]:
    return {
        "input_domain": {
            "kind": "box_v1",
            "value_space": "physical_input_dict_tree_v1",
            "min_tree": {"parameters": min_value},
            "max_tree": {"parameters": max_value},
        },
    }


def test_bundle_without_domain_omits_input_domain_metadata(tmp_path):
    out = Emulator.from_config(_array_cfg()).save_bundle(
        tmp_path / "bundle_without_domain"
    )
    metadata = json.loads((out / "metadata.json").read_text())
    assert "input_domain" not in metadata["spec"]


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [
        (0.0, 1.0),
        ([0.0], [1.0]),
        ([0.0, 1.0], [2.0, 3.0]),
        ([[0.0, 1.0]], [[2.0, 3.0]]),
        ([[[0.0, 1.0]]], [[[2.0, 3.0]]]),
    ],
)
def test_array_input_domain_accepts_shared_and_per_last_axis_broadcast_forms(
    tmp_path, min_value, max_value
):
    emu = Emulator.from_config(_array_cfg(x_dim=2))
    out = emu.save_bundle(
        tmp_path / "bundle_accepts_domain",
        spec=_input_domain_spec(min_value, max_value),
    )
    loaded = Emulator.from_bundle(out)
    assert loaded.spec["input_domain"]["storage"]["layout"] == "split_minmax_tree_v1"


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [
        ([[0.0], [1.0]], [[2.0], [3.0]]),
        ([[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]),
    ],
)
def test_array_input_domain_rejects_non_singleton_leading_dims(
    tmp_path, min_value, max_value
):
    emu = Emulator.from_config(_array_cfg(x_dim=2))
    with pytest.raises(ValueError, match="singleton leading dimensions"):
        emu.save_bundle(
            tmp_path / "bundle_rejects_domain",
            spec=_input_domain_spec(min_value, max_value),
        )


def test_describe_domain_uses_hydrated_min_tree(tmp_path):
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"parameters": None},
                channel_names_tree={"parameters": ["Teff", "logg"]},
                channel_units_tree={"parameters": ["K", "dex"]},
            )
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
    )
    emu = Emulator.from_config(cfg)
    out = emu.save_bundle(
        tmp_path / "bundle_domain_summary",
        spec=_input_domain_spec([3500.0, 0.0], [8000.0, 5.0]),
    )
    loaded = Emulator.from_bundle(out)
    summary = loaded.describe_domain()
    assert "input_domain:" in summary
    assert "kind: box_v1" in summary
    assert "value_space: physical_input_dict_tree_v1" in summary
    assert "parameters:" in summary
    assert "Teff: [3500, 8000] K" in summary
    assert "logg: [0, 5] dex" in summary


def test_array_input_domain_allows_zero_width_leaf(tmp_path):
    emu = Emulator.from_config(_array_cfg(x_dim=2))
    out = emu.save_bundle(
        tmp_path / "bundle_domain_zero_width_ok",
        spec=_input_domain_spec([3500.0, 0.0], [3500.0, 5.0]),
    )
    loaded = Emulator.from_bundle(out)
    assert loaded.spec["input_domain"]["max_tree"]["parameters"][0] == 3500.0


def test_transformer_wavelength_leaf_accepts_all_singleton_shared_forms(tmp_path):
    emu = Emulator.from_config(_transformer_cfg())
    out = emu.save_bundle(
        tmp_path / "bundle_transformer_domain_ok",
        spec={
            "input_domain": {
                "kind": "box_v1",
                "value_space": "physical_input_dict_tree_v1",
                "min_tree": {
                    "parameters": [0.0, 0.0, 0.0],
                    "wavelengths": [[5000.0]],
                },
                "max_tree": {
                    "parameters": [1.0, 1.0, 1.0],
                    "wavelengths": [[7000.0]],
                },
            },
        },
    )
    loaded = Emulator.from_bundle(out)
    assert loaded.spec["input_domain"]["min_tree"]["wavelengths"] == [[5000.0]]


def test_transformer_wavelength_leaf_rejects_non_shared_shape(tmp_path):
    emu = Emulator.from_config(_transformer_cfg())
    with pytest.raises(ValueError, match="shared form with only singleton dimensions"):
        emu.save_bundle(
            tmp_path / "bundle_transformer_domain_bad",
            spec={
                "input_domain": {
                    "kind": "box_v1",
                    "value_space": "physical_input_dict_tree_v1",
                    "min_tree": {
                        "parameters": [0.0, 0.0, 0.0],
                        "wavelengths": [5000.0, 6000.0],
                    },
                    "max_tree": {
                        "parameters": [1.0, 1.0, 1.0],
                        "wavelengths": [7000.0, 8000.0],
                    },
                },
            },
        )
