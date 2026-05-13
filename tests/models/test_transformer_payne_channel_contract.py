from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, ModelSpec, RootConfig
from astro_emulators_toolkit.models.transformer_payne_batch import (
    TransformerPayneIntensityDeviceBatchTransform,
)

jax.config.update("jax_enable_x64", True)


def _cfg() -> RootConfig:
    return RootConfig(
        io=IOSpec(
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                channel_names_tree={"flux": ["log_flux_lines", "log_flux_continuum"]},
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
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 3},
        ),
    )


def test_intensity_output_order_must_match_runtime_channel_contract(tmp_path):
    emu = Emulator.from_config(_cfg())
    with pytest.raises(
        ValueError, match="output_order must match runtime channel dataset-key contract"
    ):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={
                "lines": np.linspace(0.2, 1.0, 8, dtype=np.float32),
                "continuum": np.linspace(0.2, 1.0, 8, dtype=np.float32),
            },
            n_wavelength=4,
            eval_wavelength_grid=np.linspace(0.3, 0.9, 4, dtype=np.float32),
            output_order=("continuum", "lines"),
        )


def test_intensity_output_order_matching_runtime_channel_contract_succeeds(tmp_path):
    emu = Emulator.from_config(_cfg())
    transform = emu.make_device_batch_transform(
        mode="intensity",
        common_waves={
            "lines": np.linspace(0.2, 1.0, 8, dtype=np.float32),
            "continuum": np.linspace(0.2, 1.0, 8, dtype=np.float32),
        },
        n_wavelength=4,
        eval_wavelength_grid=np.linspace(0.3, 0.9, 4, dtype=np.float32),
        output_order=("lines", "continuum"),
    )
    assert isinstance(transform, TransformerPayneIntensityDeviceBatchTransform)
    assert callable(transform)


def test_transformer_channel_mapping_written_and_preserved_on_bundle_reload(tmp_path):
    emu = Emulator.from_config(_cfg())
    out = emu.save_bundle(tmp_path / "bundle_tp_channels")
    metadata = json.loads((out / "metadata.json").read_text())
    expected = [
        {"name": "log_flux_lines", "dataset_key": "lines"},
        {"name": "log_flux_continuum", "dataset_key": "continuum"},
    ]
    assert metadata["runtime_contract"]["transformer_payne_channels"] == expected
    assert metadata["runtime_contract"]["role_paths"]["output_leaf"] == "outputs/flux"

    loaded = Emulator.from_bundle(out)
    assert loaded.bundle_metadata is not None
    assert (
        loaded.bundle_metadata["runtime_contract"]["transformer_payne_channels"]
        == expected
    )
