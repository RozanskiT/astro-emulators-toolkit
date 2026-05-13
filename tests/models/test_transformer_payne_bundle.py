from __future__ import annotations

import json

import jax
import numpy as np

from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.emulator import Emulator

jax.config.update("jax_enable_x64", True)


class TinyTransformerDataset:
    def __init__(
        self,
        *,
        n_samples: int,
        n_params: int,
        n_wavelength: int,
        channels: int,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        self.params = rng.normal(size=(n_samples, n_params)).astype(np.float32)
        self.wavelengths = rng.uniform(0.5, 3.0, size=(n_samples, n_wavelength)).astype(
            np.float64
        )
        self.wavelengths.sort(axis=1)

        base = (
            np.sin(self.wavelengths * (0.6 + 0.2 * self.params[:, 0:1]))
            + 0.03 * np.sum(self.params, axis=1, keepdims=True)
        ).astype(np.float32)
        if channels == 1:
            self.y = base
        else:
            lines = base + 0.02 * np.cos(self.wavelengths)
            continuum = base - 0.02 * np.sin(self.wavelengths)
            self.y = np.stack([lines, continuum], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def get_batch(self, idx):
        return {
            "x": {"parameters": self.params[idx], "wavelengths": self.wavelengths[idx]},
            "y": {"flux": self.y[idx]},
        }


def _cfg(tmp_path, *, channels: int, explicit_io: bool = False) -> RootConfig:
    inputs = None
    outputs = None
    if explicit_io:
        inputs = IOTreeSpec(
            structure_tree={"parameters": None, "wavelengths": None},
            channel_names_tree={
                "parameters": ("teff", "logg", "feh", "vmic"),
                "wavelengths": None,
            },
        )
    if channels > 1:
        outputs = IOTreeSpec(
            structure_tree={"flux": None},
            channel_names_tree={"flux": ("log_flux_lines", "log_flux_continuum")},
        )
    return RootConfig(
        seed=0,
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": channels,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dim_ff_multiplier": 2,
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 4},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(
                tmp_path / f"run_{channels}ch_{'named' if explicit_io else 'default'}"
            ),
            batch_size=8,
            num_steps=2,
            steps_per_epoch=1,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=50,
        ),
        io=IOSpec(
            inputs=inputs,
            outputs=outputs,
        ),
    )


def test_transformer_payne_flux_bundle_roundtrip_preserves_runtime_contract(tmp_path):
    ds = TinyTransformerDataset(
        n_samples=16, n_params=4, n_wavelength=6, channels=1, seed=0
    )
    emu = Emulator.from_config(_cfg(tmp_path, channels=1)).configure_training()
    emu.fit(ds, validation_dataset=ds)

    out = emu.save_bundle(tmp_path / "bundle_transformer_flux")
    metadata = json.loads((out / "metadata.json").read_text())
    runtime = metadata["runtime_contract"]

    assert runtime["role_paths"] == {
        "parameter_leaf": "inputs/parameters",
        "wavelength_leaf": "inputs/wavelengths",
        "output_leaf": "outputs/flux",
    }
    assert runtime["affine_leaf_specs"]["outputs/flux"]["mode"] == "scalar_only"

    loaded = Emulator.from_bundle(out)
    pred = loaded.predict(
        {"parameters": ds.params[:2], "wavelengths": ds.wavelengths[:2]}
    )

    assert set(pred) == {"flux"}
    assert pred["flux"].shape == (2, 6)


def test_transformer_payne_intensity_bundle_roundtrip_preserves_runtime_contract(
    tmp_path,
):
    ds = TinyTransformerDataset(
        n_samples=16, n_params=4, n_wavelength=7, channels=2, seed=1
    )
    emu = Emulator.from_config(_cfg(tmp_path, channels=2)).configure_training()
    emu.fit(ds, validation_dataset=ds)

    out = emu.save_bundle(tmp_path / "bundle_transformer_intensity")
    metadata = json.loads((out / "metadata.json").read_text())
    runtime = metadata["runtime_contract"]

    assert runtime["role_paths"] == {
        "parameter_leaf": "inputs/parameters",
        "wavelength_leaf": "inputs/wavelengths",
        "output_leaf": "outputs/flux",
    }
    assert runtime["transformer_payne_channels"] == [
        {"name": "log_flux_lines", "dataset_key": "lines"},
        {"name": "log_flux_continuum", "dataset_key": "continuum"},
    ]
    assert runtime["affine_leaf_specs"]["outputs/flux"]["mode"] == "scalar_or_last_axis"

    loaded = Emulator.from_bundle(out)
    pred = loaded.predict(
        {"parameters": ds.params[:3], "wavelengths": ds.wavelengths[:3]}
    )

    assert set(pred) == {"flux"}
    assert pred["flux"].shape == (3, 7, 2)


def test_transformer_payne_bundle_roundtrip_preserves_named_io_contract(tmp_path):
    ds = TinyTransformerDataset(
        n_samples=16, n_params=4, n_wavelength=7, channels=2, seed=2
    )
    emu = Emulator.from_config(
        _cfg(tmp_path, channels=2, explicit_io=True)
    ).configure_training()
    emu.fit(ds, validation_dataset=ds)

    out = emu.save_bundle(tmp_path / "bundle_transformer_named_io")
    loaded = Emulator.from_bundle(out)
    pred = loaded.predict(
        {"parameters": ds.params[:3], "wavelengths": ds.wavelengths[:3]}
    )

    assert tuple(pred.keys()) == ("flux",)
    assert pred["flux"].shape == (3, 7, 2)
    assert loaded.input_channel_names_tree == {
        "parameters": ["teff", "logg", "feh", "vmic"],
        "wavelengths": None,
    }
    assert loaded.output_channel_names_tree == {
        "flux": ["log_flux_lines", "log_flux_continuum"],
    }
