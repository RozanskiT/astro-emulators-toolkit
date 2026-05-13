from __future__ import annotations

import numpy as np
import pytest
from flax import nnx

from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.experimental.models.explicit_wavelength_mlp import (
    ExplicitWavelengthMLP,
    ExplicitWavelengthMLPConfig,
)


class ExplicitWavelengthDataset:
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
        self.wavelengths = rng.uniform(0.2, 4.0, size=(n_samples, n_wavelength)).astype(
            np.float32
        )
        self.wavelengths.sort(axis=1)

        base = np.sin(
            self.wavelengths * (0.7 + 0.3 * self.params[:, 0:1])
        ) + 0.05 * np.sum(self.params, axis=1, keepdims=True)
        if channels == 1:
            self.y = base.astype(np.float32)
        else:
            ys = [
                base + 0.1 * (c + 1) * np.cos(self.wavelengths * (0.5 + 0.1 * c))
                for c in range(channels)
            ]
            self.y = np.stack(ys, axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def get_batch(self, idx):
        return {
            "x": {
                "parameters": self.params[idx],
                "wavelengths": self.wavelengths[idx],
            },
            "y": {"predictions": self.y[idx]},
        }


def test_explicit_wavelength_mlp_shape_mapping_for_scalar_vector_and_batch():
    model_flux = ExplicitWavelengthMLP(
        in_dim=4,
        out_dim=1,
        cfg=ExplicitWavelengthMLPConfig(channels=1, wavelength_embedding_dim=16),
        rngs=nnx.Rngs(0),
    )

    params = np.ones((3, 4), dtype=np.float32)
    wavelengths = np.linspace(0.2, 2.0, 15, dtype=np.float32).reshape(3, 5)

    scalar_out = model_flux._predict_scalar_wavelength(wavelengths[0, 0], params[0])
    assert scalar_out.shape == ()

    vector_out = model_flux._predict_sample(wavelengths[0], params[0])
    assert vector_out.shape == (5,)

    batch_out = model_flux((params, wavelengths))
    assert batch_out.shape == (3, 5)

    model_intensity = ExplicitWavelengthMLP(
        in_dim=4,
        out_dim=1,
        cfg=ExplicitWavelengthMLPConfig(channels=3, wavelength_embedding_dim=16),
        rngs=nnx.Rngs(1),
    )
    batch_out_intensity = model_intensity((params, wavelengths))
    assert batch_out_intensity.shape == (3, 5, 3)


def test_explicit_wavelength_mlp_emulator_fit_and_predict_for_flux_and_intensity(
    tmp_path,
):
    n_params = 5
    n_wave = 8

    flux_ds = ExplicitWavelengthDataset(
        n_samples=48, n_params=n_params, n_wavelength=n_wave, channels=1, seed=2
    )
    flux_cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp",
            params={"channels": 1, "wavelength_embedding_dim": 16},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run_explicit_flux"),
            batch_size=12,
            num_steps=6,
            steps_per_epoch=2,
            logging_interval_steps=1,
            evaluation_interval_steps=3,
            checkpoint_interval_steps=50,
        ),
        io=IOSpec(),
    )

    flux_emu = Emulator.from_config(flux_cfg).configure_training()
    flux_history = flux_emu.fit(flux_ds, validation_dataset=flux_ds)
    flux_pred = flux_emu.predict(
        {
            "parameters": flux_ds.params[:4],
            "wavelengths": flux_ds.wavelengths[:4],
        }
    )["predictions"]
    assert flux_pred.shape == (4, n_wave)
    with pytest.raises(ValueError, match="canonical dict-tree inputs"):
        flux_emu.predict((flux_ds.params[:4], flux_ds.wavelengths[:4]))
    assert len(flux_history.logs["training_loss"]) == flux_cfg.training.num_steps

    intensity_ds = ExplicitWavelengthDataset(
        n_samples=48, n_params=n_params, n_wavelength=n_wave, channels=3, seed=7
    )
    intensity_cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp",
            params={"channels": 3, "wavelength_embedding_dim": 16},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run_explicit_intensity"),
            batch_size=12,
            num_steps=6,
            steps_per_epoch=2,
            logging_interval_steps=1,
            evaluation_interval_steps=3,
            checkpoint_interval_steps=50,
        ),
        io=IOSpec(),
    )

    intensity_emu = Emulator.from_config(intensity_cfg).configure_training()
    intensity_history = intensity_emu.fit(intensity_ds, validation_dataset=intensity_ds)
    intensity_pred = intensity_emu.predict(
        {
            "parameters": intensity_ds.params[:4],
            "wavelengths": intensity_ds.wavelengths[:4],
        }
    )["predictions"]
    assert intensity_pred.shape == (4, n_wave, 3)
    with pytest.raises(ValueError, match="canonical dict-tree inputs"):
        intensity_emu.predict((intensity_ds.params[:4], intensity_ds.wavelengths[:4]))
    assert (
        len(intensity_history.logs["training_loss"]) == intensity_cfg.training.num_steps
    )
