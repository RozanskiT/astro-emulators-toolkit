from __future__ import annotations

import jax
import numpy as np
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
from astro_emulators_toolkit.models.transformer_payne import (
    TransformerPayne,
    TransformerPayneConfig,
    _frequency_encoding,
)

jax.config.update("jax_enable_x64", True)


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
        self.params = rng.normal(size=(n_samples, n_params)).astype(np.float64)
        self.wavelengths = rng.uniform(0.2, 4.0, size=(n_samples, n_wavelength)).astype(
            np.float64
        )
        self.wavelengths.sort(axis=1)

        base = np.sin(
            self.wavelengths * (0.7 + 0.3 * self.params[:, 0:1])
        ) + 0.05 * np.sum(self.params, axis=1, keepdims=True)
        if channels == 1:
            self.y = base.astype(np.float64)
        else:
            ys = [
                base + 0.1 * (c + 1) * np.cos(self.wavelengths * (0.5 + 0.1 * c))
                for c in range(channels)
            ]
            self.y = np.stack(ys, axis=-1).astype(np.float64)

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def get_batch(self, idx):
        return {
            "x": {"parameters": self.params[idx], "wavelengths": self.wavelengths[idx]},
            "y": {"flux": self.y[idx]},
        }


def test_transformer_payne_allows_float32_dtype_by_default():
    model = TransformerPayne(
        in_dim=4,
        out_dim=1,
        cfg=TransformerPayneConfig(
            dtype="float32", dim=32, dim_head=8, no_layers=2, no_tokens=4, channels=1
        ),
        rngs=nnx.Rngs(0),
    )
    params = np.ones((2, 4), dtype=np.float32)
    wavelengths = np.linspace(0.2, 2.0, 10, dtype=np.float64).reshape(2, 5)
    out = model((params, wavelengths))
    assert out.shape == (2, 5)
    assert out.dtype == np.float32
    assert model.wavelength_dtype == np.float64


def test_transformer_payne_bias_sections_are_independently_configurable():
    model = TransformerPayne(
        in_dim=4,
        out_dim=1,
        cfg=TransformerPayneConfig(
            dim=8,
            dim_head=4,
            no_layers=1,
            no_tokens=2,
            channels=1,
            bias_dense=False,
            bias_parameter_embedding=True,
            bias_feed_forward=False,
            bias_output_head=True,
            dtype="float64",
        ),
        rngs=nnx.Rngs(0),
    )

    assert model.param_embedding.b0 is not None
    assert model.param_embedding.b1 is not None
    assert model.ff_layers[0].in_proj.b is None
    assert model.ff_layers[0].out_proj.b is None
    assert model.head.proj0.b is not None
    assert model.head.proj1.b is not None


def test_transformer_payne_defaults_disable_all_biases():
    cfg = TransformerPayneConfig(
        dim=8,
        dim_head=4,
        no_layers=1,
        no_tokens=2,
        channels=1,
        dtype="float64",
    )
    model = TransformerPayne(in_dim=4, out_dim=1, cfg=cfg, rngs=nnx.Rngs(0))

    assert cfg.use_parameter_embedding_bias is False
    assert cfg.use_feed_forward_bias is False
    assert cfg.use_output_head_bias is False
    assert cfg.bias_attention is False

    assert model.param_embedding.b0 is None
    assert model.param_embedding.b1 is None
    assert model.ff_layers[0].in_proj.b is None
    assert model.ff_layers[0].out_proj.b is None
    assert model.head.proj0.b is None
    assert model.head.proj1.b is None
    assert model.attn_layers[0].bq is None
    assert model.attn_layers[0].bk is None
    assert model.attn_layers[0].bv is None
    assert model.attn_layers[0].bo is None


def test_frequency_encoding_keeps_sub_float32_wavelength_differences_before_cast():
    wavelengths64 = np.array([[5000.1234560, 5000.1234564]], dtype=np.float64)
    wavelengths32 = wavelengths64.astype(np.float32)

    encoded64 = _frequency_encoding(
        wavelengths64,
        min_period=3e-2,
        max_period=3e-2,
        dim=1,
        wavelength_dtype=np.float64,
        output_dtype=np.float32,
    )
    encoded32 = _frequency_encoding(
        wavelengths32,
        min_period=3e-2,
        max_period=3e-2,
        dim=1,
        wavelength_dtype=np.float32,
        output_dtype=np.float32,
    )

    assert float(encoded64[0, 0, 0]) != float(encoded64[0, 1, 0])
    assert float(encoded32[0, 0, 0]) == float(encoded32[0, 1, 0])


def test_transformer_payne_shapes_for_flux_and_intensity():
    params = np.ones((3, 4), dtype=np.float64)
    wavelengths = np.linspace(0.2, 2.0, 15, dtype=np.float64).reshape(3, 5)

    flux_model = TransformerPayne(
        in_dim=4,
        out_dim=1,
        cfg=TransformerPayneConfig(
            dim=32, dim_head=8, no_layers=2, no_tokens=4, channels=1, dtype="float64"
        ),
        rngs=nnx.Rngs(0),
    )
    flux_out = flux_model((params, wavelengths))
    assert flux_out.shape == (3, 5)

    intensity_model = TransformerPayne(
        in_dim=4,
        out_dim=1,
        cfg=TransformerPayneConfig(
            dim=32, dim_head=8, no_layers=2, no_tokens=4, channels=3, dtype="float64"
        ),
        rngs=nnx.Rngs(1),
    )
    intensity_out = intensity_model((params, wavelengths))
    assert intensity_out.shape == (3, 5, 3)


def test_transformer_payne_emulator_fit_predict_flux_and_intensity(tmp_path):
    n_params = 5
    n_wave = 8

    flux_ds = ExplicitWavelengthDataset(
        n_samples=48, n_params=n_params, n_wavelength=n_wave, channels=1, seed=2
    )
    flux_cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 1,
                "dim": 32,
                "dim_head": 8,
                "no_layers": 2,
                "no_tokens": 4,
                "dim_ff_multiplier": 2,
                "dtype": "float64",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run_transformer_flux"),
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
        {"parameters": flux_ds.params[:4], "wavelengths": flux_ds.wavelengths[:4]}
    )
    assert flux_pred["flux"].shape == (4, n_wave)
    assert len(flux_history.logs["training_loss"]) == flux_cfg.training.num_steps

    intensity_ds = ExplicitWavelengthDataset(
        n_samples=48, n_params=n_params, n_wavelength=n_wave, channels=3, seed=7
    )
    intensity_cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 3,
                "dim": 32,
                "dim_head": 8,
                "no_layers": 2,
                "no_tokens": 4,
                "dim_ff_multiplier": 2,
                "dtype": "float64",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run_transformer_intensity"),
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
    )
    assert intensity_pred["flux"].shape == (4, n_wave, 3)
    assert (
        len(intensity_history.logs["training_loss"]) == intensity_cfg.training.num_steps
    )
