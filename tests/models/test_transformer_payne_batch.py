from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.models.transformer_payne_batch import (
    TransformerPayneFluxDeviceBatchTransform,
    TransformerPayneIntensityDeviceBatchTransform,
)

jax.config.update("jax_enable_x64", True)


def _transformer_cfg(
    tmp_path, *, channels: int = 1, y_names: tuple[str, ...] | None = None
) -> RootConfig:
    outputs = None
    if y_names is not None:
        outputs = IOTreeSpec(
            structure_tree={"flux": None},
            channel_names_tree={"flux": y_names} if len(y_names) > 1 else None,
        )
    return RootConfig(
        seed=0,
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"parameters": None, "wavelengths": None},
                channel_names_tree={
                    "parameters": ("label_0", "label_1", "label_2"),
                    "wavelengths": None,
                },
            ),
            outputs=outputs,
        ),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": channels,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 4,
                "dtype": "float32",
            },
            init_hints={"parameter_dim": 3},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=2,
            batch_size=4,
            evaluation_interval_steps=1,
        ),
    )


def test_emulator_make_batch_transform_delegates_to_transformer_family(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=1)
    emu = Emulator.from_config(cfg)
    wave = np.linspace(0.2, 1.5, num=12, dtype=np.float64)
    transform = emu.make_device_batch_transform(
        wavelength_grid=wave,
        n_wavelength=6,
        eval_wavelength_grid=np.linspace(0.3, 1.0, 6, dtype=np.float64),
    )
    assert isinstance(transform, TransformerPayneFluxDeviceBatchTransform)

    batch = {
        "x": {"parameters": jnp.zeros((2, 3), dtype=jnp.float32)},
        "y": {"flux": jnp.ones((2, 12), dtype=jnp.float32)},
        "sample_weight": jnp.array([1.0, 2.0]),
    }
    out = transform(batch, rng=jax.random.key(0), train=True)
    assert out["x"]["parameters"].shape == (2, 3)
    assert out["x"]["wavelengths"].shape == (2, 6)
    assert out["y"]["flux"].shape == (2, 6)
    assert out["x"]["wavelengths"].dtype == jnp.float64
    assert out["y"]["flux"].dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(out["sample_weight"]), np.asarray(batch["sample_weight"])
    )


def test_flux_device_batch_transform_for_init_matches_eval_path(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=1)
    emu = Emulator.from_config(cfg)
    eval_grid = np.linspace(0.3, 1.0, 6, dtype=np.float64)
    transform = emu.make_device_batch_transform(
        wavelength_grid=np.linspace(0.2, 1.5, num=12, dtype=np.float64),
        n_wavelength=6,
        eval_wavelength_grid=eval_grid,
    )

    batch = {
        "x": {"parameters": jnp.zeros((2, 3), dtype=jnp.float32)},
        "y": {"flux": jnp.ones((2, 12), dtype=jnp.float32)},
    }
    init_out = transform.for_init(batch)
    eval_out = transform(batch, rng=jax.random.key(0), train=False)

    np.testing.assert_allclose(
        np.asarray(init_out["x"]["wavelengths"]),
        np.asarray(eval_out["x"]["wavelengths"]),
    )
    np.testing.assert_allclose(
        np.asarray(init_out["y"]["flux"]), np.asarray(eval_out["y"]["flux"])
    )


def test_intensity_empty_overlap_fails(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=2)
    emu = Emulator.from_config(cfg)
    with pytest.raises(ValueError, match="channel overlap interval"):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={
                "lines": np.linspace(0.1, 0.3, 8, dtype=np.float32),
                "continuum": np.linspace(0.5, 0.8, 10, dtype=np.float32),
            },
            n_wavelength=6,
            eval_wavelength_grid=np.linspace(0.5, 0.7, 6, dtype=np.float32),
            output_order=("lines", "continuum"),
        )


def test_non_monotonic_source_waves_fail(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=1)
    emu = Emulator.from_config(cfg)
    with pytest.raises(ValueError, match="strictly monotonic increasing"):
        emu.make_device_batch_transform(
            wavelength_grid=np.array([0.1, 0.2, 0.19], dtype=np.float32),
            n_wavelength=3,
            eval_wavelength_grid=np.array([0.12, 0.15, 0.18], dtype=np.float32),
        )

    cfg_i = _transformer_cfg(tmp_path, channels=2)
    emu_i = Emulator.from_config(cfg_i)
    with pytest.raises(ValueError, match="strictly monotonic increasing"):
        emu_i.make_device_batch_transform(
            mode="intensity",
            common_waves={
                "lines": np.array([0.2, 0.3, 0.25], dtype=np.float32),
                "continuum": np.array([0.21, 0.31, 0.41], dtype=np.float32),
            },
            n_wavelength=3,
            eval_wavelength_grid=np.array([0.22, 0.27, 0.3], dtype=np.float32),
            output_order=("lines", "continuum"),
        )


def test_intensity_eval_grid_outside_overlap_fails_by_default(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=2)
    emu = Emulator.from_config(cfg)
    with pytest.raises(ValueError, match="allow_eval_outside_overlap"):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={
                "lines": np.linspace(0.2, 1.0, 8, dtype=np.float32),
                "continuum": np.linspace(0.25, 1.2, 10, dtype=np.float32),
            },
            n_wavelength=4,
            eval_wavelength_grid=np.linspace(0.2, 1.1, 4, dtype=np.float32),
            output_order=("lines", "continuum"),
        )


def test_intensity_output_order_mismatch_fails(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=2, y_names=("a", "b"))
    emu = Emulator.from_config(cfg)

    with pytest.raises(ValueError, match="output_order length"):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={"lines": np.linspace(0.2, 1.0, 8, dtype=np.float32)},
            n_wavelength=3,
            eval_wavelength_grid=np.linspace(0.3, 0.9, 3, dtype=np.float32),
            output_order=("lines",),
        )

    with pytest.raises(ValueError, match="keys not present in output_order"):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={
                "lines": np.linspace(0.2, 1.0, 8, dtype=np.float32),
                "continuum": np.linspace(0.2, 1.0, 8, dtype=np.float32),
                "extra": np.linspace(0.2, 1.0, 8, dtype=np.float32),
            },
            n_wavelength=3,
            eval_wavelength_grid=np.linspace(0.3, 0.9, 3, dtype=np.float32),
            output_order=("lines", "continuum"),
        )


def test_intensity_missing_common_waves_keys_fail_cleanly(tmp_path):
    cfg = _transformer_cfg(tmp_path, channels=2)
    emu = Emulator.from_config(cfg)

    with pytest.raises(ValueError, match="missing required keys"):
        emu.make_device_batch_transform(
            mode="intensity",
            common_waves={"lines": np.linspace(0.2, 1.0, 8, dtype=np.float32)},
            n_wavelength=3,
            eval_wavelength_grid=np.linspace(0.3, 0.9, 3, dtype=np.float32),
            output_order=("lines", "continuum"),
        )


def test_intensity_transform_uses_dict_y_contract_even_with_array_x(tmp_path):
    cfg = _transformer_cfg(
        tmp_path, channels=2, y_names=("log_flux_lines", "log_flux_continuum")
    )
    emu = Emulator.from_config(cfg)
    transform = emu.make_device_batch_transform(
        mode="intensity",
        common_waves={
            "lines": np.linspace(0.2, 1.0, 8, dtype=np.float32),
            "continuum": np.linspace(0.25, 1.05, 10, dtype=np.float32),
        },
        n_wavelength=4,
        eval_wavelength_grid=np.linspace(0.3, 0.9, 4, dtype=np.float32),
        output_order=("lines", "continuum"),
    )
    assert isinstance(transform, TransformerPayneIntensityDeviceBatchTransform)

    batch = {
        "x": jnp.zeros((2, 3), dtype=jnp.float32),
        "y": {
            "lines": jnp.ones((2, 8), dtype=jnp.float32),
            "continuum": jnp.ones((2, 10), dtype=jnp.float32),
        },
    }
    out = transform(batch, rng=jax.random.key(0), train=False)

    assert isinstance(out["y"], dict)
    assert out["y"]["flux"].shape == (2, 4, 2)


def test_transformer_payne_float32_fit_predict_smoke(tmp_path):
    cfg = _transformer_cfg(
        tmp_path, channels=2, y_names=("log_flux_lines", "log_flux_continuum")
    )
    emu = Emulator.from_config(cfg).configure_training()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    wave_lines = np.linspace(0.2, 1.3, num=15, dtype=np.float64)
    wave_cont = np.linspace(0.25, 1.25, num=11, dtype=np.float64)
    y_lines = np.sin(x[:, :1] + wave_lines[None, :]).astype(np.float32)
    y_cont = np.cos(x[:, :1] + wave_cont[None, :]).astype(np.float32)

    ds = TreeArrayDataset(
        x={"parameters": x}, y={"lines": y_lines, "continuum": y_cont}
    )
    eval_wave = np.linspace(0.3, 1.2, num=8, dtype=np.float64)
    transform = emu.make_device_batch_transform(
        mode="intensity",
        common_waves={"lines": wave_lines, "continuum": wave_cont},
        n_wavelength=8,
        eval_wavelength_grid=eval_wave,
        output_order=("lines", "continuum"),
    )
    sample = transform(ds.get_batch(np.arange(4)), rng=jax.random.key(7), train=True)
    assert sample["x"]["wavelengths"].dtype == jnp.float64
    assert sample["y"]["flux"].dtype == jnp.float32
    emu.fit(ds, validation_dataset=ds, device_batch_transform=transform)

    pred = emu.predict(
        {
            "parameters": x[:2],
            "wavelengths": np.broadcast_to(eval_wave[None, :], (2, 8)),
        }
    )
    assert pred["flux"].shape == (2, 8, 2)
