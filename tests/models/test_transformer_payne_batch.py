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
    _sample_random_wave,
)

jax.config.update("jax_enable_x64", True)


def test_flux_transform_passthrough_eval_returns_batch_unchanged():
    wave = np.linspace(0.0, 1.0, 32)
    transform = TransformerPayneFluxDeviceBatchTransform(
        wavelength_grid=wave,
        n_wavelength=8,
        eval_mode="passthrough",
    )
    params = np.zeros((2, 3), dtype=np.float64)

    # Validation batches carry precomputed per-example query wavelengths and
    # matching targets (the original per-example random-wavelength protocol).
    query = np.sort(np.random.default_rng(0).uniform(0.0, 1.0, size=(2, 8)))
    eval_batch = {"x": (params, query), "y": np.ones((2, 8), dtype=np.float64)}

    out = transform(eval_batch, train=False)
    assert out["x"] is eval_batch["x"]
    assert out["y"] is eval_batch["y"]

    # for_init follows the eval path, so initialization uses the dataset's
    # explicit wavelengths too.
    init = transform.for_init(eval_batch)
    assert init["x"] is eval_batch["x"]

    # Training batches carry full source spectra; the transform still samples
    # random query wavelengths and interpolates.
    train_batch = {
        "x": (params, np.zeros((2, 8), dtype=np.float64)),
        "y": np.ones((2, wave.shape[0]), dtype=np.float64),
    }
    trained = transform(train_batch, train=True, rng=jax.random.PRNGKey(0))
    assert trained["x"] is not train_batch["x"]
    assert np.asarray(trained["x"][1]).shape == (2, 8)
    assert np.asarray(trained["y"]).shape == (2, 8)


def test_flux_transform_passthrough_allows_missing_eval_grid():
    wave = np.linspace(0.0, 1.0, 32)
    # n_wavelength != len(wavelength_grid) requires an eval grid in "grid"
    # mode but not in "passthrough" mode.
    with pytest.raises(ValueError, match="Evaluation wavelength grid"):
        TransformerPayneFluxDeviceBatchTransform(wavelength_grid=wave, n_wavelength=8)
    TransformerPayneFluxDeviceBatchTransform(
        wavelength_grid=wave, n_wavelength=8, eval_mode="passthrough"
    )


def test_flux_transform_rejects_unknown_eval_mode():
    wave = np.linspace(0.0, 1.0, 8)
    with pytest.raises(ValueError, match="eval_mode"):
        TransformerPayneFluxDeviceBatchTransform(
            wavelength_grid=wave, n_wavelength=8, eval_mode="random"
        )


def test_frequency_encoding_span_mismatch_warns_for_raw_angstrom_grid():
    import warnings as _warnings

    from astro_emulators_toolkit.config import ModelSpec, RootConfig
    from astro_emulators_toolkit.models.runtime_adapters import (
        _warn_on_frequency_encoding_span_mismatch,
    )

    cfg = RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={"min_period": 3e-2, "max_period": 30.0},
        )
    )

    # Raw Angstrom coordinate: span 6000 >> max_period 30.
    with pytest.warns(UserWarning, match="span"):
        _warn_on_frequency_encoding_span_mismatch(
            cfg, wavelength_arrays=(np.linspace(3000.0, 9000.0, 200),)
        )

    # log10-lambda-like coordinate with matched periods: no warning.
    log_cfg = RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={"min_period": 1e-6, "max_period": 10.0},
        )
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        _warn_on_frequency_encoding_span_mismatch(
            log_cfg, wavelength_arrays=(np.linspace(3.477, 3.954, 1000),)
        )


def test_frequency_encoding_min_period_warning_for_coarse_encoding():
    from astro_emulators_toolkit.config import ModelSpec, RootConfig
    from astro_emulators_toolkit.models.runtime_adapters import (
        _warn_on_frequency_encoding_span_mismatch,
    )

    cfg = RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={"min_period": 0.5, "max_period": 30.0},
        )
    )
    # Grid spacing ~0.005 with min_period 0.5: encoding cannot resolve
    # grid-scale structure.
    with pytest.warns(UserWarning, match="min_period"):
        _warn_on_frequency_encoding_span_mismatch(
            cfg, wavelength_arrays=(np.linspace(0.0, 1.0, 200),)
        )


def test_sample_random_wave_is_unsorted_raw_uniform_stream():
    # Pin the no-sort contract: sorting was removed because it is unnecessary
    # (jnp.interp accepts unsorted queries; the model treats query wavelengths
    # independently) and its CUB custom call destabilized long CUDA runs.
    rng = jax.random.PRNGKey(0)
    wave = _sample_random_wave(
        rng, 4, 16, min_w=1.0, max_w=2.0, dtype=jnp.dtype(jnp.float64)
    )
    expected = jax.random.uniform(
        rng, shape=(4, 16), minval=1.0, maxval=2.0, dtype=jnp.float64
    )
    assert wave.shape == (4, 16)
    assert bool(jnp.all(wave == expected))
    assert bool(jnp.all((wave >= 1.0) & (wave < 2.0)))


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
