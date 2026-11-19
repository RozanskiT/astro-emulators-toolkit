"""Direct interpolation, joint training and bundle contracts of the experimental model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle import write_model_card
from astro_emulators_toolkit.config import OptimConfig
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.experimental.models.interpolated_payne import (
    InterpolatedPayne,
    InterpolatedPayneConfig,
    make_exojax_predictor,
    make_native_apply,
)
from astro_emulators_toolkit.presets import (
    transformer_payne_flux,
    transformer_payne_intensity,
)

jax.config.update("jax_enable_x64", True)
FAMILY = "experimental/interpolated_payne"


@pytest.mark.parametrize("channels", [1, 2])
def test_segmented_interpolation_and_gradients(channels):
    cfg = InterpolatedPayneConfig(
        grid_segments=((0, 2, 3), (4, 7, 3)),
        rank=1,
        hidden_sizes=(4,),
        channels=channels,
        log10_bounds=(0, 2),
    )
    model = InterpolatedPayne(in_dim=1, out_dim=1, cfg=cfg, rngs=nnx.Rngs(1))
    log_i = np.array([0, 1, 0, 2, 0, 1], dtype=np.float32)
    model.mean[...] = jnp.asarray(np.repeat(log_i, channels))
    context = jnp.zeros((2, 1), dtype=jnp.float32)
    w = jnp.array([7.0, 0.0, 0.5, 1.0, 2.0, 3.0, -0.1, 4.0, 5.5, 7.1, 0.5])
    expected = np.log10([10, 1, 5.5, 10, 1, np.nan, np.nan, 100, 1, np.nan, 5.5]) / 2
    expected = np.stack((expected, expected[::-1]))
    if channels > 1:
        expected = np.repeat(expected[..., None], channels, axis=-1)
    y = model.predict_queries(context, jnp.stack((w, w[::-1])))
    np.testing.assert_allclose(y, expected, atol=1e-6, equal_nan=True)
    shared = model.predict_queries(context, w)
    for row in shared:
        np.testing.assert_allclose(row, expected[0], atol=1e-6, equal_nan=True)

    def query(q):
        return model.predict_one(context, q).sum()

    # Analytic left/right segment endpoint slopes in canonical log units.
    slopes = jax.grad(query)(jnp.array([0.0, 7.0]))
    np.testing.assert_allclose(
        slopes, channels * np.array([9.0, 6.0 / 10]) / (2 * np.log(10)), rtol=2e-6
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_joint_fit_frozen_jax_and_bundle(tmp_path, channels):
    wave = np.linspace(0, 1, 12)
    p = np.random.default_rng(8).uniform(size=(16, 3)).astype("f")
    y = np.sin(wave + p[:, :1]).astype("f")
    preset = transformer_payne_flux if channels == 1 else transformer_payne_intensity
    cfg = preset(profile="smoke", workdir=str(tmp_path / "run"))
    params = dict(
        grid_segments=[[0, 1, 12]],
        rank=2,
        channels=channels,
        hidden_sizes=[8],
    )
    cfg = replace(
        cfg,
        model=replace(
            cfg.model, name=FAMILY, params=params, init_hints={"parameter_dim": 3}
        ),
        optim=OptimConfig(name="adam", lr=0.005),
        training=replace(cfg.training, num_steps=2, batch_size=4),
    )
    emu = Emulator.from_config(cfg)
    if channels == 1:
        transform = emu.make_device_batch_transform(
            wavelength_grid=wave, n_wavelength=4, eval_wavelength_grid=wave[:4]
        )
        ds = TreeArrayDataset(x={"parameters": p}, y={"flux": y})
    else:
        transform = emu.make_device_batch_transform(
            mode="intensity",
            common_waves={"lines": wave, "continuum": wave},
            n_wavelength=4,
            eval_wavelength_grid=wave[:4],
            output_order=("lines", "continuum"),
        )
        ds = TreeArrayDataset(x={"parameters": p}, y={"lines": y, "continuum": 2 * y})
    core = emu.params["core_model"]
    basis = np.array(core["basis"])
    kernel = np.array(core["head"]["layers"][0]["kernel"])
    h = emu.fit(ds, callbacks=[], device_batch_transform=transform)
    assert np.isfinite(h.logs["training_loss"]).all()
    core = emu.params["core_model"]
    assert np.any(core["basis"] != basis) and np.any(
        core["head"]["layers"][0]["kernel"] != kernel
    )
    x = {"parameters": p[:2], "wavelengths": np.stack([wave, wave[::-1]])}
    expected = emu.predict(x)["flux"]
    path = emu.save_bundle(tmp_path / "bundle")
    loaded = Emulator.from_bundle(path)
    np.testing.assert_array_equal(loaded.predict(x)["flux"], expected)
    assert loaded.model_family_id == "experimental_interpolated_payne_v1"
    card = write_model_card(path).read_text()
    assert "JAX_ENABLE_X64=1" in card and "development toolkit" in card
    frozen = loaded.make_frozen_apply(jit=True)
    np.testing.assert_allclose(frozen(x)["flux"], expected, rtol=2e-5, atol=2e-6)
    g = jax.grad(lambda q: frozen({**x, "parameters": q})["flux"].sum())(
        jnp.asarray(p[:2])
    )
    assert np.isfinite(g).all() and np.any(g != 0)


@pytest.mark.parametrize(
    "segments", [[], [[0, 1, 1]], [[0, 1, 3], [1, 2, 3]], [[1, 0, 3]]]
)
def test_bad_grid(segments):
    with pytest.raises(ValueError):
        InterpolatedPayneConfig(grid_segments=segments, rank=1)


def test_exojax_physical_predictor_from_bundle(tmp_path):
    from astro_emulators_toolkit.config import MinMaxTreeSpec

    preset = transformer_payne_flux(profile="smoke")
    lo = [1000.0, 0.05, -6.0, -6.0, -6.0, -10.0, -3.0, 0.0]
    hi = [1700.0, 0.2, -1.0, -1.0, -1.0, -5.0, 5.0, 1.0]
    scaling = MinMaxTreeSpec(
        min_tree={"parameters": lo, "wavelengths": 1.0},
        max_tree={"parameters": hi, "wavelengths": 2.0},
    )
    # These settings also exercise the config and parameter layout of the trained bundle.
    settings = dict(
        grid_segments=[[1.0, 2.0, 128]],
        log10_bounds=[-2.0, 3.0],
        features="exojax_v1",
        trainable_basis=True,
        coefficient_head="mlp",
        interpolation="linear",
        full_precision_modes=None,
    )
    cfg = replace(
        preset,
        model=replace(
            preset.model, name=FAMILY, params=settings, init_hints={"parameter_dim": 8}
        ),
        io=replace(preset.io, reference_scaling_inputs=scaling),
    )
    emu = Emulator.from_config(cfg)
    loaded = Emulator.from_bundle(
        emu.save_bundle(
            tmp_path / "intensity",
            extras={
                "wavelength_coordinate": "log10_angstrom_legacy_roundtrip",
                "preprocessing": {"legacy_wavelength_bounds": [10.0, 100.0]},
            },
        )
    )
    assert loaded.params["core_model"]["basis"].shape == (128, 128)
    predict = make_exojax_predictor(loaded)
    low, span = jnp.asarray(lo), jnp.asarray(hi) - jnp.asarray(lo)
    labels = low + span * jnp.array([[0.3], [0.6]])
    # Midway in log wavelength between two native knots.
    q = 10 ** ((32.5 / 127) + 1)
    native = make_native_apply(loaded, linear_intensity=True)(
        ((labels - low) / span).astype(jnp.float32)
    )
    expected = (native[:, 32] + native[:, 33]) / 2
    intensity = predict(labels, jnp.array([q, 9.0]))
    assert intensity.dtype == jnp.float64 and np.isnan(intensity[:, 1]).all()
    np.testing.assert_allclose(intensity[:, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(
        predict(labels[0], jnp.array([q])), intensity[:1, :1], rtol=1e-6
    )
    dp, dw = jax.grad(lambda p, w: predict(p, w).sum(), argnums=(0, 1))(
        labels, jnp.array([q])
    )
    assert np.isfinite(dp).all() and np.any(dp != 0)
    np.testing.assert_allclose(
        dw, ((native[:, 33] - native[:, 32]) * 127 / (q * np.log(10))).sum(), rtol=2e-5
    )


def test_reject_unsupported_prototype_settings():
    with pytest.raises(ValueError, match="interpolation"):
        InterpolatedPayneConfig.from_dict(
            {"grid_segments": [[0, 1, 128]], "interpolation": "cubic"}
        )
