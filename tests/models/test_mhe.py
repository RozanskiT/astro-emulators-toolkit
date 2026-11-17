"""Focused native contracts for the two experimental MHE families."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    ModelSpec,
    OptimConfig,
    RootConfig,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.experimental.models.mhe_config import MHEConfig
from astro_emulators_toolkit.experimental.models.mhe_layers import MHEHashEncoding
from astro_emulators_toolkit.presets import (
    transformer_payne_flux,
    transformer_payne_intensity,
)
from astro_emulators_toolkit.resolver import get_supported_experimental_model_names

jax.config.update("jax_enable_x64", True)
FAMILIES = ("mhe_mlp", "mhe_transformer_payne")


def tiny_params(family, **extra):
    params = dict(
        min_period=0.02, max_period=2.0, mhe=dict(levels=3, log2_hashmap_size=4)
    )
    params.update(
        dict(hidden_sizes=[8, 7], parameter_hidden_sizes=[6])
        if family == "mhe_mlp"
        else dict(dim=8, dim_head=4, no_tokens=2, no_layers=1)
    )
    params.update(extra)
    return params


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("channels", (1, 2))
def test_native_training_batch_transforms_and_bundle(tmp_path, family, channels):
    preset = transformer_payne_intensity if channels > 1 else transformer_payne_flux
    cfg = preset(profile="smoke", workdir=str(tmp_path / "run"))
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            name=f"experimental/{family}",
            params=tiny_params(family, channels=channels),
        ),
        optim=OptimConfig(name="adam", lr=0.01),
        training=replace(cfg.training, batch_size=4, num_steps=3),
    )
    wave = np.linspace(0.1, 0.9, 9, dtype=np.float64)
    labels = np.random.default_rng(4).uniform(size=(8, 3)).astype(np.float32)
    target = np.sin(wave[None, :] + labels[:, :1]).astype(np.float32)
    assert cfg.model.name in get_supported_experimental_model_names()
    emu = Emulator.from_config(cfg)
    if channels == 1:
        transform = emu.make_device_batch_transform(
            wavelength_grid=wave, n_wavelength=5, eval_wavelength_grid=wave[::2]
        )
        ds = TreeArrayDataset(x={"parameters": labels}, y={"flux": target})
    else:
        transform = emu.make_device_batch_transform(
            mode="intensity",
            common_waves={"lines": wave, "continuum": wave},
            n_wavelength=5,
            eval_wavelength_grid=wave[::2],
            output_order=("lines", "continuum"),
        )
        ds = TreeArrayDataset(
            x={"parameters": labels}, y={"lines": target, "continuum": target * 0.5}
        )
    # Initialization uses the transform's eval path, just like TransformerPayne.
    batch = transform.for_init(ds.get_batch(np.arange(4)))
    emu.initialize(inputs=batch["x"], outputs=batch["y"])
    before = jax.tree.map(lambda x: np.asarray(x).copy(), emu.params)
    history = emu.fit(
        ds, validation_dataset=ds, callbacks=[], device_batch_transform=transform
    )
    assert len(history.logs["training_loss"]) == 3
    assert np.isfinite(history.logs["training_loss"]).all()
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(emu.params))
    )
    assert not np.array_equal(
        before["core_model"]["wavelength_encoder"]["table"],
        emu.params["core_model"]["wavelength_encoder"]["table"],
    )
    x = {"parameters": labels[:2], "wavelengths": np.broadcast_to(wave, (2, len(wave)))}
    pred = emu.predict(x)["flux"]
    assert pred.shape == ((2, 9) if channels == 1 else (2, 9, channels))
    assert pred.dtype == np.float32
    frozen = emu.make_frozen_apply(jit=True)
    grad = jax.grad(lambda p: jnp.sum(frozen({**x, "parameters": p})["flux"]))(
        jnp.asarray(x["parameters"])
    )
    assert np.isfinite(grad).all() and np.any(grad != 0)
    path = emu.save_bundle(tmp_path / "bundle", extras={"wavelengths": wave})
    restored = Emulator.from_bundle(path)
    assert restored.model_family_id == f"experimental_{family}_v1"
    assert (
        restored.bundle_metadata["runtime_contract"]
        == emu.bundle_metadata["runtime_contract"]
    )
    np.testing.assert_array_equal(restored.predict(x)["flux"], pred)
    np.testing.assert_allclose(
        jax.jit(restored.make_frozen_apply())(x)["flux"], pred, atol=2e-6, rtol=2e-5
    )


def test_hash_interpolation_collisions_boundaries_and_derivatives():
    encoding = MHEHashEncoding(
        rngs=nnx.Rngs(0),
        min_period=0.25,
        max_period=1.0,
        cfg=MHEConfig(levels=3, features_per_level=2, log2_hashmap_size=2),
        dtype="float64",
    )
    table = (
        np.arange(sum(encoding.table_sizes) * 2, dtype=np.float64).reshape(-1, 2) ** 2
    )
    encoding.table[...] = jnp.asarray(table)
    wave = np.array([-0.0625, 0.0, 0.0625, 0.5, 1.0, 1.125], dtype=np.float64)
    expected = []
    for w in wave:
        row = []
        for width, size, offset in zip(
            encoding.cell_widths, encoding.table_sizes, encoding.offsets
        ):
            position = w / width
            lower = int(np.floor(position))
            a, b = table[offset + lower % size], table[offset + (lower + 1) % size]
            row.extend(a + (position - lower) * (b - a))
        expected.append(row)
    np.testing.assert_allclose(encoding(wave), expected, atol=1e-12)
    point, eps = 0.073, 1e-7  # away from piecewise-linear knots
    analytic = jax.jacrev(encoding)(jnp.array(point, dtype=jnp.float64))
    numeric = (encoding(point + eps) - encoding(point - eps)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_native_checkpoint_resume_matches_uninterrupted_training(tmp_path, family):
    cfg = RootConfig(
        model=ModelSpec(name=f"experimental/{family}", params=tiny_params(family)),
        optim=OptimConfig(name="adam", lr=0.001),
        training=TrainConfig(
            workdir=str(tmp_path / "resumed"),
            batch_size=4,
            num_steps=2,
            checkpoint_interval_steps=1,
            max_saved_checkpoints=2,
        ),
    )
    labels = np.random.default_rng(3).uniform(size=(8, 3)).astype(np.float32)
    waves = np.broadcast_to(np.linspace(0.1, 0.9, 4), (8, 4))
    ds = TreeArrayDataset(
        x={"parameters": labels, "wavelengths": waves},
        y={"flux": np.sin(labels[:, :1] + waves).astype(np.float32)},
    )
    first = Emulator.from_config(cfg)
    first.fit(ds, max_steps=1)
    resumed = Emulator.from_config(cfg)
    resumed.fit(ds, resume=True)
    full = Emulator.from_config(
        replace(cfg, training=replace(cfg.training, workdir=str(tmp_path / "full")))
    )
    full.fit(ds)
    for expected, actual in zip(
        jax.tree.leaves(full.params), jax.tree.leaves(resumed.params)
    ):
        np.testing.assert_array_equal(actual, expected)
