"""One downstream intensity workflow for the four wavelength-explicit model families."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOTreeSpec, MinMaxTreeSpec
from astro_emulators_toolkit.experimental.exojax import make_exojax_predictor
from astro_emulators_toolkit.presets import transformer_payne_flux

jax.config.update("jax_enable_x64", True)

TRANSFORMER = dict(dim=8, dim_head=4, no_layers=1, no_tokens=2)
HASH = dict(mhe=dict(levels=3, log2_hashmap_size=4, wavelength_range=[1.0, 2.0]))
FAMILIES = {
    "transformer_payne": TRANSFORMER,
    "experimental/mhe_mlp": dict(hidden_sizes=[8], **HASH),
    "experimental/mhe_transformer_payne": dict(**TRANSFORMER, **HASH),
    "experimental/interpolated_payne": dict(
        grid_segments=[[1.0, 1.5, 64], [1.7, 2.0, 64]],
        hidden_sizes=[8],
        log10_bounds=[-2.0, 3.0],
    ),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_same_retrieval_after_bundle_swap(tmp_path, family):
    preset = transformer_payne_flux(profile="smoke")
    affine = family == "transformer_payne"
    low, high = np.arange(1.0, 9.0), np.arange(2.0, 10.0) + 2
    cfg = replace(
        preset,
        model=replace(
            preset.model,
            name=family,
            params=FAMILIES[family],
            init_hints={"parameter_dim": 8},
        ),
        io=replace(
            preset.io,
            inputs=IOTreeSpec(structure_tree={"parameters": None, "wavelengths": None}),
            outputs=IOTreeSpec(structure_tree={"log10_intensity": None}),
            reference_scaling_inputs=MinMaxTreeSpec(
                min_tree={
                    "parameters": low.tolist(),
                    "wavelengths": 10.0 if affine else 0.0,
                },
                max_tree={
                    "parameters": high.tolist(),
                    "wavelengths": 100.0 if affine else 1.0,
                },
            ),
            reference_scaling_outputs=MinMaxTreeSpec(
                min_tree={"log10_intensity": -2.0}, max_tree={"log10_intensity": 3.0}
            ),
            input_domain=None,
        ),
    )
    extras = {"chip_intervals_angstrom": [[10.0, 10**1.5], [10**1.7, 100.0]]}
    if not affine:
        extras.update(
            wavelength_coordinate="log10_angstrom_legacy_roundtrip",
            preprocessing={"legacy_wavelength_bounds": [10.0, 100.0]},
        )
    emu = Emulator.from_bundle(
        Emulator.from_config(cfg).save_bundle(tmp_path / "bundle", extras=extras)
    )
    assert emu.bundle_metadata["runtime_contract"]["role_paths"] == {
        "parameter_leaf": "inputs/parameters",
        "wavelength_leaf": "inputs/wavelengths",
        "output_leaf": "outputs/log10_intensity",
    }
    physical = make_exojax_predictor(emu)
    frozen = emu.make_frozen_apply()
    p = jnp.asarray(low + np.array([[0.25], [0.6]]) * (high - low))
    w = jnp.array([79.2, 19.3, 24.9, 19.3])

    def canonical_inputs(p, w):
        q = (
            (w - 10.0) / 90.0
            if affine
            else jnp.log10(10.0 + ((w - 10.0) / 90.0) * 90.0)
        )
        return {
            "parameters": ((p - low) / (high - low)).astype(jnp.float32),
            "wavelengths": jnp.broadcast_to(q, (len(p), len(w))),
        }

    def reference(p, w):
        y = frozen(canonical_inputs(p, w))["log10_intensity"].astype(jnp.float64)
        return 10 ** (5 * y - 2)

    x = canonical_inputs(p, w)
    x["wavelengths"] = x["wavelengths"].at[1].set(x["wavelengths"][1, ::-1])
    host = emu.predict(x)["log10_intensity"]
    for apply in (emu.apply_jax, frozen):
        np.testing.assert_allclose(
            apply(x)["log10_intensity"], host, rtol=1e-5, atol=1e-6
        )

    for key in ("parameters", "wavelengths"):
        one_row = {**x, key: x[key][:1]}
        repeated = {**x, key: jnp.broadcast_to(one_row[key], x[key].shape)}
        np.testing.assert_allclose(
            frozen(one_row)["log10_intensity"],
            frozen(repeated)["log10_intensity"],
            rtol=1e-5,
            atol=1e-6,
        )
    y = physical(p, w)
    assert y.shape == (2, 4) and y.dtype == jnp.float64
    np.testing.assert_allclose(y, reference(p, w), rtol=3e-6)
    np.testing.assert_allclose(physical(p[0], w), y[:1], rtol=3e-6)
    gaps = jnp.array([9.0, 40.0, 101.0])
    assert np.isnan(physical(p, gaps)).all()

    def grads(predict):
        return jax.grad(lambda a, q: jnp.log(predict(a, q)).sum(), argnums=(0, 1))(p, w)

    for g, ref in zip(grads(physical), grads(reference), strict=True):
        assert np.isfinite(g).all()
        np.testing.assert_allclose(g, ref, rtol=2e-4, atol=2e-5)
    with pytest.raises(ValueError, match="wavelengths"):
        physical(p, w[None, :])


@pytest.mark.parametrize("family", FAMILIES)
def test_custom_multichannel_role_paths(tmp_path, family):
    preset = transformer_payne_flux(profile="smoke")
    cfg = replace(
        preset,
        model=replace(
            preset.model,
            name=family,
            params={**FAMILIES[family], "channels": 2},
            init_hints={"parameter_dim": 2},
        ),
        io=replace(
            preset.io,
            inputs=IOTreeSpec(
                structure_tree={"stars": {"labels": None}, "obs": {"wavelengths": None}}
            ),
            outputs=IOTreeSpec(
                structure_tree={"spectrum": {"intensity": None}},
                channel_names_tree={"spectrum": {"intensity": ["lines", "continuum"]}},
            ),
            input_domain=None,
            reference_scaling_inputs=None,
            reference_scaling_outputs=None,
        ),
    )
    emu = Emulator.from_bundle(
        Emulator.from_config(cfg).save_bundle(tmp_path / "bundle")
    )
    assert (
        emu.bundle_metadata["runtime_contract"]["role_paths"]["output_leaf"]
        == "outputs/spectrum/intensity"
    )
    x = {
        "stars": {"labels": np.ones((1, 2))},
        "obs": {"wavelengths": np.array([[1.27, 1.81]])},
    }
    y = emu.apply_jax(x)["spectrum"]["intensity"]
    assert y.shape == (1, 2, 2) and np.isfinite(y).all()
