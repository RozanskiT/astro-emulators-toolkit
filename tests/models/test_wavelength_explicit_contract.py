"""One downstream intensity workflow for the four wavelength-explicit model families."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOTreeSpec
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
            reference_scaling_inputs=None,
            reference_scaling_outputs=None,
            input_domain=None,
        ),
    )
    emu = Emulator.from_bundle(
        Emulator.from_config(cfg).save_bundle(tmp_path / "bundle")
    )
    assert emu.bundle_metadata["runtime_contract"]["role_paths"] == {
        "parameter_leaf": "inputs/parameters",
        "wavelength_leaf": "inputs/wavelengths",
        "output_leaf": "outputs/log10_intensity",
    }
    frozen = emu.make_frozen_apply()
    wavelengths = jnp.array([1.87, 1.23, 1.79, 1.23], dtype=jnp.float64)
    x = {
        "parameters": jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 16,
        "wavelengths": jnp.stack((wavelengths, wavelengths[::-1])),
    }
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
    grads = jax.grad(lambda inputs: frozen(inputs)["log10_intensity"].sum())(x)
    for key, grad in grads.items():
        assert grad.shape == x[key].shape and np.isfinite(grad).all()


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
