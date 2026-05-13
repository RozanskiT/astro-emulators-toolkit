from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit.inference.compose import (
    downgrade_spectral_resolution,
)
from astro_emulators_toolkit.inference import compose


def _log_wavelength_grid(n: int = 128) -> jax.Array:
    return jnp.linspace(jnp.log10(5000.0), jnp.log10(5010.0), n)


def _sinusoid(log_wavelength: jax.Array) -> jax.Array:
    grid = jnp.linspace(0.0, 1.0, log_wavelength.shape[0])
    return 1.0 + 0.2 * jnp.sin(12.0 * jnp.pi * grid)


def _smooth_value(log_wavelength: jax.Array, value: jax.Array, *, axis: int = -1):
    wrapped = downgrade_spectral_resolution(
        lambda: {"flux": value},
        log_wavelength,
        resolution=5000.0,
        axis=axis,
        jit=False,
    )
    return wrapped()["flux"]


def test_downgrade_spectral_resolution_smooths_last_axis_for_flux_batches() -> None:
    log_wavelength = _log_wavelength_grid()
    flux = _sinusoid(log_wavelength)
    batched = jnp.stack([flux, flux])

    smoothed = _smooth_value(log_wavelength, batched)

    assert smoothed.shape == batched.shape
    assert float(jnp.max(jnp.abs(smoothed[0] - flux))) > 1.0e-4
    expected = _smooth_value(log_wavelength, flux)
    np.testing.assert_allclose(
        np.asarray(smoothed[0]),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(smoothed[1]),
        np.asarray(smoothed[0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_downgrade_spectral_resolution_smooths_second_to_last_axis_for_channels() -> (
    None
):
    log_wavelength = _log_wavelength_grid()
    signal = _sinusoid(log_wavelength)
    intensity_one = jnp.stack([signal, signal], axis=-1)
    intensity = jnp.stack([intensity_one, intensity_one])

    smoothed = _smooth_value(log_wavelength, intensity, axis=-2)
    expected = _smooth_value(log_wavelength, signal)

    assert smoothed.shape == intensity.shape
    np.testing.assert_allclose(
        np.asarray(smoothed[0, :, 0]),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(smoothed[0, :, 1]),
        np.asarray(smoothed[0, :, 0]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(smoothed[1]),
        np.asarray(smoothed[0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_downgrade_spectral_resolution_postprocesses_selected_leaf() -> None:
    log_wavelength = _log_wavelength_grid()
    flux = jnp.stack(
        [
            _sinusoid(log_wavelength),
            0.8 * _sinusoid(log_wavelength),
        ]
    )
    aux = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)

    def apply_fn(x, scale):
        return {
            "spectra": {"flux": scale * x["flux"]},
            "aux": aux,
        }

    wrapped = downgrade_spectral_resolution(
        apply_fn,
        log_wavelength,
        resolution=5000.0,
        output_path="spectra/flux",
        jit=False,
    )

    result = wrapped({"flux": flux}, jnp.asarray(2.0, dtype=jnp.float32))
    expected = _smooth_value(log_wavelength, 2.0 * flux)

    np.testing.assert_allclose(
        np.asarray(result["spectra"]["flux"]),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(result["aux"]), np.asarray(aux))


def test_downgrade_spectral_resolution_smooths_all_leaves_with_bare_name() -> None:
    log_wavelength = _log_wavelength_grid()
    flux = jnp.stack(
        [
            _sinusoid(log_wavelength),
            0.5 * _sinusoid(log_wavelength),
        ]
    )

    def apply_fn():
        return {
            "blue": {"flux": flux},
            "red": {"flux": 2.0 * flux},
            "continuum": jnp.ones_like(flux),
        }

    wrapped = downgrade_spectral_resolution(
        apply_fn,
        log_wavelength,
        resolution=5000.0,
        output_path="flux",
        jit=False,
    )

    result = wrapped()
    expected_blue = _smooth_value(log_wavelength, flux)
    expected_red = _smooth_value(log_wavelength, 2.0 * flux)

    np.testing.assert_allclose(
        np.asarray(result["blue"]["flux"]),
        np.asarray(expected_blue),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result["red"]["flux"]),
        np.asarray(expected_red),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result["continuum"]),
        np.asarray(jnp.ones_like(flux)),
    )


def test_downgrade_spectral_resolution_uses_axis_tree() -> None:
    log_wavelength = _log_wavelength_grid()
    signal = _sinusoid(log_wavelength)
    flux = jnp.stack([signal, 0.8 * signal])
    intensity = jnp.stack([flux, flux], axis=-1)
    raw = jnp.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)

    def apply_fn():
        return {
            "flux": flux,
            "intensity": intensity,
            "raw": raw,
            "nested": {
                "flux": 0.5 * flux,
                "raw": raw + 1.0,
            },
        }

    axis_tree = {
        "flux": -1,
        "intensity": -2,
        "raw": None,
        "nested": {
            "flux": -1,
            "raw": None,
        },
    }
    wrapped = downgrade_spectral_resolution(
        apply_fn,
        log_wavelength,
        resolution=5000.0,
        axis_tree=axis_tree,
        jit=False,
    )

    result = wrapped()

    np.testing.assert_allclose(
        np.asarray(result["flux"]),
        np.asarray(_smooth_value(log_wavelength, flux)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result["intensity"]),
        np.asarray(_smooth_value(log_wavelength, intensity, axis=-2)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result["nested"]["flux"]),
        np.asarray(_smooth_value(log_wavelength, 0.5 * flux)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(result["raw"]), np.asarray(raw))
    np.testing.assert_allclose(
        np.asarray(result["nested"]["raw"]), np.asarray(raw + 1.0)
    )


def test_downgrade_spectral_resolution_returned_function_is_jittable() -> None:
    log_wavelength = _log_wavelength_grid()
    flux = _sinusoid(log_wavelength)[None, :]

    def apply_fn(scale):
        return {"flux": scale * flux}

    wrapped = downgrade_spectral_resolution(
        apply_fn,
        log_wavelength,
        resolution=5000.0,
    )

    value = wrapped(jnp.asarray(1.5, dtype=jnp.float32))["flux"]
    grad = jax.grad(lambda scale: jnp.sum(wrapped(scale)["flux"]))(
        jnp.asarray(1.5, dtype=jnp.float32)
    )

    assert value.shape == flux.shape
    assert jnp.isfinite(grad)


def test_downgrade_spectral_resolution_validates_log_wavelength_grid_once() -> None:
    with pytest.raises(ValueError, match="uniformly spaced"):
        downgrade_spectral_resolution(
            lambda: {"flux": jnp.ones(3, dtype=jnp.float32)},
            jnp.asarray([1.0, 1.1, 1.4], dtype=jnp.float32),
            resolution=5000.0,
        )


def test_downgrade_spectral_resolution_kernel_does_not_keep_wavelength_vector() -> None:
    log_wavelength = _log_wavelength_grid()
    shorter_flux = _sinusoid(_log_wavelength_grid(n=64))[None, :]
    wrapped = downgrade_spectral_resolution(
        lambda: {"flux": shorter_flux},
        log_wavelength,
        resolution=5000.0,
        jit=False,
    )

    result = wrapped()["flux"]

    assert result.shape == shorter_flux.shape
    assert float(jnp.max(jnp.abs(result - shorter_flux))) > 1.0e-4


def test_only_downgrade_resolution_helper_is_public() -> None:
    assert "downgrade_spectral_resolution" in compose.__all__
    assert "apply_spectral_resolution" not in compose.__all__
