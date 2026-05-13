from __future__ import annotations

import jax
import numpy as np
import pytest

from astro_emulators_toolkit.models.transformer_payne_batch import (
    make_flux_batch_transform,
    make_intensity_batch_transform,
)


def test_default_overlap_sampling_uses_supported_interval():
    transform = make_intensity_batch_transform(
        common_waves={
            "lines": np.linspace(0.1, 1.0, 8, dtype=np.float32),
            "continuum": np.linspace(0.4, 1.4, 10, dtype=np.float32),
        },
        n_wavelength=6,
        eval_wavelength_grid=np.linspace(0.45, 0.95, 6, dtype=np.float32),
        output_order=("lines", "continuum"),
    )

    batch = {
        "x": np.zeros((5, 3), dtype=np.float32),
        "y": {
            "lines": np.ones((5, 8), dtype=np.float32),
            "continuum": np.ones((5, 10), dtype=np.float32),
        },
    }
    out = transform(batch, rng=jax.random.key(0), train=True)
    sampled = np.asarray(out["x"][1])
    assert float(sampled.min()) >= 0.4 - 1e-6
    assert float(sampled.max()) <= 1.0 + 1e-6


def test_override_outside_support_raises_error():
    with pytest.raises(ValueError, match="allow_extrapolation"):
        make_flux_batch_transform(
            wavelength_grid=np.linspace(5000.0, 5100.0, 16, dtype=np.float32),
            n_wavelength=8,
            eval_wavelength_grid=np.linspace(5005.0, 5095.0, 8, dtype=np.float32),
            min_w=4990.0,
            max_w=5100.0,
        )


def test_override_outside_support_allows_with_explicit_flag():
    transform = make_flux_batch_transform(
        wavelength_grid=np.linspace(5000.0, 5100.0, 16, dtype=np.float32),
        n_wavelength=8,
        eval_wavelength_grid=np.linspace(5005.0, 5095.0, 8, dtype=np.float32),
        min_w=4990.0,
        max_w=5100.0,
        allow_extrapolation=True,
    )

    batch = {
        "x": np.zeros((4, 3), dtype=np.float32),
        "y": np.ones((4, 16), dtype=np.float32),
    }
    out = transform(batch, rng=jax.random.key(1), train=True)
    sampled = np.asarray(out["x"][1])
    assert float(sampled.min()) >= 4990.0 - 1e-6
    assert float(sampled.max()) <= 5100.0 + 1e-6
