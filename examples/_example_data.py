"""Shared data loaders/split helpers for example scripts.

Helpers preserve on-disk dtypes and use randomized-in-domain datasets for
training/validation splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EXAMPLES_DIR / "examples_datasets"

ISOCHRONE_X_COLUMNS = ("eep", "initial_mass", "feh")
ISOCHRONE_Y_COLUMNS = (
    "log_Teff",
    "log_g",
    "log10_isochrone_age_yr",
    "Gaia_G_EDR3",
    "Gaia_BP_EDR3",
    "Gaia_RP_EDR3",
)
FLUX_INPUT_LEAF = "parameters"
FLUX_OUTPUT_LEAF = "flux"
ISOCHRONE_INPUT_LEAF = "parameters"
ISOCHRONE_OUTPUT_LEAF = "targets"


def _split_idx(
    n_rows: int, *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = max(1, int(round(n_rows * val_fraction)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_randomized_flux_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = DATASETS_DIR / "irregular_flux"
    return (
        np.load(base / "parameters.npy"),
        np.load(base / "normalized_flux.npy"),
        np.load(base / "wavelength.npy"),
    )


def load_randomized_flux() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    x, y, wave = load_randomized_flux_arrays()
    return ({FLUX_INPUT_LEAF: x}, {FLUX_OUTPUT_LEAF: y}, wave)


def split_randomized_flux_arrays(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, wave = load_randomized_flux_arrays()
    train_idx, val_idx = _split_idx(len(x), val_fraction=val_fraction, seed=seed)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], wave


def split_randomized_flux(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    x_train, y_train, x_val, y_val, wave = split_randomized_flux_arrays(
        val_fraction=val_fraction, seed=seed
    )
    return (
        {FLUX_INPUT_LEAF: x_train},
        {FLUX_OUTPUT_LEAF: y_train},
        {FLUX_INPUT_LEAF: x_val},
        {FLUX_OUTPUT_LEAF: y_val},
        wave,
    )


def load_randomized_intensity_arrays() -> tuple[
    np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    base = DATASETS_DIR / "irregular_intensity"
    y = {
        "lines": np.load(base / "intensity_lines.npy"),
        "continuum": np.load(base / "intensity_continuum.npy"),
    }
    wave = {
        "lines": np.load(base / "wavelength_lines.npy"),
        "continuum": np.load(base / "wavelength_continuum.npy"),
    }
    x = np.load(base / "parameters.npy")
    validate_intensity_payload(y, wave)
    return (x, y, wave)


def load_randomized_intensity() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    x, y, wave = load_randomized_intensity_arrays()
    return ({FLUX_INPUT_LEAF: x}, y, wave)


def validate_intensity_payload(
    y: dict[str, np.ndarray], wave: dict[str, np.ndarray]
) -> None:
    required = ("lines", "continuum")
    if set(y) != set(required):
        raise ValueError(
            f"Intensity payload y keys must be {required}, got {tuple(sorted(y))}."
        )
    if set(wave) != set(required):
        raise ValueError(
            f"Intensity payload wave keys must be {required}, got {tuple(sorted(wave))}."
        )
    n_samples = y["lines"].shape[0]
    for name in required:
        if y[name].ndim != 2:
            raise ValueError(
                f"Intensity payload y['{name}'] must have shape (batch, n_wave), got {y[name].shape}."
            )
        if wave[name].ndim != 1:
            raise ValueError(
                f"Intensity payload wave['{name}'] must be 1D, got {wave[name].shape}."
            )
        if y[name].shape[0] != n_samples:
            raise ValueError(
                "Intensity payload channels must have matching batch dimension."
            )
        if y[name].shape[1] != wave[name].shape[0]:
            raise ValueError(
                f"Intensity payload channel '{name}' wave-length mismatch."
            )
        if np.any(np.diff(wave[name].astype(np.float64)) <= 0.0):
            raise ValueError(
                f"Intensity payload wave['{name}'] must be strictly increasing."
            )


def split_randomized_intensity_arrays(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    x, y, wave = load_randomized_intensity_arrays()
    train_idx, val_idx = _split_idx(len(x), val_fraction=val_fraction, seed=seed)
    y_train = {name: values[train_idx] for name, values in y.items()}
    y_val = {name: values[val_idx] for name, values in y.items()}
    return x[train_idx], y_train, x[val_idx], y_val, wave


def split_randomized_intensity(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    x_train, y_train, x_val, y_val, wave = split_randomized_intensity_arrays(
        val_fraction=val_fraction, seed=seed
    )
    return ({FLUX_INPUT_LEAF: x_train}, y_train, {FLUX_INPUT_LEAF: x_val}, y_val, wave)


def load_isochrones_table() -> tuple[np.ndarray, tuple[str, ...]]:
    table = np.load(DATASETS_DIR / "isochrones" / "mist_isochrones_dev.npy")
    meta = json.loads(
        (DATASETS_DIR / "isochrones" / "mist_isochrones_dev_columns.json").read_text()
    )
    return table, tuple(meta["columns"])


def split_isochrones_named_arrays(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table, columns = load_isochrones_table()
    name_to_idx = {name: i for i, name in enumerate(columns)}
    x_idx = [name_to_idx[name] for name in ISOCHRONE_X_COLUMNS]
    y_idx = [name_to_idx[name] for name in ISOCHRONE_Y_COLUMNS]
    x = table[:, x_idx]
    y = table[:, y_idx]
    train_idx, val_idx = _split_idx(len(x), val_fraction=val_fraction, seed=seed)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def split_isochrones_named(
    *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    x_train, y_train, x_val, y_val = split_isochrones_named_arrays(
        val_fraction=val_fraction, seed=seed
    )
    return (
        {ISOCHRONE_INPUT_LEAF: x_train},
        {ISOCHRONE_OUTPUT_LEAF: y_train},
        {ISOCHRONE_INPUT_LEAF: x_val},
        {ISOCHRONE_OUTPUT_LEAF: y_val},
    )
