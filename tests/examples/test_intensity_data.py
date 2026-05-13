from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "examples"))
from _example_data import (
    load_randomized_intensity,
    split_randomized_intensity,
    validate_intensity_payload,
)


def test_load_randomized_intensity_returns_lines_and_continuum() -> None:
    x, y, wave = load_randomized_intensity()

    assert set(x) == {"parameters"}
    assert set(y) == {"lines", "continuum"}
    assert set(wave) == {"lines", "continuum"}
    assert y["lines"].shape[0] == x["parameters"].shape[0]
    assert y["continuum"].shape[0] == x["parameters"].shape[0]
    assert y["lines"].shape[1] == wave["lines"].shape[0]
    assert y["continuum"].shape[1] == wave["continuum"].shape[0]
    validate_intensity_payload(y, wave)


def test_split_randomized_intensity_preserves_both_outputs() -> None:
    x_train, y_train, x_val, y_val, wave = split_randomized_intensity(
        val_fraction=0.2, seed=11
    )

    assert set(y_train) == {"lines", "continuum"}
    assert set(y_val) == {"lines", "continuum"}
    assert set(wave) == {"lines", "continuum"}

    assert y_train["lines"].shape[0] == x_train["parameters"].shape[0]
    assert y_train["continuum"].shape[0] == x_train["parameters"].shape[0]
    assert y_val["lines"].shape[0] == x_val["parameters"].shape[0]
    assert y_val["continuum"].shape[0] == x_val["parameters"].shape[0]

    x_train_2, _, x_val_2, _, _ = split_randomized_intensity(val_fraction=0.2, seed=11)
    assert np.array_equal(x_train["parameters"], x_train_2["parameters"])
    assert np.array_equal(x_val["parameters"], x_val_2["parameters"])
