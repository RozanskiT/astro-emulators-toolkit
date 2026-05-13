from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "examples"))
from _example_data import load_randomized_intensity, validate_intensity_payload


def test_intensity_plotting_smoke() -> None:
    x, y, wave = load_randomized_intensity()
    validate_intensity_payload(y, wave)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.plot(wave["lines"], y["lines"][0])
    ax1.plot(wave["continuum"], y["continuum"][0])
    fig.canvas.draw()
    assert x["parameters"].shape[0] == y["lines"].shape[0]
    plt.close(fig)
