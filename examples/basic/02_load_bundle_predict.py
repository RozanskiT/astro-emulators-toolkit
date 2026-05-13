"""Load the shipped reference bundle and predict full-spectrum flux.

Data: reference bundle built from examples/examples_datasets/irregular_flux.
Creates: nothing.
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator, normalize_tree

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_bundle import require_reference_bundle


def main() -> None:
    bundle = require_reference_bundle()
    x_physical = np.asarray(
        [
            [5600.0, 4.2, -0.1],
            [5000.0, 3.8, 0.0],
        ],
        dtype=np.float32,
    )
    emu = Emulator.from_bundle(bundle)
    ref = emu.reference_scaling_inputs or {}
    x_scaled = normalize_tree(
        {"parameters": x_physical}, ref["min_tree"], ref["max_tree"]
    )
    flux = emu.predict(x_scaled)["flux"]
    n_wave = len(emu.bundle_extras.get("wavelength_angstrom", []))
    print("Bundle:", bundle)
    print("Input preprocessing:", "normalize_tree from bundle reference_scaling_inputs")
    print("Prediction shape:", flux.shape)
    print("First spectrum pixel count:", flux.shape[1])
    print("Wavelengths tracked in extras:", n_wave)


if __name__ == "__main__":
    main()
