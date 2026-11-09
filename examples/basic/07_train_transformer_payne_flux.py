"""Train transformer_payne for wavelength-explicit 500-pixel flux prediction.

Data: irregular_flux split into train/validation from same distribution.
Creates: examples/runs/basic_transformer_payne_flux/bundle
Runtime: ~20-40s on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import transformer_payne_flux

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux


def main() -> None:
    x_train, y_train, x_val, y_val, wave = split_randomized_flux(
        val_fraction=0.1, seed=2
    )

    profile = "smoke"
    cfg = transformer_payne_flux(
        workdir=str(Path("examples/runs/basic_transformer_payne_flux")),
        profile=profile,
    )

    emu = Emulator.from_config(cfg)
    transform = emu.make_device_batch_transform(
        wavelength_grid=wave, n_wavelength=wave.shape[0]
    )
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        device_batch_transform=transform,
    )

    inference_wave = np.broadcast_to(wave[None, :], (2, wave.shape[0]))
    pred = emu.predict(
        {"parameters": x_val["parameters"][:2], "wavelengths": inference_wave}
    )["flux"]
    bundle_dir = emu.save_bundle()
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", profile)
    print("Pred shape:", pred.shape)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
