"""Train experimental explicit_wavelength_mlp on randomized flux.

Data: irregular_flux split into train/validation from same distribution.
Creates: examples/runs/exp_explicit_wavelength_flux/bundle.
Runtime: ~20-40s on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays

PARAMETER_NAMES = ("teff", "logg", "feh")
Y_NAME = "flux"


def main() -> None:
    x_train, y_train, x_val, y_val, wave = split_randomized_flux_arrays(
        val_fraction=0.1, seed=5
    )
    cfg = RootConfig(
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp",
            params={
                "parameter_hidden_dim": 128,
                "joint_hidden_dim": 128,
                "wavelength_embedding_dim": 16,
                "activation": "gelu",
                "dtype": "float32",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adamw", lr=1e-3),
        training=TrainConfig(
            workdir=str(Path("examples/runs/exp_explicit_wavelength_flux")),
            batch_size=128,
            num_steps=20,
            evaluation_interval_steps=10,
        ),
    )
    emu = Emulator.from_config(cfg)
    transform = emu.make_device_batch_transform(
        wavelength_grid=wave, n_wavelength=wave.shape[0]
    )
    history = emu.fit(
        TreeArrayDataset(x={"parameters": x_train}, y={"predictions": y_train}),
        validation_dataset=TreeArrayDataset(
            x={"parameters": x_val}, y={"predictions": y_val}
        ),
        device_batch_transform=transform,
    )
    inference_wave = np.broadcast_to(
        wave[None, :].astype(np.float32), (2, wave.shape[0])
    )
    pred = emu.predict({"parameters": x_val[:2], "wavelengths": inference_wave})[
        "predictions"
    ]
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Pred shape:", pred.shape)


if __name__ == "__main__":
    main()
