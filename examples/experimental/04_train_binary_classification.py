"""Train experimental binary classification on randomized flux labels.

Data: irregular_flux split into train/validation from same distribution.
Creates: examples/runs/exp_binary_classification/bundle.
Runtime: ~10-20s on CPU.
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


def _labels_from_feh(x: np.ndarray) -> np.ndarray:
    return (x[:, 2] > 0.0).astype(np.float32)[:, None]


def main() -> None:
    x_train, _, x_val, _, _ = split_randomized_flux_arrays(val_fraction=0.1, seed=8)
    cfg = RootConfig(
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (64, 64), "activation": "gelu", "dtype": "float32"},
        ),
        task=TaskSpec(name="experimental/binary_classification", params={}),
        optim=OptimConfig(name="adamw", lr=5e-4),
        training=TrainConfig(
            workdir=str(Path("examples/runs/exp_binary_classification")),
            batch_size=128,
            num_steps=30,
            logging_interval_steps=10,
            evaluation_interval_steps=10,
        ),
    )
    emu = Emulator.from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(
            x={"parameters": x_train}, y={"predictions": _labels_from_feh(x_train)}
        ),
        validation_dataset=TreeArrayDataset(
            x={"parameters": x_val}, y={"predictions": _labels_from_feh(x_val)}
        ),
    )
    pred = emu.make_frozen_apply(jit=False)({"parameters": x_val[:4]})["predictions"]
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Probabilities:", pred[:, 0].tolist())
    print("Saved bundle:", emu.save_bundle())


if __name__ == "__main__":
    main()
