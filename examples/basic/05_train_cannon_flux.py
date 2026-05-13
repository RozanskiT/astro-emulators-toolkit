"""Train the Cannon baseline on randomized flux train/validation split.

Data: irregular_flux split into train/validation from same distribution.
Creates: examples/runs/basic_cannon_flux/bundle
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import cannon_flux
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux


def main() -> None:
    x_train, y_train, x_val, y_val, _ = split_randomized_flux(val_fraction=0.1, seed=1)
    cfg = cannon_flux(
        workdir=str(Path("examples/runs/basic_cannon_flux")),
        profile="smoke",
    )
    emu = Emulator.from_config(cfg)
    callbacks = build_callbacks_from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        callbacks=callbacks,
    )
    pred = emu.predict({"parameters": x_val["parameters"][:2]})["flux"]
    bundle_dir = emu.save_bundle()
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", "smoke")
    print("Pred shape:", pred.shape)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
