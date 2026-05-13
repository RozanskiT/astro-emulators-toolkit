"""Inspect training internals and history keys after a short run.

Data: irregular_flux randomized train/validation split.
Creates: examples/runs/advanced_training_internals.
Runtime: ~10s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import payne_flux_mlp
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux


def main() -> None:
    x_train, y_train, x_val, y_val, _ = split_randomized_flux(val_fraction=0.1, seed=4)
    cfg = payne_flux_mlp(
        workdir="examples/runs/advanced_training_internals",
        profile="smoke",
    )
    emu = Emulator.from_config(cfg)
    callbacks = build_callbacks_from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        callbacks=callbacks,
    )
    print("History keys:", sorted(history.logs.keys()))
    print("Last fit method:", emu.last_fit_method)


if __name__ == "__main__":
    main()
