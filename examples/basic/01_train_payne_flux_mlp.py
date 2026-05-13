"""Train a stable MLP flux emulator and predict full-spectrum flux.

Data: irregular_flux split into train/validation from the same randomized distribution.
Creates: examples/runs/basic_payne_flux_mlp/bundle
Runtime: ~10-20s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import payne_flux_mlp
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_smoke import smoke_value
from _example_data import split_randomized_flux


def main() -> None:
    workdir = smoke_value(
        "examples/runs/basic_payne_flux_mlp",
        smoke="examples/runs/basic_payne_flux_mlp_smoke",
    )
    x_train, y_train, x_val, y_val, wave = split_randomized_flux(
        val_fraction=0.1, seed=0
    )
    cfg = payne_flux_mlp(
        workdir=workdir,
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
    bundle = emu.save_bundle()
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", "smoke")
    print("Predicted full-spectrum shape:", pred.shape)
    print("Predicted wavelengths:", wave.shape[0])
    print("Bundle:", bundle)


if __name__ == "__main__":
    main()
