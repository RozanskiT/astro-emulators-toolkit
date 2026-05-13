"""Train an isochrone MLP using named-column selection.

Data: isochrones table split into train/validation.
Creates: examples/runs/basic_isochrone_mlp/bundle
Runtime: ~10-20s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import isochrone_mlp
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import (
    ISOCHRONE_X_COLUMNS,
    split_isochrones_named,
)


def main() -> None:
    x_train, y_train, x_val, y_val = split_isochrones_named(val_fraction=0.1, seed=0)
    cfg = isochrone_mlp(
        workdir=str(Path("examples/runs/basic_isochrone_mlp")),
        profile="smoke",
    )

    emu = Emulator.from_config(cfg)
    callbacks = build_callbacks_from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        callbacks=callbacks,
    )
    pred = emu.predict({"parameters": x_val["parameters"][:4]})["targets"]
    bundle_dir = emu.save_bundle()
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", "smoke")
    print("Prediction shape:", pred.shape)
    print("Input channel reference:", ISOCHRONE_X_COLUMNS)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
