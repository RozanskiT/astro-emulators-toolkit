"""Inspect training internals: history keys and explicit step schedules.

Data: irregular_flux randomized train/validation split.
Creates: examples/runs/advanced_training_internals.
Runtime: ~10s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import payne_flux_mlp

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux


def main() -> None:
    x_train, y_train, x_val, y_val, _ = split_randomized_flux(val_fraction=0.1, seed=4)
    cfg = payne_flux_mlp(
        workdir="examples/runs/advanced_training_internals",
        profile="smoke",
    )
    # Explicit absolute step schedules are unioned with the periodic
    # *_interval_steps cadence: here steps {1, 2} log/evaluate in addition to
    # every third step.
    cfg = cfg.with_updates(
        training={
            "logging_interval_steps": 3,
            "logging_steps": (1, 2),
            "evaluation_interval_steps": 3,
            "evaluation_steps": (1, 2),
        }
    )
    emu = Emulator.from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
    )
    print("History keys:", sorted(history.logs.keys()))
    print(
        "Recorded training steps:",
        tuple(int(step) for step in history.logs.get("training_step", ())),
    )
    print(
        "Recorded validation steps:",
        tuple(int(step) for step in history.logs.get("validation_step", ())),
    )
    print("Last fit method:", emu.last_fit_method)


if __name__ == "__main__":
    main()
