"""Resume internal training state from the run produced in basic/01.

Data: irregular_flux randomized train/validation split.
Creates: examples/runs/basic_payne_flux_mlp/bundle_resumed.
Runtime: ~10s on CPU after basic/01 has run.
Notes: resume=True uses trainer checkpoint/run-management state from run_config.json
and checkpoints/, not the portable bundle sharing contract.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import load_config
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.training import Callback, build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_smoke import smoke_value
from _example_data import split_randomized_flux


class FinalStepRecorder(Callback):
    def __init__(self) -> None:
        self.step: int | None = None

    def on_train_end(self, logs=None):
        if logs is not None and "step" in logs:
            self.step = int(logs["step"])


def main() -> None:
    run_dir = Path(
        smoke_value(
            "examples/runs/basic_payne_flux_mlp",
            smoke="examples/runs/basic_payne_flux_mlp_smoke",
        )
    )
    cfg = load_config(run_dir / "run_config.json")
    original_target_steps = int(cfg.training.num_steps)
    extra_steps = 10
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            num_steps=original_target_steps + extra_steps,
        ),
    )
    x_train, y_train, x_val, y_val, _ = split_randomized_flux(val_fraction=0.1, seed=0)

    emu = Emulator.from_config(cfg)
    callbacks = build_callbacks_from_config(cfg)
    final_step = FinalStepRecorder()
    callbacks.append(final_step)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        resume=True,
        max_steps=extra_steps,
        callbacks=callbacks,
    )
    out = emu.save_bundle(run_dir / "bundle_resumed")
    print("Original target step:", original_target_steps)
    print("Resumed final step:", final_step.step)
    print(
        "Resumed logged training records:",
        len(history.logs.get("training_loss", [])),
    )
    print("Bundle:", out)


if __name__ == "__main__":
    main()
