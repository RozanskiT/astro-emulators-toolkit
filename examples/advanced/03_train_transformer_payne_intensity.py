"""Train transformer_payne on randomized intensity spectra.

Data: irregular_intensity split into train/validation from same distribution.
Creates: examples/runs/advanced_transformer_payne_intensity/bundle.
Runtime: ~20-40s on CPU.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import transformer_payne_intensity
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_smoke import smoke_value
from _example_data import split_randomized_intensity


def main() -> None:
    x_train, y_train, x_val, y_val, wave = split_randomized_intensity(
        val_fraction=0.1, seed=3
    )
    wave_lines = wave["lines"]
    wave_cont = wave["continuum"]
    workdir = smoke_value(
        "examples/runs/advanced_transformer_payne_intensity",
        smoke="examples/runs/advanced_transformer_payne_intensity_smoke",
    )
    cfg = transformer_payne_intensity(
        channels=2,
        workdir=workdir,
        profile="smoke",
    )
    explicit_logging_steps = tuple(
        step for step in (1, 2, 4, 8, 16, 32) if step <= int(cfg.training.num_steps)
    )
    explicit_evaluation_steps = tuple(
        step for step in (2, 4, 8, 16, 32) if step <= int(cfg.training.num_steps)
    )
    explicit_checkpoint_steps = tuple(
        step for step in (2, 8, 16, 32) if step <= int(cfg.training.num_steps)
    )
    logging_interval_steps = 3
    evaluation_interval_steps = 3
    checkpoint_interval_steps = 3
    cfg = cfg.with_updates(
        training=replace(
            cfg.training,
            logging_interval_steps=logging_interval_steps,
            evaluation_interval_steps=evaluation_interval_steps,
            logging_steps=explicit_logging_steps,
            evaluation_steps=explicit_evaluation_steps,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_steps=explicit_checkpoint_steps,
            max_saved_checkpoints=None,
        )
    )
    channel_names = tuple(cfg.io.outputs.channel_names_tree["flux"])

    n_eval_wave = 32
    eval_wave = np.linspace(
        max(wave_lines[0], wave_cont[0]),
        min(wave_lines[-1], wave_cont[-1]),
        num=n_eval_wave,
        dtype=np.float64,
    )
    emu = Emulator.from_config(cfg)
    transform = emu.make_device_batch_transform(
        mode="intensity",
        common_waves={"lines": wave_lines, "continuum": wave_cont},
        n_wavelength=n_eval_wave,
        eval_wavelength_grid=eval_wave,
        output_order=("lines", "continuum"),
    )
    callbacks = build_callbacks_from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        device_batch_transform=transform,
        callbacks=callbacks,
    )
    checkpoint_dir = Path(cfg.training.workdir) / "checkpoints"
    recorded_checkpoint_steps = tuple(
        sorted(
            int(path.name)
            for path in checkpoint_dir.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and int(path.name) <= int(cfg.training.num_steps)
        )
    )
    inference_wave = np.broadcast_to(eval_wave[None, :], (2, n_eval_wave))
    pred = emu.make_frozen_apply(jit=False)(
        {"parameters": x_val["parameters"][:2], "wavelengths": inference_wave}
    )["flux"]
    bundle_dir = emu.save_bundle()
    print("Logged steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", "smoke")
    print("Periodic training log interval:", cfg.training.logging_interval_steps)
    print("Periodic validation interval:", cfg.training.evaluation_interval_steps)
    print("Periodic checkpoint interval:", cfg.training.checkpoint_interval_steps)
    print("Explicit training log steps:", cfg.training.logging_steps)
    print("Explicit validation steps:", cfg.training.evaluation_steps)
    print("Explicit checkpoint steps:", cfg.training.checkpoint_steps)
    print("Max saved checkpoints:", cfg.training.max_saved_checkpoints)
    print(
        "Recorded training steps:",
        tuple(int(step) for step in history.logs.get("training_step", ())),
    )
    print(
        "Recorded validation steps:",
        tuple(int(step) for step in history.logs.get("validation_step", ())),
    )
    print("Recorded checkpoint steps:", recorded_checkpoint_steps)
    print("Pred shape:", pred.shape)
    print("Output channels:", channel_names)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
