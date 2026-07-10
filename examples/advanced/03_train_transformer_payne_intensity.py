"""Train transformer_payne on randomized intensity spectra.

Data: irregular_intensity split into train/validation from same distribution.
Creates: examples/runs/advanced_transformer_payne_intensity/bundle.
Runtime: ~20-40s on CPU.

Explicit logging/evaluation/checkpoint step schedules are demonstrated in
examples/advanced/05_training_internals.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import transformer_payne_intensity

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
    profile = "smoke"
    cfg = transformer_payne_intensity(
        channels=2,
        workdir=workdir,
        profile=profile,
    )
    channel_names = tuple(cfg.io.outputs.channel_names_tree["flux"])

    # The two intensity channels live on different source grids; the device
    # batch transform interpolates both onto shared query wavelengths.
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
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
        device_batch_transform=transform,
    )

    inference_wave = np.broadcast_to(eval_wave[None, :], (2, n_eval_wave))
    pred = emu.make_frozen_apply(jit=False)(
        {"parameters": x_val["parameters"][:2], "wavelengths": inference_wave}
    )["flux"]
    bundle_dir = emu.save_bundle()
    print("Logged steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", profile)
    print("Pred shape:", pred.shape)
    print("Output channels:", channel_names)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
