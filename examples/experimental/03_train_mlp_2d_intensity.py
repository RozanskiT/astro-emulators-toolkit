"""Train experimental MLP-2D intensity model on randomized intensity data.

Data: irregular_intensity split into train/validation from same distribution.
Creates: examples/runs/exp_mlp2d_intensity/bundle.
Runtime: ~20-40s on CPU.
Notes: labels are min-max scaled, and the target is log10(line intensity) min-max scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator, denormalize_tree, normalize_tree
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_intensity_arrays

PARAMETER_NAMES = ("teff", "logg", "feh", "mu")
PARAMETER_MEANINGS = (
    "effective temperature",
    "surface gravity",
    "metallicity [Fe/H]",
    "cosine of viewing angle",
)
PARAMETER_UNITS = ("K", "dex", "dex", "dimensionless")
Y_NAME = "predictions"
INPUT_SCALE_BOUNDS = {
    "teff": (4500.0, 7000.0),
    "logg": (2.5, 5.0),
    "feh": (-0.3, 0.3),
    "mu": (0.001, 1.0),
}
LOG_CLIP_MIN = np.finfo(np.float32).tiny


def _scale_tree(
    values: dict[str, np.ndarray],
    *,
    min_tree: dict[str, np.ndarray | float],
    max_tree: dict[str, np.ndarray | float],
):
    return normalize_tree(values, min_tree, max_tree)


def _unscale_tree(
    values: dict[str, np.ndarray],
    *,
    min_tree: dict[str, np.ndarray | float],
    max_tree: dict[str, np.ndarray | float],
):
    return denormalize_tree(values, min_tree, max_tree)


def main() -> None:
    x_train, y_train, x_val, y_val, _ = split_randomized_intensity_arrays(
        val_fraction=0.1, seed=7
    )
    y_train_lines = y_train["lines"].astype(np.float32)[..., None]
    y_val_lines = y_val["lines"].astype(np.float32)[..., None]

    input_scale_min = np.asarray(
        [INPUT_SCALE_BOUNDS[name][0] for name in PARAMETER_NAMES], dtype=np.float32
    )
    input_scale_max = np.asarray(
        [INPUT_SCALE_BOUNDS[name][1] for name in PARAMETER_NAMES], dtype=np.float32
    )
    x_scaling_min = {"parameters": input_scale_min}
    x_scaling_max = {"parameters": input_scale_max}
    x_train_scaled = _scale_tree(
        {"parameters": x_train.astype(np.float32)},
        min_tree=x_scaling_min,
        max_tree=x_scaling_max,
    )["parameters"]
    x_val_scaled = _scale_tree(
        {"parameters": x_val.astype(np.float32)},
        min_tree=x_scaling_min,
        max_tree=x_scaling_max,
    )["parameters"]

    train_log_lines = np.log10(np.clip(y_train_lines, LOG_CLIP_MIN, None)).astype(
        np.float32
    )
    val_log_lines = np.log10(np.clip(y_val_lines, LOG_CLIP_MIN, None)).astype(
        np.float32
    )
    log_line_min = float(min(train_log_lines.min(), val_log_lines.min()))
    log_line_max = float(max(train_log_lines.max(), val_log_lines.max()))
    y_scaling_min = {Y_NAME: log_line_min}
    y_scaling_max = {Y_NAME: log_line_max}
    y_train_scaled = _scale_tree(
        {Y_NAME: train_log_lines}, min_tree=y_scaling_min, max_tree=y_scaling_max
    )[Y_NAME]
    y_val_scaled = _scale_tree(
        {Y_NAME: val_log_lines}, min_tree=y_scaling_min, max_tree=y_scaling_max
    )[Y_NAME]

    print("Manual input scaling bounds:")
    for name, unit, lo, hi in zip(
        PARAMETER_NAMES, PARAMETER_UNITS, input_scale_min, input_scale_max
    ):
        print(f"  {name} [{unit}]: min={float(lo):.6g}, max={float(hi):.6g}")
    print("Target scaling:")
    print(f"  log10(lines) min={log_line_min:.6g}, max={log_line_max:.6g}")
    print(f"Train samples: {x_train_scaled.shape[0]}")
    print(f"Validation samples: {x_val_scaled.shape[0]}")

    cfg = RootConfig(
        model=ModelSpec(
            name="experimental/mlp_2d_regression",
            params={
                "hidden_sizes": (128, 128),
                "activation": "gelu",
                "dtype": "float32",
                "channels": 1,
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adamw", lr=1e-3),
        training=TrainConfig(
            workdir=str(Path("examples/runs/exp_mlp2d_intensity")),
            batch_size=128,
            num_steps=20,
            evaluation_interval_steps=10,
        ),
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"parameters": None},
                channel_names_tree={"parameters": list(PARAMETER_NAMES)},
                leaf_meanings_tree={"parameters": "stellar labels"},
                channel_meanings_tree={"parameters": list(PARAMETER_MEANINGS)},
                channel_units_tree={"parameters": list(PARAMETER_UNITS)},
            ),
            outputs=IOTreeSpec(
                structure_tree={Y_NAME: None},
                leaf_meanings_tree={
                    Y_NAME: "min-max scaled log10 line intensity on the shared wavelength grid"
                },
                channel_units_tree={Y_NAME: ["dimensionless"]},
            ),
        ),
    )
    emu = Emulator.from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x={"parameters": x_train_scaled}, y={Y_NAME: y_train_scaled}),
        validation_dataset=TreeArrayDataset(
            x={"parameters": x_val_scaled}, y={Y_NAME: y_val_scaled}
        ),
    )
    pred_scaled = emu.predict({"parameters": x_val_scaled[:2]})[Y_NAME]
    pred_log = _unscale_tree(
        {Y_NAME: pred_scaled}, min_tree=y_scaling_min, max_tree=y_scaling_max
    )[Y_NAME]
    pred = np.power(np.float32(10.0), np.asarray(pred_log, dtype=np.float32))
    example_mae_log = float(np.mean(np.abs(np.asarray(pred_log[0]) - val_log_lines[0])))

    print("Steps:", len(history.logs.get("training_loss", [])))
    print(
        "Validation loss (scaled log-space):",
        float(history.logs.get("validation_loss", [np.nan])[-1]),
    )
    print("Example 0 MAE in log10(line intensity):", example_mae_log)
    print("Pred shape:", pred.shape)


if __name__ == "__main__":
    main()
