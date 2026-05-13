"""Train a longer Payne-style flux MLP with SOAP, CSV logs, checkpoints, and bundle scaling metadata.

Data: irregular_flux split into train/validation from the same randomized distribution.
Creates: examples/runs/development_payne_flux_mlp_soap/{bundle,checkpoints,history_*.csv,training_validation_curves.png,validation_example_0.png}
Runtime: minutes on CPU for 10,000 steps.
Requires: `uv sync` from a source checkout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astro_emulators_toolkit import Emulator
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
from astro_emulators_toolkit.training import (
    CSVLogger,
    ModelCheckpoint,
    ProgressBarLogger,
)

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "development_payne_flux_mlp_soap"
TRAIN_CSV = RUN_DIR / "history_train.csv"
VAL_CSV = RUN_DIR / "history_val.csv"
CURVES_PATH = RUN_DIR / "training_validation_curves.png"
PLOT_PATH = RUN_DIR / "validation_example_0.png"

PARAMETER_NAMES = ("teff", "logg", "feh")
PARAMETER_MEANINGS = ("effective temperature", "surface gravity", "metallicity [Fe/H]")
PARAMETER_UNITS = ("K", "dex", "dex")
Y_NAME = "flux"

INPUT_SCALE_BOUNDS = {
    "teff": (4500.0, 7000.0),
    "logg": (2.5, 5.0),
    "feh": (-0.3, 0.3),
}

SEED = 0
VAL_FRACTION = 0.1
NUM_STEPS = 10_000
BATCH_SIZE = 128
LOG_EVERY = 50
VAL_EVERY = 500
CHECKPOINT_EVERY = 500
MAX_SAVED_CHECKPOINTS = 5
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = NUM_STEPS // 10


def _save_training_curves(history_logs: dict[str, list[float]]) -> None:
    train_steps = np.asarray(history_logs.get("training_step", ()), dtype=np.float32)
    train_loss = np.asarray(history_logs.get("training_loss", ()), dtype=np.float32)
    val_steps = np.asarray(history_logs.get("validation_step", ()), dtype=np.float32)
    val_loss = np.asarray(history_logs.get("validation_loss", ()), dtype=np.float32)

    n_train = min(train_steps.size, train_loss.size)
    n_val = min(val_steps.size, val_loss.size)
    if n_train == 0 and n_val == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if n_train > 0:
        ax.plot(
            train_steps[:n_train],
            train_loss[:n_train],
            label="train loss",
            lw=1.5,
            alpha=0.9,
        )
    if n_val > 0:
        ax.plot(
            val_steps[:n_val],
            val_loss[:n_val],
            label="validation loss",
            lw=1.8,
            marker="o",
            ms=3.5,
        )

    all_positive = True
    if n_train > 0:
        all_positive = all_positive and bool(np.all(train_loss[:n_train] > 0.0))
    if n_val > 0:
        all_positive = all_positive and bool(np.all(val_loss[:n_val] > 0.0))
    if all_positive:
        ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    try:
        import orbax.checkpoint  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing base dependency 'orbax-checkpoint'. Reinstall the toolkit base dependencies "
            "(for example with `uv sync` from a source checkout)."
        ) from exc

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, wave = split_randomized_flux_arrays(
        val_fraction=VAL_FRACTION, seed=SEED
    )

    input_scale_min = np.asarray(
        [INPUT_SCALE_BOUNDS[name][0] for name in PARAMETER_NAMES], dtype=np.float32
    )
    input_scale_max = np.asarray(
        [INPUT_SCALE_BOUNDS[name][1] for name in PARAMETER_NAMES], dtype=np.float32
    )
    x_train = (
        (x_train.astype(np.float32) - input_scale_min)
        / (input_scale_max - input_scale_min)
    ).astype(np.float32)
    x_val = (
        (x_val.astype(np.float32) - input_scale_min)
        / (input_scale_max - input_scale_min)
    ).astype(np.float32)

    print("Manual input scaling bounds:")
    for name, unit, lo, hi in zip(
        PARAMETER_NAMES, PARAMETER_UNITS, input_scale_min, input_scale_max
    ):
        print(f"  {name} [{unit}]: min={float(lo):.6g}, max={float(hi):.6g}")

    print(f"Train samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Wavelength points: {wave.shape[0]}")

    cfg = RootConfig(
        seed=SEED,
        model=ModelSpec(
            name="mlp",
            params={
                "hidden_sizes": (128, 128),
                "activation": "gelu",
                "dtype": "float32",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(
            name="soap",
            lr=MAX_LR,
            schedule="cosine",
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
        ),
        training=TrainConfig(
            workdir=str(RUN_DIR),
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            val_fraction=VAL_FRACTION,
            logging_interval_steps=LOG_EVERY,
            evaluation_interval_steps=VAL_EVERY,
            checkpoint_interval_steps=CHECKPOINT_EVERY,
            max_saved_checkpoints=MAX_SAVED_CHECKPOINTS,
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
                    Y_NAME: "continuum-normalized flux vector on the shared wavelength grid"
                },
                leaf_units_tree={Y_NAME: "dimensionless"},
            ),
        ),
    )

    print("Training configuration:")
    hidden_sizes = tuple(
        int(width) for width in dict(cfg.model.params).get("hidden_sizes", ())
    )
    architecture = (x_train.shape[1], *hidden_sizes, y_train.shape[1])
    print("  model: mlp")
    print(f"  architecture: {' -> '.join(str(width) for width in architecture)}")
    print("  optimizer: soap")
    print("  schedule: linear warmup + cosine decay")
    print(f"  max learning rate: {cfg.optim.lr:.1e}")
    print(f"  warmup steps: {cfg.optim.warmup_steps}")
    print(f"  total steps: {cfg.training.num_steps}")
    print(f"  validation every: {cfg.training.evaluation_interval_steps} steps")
    print(f"  checkpoint every: {cfg.training.checkpoint_interval_steps} steps")

    emu = Emulator.from_config(cfg).configure_training()
    history = emu.fit(
        TreeArrayDataset(x={"parameters": x_train}, y={Y_NAME: y_train}),
        validation_dataset=TreeArrayDataset(x={"parameters": x_val}, y={Y_NAME: y_val}),
        callbacks=[
            ProgressBarLogger(total_steps=cfg.training.num_steps),
            CSVLogger(TRAIN_CSV, split="train"),
            CSVLogger(VAL_CSV, split="val"),
            ModelCheckpoint(
                every_n_steps=cfg.training.checkpoint_interval_steps,
                explicit_steps=cfg.training.checkpoint_steps,
            ),
        ],
    )

    _save_training_curves(history.logs)

    named_pred = emu.make_frozen_apply(jit=False)({"parameters": x_val[:1]})
    pred = np.asarray(named_pred[Y_NAME][0])

    fig, (ax_flux, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": (3, 1)}
    )
    ax_flux.plot(wave, y_val[0], label="ground truth", lw=1.4)
    ax_flux.plot(wave, pred, label="prediction", lw=1.1, alpha=0.85)
    ax_flux.set_ylabel("Normalized flux")
    ax_flux.set_title("Validation example 0")
    ax_flux.grid(alpha=0.2)
    ax_flux.legend()
    ax_resid.plot(wave, pred - y_val[0], color="0.15", lw=1.0)
    ax_resid.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax_resid.set_xlabel("Wavelength [Angstrom]")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    input_tree = {
        name: {"min": float(lo), "max": float(hi)}
        for name, lo, hi in zip(PARAMETER_NAMES, input_scale_min, input_scale_max)
    }
    input_min = [float(input_tree[name]["min"]) for name in PARAMETER_NAMES]
    input_max = [float(input_tree[name]["max"]) for name in PARAMETER_NAMES]
    bundle_dir = emu.save_bundle(
        spec={
            "inputs": {
                "structure_tree": {"parameters": None},
                "channel_names_tree": {"parameters": list(PARAMETER_NAMES)},
                "leaf_meanings_tree": {"parameters": "stellar labels"},
                "channel_meanings_tree": {"parameters": list(PARAMETER_MEANINGS)},
                "channel_units_tree": {"parameters": list(PARAMETER_UNITS)},
            },
            "outputs": {
                "structure_tree": {Y_NAME: None},
                "leaf_meanings_tree": {
                    Y_NAME: "continuum-normalized flux vector on the shared wavelength grid"
                },
                "leaf_units_tree": {Y_NAME: "dimensionless"},
            },
            "input_domain": {
                "kind": "box_v1",
                "value_space": "physical_input_dict_tree_v1",
                "min_tree": {"parameters": input_min},
                "max_tree": {"parameters": input_max},
            },
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"parameters": input_min},
                "max_tree": {"parameters": input_max},
            },
            "reference_scaling_outputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {
                    Y_NAME: np.zeros((y_train.shape[1],), dtype=np.float32).tolist()
                },
                "max_tree": {
                    Y_NAME: np.ones((y_train.shape[1],), dtype=np.float32).tolist()
                },
            },
        },
        extras={
            "wavelength_angstrom": [float(v) for v in wave],
            "notes": "Inputs are min-max scaled before training using the fixed bounds stored in this bundle. Output reference scaling metadata is recorded separately and is identity.",
        },
    )

    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Validation evaluations:", len(history.logs.get("validation_loss", [])))
    print("Named prediction keys:", tuple(named_pred.keys()))
    print("Named flux shape:", named_pred[Y_NAME].shape)
    print("Training CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Checkpoints:", RUN_DIR / "checkpoints")
    print("Saved curves:", CURVES_PATH)
    print("Saved plot:", PLOT_PATH)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
