"""Train a longer isochrone MLP with SOAP, CSV logs, checkpoints, and bundle scaling metadata.

Data: isochrones table split into train/validation.
Creates: examples/runs/development_isochrone_mlp_soap/{bundle,checkpoints,history_*.csv,training_validation_curves.png,validation_parity.png}
Runtime: minutes on CPU for 10,000 steps.
Requires: `uv sync` from a source checkout.
"""

from __future__ import annotations

import math
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
from _example_data import (
    ISOCHRONE_X_COLUMNS,
    ISOCHRONE_Y_COLUMNS,
    split_isochrones_named_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "development_isochrone_mlp_soap"
TRAIN_CSV = RUN_DIR / "history_train.csv"
VAL_CSV = RUN_DIR / "history_val.csv"
CURVES_PATH = RUN_DIR / "training_validation_curves.png"
PLOT_PATH = RUN_DIR / "validation_parity.png"

PARAMETER_NAMES = ISOCHRONE_X_COLUMNS
PARAMETER_MEANINGS = (
    "equivalent evolutionary point",
    "initial stellar mass",
    "metallicity [Fe/H]",
)
PARAMETER_UNITS = ("index", "Msun", "dex")
TARGET_NAMES = ISOCHRONE_Y_COLUMNS
TARGET_MEANINGS = (
    "log10 effective temperature",
    "surface gravity",
    "log10 isochrone age",
    "Gaia G-band absolute magnitude",
    "Gaia BP-band absolute magnitude",
    "Gaia RP-band absolute magnitude",
)
TARGET_UNITS = ("log10(K)", "dex", "log10(yr)", "mag", "mag", "mag")

INPUT_SCALE_BOUNDS = {
    "eep": (202.0, 605.0),
    "initial_mass": (0.1, 300.0),
    "feh": (-0.5, 0.5),
}
OUTPUT_SCALE_BOUNDS = {
    "log_Teff": (3.4, 5.1),
    "log_g": (-0.4, 5.3),
    "log10_isochrone_age_yr": (5.0, 10.3),
    "Gaia_G_EDR3": (-11.0, 14.5),
    "Gaia_BP_EDR3": (-11.0, 17.2),
    "Gaia_RP_EDR3": (-11.0, 13.0),
}

SEED = 0
VAL_FRACTION = 0.1
NUM_STEPS = 10_000
BATCH_SIZE = 128
LOG_EVERY = 50
VAL_EVERY = 500
CHECKPOINT_EVERY = 500
MAX_SAVED_CHECKPOINTS = 5
MAX_LR = 3e-3
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = NUM_STEPS // 10


def _scale_array(x: np.ndarray, *, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mins) / (maxs - mins)).astype(np.float32)


def _unscale_array(x: np.ndarray, *, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) * (maxs - mins) + mins).astype(np.float32)


def _save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    n_panels = len(TARGET_NAMES) + 1
    n_cols = 3
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (name, unit) in enumerate(zip(TARGET_NAMES, TARGET_UNITS)):
        ax = axes[i]
        true = y_true[:, i]
        pred = y_pred[:, i]
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        span = hi - lo
        pad = 0.05 * span if span > 0.0 else 1.0
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        ax.scatter(true, pred, s=8, alpha=0.25, linewidths=0.0)
        ax.plot(
            (lo - pad, hi + pad), (lo - pad, hi + pad), color="0.2", lw=1.0, ls="--"
        )
        ax.set_title(f"{name}\nMAE={mae:.4g}, RMSE={rmse:.4g}")
        label = f"[{unit}]" if unit else ""
        ax.set_xlabel(f"Truth {label}".strip())
        ax.set_ylabel(f"Prediction {label}".strip())
        ax.grid(alpha=0.2)

    residual = y_pred[0] - y_true[0]
    summary = [
        "Validation example 0 residuals:",
        *[
            f"{name}: {float(delta):+.4f} {unit}".rstrip()
            for name, delta, unit in zip(TARGET_NAMES, residual, TARGET_UNITS)
        ],
    ]
    axes[-1].axis("off")
    axes[-1].text(0.0, 1.0, "\n".join(summary), va="top", ha="left", family="monospace")
    for ax in axes[len(TARGET_NAMES) + 1 :]:
        ax.axis("off")

    fig.suptitle("Isochrone SOAP MLP validation parity", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


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
    ax.set_ylabel("Loss (scaled target space)")
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

    x_train, y_train, x_val, y_val = split_isochrones_named_arrays(
        val_fraction=VAL_FRACTION, seed=SEED
    )

    input_scale_min = np.asarray(
        [INPUT_SCALE_BOUNDS[name][0] for name in PARAMETER_NAMES], dtype=np.float32
    )
    input_scale_max = np.asarray(
        [INPUT_SCALE_BOUNDS[name][1] for name in PARAMETER_NAMES], dtype=np.float32
    )
    output_scale_min = np.asarray(
        [OUTPUT_SCALE_BOUNDS[name][0] for name in TARGET_NAMES], dtype=np.float32
    )
    output_scale_max = np.asarray(
        [OUTPUT_SCALE_BOUNDS[name][1] for name in TARGET_NAMES], dtype=np.float32
    )
    x_train = _scale_array(x_train, mins=input_scale_min, maxs=input_scale_max)
    x_val = _scale_array(x_val, mins=input_scale_min, maxs=input_scale_max)
    y_train_scaled = _scale_array(y_train, mins=output_scale_min, maxs=output_scale_max)
    y_val_scaled = _scale_array(y_val, mins=output_scale_min, maxs=output_scale_max)

    print(f"Train samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Target dimensions: {y_train.shape[1]}")

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
                leaf_meanings_tree={"parameters": "isochrone conditioning parameters"},
                channel_meanings_tree={"parameters": list(PARAMETER_MEANINGS)},
                channel_units_tree={"parameters": list(PARAMETER_UNITS)},
            ),
            outputs=IOTreeSpec(
                structure_tree={"predictions": None},
                channel_names_tree={"predictions": list(TARGET_NAMES)},
                leaf_meanings_tree={"predictions": "scaled isochrone target vector"},
                channel_meanings_tree={"predictions": list(TARGET_MEANINGS)},
                channel_units_tree={"predictions": list(TARGET_UNITS)},
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
        TreeArrayDataset(x={"parameters": x_train}, y={"predictions": y_train_scaled}),
        validation_dataset=TreeArrayDataset(
            x={"parameters": x_val}, y={"predictions": y_val_scaled}
        ),
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

    named_pred = emu.make_frozen_apply(jit=False)({"parameters": x_val})
    pred_scaled = np.asarray(named_pred["predictions"])
    pred = _unscale_array(pred_scaled, mins=output_scale_min, maxs=output_scale_max)
    _save_training_curves(history.logs)
    _save_parity_plot(y_val, pred)

    mae = np.mean(np.abs(pred - y_val), axis=0)
    rmse = np.sqrt(np.mean((pred - y_val) ** 2, axis=0))

    input_min = [float(v) for v in input_scale_min]
    input_max = [float(v) for v in input_scale_max]
    output_min = [float(v) for v in output_scale_min]
    output_max = [float(v) for v in output_scale_max]
    bundle_dir = emu.save_bundle(
        spec={
            "inputs": {
                "structure_tree": {"parameters": None},
                "channel_names_tree": {"parameters": list(PARAMETER_NAMES)},
                "leaf_meanings_tree": {
                    "parameters": "isochrone conditioning parameters"
                },
                "channel_meanings_tree": {"parameters": list(PARAMETER_MEANINGS)},
                "channel_units_tree": {"parameters": list(PARAMETER_UNITS)},
            },
            "outputs": {
                "structure_tree": {"predictions": None},
                "channel_names_tree": {"predictions": list(TARGET_NAMES)},
                "leaf_meanings_tree": {"predictions": "scaled isochrone target vector"},
                "channel_meanings_tree": {"predictions": list(TARGET_MEANINGS)},
                "channel_units_tree": {"predictions": list(TARGET_UNITS)},
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
                "min_tree": {"predictions": output_min},
                "max_tree": {"predictions": output_max},
            },
        },
        extras={
            "notes": (
                "Inputs and outputs are min-max scaled before training using the fixed bounds stored in this bundle. "
                "Use the recorded input/output reference scaling blocks to map between physical and model spaces."
            ),
        },
    )

    print("Validation metrics:")
    for name, unit, mae_i, rmse_i in zip(TARGET_NAMES, TARGET_UNITS, mae, rmse):
        print(f"  {name} [{unit}]: MAE={float(mae_i):.6g}, RMSE={float(rmse_i):.6g}")
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Validation evaluations:", len(history.logs.get("validation_loss", [])))
    print("Named prediction keys:", tuple(named_pred.keys()))
    print("Prediction shape:", pred.shape)
    print("Training CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Checkpoints:", RUN_DIR / "checkpoints")
    print("Saved curves:", CURVES_PATH)
    print("Saved plot:", PLOT_PATH)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
