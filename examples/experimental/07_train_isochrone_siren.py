"""Train an experimental SIREN emulator on the example isochrone table.

Data: examples/examples_datasets/isochrones/mist_isochrones_dev.npy.
Creates: examples/runs/experimental_isochrone_siren/{bundle,run_config.json}.
Runtime: ~10-20s on CPU; much shorter with ASTRO_EMU_EXAMPLE_SMOKE=1.
Optional extras: none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    MinMaxTreeSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.training import build_callbacks_from_config

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import (
    ISOCHRONE_X_COLUMNS,
    ISOCHRONE_Y_COLUMNS,
    split_isochrones_named_arrays,
)
from _example_smoke import smoke_value

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "experimental_isochrone_siren"

SEED = 0
VAL_FRACTION = 0.1
NUM_STEPS = smoke_value(800, smoke=20)
BATCH_SIZE = 128
MAX_LR = 1e-4
WARMUP_STEPS = max(1, NUM_STEPS // 10)


def _scale_array(x: np.ndarray, *, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mins) / (maxs - mins)).astype(np.float32)


def _unscale_array(x: np.ndarray, *, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) * (maxs - mins) + mins).astype(np.float32)


def _minmax_from_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate(arrays, axis=0).astype(np.float32)
    return joined.min(axis=0), joined.max(axis=0)


def main() -> None:
    x_train, y_train, x_val, y_val = split_isochrones_named_arrays(
        val_fraction=VAL_FRACTION, seed=SEED
    )
    input_min, input_max = _minmax_from_arrays(x_train, x_val)
    output_min, output_max = _minmax_from_arrays(y_train, y_val)

    x_train_scaled = _scale_array(x_train, mins=input_min, maxs=input_max)
    x_val_scaled = _scale_array(x_val, mins=input_min, maxs=input_max)
    y_train_scaled = _scale_array(y_train, mins=output_min, maxs=output_max)
    y_val_scaled = _scale_array(y_val, mins=output_min, maxs=output_max)

    cfg = RootConfig(
        seed=SEED,
        model=ModelSpec(
            name="experimental/siren",
            params={
                "hidden_sizes": (128, 128, 128),
                "omega0_first": 12.0,
                "omega0_hidden": 1.0,
                "dtype": "float32",
            },
            init_hints={
                "input_last_axis": len(ISOCHRONE_X_COLUMNS),
                "output_last_axis": len(ISOCHRONE_Y_COLUMNS),
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(
            name="adamw",
            lr=MAX_LR,
            schedule="cosine",
            warmup_steps=WARMUP_STEPS,
            weight_decay=1e-6,
        ),
        training=TrainConfig(
            workdir=str(RUN_DIR),
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            val_fraction=VAL_FRACTION,
            logging_interval_steps=smoke_value(50, smoke=5),
            evaluation_interval_steps=smoke_value(100, smoke=10),
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
        ),
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"parameters": None},
                channel_names_tree={"parameters": list(ISOCHRONE_X_COLUMNS)},
            ),
            outputs=IOTreeSpec(
                structure_tree={"targets": None},
                channel_names_tree={"targets": list(ISOCHRONE_Y_COLUMNS)},
            ),
            reference_scaling_inputs=MinMaxTreeSpec(
                min_tree={"parameters": input_min.tolist()},
                max_tree={"parameters": input_max.tolist()},
            ),
            reference_scaling_outputs=MinMaxTreeSpec(
                min_tree={"targets": output_min.tolist()},
                max_tree={"targets": output_max.tolist()},
            ),
            input_domain=MinMaxTreeSpec(
                min_tree={"parameters": input_min.tolist()},
                max_tree={"parameters": input_max.tolist()},
            ),
        ),
    )

    print("Experimental model:", cfg.model.name)
    print(f"Train samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Steps: {cfg.training.num_steps}")
    print(f"Max learning rate: {cfg.optim.lr:.1e}")
    print(f"Warmup steps: {cfg.optim.warmup_steps}")

    emu = Emulator.from_config(cfg).configure_training()
    history = emu.fit(
        TreeArrayDataset(
            x={"parameters": x_train_scaled},
            y={"targets": y_train_scaled},
        ),
        validation_dataset=TreeArrayDataset(
            x={"parameters": x_val_scaled},
            y={"targets": y_val_scaled},
        ),
        callbacks=build_callbacks_from_config(cfg),
    )

    pred_scaled = emu.predict({"parameters": x_val_scaled})["targets"]
    pred = _unscale_array(pred_scaled, mins=output_min, maxs=output_max)
    mae = np.mean(np.abs(pred - y_val), axis=0)

    bundle_dir = emu.save_bundle()
    print("Final validation loss:", history.logs.get("validation_loss", [None])[-1])
    print("Mean physical MAE:", float(mae.mean()))
    print("Prediction shape:", pred.shape)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
