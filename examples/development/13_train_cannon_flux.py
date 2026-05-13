"""Train a Cannon flux emulator with closed-form solver, CSV logs, plots, and bundle scaling metadata.

Data: irregular_flux split into train/validation from the same randomized distribution.
Creates: examples/runs/development_cannon_flux/{bundle,history_*.csv,validation_example_0.png}
Runtime: a few seconds on CPU.
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
    RootConfig,
    SolverConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.models.cannon import cannon_feature_dim
from astro_emulators_toolkit.training import CSVLogger, ProgressBarLogger

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "development_cannon_flux"
TRAIN_CSV = RUN_DIR / "history_train.csv"
VAL_CSV = RUN_DIR / "history_val.csv"
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
RIDGE = 1e-4


def _save_validation_flux_plot(
    wave: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    fig, (ax_flux, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": (3, 1)}
    )
    ax_flux.plot(wave, y_true, label="ground truth", lw=1.4)
    ax_flux.plot(wave, y_pred, label="prediction", lw=1.1, alpha=0.85)
    ax_flux.set_ylabel("Normalized flux")
    ax_flux.set_title("Validation example 0")
    ax_flux.grid(alpha=0.2)
    ax_flux.legend()

    ax_resid.plot(wave, y_pred - y_true, color="0.15", lw=1.0)
    ax_resid.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax_resid.set_xlabel("Wavelength [Angstrom]")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
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
            name="cannon", params={"include_bias": True, "dtype": "float32"}
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        solver=SolverConfig(
            name="closed_form_linear",
            params={"ridge": RIDGE, "regularize_intercept": False},
        ),
        training=TrainConfig(
            workdir=str(RUN_DIR),
            batch_size=256,
            num_steps=1,
            val_fraction=VAL_FRACTION,
            logging_interval_steps=1,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
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

    feature_dim = cannon_feature_dim(x_train.shape[1], include_bias=True)
    print("Training configuration:")
    print("  model: cannon")
    print(
        f"  feature space: {x_train.shape[1]} parameters -> {feature_dim} cannon features -> {y_train.shape[1]} outputs"
    )
    print(f"  solver: {cfg.solver.name}")
    print(f"  ridge regularization: {cfg.solver.params['ridge']:.1e}")
    print(f"  regularize intercept: {cfg.solver.params['regularize_intercept']}")

    emu = Emulator.from_config(cfg).configure_training()
    history = emu.fit(
        TreeArrayDataset(x={"parameters": x_train}, y={Y_NAME: y_train}),
        validation_dataset=TreeArrayDataset(x={"parameters": x_val}, y={Y_NAME: y_val}),
        callbacks=[
            ProgressBarLogger(total_steps=cfg.training.num_steps),
            CSVLogger(TRAIN_CSV, split="train"),
            CSVLogger(VAL_CSV, split="val"),
        ],
    )

    named_pred = emu.make_frozen_apply(jit=False)({"parameters": x_val[:1]})
    pred = np.asarray(named_pred[Y_NAME][0])
    _save_validation_flux_plot(wave, y_val[0], pred)

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
            "notes": (
                "Cannon is trained with the closed_form_linear solver. Inputs are min-max scaled before fitting "
                "using the fixed bounds stored in this bundle. The intercept column is left unregularized, while "
                "quadratic and linear feature weights use the configured ridge value. Output reference scaling "
                "metadata is recorded separately and is identity."
            ),
        },
    )

    print("Fit method:", emu.last_fit_method)
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Validation evaluations:", len(history.logs.get("validation_loss", [])))
    print("Named prediction keys:", tuple(named_pred.keys()))
    print("Named flux shape:", named_pred[Y_NAME].shape)
    print("Training CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Saved plot:", PLOT_PATH)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
