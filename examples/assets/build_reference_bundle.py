"""Build raw and released Payne flux reference bundle assets.

This script always retrains the reference model locally, writes a clean
intermediate bundle to examples/runs/reference_bundle_builder/bundle, copies
that bundle into examples/assets/reference_bundle_raw, and then prepares
examples/assets/reference_bundle_release from the raw bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle import prepare_bundle_release
from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, RootConfig
from astro_emulators_toolkit.presets import payne_flux_mlp

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "assets" / "reference_bundle_raw"
DEFAULT_RELEASED_OUTPUT_DIR = (
    REPO_ROOT / "examples" / "assets" / "reference_bundle_release"
)
RUN_DIR = REPO_ROOT / "examples" / "runs" / "reference_bundle_builder"
RUN_BUNDLE_DIR = RUN_DIR / "bundle"
TRAIN_CSV = RUN_DIR / "history_train.csv"
VAL_CSV = RUN_DIR / "history_val.csv"
CURVES_PATH = RUN_DIR / "training_validation_curves.png"
PLOT_PATH = RUN_DIR / "validation_example_0.png"
LEGACY_REFERENCE_SCALING_SIDECAR = "reference_scaling.safetensors"

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
REFERENCE_RELEASE_NAME = "payne-flux-reference-example"
REFERENCE_RELEASE_VERSION = "0.1.0"


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _drop_legacy_bundle_sidecars(bundle_dir: Path) -> None:
    _remove_path(bundle_dir / LEGACY_REFERENCE_SCALING_SIDECAR)


def _copy_bundle(source_dir: Path, output_dir: Path) -> Path:
    _remove_path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir)
    _drop_legacy_bundle_sidecars(output_dir)
    return output_dir


def prepare_for_release(bundle_dir: Path, released_output_dir: Path) -> Path:
    _remove_path(released_output_dir)
    prepared = prepare_bundle_release(
        bundle_dir,
        path=released_output_dir,
        release_name=REFERENCE_RELEASE_NAME,
        release_version=REFERENCE_RELEASE_VERSION,
    )
    _drop_legacy_bundle_sidecars(prepared)
    print(
        "Prepared release bundle:",
        f"{REFERENCE_RELEASE_NAME}@{REFERENCE_RELEASE_VERSION}",
    )
    return prepared


def _save_training_curves(history_logs: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def _train_reference_source_bundle() -> Path:
    try:
        import orbax.checkpoint  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing base dependency 'orbax-checkpoint'. Reinstall the toolkit base dependencies "
            "(for example with `uv sync` from a source checkout)."
        ) from exc

    from astro_emulators_toolkit.data import TreeArrayDataset
    from astro_emulators_toolkit.training import (
        CSVLogger,
        ModelCheckpoint,
        ProgressBarLogger,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _remove_path(RUN_DIR)
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

    cfg = payne_flux_mlp(
        workdir=str(RUN_DIR),
        profile="cpu_recommended",
    )
    cfg = RootConfig(
        schema_version=cfg.schema_version,
        seed=cfg.seed,
        model=cfg.model,
        task=cfg.task,
        solver=cfg.solver,
        optim=cfg.optim,
        training=cfg.training,
        bundle=cfg.bundle,
        hub=cfg.hub,
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
            reference_scaling_inputs=cfg.io.reference_scaling_inputs,
            reference_scaling_outputs=cfg.io.reference_scaling_outputs,
            input_domain=cfg.io.input_domain,
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

    pred = np.asarray(emu.predict({"parameters": x_val[:1]})[Y_NAME][0])

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
        RUN_BUNDLE_DIR,
        spec={
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
                "min_tree": {Y_NAME: 0.0},
                "max_tree": {Y_NAME: 1.0},
            },
        },
        extras={
            "wavelength_angstrom": [float(v) for v in wave],
            "notes": "Inputs are min-max scaled before training using the fixed bounds stored in this bundle. Output reference scaling metadata is recorded separately and is identity.",
        },
    )
    _drop_legacy_bundle_sidecars(bundle_dir)

    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Validation evaluations:", len(history.logs.get("validation_loss", [])))
    print("Named prediction keys:", (Y_NAME,))
    print("Named flux shape:", pred.shape)
    print("Training CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Checkpoints:", RUN_DIR / "checkpoints")
    print("Saved curves:", CURVES_PATH)
    print("Saved plot:", PLOT_PATH)
    print("Saved source bundle:", bundle_dir)
    return bundle_dir


def build_reference_bundle(output_dir: Path) -> Path:
    trained_source = _train_reference_source_bundle()
    copied = _copy_bundle(trained_source, output_dir)
    print("Saved raw bundle:", copied)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the raw reference bundle asset.",
    )
    parser.add_argument(
        "--released-output",
        type=Path,
        default=DEFAULT_RELEASED_OUTPUT_DIR,
        help="Destination for the released reference bundle asset.",
    )
    args = parser.parse_args()
    bundle_path = build_reference_bundle(args.output.resolve())
    released_bundle_path = prepare_for_release(
        bundle_path, args.released_output.resolve()
    )
    print("Bundle:", bundle_path)
    print("Released bundle:", released_bundle_path)


if __name__ == "__main__":
    main()
