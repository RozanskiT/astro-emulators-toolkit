"""Train transformer_payne for intensity with SOAP, CSV logs, checkpoints, and bundle metadata.

Data: irregular_intensity split into train/validation from the same randomized distribution.
Creates: examples/runs/development_transformer_payne_intensity_soap/{bundle,checkpoints,history_*.csv,training_validation_curves.png,validation_example_0.png}
Runtime: minutes on CPU.
Requires: `uv sync` from a source checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
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
from astro_emulators_toolkit.data import DeviceBatchTransform, TreeArrayDataset
from astro_emulators_toolkit.training import (
    CSVLogger,
    ModelCheckpoint,
    ProgressBarLogger,
)

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_intensity_arrays

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO_ROOT / "examples" / "runs" / "development_transformer_payne_intensity_soap"
)
TRAIN_CSV = RUN_DIR / "history_train.csv"
VAL_CSV = RUN_DIR / "history_val.csv"
CURVES_PATH = RUN_DIR / "training_validation_curves.png"
PLOT_PATH = RUN_DIR / "validation_example_0.png"

PARAMETER_NAMES = ("teff", "logg", "feh", "mu")
PARAMETER_MEANINGS = (
    "effective temperature",
    "surface gravity",
    "metallicity [Fe/H]",
    "cosine of viewing angle",
)
PARAMETER_UNITS = ("K", "dex", "dex", "dimensionless")
CHANNEL_NAMES = ("normalized_intensity", "log10_continuum_minmax")

INPUT_SCALE_BOUNDS = {
    "teff": (4500.0, 7000.0),
    "logg": (2.5, 5.0),
    "feh": (-0.3, 0.3),
    "mu": (0.001, 1.0),
}

SEED = 0
VAL_FRACTION = 0.1
NUM_STEPS = 3_000
BATCH_SIZE = 64
LOG_EVERY = 50
VAL_EVERY = 200
CHECKPOINT_EVERY = 1000
MAX_SAVED_CHECKPOINTS = 5
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = NUM_STEPS // 10
MIN_PERIOD = 3e-2
MAX_PERIOD = 30.0
N_WAVELENGTH = 500
LOG_CLIP_MIN = np.finfo(np.float32).tiny


@dataclass(frozen=True)
class NormalizedIntensityDeviceBatchTransform:
    base_transform: DeviceBatchTransform
    log_cont_min: float
    log_cont_max: float

    def for_init(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self(batch, train=False, rng=None)

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        rng: Any,
        train: bool,
    ) -> dict[str, Any]:
        transformed = self.base_transform(batch, rng=rng, train=train)
        out = dict(transformed)
        flux = transformed["y"]["flux"]
        lines = flux[..., 0]
        continuum = jnp.clip(flux[..., 1], LOG_CLIP_MIN, None)
        normalized_intensity = lines / continuum
        log_continuum = jnp.log10(continuum)
        continuum_minmax = (log_continuum - self.log_cont_min) / (
            self.log_cont_max - self.log_cont_min
        )
        y_out = dict(transformed["y"])
        y_out["flux"] = jnp.stack((normalized_intensity, continuum_minmax), axis=-1)
        out["y"] = y_out
        return out


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


def _save_validation_intensity_plot(
    wave: np.ndarray,
    y_true_norm_intensity: np.ndarray,
    y_true_cont_minmax: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    ax_lines, ax_cont = axes[0]
    ax_lines_resid, ax_cont_resid = axes[1]

    pred_norm_intensity = y_pred[:, 0]
    pred_cont_minmax = y_pred[:, 1]

    ax_lines.plot(wave, y_true_norm_intensity, label="ground truth", lw=1.3)
    ax_lines.plot(wave, pred_norm_intensity, label="prediction", lw=1.0, alpha=0.85)
    ax_lines.set_title("Normalized intensity (lines / continuum)")
    ax_lines.set_ylabel("Dimensionless")
    ax_lines.grid(alpha=0.2)
    ax_lines.legend()

    ax_cont.plot(wave, y_true_cont_minmax, label="ground truth", lw=1.3)
    ax_cont.plot(wave, pred_cont_minmax, label="prediction", lw=1.0, alpha=0.85)
    ax_cont.set_title("log10(continuum) min-max")
    ax_cont.grid(alpha=0.2)
    ax_cont.legend()

    ax_lines_resid.plot(
        wave, pred_norm_intensity - y_true_norm_intensity, color="0.15", lw=1.0
    )
    ax_lines_resid.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax_lines_resid.set_xlabel("Wavelength [Angstrom]")
    ax_lines_resid.set_ylabel("Residual")
    ax_lines_resid.grid(alpha=0.2)

    ax_cont_resid.plot(
        wave, pred_cont_minmax - y_true_cont_minmax, color="0.15", lw=1.0
    )
    ax_cont_resid.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax_cont_resid.set_xlabel("Wavelength [Angstrom]")
    ax_cont_resid.grid(alpha=0.2)

    fig.suptitle("Validation example 0")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def _print_axiswise_validation(history_logs: dict[str, list[float]]) -> None:
    print("Validation metrics (axis-wise by channel):")
    for i, channel_name in enumerate(CHANNEL_NAMES):
        mse_series = history_logs.get(f"validation_mse_channel_{i}", [])
        mae_series = history_logs.get(f"validation_mae_channel_{i}", [])
        mse_val = float(mse_series[-1]) if mse_series else float("nan")
        mae_val = float(mae_series[-1]) if mae_series else float("nan")
        print(f"  {channel_name}: mse={mse_val:.6g}, mae={mae_val:.6g}")


def _scale_log_channel(
    values: np.ndarray, *, log_min: float, log_max: float
) -> np.ndarray:
    return ((values.astype(np.float32) - log_min) / (log_max - log_min)).astype(
        np.float32
    )


def _unscale_log_channel(
    values: np.ndarray, *, log_min: float, log_max: float
) -> np.ndarray:
    return (values.astype(np.float32) * (log_max - log_min) + log_min).astype(
        np.float32
    )


def main() -> None:
    try:
        import orbax.checkpoint  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing base dependency 'orbax-checkpoint'. Reinstall the toolkit base dependencies "
            "(for example with `uv sync` from a source checkout)."
        ) from exc

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, wave = split_randomized_intensity_arrays(
        val_fraction=VAL_FRACTION, seed=SEED
    )
    wave_lines = wave["lines"].astype(np.float64)
    wave_cont = wave["continuum"].astype(np.float64)
    eval_wave = np.linspace(
        max(float(wave_lines[0]), float(wave_cont[0])),
        min(float(wave_lines[-1]), float(wave_cont[-1])),
        num=N_WAVELENGTH,
        dtype=np.float64,
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

    train_log_cont = np.log10(
        np.clip(y_train["continuum"].astype(np.float32), LOG_CLIP_MIN, None)
    )
    val_log_cont = np.log10(
        np.clip(y_val["continuum"].astype(np.float32), LOG_CLIP_MIN, None)
    )
    log_cont_min = min(float(train_log_cont.min()), float(val_log_cont.min()))
    log_cont_max = max(float(train_log_cont.max()), float(val_log_cont.max()))

    print("Manual input scaling bounds:")
    for name, unit, lo, hi in zip(
        PARAMETER_NAMES, PARAMETER_UNITS, input_scale_min, input_scale_max
    ):
        print(f"  {name} [{unit}]: min={float(lo):.6g}, max={float(hi):.6g}")

    print(f"Train samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Lines wavelength points: {wave_lines.shape[0]}")
    print(f"Continuum wavelength points: {wave_cont.shape[0]}")
    print(f"Evaluation wavelength points: {eval_wave.shape[0]}")
    print("Continuum log10 min-max normalization bounds:")
    print(f"  min={log_cont_min:.6g}, max={log_cont_max:.6g}")

    cfg = RootConfig(
        seed=SEED,
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 2,
                "dim": 32,
                "dim_head": 32,
                "no_layers": 2,
                "no_tokens": 4,
                "dim_ff_multiplier": 2,
                "min_period": MIN_PERIOD,
                "max_period": MAX_PERIOD,
                "dtype": "float32",
            },
        ),
        task=TaskSpec(
            name="regression",
            params={
                "loss": "mse",
                "metrics": ("mse", "mae"),
                "metric_axes": {"global": "all", "channel": [0]},
            },
        ),
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
                structure_tree={"parameters": None, "wavelengths": None},
                channel_names_tree={
                    "parameters": list(PARAMETER_NAMES),
                    "wavelengths": None,
                },
                leaf_meanings_tree={
                    "parameters": "stellar labels",
                    "wavelengths": "query wavelengths",
                },
                leaf_units_tree={"parameters": None, "wavelengths": "angstrom"},
                channel_meanings_tree={
                    "parameters": list(PARAMETER_MEANINGS),
                    "wavelengths": None,
                },
                channel_units_tree={
                    "parameters": list(PARAMETER_UNITS),
                    "wavelengths": None,
                },
            ),
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                channel_names_tree={"flux": list(CHANNEL_NAMES)},
                leaf_meanings_tree={"flux": "transformed intensity target channels"},
                channel_meanings_tree={
                    "flux": [
                        "line intensity divided by continuum intensity",
                        "min-max normalized log10 continuum intensity",
                    ]
                },
                channel_units_tree={"flux": ["dimensionless", "dimensionless"]},
            ),
        ),
    )

    model_params = dict(cfg.model.params)
    print("Training configuration:")
    print("  model: transformer_payne")
    print(
        "  architecture: "
        f"tokens={model_params['no_tokens']}, layers={model_params['no_layers']}, "
        f"dim={model_params['dim']}, dim_head={model_params['dim_head']}, ff_mult={model_params['dim_ff_multiplier']}"
    )
    print(
        f"  wavelength encoding periods: min={model_params['min_period']:.1e}, max={model_params['max_period']:.1e}"
    )
    print("  optimizer: soap")
    print("  schedule: linear warmup + cosine decay")
    print(f"  max learning rate: {cfg.optim.lr:.1e}")
    print(f"  warmup steps: {cfg.optim.warmup_steps}")
    print(f"  total steps: {cfg.training.num_steps}")
    print(f"  validation every: {cfg.training.evaluation_interval_steps} steps")
    print(f"  checkpoint every: {cfg.training.checkpoint_interval_steps} steps")

    emu = Emulator.from_config(cfg).configure_training()
    device_batch_transform = NormalizedIntensityDeviceBatchTransform(
        base_transform=emu.make_device_batch_transform(
            mode="intensity",
            common_waves={"lines": wave_lines, "continuum": wave_cont},
            n_wavelength=eval_wave.shape[0],
            eval_wavelength_grid=eval_wave,
            output_order=("lines", "continuum"),
            min_w=float(eval_wave[0]),
            max_w=float(eval_wave[-1]),
        ),
        log_cont_min=log_cont_min,
        log_cont_max=log_cont_max,
    )

    history = emu.fit(
        TreeArrayDataset(x={"parameters": x_train}, y=y_train),
        validation_dataset=TreeArrayDataset(x={"parameters": x_val}, y=y_val),
        device_batch_transform=device_batch_transform,
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

    inference_wave = np.broadcast_to(eval_wave[None, :], (1, eval_wave.shape[0]))
    named_pred = emu.make_frozen_apply(jit=False)(
        {"parameters": x_val[:1], "wavelengths": inference_wave}
    )
    pred = np.asarray(named_pred["flux"][0])
    pred_log_cont = _unscale_log_channel(
        pred[:, 1], log_min=log_cont_min, log_max=log_cont_max
    )

    y_true_lines = np.interp(
        eval_wave, wave_lines, y_val["lines"][0].astype(np.float64)
    ).astype(np.float32)
    y_true_cont = np.interp(
        eval_wave, wave_cont, y_val["continuum"][0].astype(np.float64)
    ).astype(np.float32)
    y_true_norm_intensity = y_true_lines / np.clip(y_true_cont, LOG_CLIP_MIN, None)
    y_true_cont_minmax = _scale_log_channel(
        np.log10(np.clip(y_true_cont, LOG_CLIP_MIN, None)),
        log_min=log_cont_min,
        log_max=log_cont_max,
    )
    _save_validation_intensity_plot(
        eval_wave, y_true_norm_intensity, y_true_cont_minmax, pred
    )

    parameter_tree = {
        name: {"min": float(lo), "max": float(hi)}
        for name, lo, hi in zip(PARAMETER_NAMES, input_scale_min, input_scale_max)
    }
    parameter_min = [float(parameter_tree[name]["min"]) for name in PARAMETER_NAMES]
    parameter_max = [float(parameter_tree[name]["max"]) for name in PARAMETER_NAMES]
    bundle_dir = emu.save_bundle(
        spec={
            "inputs": {
                "structure_tree": {"parameters": None, "wavelengths": None},
                "channel_names_tree": {
                    "parameters": list(PARAMETER_NAMES),
                    "wavelengths": None,
                },
                "leaf_meanings_tree": {
                    "parameters": "stellar labels",
                    "wavelengths": "query wavelengths",
                },
                "leaf_units_tree": {"parameters": None, "wavelengths": "angstrom"},
                "channel_meanings_tree": {
                    "parameters": list(PARAMETER_MEANINGS),
                    "wavelengths": None,
                },
                "channel_units_tree": {
                    "parameters": list(PARAMETER_UNITS),
                    "wavelengths": None,
                },
            },
            "outputs": {
                "structure_tree": {"flux": None},
                "channel_names_tree": {"flux": list(CHANNEL_NAMES)},
                "leaf_meanings_tree": {"flux": "transformed intensity target channels"},
                "channel_meanings_tree": {
                    "flux": [
                        "line intensity divided by continuum intensity",
                        "min-max normalized log10 continuum intensity",
                    ]
                },
                "channel_units_tree": {"flux": ["dimensionless", "dimensionless"]},
            },
            "input_domain": {
                "kind": "box_v1",
                "value_space": "physical_input_dict_tree_v1",
                "min_tree": {
                    "parameters": parameter_min,
                    "wavelengths": float(eval_wave[0]),
                },
                "max_tree": {
                    "parameters": parameter_max,
                    "wavelengths": float(eval_wave[-1]),
                },
            },
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {
                    "parameters": parameter_min,
                    "wavelengths": float(eval_wave[0]),
                },
                "max_tree": {
                    "parameters": parameter_max,
                    "wavelengths": float(eval_wave[-1]),
                },
            },
            "reference_scaling_outputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "outputs",
                "source_space": "canonical_output_dict_tree_v1",
                "target_space": "physical_output_dict_tree_v1",
                "min_tree": {"flux": [0.0, log_cont_min]},
                "max_tree": {"flux": [1.0, log_cont_max]},
            },
        },
        extras={
            "evaluation_wavelength_angstrom": [float(v) for v in eval_wave],
            "notes": (
                "TransformerPayne intensity model is trained with random wavelength sampling in-bounds during training and "
                "evaluated on the fixed shared wavelength grid recorded in this bundle. Targets are transformed to "
                "two channels: (1) normalized intensity lines/continuum, and (2) log10 continuum min-max normalized."
            ),
        },
    )

    print("Fit method:", emu.last_fit_method)
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Validation evaluations:", len(history.logs.get("validation_loss", [])))
    _print_axiswise_validation(history.logs)
    print("Named prediction keys:", tuple(named_pred.keys()))
    print("Named flux shape:", named_pred["flux"].shape)
    print("Channel semantics:", CHANNEL_NAMES)
    print(
        "Predicted log10 continuum range (unscaled):",
        f"{float(pred_log_cont.min()):.6g} .. {float(pred_log_cont.max()):.6g}",
    )
    print("Training CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Checkpoints:", RUN_DIR / "checkpoints")
    print("Saved curves:", CURVES_PATH)
    print("Saved plot:", PLOT_PATH)
    print("Saved bundle:", bundle_dir)


if __name__ == "__main__":
    main()
