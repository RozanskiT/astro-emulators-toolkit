"""Benchmark transformer_payne flux training and validation hot paths to JSON.

This maintainer-focused script targets the stable transformer_payne flux workflow
and records small component benchmarks before and after performance changes.

Typical usage:

    uv run python examples/development/16_benchmark_transformer_payne_flux_training_components.py --label baseline
    uv run python examples/development/16_benchmark_transformer_payne_flux_training_components.py --label after_eval_jit --compare-to examples/runs/development_benchmarks/baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

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
from astro_emulators_toolkit.data import TreeArrayDataset, DataLoader
from astro_emulators_toolkit.data.protocols import (
    call_device_batch_transform,
    init_batch_via_device_transform,
)
from astro_emulators_toolkit.optimizers import make_tx
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "examples" / "runs" / "development_benchmarks"

PARAMETER_NAMES = ("teff", "logg", "feh")
PARAMETER_MEANINGS = ("effective temperature", "surface gravity", "metallicity [Fe/H]")
PARAMETER_UNITS = ("K", "dex", "dex")
INPUT_SCALE_BOUNDS = {
    "teff": (4500.0, 7000.0),
    "logg": (2.5, 5.0),
    "feh": (-0.3, 0.3),
}
Y_NAME = "flux"
SEED = 0
VAL_FRACTION = 0.1
DEFAULT_BATCH_SIZE = 64
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 15
DEFAULT_SWEEP_REPEAT = 5
DEFAULT_TRAIN_SAMPLES = 4096
DEFAULT_VAL_SAMPLES = 512
MAX_LR = 3e-3
WEIGHT_DECAY = 1e-5
NUM_STEPS = 256
WARMUP_STEPS = NUM_STEPS // 10
MIN_PERIOD = 3e-2
MAX_PERIOD = 30.0


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    num_samples: int


def _block_tree(tree: Any) -> Any:
    def _block_leaf(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_block_leaf, tree)


def _stats(samples_ms: list[float]) -> BenchmarkStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return BenchmarkStats(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p90_ms=float(np.percentile(arr, 90.0)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
        num_samples=int(arr.size),
    )


def _time_fn(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> BenchmarkStats:
    for _ in range(int(warmup)):
        fn()
    samples_ms: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples_ms)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build_cfg(*, batch_size: int) -> RootConfig:
    return RootConfig(
        seed=SEED,
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 1,
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
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(
            name="soap",
            lr=MAX_LR,
            schedule="cosine",
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
        ),
        training=TrainConfig(
            workdir=str(RUN_DIR / "_scratch"),
            batch_size=batch_size,
            num_steps=NUM_STEPS,
            val_fraction=VAL_FRACTION,
            logging_interval_steps=0,
            evaluation_interval_steps=64,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
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
                structure_tree={Y_NAME: None},
                leaf_meanings_tree={
                    Y_NAME: "continuum-normalized flux evaluated on the queried wavelength grid"
                },
                leaf_units_tree={Y_NAME: "dimensionless"},
            ),
        ),
    )


def _build_workload(
    *,
    batch_size: int,
    train_samples: int,
    val_samples: int,
):
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
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    wave = wave.astype(np.float64)

    n_train = min(int(train_samples), int(x_train.shape[0]))
    n_val = min(int(val_samples), int(x_val.shape[0]))
    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            "Benchmark workload requires at least one training and validation sample."
        )

    train_dataset = TreeArrayDataset(
        x={"parameters": x_train[:n_train]}, y={Y_NAME: y_train[:n_train]}
    )
    val_dataset = TreeArrayDataset(
        x={"parameters": x_val[:n_val]}, y={Y_NAME: y_val[:n_val]}
    )
    cfg = _build_cfg(batch_size=batch_size)

    sample_batch = train_dataset.get_batch(np.asarray([0], dtype=np.int64))
    emu = Emulator.from_config(cfg)
    transform = emu.make_device_batch_transform(
        mode="flux",
        wavelength_grid=wave,
        eval_wavelength_grid=wave,
        n_wavelength=wave.shape[0],
        min_w=float(wave[0]),
        max_w=float(wave[-1]),
    )
    init_batch = init_batch_via_device_transform(transform, sample_batch)
    emu.initialize(inputs=init_batch["x"], outputs=init_batch["y"])
    emu = emu.configure_training(optimizer=make_tx(cfg))

    return {
        "cfg": cfg,
        "emu": emu,
        "transform": transform,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "wave": wave,
        "n_train": n_train,
        "n_val": n_val,
    }


def _build_train_step(*, graphdef, task, tx, device_batch_transform):
    @jax.jit
    def train_step(state: TrainState, batch: dict[str, Any]):
        state, step_key = state.next_rng()
        model_key = step_key
        if device_batch_transform is not None:
            transform_key, model_key = jax.random.split(step_key)
            original_batch = batch
            transformed = call_device_batch_transform(
                device_batch_transform,
                batch,
                rng=transform_key,
                train=True,
            )
            batch = trainer._merge_batch_metadata(original_batch, transformed)

        rngs = nnx.Rngs(dropout=model_key)

        def loss_fn(params, model_state):
            full_state = nnx.merge_state(params, model_state)
            pred, (_, new_full_state) = nnx.call((graphdef, full_state))(
                batch["x"], train=True, rngs=rngs
            )
            _, new_model_state = (
                nnx.split_state(new_full_state, nnx.Param, ...)
                if new_full_state is not None
                else (None, model_state)
            )
            loss, metrics = task.loss_and_metrics(pred, batch)
            return loss, (metrics, new_model_state)

        (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(state.params, state.model_state)
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + jnp.array(1, dtype=state.step.dtype),
            params=new_params,
            model_state=new_model_state,
            opt_state=new_opt_state,
        )
        return new_state, {"loss": loss, **metrics}

    return train_step


def _build_eval_predict(*, graphdef, device_batch_transform):
    builder = getattr(trainer, "_build_eval_predict", None)
    if builder is not None:
        return builder(graphdef=graphdef, device_batch_transform=device_batch_transform)

    def eval_predict(state: TrainState, batch: dict[str, Any]):
        if device_batch_transform is not None:
            original_batch = batch
            transformed = call_device_batch_transform(
                device_batch_transform,
                batch,
                rng=jax.random.key(0),
                train=False,
            )
            batch = trainer._merge_batch_metadata(original_batch, transformed)
        full_state = nnx.merge_state(state.params, state.model_state)
        pred, _ = nnx.call((graphdef, full_state))(batch["x"], train=False, rngs=None)
        return pred, batch

    return eval_predict


def _benchmark_validation_sweep(
    *,
    state: TrainState,
    eval_predict,
    task,
    val_batches_device: list[dict[str, Any]],
    warmup: int,
    repeat: int,
) -> BenchmarkStats:
    def run_sweep():
        eval_state = task.init_eval_state()
        for batch in val_batches_device:
            pred, batch_eval = eval_predict(state, batch)
            eval_state = task.update_eval_state(eval_state, pred, batch_eval)
        agg = task.finalize_eval(eval_state)
        _block_tree(agg)
        return agg

    return _time_fn(run_sweep, warmup=warmup, repeat=repeat)


def _benchmark_eval_batch(
    *,
    state: TrainState,
    eval_predict,
    batch: dict[str, Any],
    warmup: int,
    repeat: int,
) -> BenchmarkStats:
    def run_batch():
        pred, batch_eval = eval_predict(state, batch)
        _block_tree((pred, batch_eval))
        return pred

    return _time_fn(run_batch, warmup=warmup, repeat=repeat)


def _compare(current: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    current_bench = current.get("benchmarks", {})
    baseline_bench = baseline.get("benchmarks", {})

    print(f"Comparison vs {baseline_path}:")
    for name in sorted(set(current_bench) & set(baseline_bench)):
        cur = float(current_bench[name]["median_ms"])
        base = float(baseline_bench[name]["median_ms"])
        if base == 0.0:
            continue
        delta_pct = 100.0 * (cur - base) / base
        print(f"  {name}: {cur:.3f} ms vs {base:.3f} ms ({delta_pct:+.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True, help="Output label used for the JSON filename."
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Optional earlier JSON file to compare against.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--sweep-repeat", type=int, default=DEFAULT_SWEEP_REPEAT)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    workload = _build_workload(
        batch_size=int(args.batch_size),
        train_samples=int(args.train_samples),
        val_samples=int(args.val_samples),
    )
    cfg = workload["cfg"]
    emu = workload["emu"]
    transform = workload["transform"]
    train_dataset = workload["train_dataset"]
    val_dataset = workload["val_dataset"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=cfg.training.shuffle,
        seed=cfg.training.shuffle_seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

    host_train_batches = [
        train_loader.train_batch(i) for i in range(max(8, int(args.repeat)))
    ]
    host_val_batches = list(val_loader.iter_eval_batches())
    if not host_val_batches:
        raise ValueError("Validation benchmark requires at least one validation batch.")

    device_train_batches = [
        trainer._to_jax_batch(batch) for batch in host_train_batches
    ]
    device_val_batches = [trainer._to_jax_batch(batch) for batch in host_val_batches]

    train_step = _build_train_step(
        graphdef=emu.graphdef,
        task=emu.task,
        tx=emu.tx,
        device_batch_transform=transform,
    )
    eval_predict = _build_eval_predict(
        graphdef=emu.graphdef,
        device_batch_transform=transform,
    )

    state_holder = {"state": emu._init_train_state(), "i": 0}

    def run_loader_train_batch():
        idx = state_holder["i"] % len(host_train_batches)
        state_holder["i"] += 1
        return train_loader.train_batch(idx)

    def run_to_jax_batch():
        idx = state_holder["i"] % len(host_train_batches)
        state_holder["i"] += 1
        batch = trainer._to_jax_batch(host_train_batches[idx])
        _block_tree(batch)
        return batch

    def run_train_step():
        idx = state_holder["i"] % len(device_train_batches)
        state_holder["i"] += 1
        new_state, logs = train_step(state_holder["state"], device_train_batches[idx])
        _block_tree(logs["loss"])
        state_holder["state"] = new_state
        return logs

    loader_stats = _time_fn(
        run_loader_train_batch, warmup=int(args.warmup), repeat=int(args.repeat)
    )
    to_jax_stats = _time_fn(
        run_to_jax_batch, warmup=int(args.warmup), repeat=int(args.repeat)
    )
    train_step_stats = _time_fn(
        run_train_step, warmup=int(args.warmup), repeat=int(args.repeat)
    )
    eval_batch_stats = _benchmark_eval_batch(
        state=state_holder["state"],
        eval_predict=eval_predict,
        batch=device_val_batches[0],
        warmup=int(args.warmup),
        repeat=int(args.repeat),
    )
    validation_sweep_stats = _benchmark_validation_sweep(
        state=state_holder["state"],
        eval_predict=eval_predict,
        task=emu.task,
        val_batches_device=device_val_batches,
        warmup=int(args.warmup),
        repeat=int(args.sweep_repeat),
    )

    payload = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "device": [str(device) for device in jax.devices()],
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "workload": {
            "model": cfg.model.name,
            "batch_size": int(args.batch_size),
            "train_samples": int(workload["n_train"]),
            "val_samples": int(workload["n_val"]),
            "num_val_batches": int(len(device_val_batches)),
            "wave_points": int(workload["wave"].shape[0]),
        },
        "settings": {
            "warmup": int(args.warmup),
            "repeat": int(args.repeat),
            "sweep_repeat": int(args.sweep_repeat),
        },
        "benchmarks": {
            "loader_train_batch": asdict(loader_stats),
            "to_jax_batch": asdict(to_jax_stats),
            "train_step": asdict(train_step_stats),
            "validation_predict_batch": asdict(eval_batch_stats),
            "validation_sweep": asdict(validation_sweep_stats),
        },
    }

    out_path = RUN_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote benchmark results to {out_path}")
    for name, stats in payload["benchmarks"].items():
        print(
            f"  {name}: median={stats['median_ms']:.3f} ms, "
            f"p90={stats['p90_ms']:.3f} ms, mean={stats['mean_ms']:.3f} ms"
        )
    if args.compare_to is not None:
        _compare(payload, args.compare_to)


if __name__ == "__main__":
    main()
