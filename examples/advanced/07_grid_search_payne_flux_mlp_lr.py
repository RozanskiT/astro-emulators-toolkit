"""Prepare configs and launch a small learning-rate grid search.

What this shows:
- build one base preset config, then derive one config per hyperparameter value;
- keep each run in a separate workdir so bundles, logs, and summaries do not collide;
- launch the single-config worker script as independent subprocesses;
- collect each run's `tuning_result.json` into a sorted `grid_results.json`.

Grid: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2.
Creates: examples/runs/advanced_payne_flux_mlp_lr_grid/{base_config.yaml,lr_*/config.yaml,lr_*/tuning_result.json,grid_results.json}.
Runtime: roughly 1-2 minutes on CPU for the default smoke profile because it launches five independent training processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from astro_emulators_toolkit.config import save_config
from astro_emulators_toolkit.presets import payne_flux_mlp

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).with_name("06_train_payne_flux_mlp_from_config.py")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "examples" / "runs" / "advanced_payne_flux_mlp_lr_grid"
)
BASE_CONFIG_FILENAME = "base_config.yaml"
RUN_CONFIG_FILENAME = "config.yaml"
RESULT_FILENAME = "tuning_result.json"
GRID_RESULTS_FILENAME = "grid_results.json"
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write YAML configs and launch an independent learning-rate scan."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where configs, run directories, and summaries are written.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "cpu_recommended"),
        default="smoke",
        help="Preset profile used to build the base config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shared random seed for all runs so only learning rate changes.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write configs; do not launch the child training processes.",
    )
    return parser.parse_args()


def _lr_label(lr: float) -> str:
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _sort_key(result: dict[str, object]) -> float:
    value = result.get("final_validation_loss")
    return float("inf") if value is None else float(value)


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = payne_flux_mlp(
        workdir=str(output_root / "template_run"),
        profile=args.profile,
    ).with_updates(seed=int(args.seed))
    base_cfg = base_cfg.with_updates(
        training=replace(
            base_cfg.training,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
        )
    )
    save_config(base_cfg, output_root / BASE_CONFIG_FILENAME)

    print("Base config:", output_root / BASE_CONFIG_FILENAME, flush=True)
    print(
        "Learning-rate grid:",
        ", ".join(_lr_label(lr) for lr in LEARNING_RATES),
        flush=True,
    )

    results: list[dict[str, object]] = []
    for lr in LEARNING_RATES:
        label = _lr_label(lr)
        run_dir = output_root / f"lr_{label}"
        run_cfg = base_cfg.with_updates(
            optim=replace(base_cfg.optim, lr=lr),
            training=replace(base_cfg.training, workdir=str(run_dir)),
        )
        config_path = run_dir / RUN_CONFIG_FILENAME
        save_config(run_cfg, config_path)
        print("Prepared:", config_path, flush=True)

        if args.prepare_only:
            continue

        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), str(config_path), "--label", label],
            cwd=str(REPO_ROOT),
            check=True,
        )

        result_path = run_dir / RESULT_FILENAME
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))

    if args.prepare_only:
        print(
            "Preparation complete. Launch the worker script on any generated config.yaml file to run a single experiment.",
            flush=True,
        )
        return

    results.sort(key=_sort_key)
    results_path = output_root / GRID_RESULTS_FILENAME
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    if results:
        best = results[0]
        print("Best run:", best.get("label"), flush=True)
        print("Best validation loss:", best.get("final_validation_loss"), flush=True)
    print("Grid summary:", results_path, flush=True)


if __name__ == "__main__":
    main()
