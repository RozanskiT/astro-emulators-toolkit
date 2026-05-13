"""Train one stable Payne-style flux MLP experiment from a config file.

What this shows:
- treat a JSON/YAML `RootConfig` as the unit of one training experiment;
- run a single config in an isolated `training.workdir`;
- write a small machine-readable `tuning_result.json` summary for a launcher
  or scheduler to collect.

Data: irregular_flux split into train/validation from the same randomized distribution.
Creates: config.training.workdir/{bundle,tuning_result.json}.
Runtime: ~10-20s on CPU for smoke-sized configs.
Used by: 07_grid_search_payne_flux_mlp_lr.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import load_config
from astro_emulators_toolkit.data import TreeArrayDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux

RESULT_FILENAME = "tuning_result.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a flux MLP from a JSON or YAML RootConfig file."
    )
    parser.add_argument("config", type=Path, help="Path to config.json/config.yaml.")
    parser.add_argument(
        "--label",
        default=None,
        help="Optional run label written into the tuning summary.",
    )
    return parser.parse_args()


def _last_metric(logs: dict[str, list[Any]], name: str) -> float | None:
    values = logs.get(name, ())
    if not values:
        return None
    return float(values[-1])


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    x_train, y_train, x_val, y_val, _ = split_randomized_flux(
        val_fraction=float(cfg.training.val_fraction),
        seed=int(cfg.seed),
    )

    emu = Emulator.from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=x_train, y=y_train),
        validation_dataset=TreeArrayDataset(x=x_val, y=y_val),
    )
    bundle_dir = emu.save_bundle()

    workdir = Path(cfg.training.workdir)
    result = {
        "label": args.label,
        "config_path": str(args.config.resolve()),
        "workdir": str(workdir.resolve()),
        "learning_rate": float(cfg.optim.lr),
        "num_steps": int(cfg.training.num_steps),
        "logged_training_steps": len(history.logs.get("training_loss", [])),
        "final_training_loss": _last_metric(history.logs, "training_loss"),
        "final_validation_loss": _last_metric(history.logs, "validation_loss"),
        "bundle_dir": str(Path(bundle_dir).resolve()),
    }
    (workdir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True))

    print("Config:", args.config)
    if args.label is not None:
        print("Run label:", args.label)
    print(f"Learning rate: {float(cfg.optim.lr):.1e}")
    print("Training steps logged:", result["logged_training_steps"])
    print("Final training loss:", result["final_training_loss"])
    print("Final validation loss:", result["final_validation_loss"])
    print("Bundle:", bundle_dir)
    print("Summary:", workdir / RESULT_FILENAME)


if __name__ == "__main__":
    main()
