"""Train an experimental wavelength-explicit family with the TransformerPayne flux workflow.

Data: shipped irregular_flux Korg/MARCS spectra, deterministic 90/10 split.
Creates: examples/runs/experimental_<model>_flux/{config.json,bundle} by default.
Runtime: seconds for the default smoke profile; minutes for cpu_recommended.
Optional extras: none. Requires JAX_ENABLE_X64=1 (set below before importing JAX).
Inputs are explicitly min/max normalized; continuum-normalized flux is unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

from astro_emulators_toolkit import Emulator, normalize_tree
from astro_emulators_toolkit.config import IOTreeSpec, MinMaxTreeSpec, save_config
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import transformer_payne_flux
from astro_emulators_toolkit.training import build_callbacks_from_config

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import split_randomized_flux_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("mhe_mlp", "mhe_transformer_payne", "interpolated_payne"),
        default="mhe_mlp",
    )
    parser.add_argument(
        "--profile", choices=("smoke", "cpu_recommended"), default="smoke"
    )
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    workdir = args.workdir or (
        Path(__file__).resolve().parents[1] / "runs" / f"experimental_{args.model}_flux"
    )
    smoke = args.profile == "smoke"
    x_train, y_train, x_val, y_val, wave = split_randomized_flux_arrays(seed=2)
    lo = {"parameters": [4500.0, 2.5, -0.3], "wavelengths": float(wave[0])}
    hi = {"parameters": [7000.0, 5.0, 0.3], "wavelengths": float(wave[-1])}

    cfg = transformer_payne_flux(profile=args.profile, workdir=str(workdir))
    # Keep the preset's task, optimizer, training, and runtime I/O contract.
    if args.model == "interpolated_payne":
        params = {"grid_segments": [[0.0, 1.0, len(wave)]]}
    else:
        params = (
            {"hidden_sizes": [16 if smoke else 128] * 3}
            if args.model == "mhe_mlp"
            else dict(cfg.model.params)
        )
        params.update(
            min_period=1.0 / (32 if smoke else 10000),
            max_period=2.0,
            mhe={
                "levels": 4 if smoke else 16,
                "log2_hashmap_size": 6 if smoke else 15,
                "wavelength_range": [0.0, 1.0],
            },
        )
    cfg = replace(
        cfg,
        model=replace(cfg.model, name=f"experimental/{args.model}", params=params),
        io=replace(
            cfg.io,
            inputs=IOTreeSpec(
                structure_tree={"parameters": None, "wavelengths": None},
                channel_names_tree={
                    "parameters": ["teff", "logg", "feh"],
                    "wavelengths": None,
                },
                channel_units_tree={
                    "parameters": ["K", "dex", "dex"],
                    "wavelengths": None,
                },
                leaf_units_tree={"parameters": None, "wavelengths": "Angstrom"},
                leaf_meanings_tree={
                    "parameters": "Physical labels, explicitly min/max normalized before use",
                    "wavelengths": "Wavelength, explicitly min/max normalized before use",
                },
            ),
            outputs=IOTreeSpec(
                structure_tree={"flux": None},
                leaf_units_tree={"flux": "dimensionless"},
                leaf_meanings_tree={"flux": "Continuum-normalized flux"},
            ),
            reference_scaling_inputs=MinMaxTreeSpec(min_tree=lo, max_tree=hi),
            reference_scaling_outputs=MinMaxTreeSpec(
                min_tree={"flux": 0.0}, max_tree={"flux": 1.0}
            ),
            input_domain=MinMaxTreeSpec(min_tree=lo, max_tree=hi),
        ),
    )
    train_inputs = normalize_tree({"parameters": x_train, "wavelengths": wave}, lo, hi)
    val_inputs = normalize_tree({"parameters": x_val, "wavelengths": wave}, lo, hi)
    wave_scaled = np.asarray(train_inputs["wavelengths"], dtype=np.float64)
    train_labels = np.asarray(train_inputs["parameters"], dtype=np.float32)
    val_labels = np.asarray(val_inputs["parameters"], dtype=np.float32)
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, workdir / "config.json")

    emu = Emulator.from_config(cfg)
    query_count = 32 if smoke else 128
    transform = emu.make_device_batch_transform(
        wavelength_grid=wave_scaled,
        n_wavelength=query_count,
        eval_wavelength_grid=np.linspace(0.0, 1.0, query_count, dtype=np.float64),
    )
    history = emu.fit(
        TreeArrayDataset(x={"parameters": train_labels}, y={"flux": y_train}),
        validation_dataset=TreeArrayDataset(
            x={"parameters": val_labels}, y={"flux": y_val}
        ),
        callbacks=build_callbacks_from_config(cfg),
        device_batch_transform=transform,
    )
    inputs = {
        "parameters": val_labels[:2],
        "wavelengths": np.broadcast_to(wave_scaled, (2, len(wave_scaled))),
    }
    pred = emu.predict(inputs)["flux"]
    bundle = emu.save_bundle(extras={"wavelength_angstrom": wave})
    # This checks portability, not held-out scientific accuracy.
    np.testing.assert_array_equal(
        Emulator.from_bundle(bundle).predict(inputs)["flux"], pred
    )
    print("Model:", cfg.model.name)
    print("Steps:", len(history.logs["training_loss"]))
    print("Prediction shape:", pred.shape)
    print("Saved bundle:", bundle)


if __name__ == "__main__":
    main()
