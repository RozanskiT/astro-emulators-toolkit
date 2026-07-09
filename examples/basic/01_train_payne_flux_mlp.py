"""Train a stable MLP flux emulator and predict full-spectrum flux.

Data: irregular_flux split into train/validation from the same randomized distribution.
Creates: examples/runs/basic_payne_flux_mlp/bundle
Runtime: ~10-20s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from astro_emulators_toolkit import Emulator, normalize_tree
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import payne_flux_mlp
from astro_emulators_toolkit.training import build_callbacks_from_config

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_smoke import smoke_value
from _example_data import split_randomized_flux


def main() -> None:
    workdir = smoke_value(
        "examples/runs/basic_payne_flux_mlp",
        smoke="examples/runs/basic_payne_flux_mlp_smoke",
    )
    x_train, y_train, x_val, y_val, wave = split_randomized_flux(
        val_fraction=0.1, seed=0
    )
    cfg = payne_flux_mlp(
        workdir=workdir,
        profile="smoke",
    )

    # Train on min-max scaled labels and record the scaling in the bundle, so
    # consumers can reproduce the training-space inputs (see example 02).
    min_tree = {"parameters": x_train["parameters"].min(axis=0)}
    max_tree = {"parameters": x_train["parameters"].max(axis=0)}

    emu = Emulator.from_config(cfg)
    callbacks = build_callbacks_from_config(cfg)
    history = emu.fit(
        TreeArrayDataset(x=normalize_tree(x_train, min_tree, max_tree), y=y_train),
        validation_dataset=TreeArrayDataset(
            x=normalize_tree(x_val, min_tree, max_tree), y=y_val
        ),
        callbacks=callbacks,
    )

    x_pred = normalize_tree(
        {"parameters": x_val["parameters"][:2]}, min_tree, max_tree
    )
    pred = emu.predict(x_pred)["flux"]
    bundle = emu.save_bundle(
        spec={
            "reference_scaling_inputs": {
                "kind": "affine_minmax_v1",
                "applies_to": "inputs",
                "source_space": "physical_input_dict_tree_v1",
                "target_space": "canonical_input_dict_tree_v1",
                "min_tree": {"parameters": [float(v) for v in min_tree["parameters"]]},
                "max_tree": {"parameters": [float(v) for v in max_tree["parameters"]]},
            },
        }
    )
    print("Steps:", len(history.logs.get("training_loss", [])))
    print("Preset profile:", "smoke")
    print("Predicted full-spectrum shape:", pred.shape)
    print("Predicted wavelengths:", wave.shape[0])
    print("Bundle:", bundle)


if __name__ == "__main__":
    main()
