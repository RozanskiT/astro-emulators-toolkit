"""Create and inspect portability-oriented bundle metadata.

Data: examples/examples_datasets/irregular_flux (for realistic domain metadata).
Creates: examples/runs/portability_bundle.
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOTreeSpec, IOSpec, ModelSpec, RootConfig

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import load_randomized_flux


def main() -> None:
    x, _, _ = load_randomized_flux()
    parameters = x["parameters"]
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"parameters": None},
                channel_names_tree={"parameters": ["teff", "logg", "feh"]},
                leaf_meanings_tree={"parameters": "stellar labels"},
                channel_meanings_tree={
                    "parameters": [
                        "effective temperature",
                        "surface gravity",
                        "metallicity",
                    ]
                },
                channel_units_tree={"parameters": ["K", "dex", "dex"]},
            ),
            outputs=IOTreeSpec(
                structure_tree={"teff_proxy": None},
                leaf_meanings_tree={"teff_proxy": "example portability output"},
                leaf_units_tree={"teff_proxy": "K"},
            ),
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 3, "output_last_axis": 1},
        ),
    )
    emu = Emulator.from_config(
        cfg,
        init_example={
            "inputs": {"parameters": parameters[:1].astype(np.float32)},
            "outputs": {
                "teff_proxy": np.asarray([[parameters[:1, 0].mean()]], dtype=np.float32)
            },
        },
    )

    spec = {
        "input_domain": {
            "kind": "box_v1",
            "value_space": "physical_input_dict_tree_v1",
            "min_tree": {
                "parameters": [
                    float(parameters[:, 0].min()),
                    float(parameters[:, 1].min()),
                    float(parameters[:, 2].min()),
                ]
            },
            "max_tree": {
                "parameters": [
                    float(parameters[:, 0].max()),
                    float(parameters[:, 1].max()),
                    float(parameters[:, 2].max()),
                ]
            },
        },
        "reference_scaling_inputs": {
            "kind": "affine_minmax_v1",
            "applies_to": "inputs",
            "source_space": "physical_input_dict_tree_v1",
            "target_space": "canonical_input_dict_tree_v1",
            "min_tree": {
                "parameters": [
                    float(parameters[:, 0].min()),
                    float(parameters[:, 1].min()),
                    float(parameters[:, 2].min()),
                ]
            },
            "max_tree": {
                "parameters": [
                    float(parameters[:, 0].max()),
                    float(parameters[:, 1].max()),
                    float(parameters[:, 2].max()),
                ]
            },
        },
        "reference_scaling_outputs": {
            "kind": "affine_minmax_v1",
            "applies_to": "outputs",
            "source_space": "canonical_output_dict_tree_v1",
            "target_space": "physical_output_dict_tree_v1",
            "min_tree": {"teff_proxy": [3200.0]},
            "max_tree": {"teff_proxy": [7000.0]},
        },
    }

    out_dir = Path("examples/runs/portability_bundle")
    emu.save_bundle(
        out_dir,
        spec=spec,
        extras={
            "companion_recipe": {
                "kind": "hf_repo_file_v1",
                "repo_id": "org/model-artifact",
                "revision": "v0.1.0",
                "path": "predict_physical.py",
                "role": "physical_inference_wrapper",
            }
        },
    )
    loaded = Emulator.from_bundle(out_dir)
    print("Bundle path:", out_dir)
    print("Companion recipe:", loaded.bundle_extras.get("companion_recipe"))
    print(
        "Input reference scaling kind:",
        (loaded.reference_scaling_inputs or {}).get("kind"),
    )
    print(
        "Output reference scaling kind:",
        (loaded.reference_scaling_outputs or {}).get("kind"),
    )


if __name__ == "__main__":
    main()
