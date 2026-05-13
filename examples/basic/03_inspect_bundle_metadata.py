"""Inspect a loaded bundle and its user-facing metadata.

Data: shipped reference bundle plus explicit input scaling from bundle metadata.
Creates: nothing.
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator, normalize_tree

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_bundle import require_reference_bundle


def main() -> None:
    bundle = require_reference_bundle()
    emu = Emulator.from_bundle(bundle)
    x_physical = np.array([[5600.0, 4.2, -0.1]], dtype=np.float32)
    ref = emu.reference_scaling_inputs or {}
    packed_scaled = normalize_tree(
        {"parameters": x_physical}, ref["min_tree"], ref["max_tree"]
    )
    pred = emu.predict(packed_scaled)
    output_keys = list(pred.keys())
    output_leaf = output_keys[0]
    input_channel_names = (emu.input_channel_names_tree or {}).get("parameters") or []
    output_channel_names = (emu.output_channel_names_tree or {}).get(output_leaf) or []

    print(emu.describe_bundle())
    print("Input channel names:", tuple(input_channel_names))
    print("Output keys:", tuple(output_keys))
    print("First output shape:", pred[output_leaf].shape)
    if output_channel_names:
        print("Output channel names:", tuple(output_channel_names))
    print("Input scaling present:", emu.reference_scaling_inputs is not None)
    print("Output scaling present:", emu.reference_scaling_outputs is not None)
    print("Bundle extras keys:", tuple(sorted(emu.bundle_extras)))
    print("Domain summary:", emu.describe_domain())


if __name__ == "__main__":
    main()
