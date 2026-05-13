"""Load a pretrained emulator bundle from Hugging Face for quick smoke testing.

Fallback guidance:
- If Hugging Face is unavailable in your environment, first run:
  1) `python examples/development/01_generate_rff_dataset.py`
  2) `python examples/development/02_train_rff_mlp.py`
- Then this script will load `examples/development/runs/rff_mlp/bundle` instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator

script_dir = Path(__file__).parent.resolve()


def main() -> None:
    repo_id = "RozanskiT/example_bundle"
    local_bundle = script_dir / "runs/rff_mlp/bundle"

    has_hf_hub = importlib.util.find_spec("huggingface_hub") is not None
    if not has_hf_hub:
        print("`huggingface_hub` is not installed in this environment.")
        print("To prepare a local fallback bundle, run:")
        print(f"  python {(script_dir / '01_generate_rff_dataset.py').as_posix()}")
        print(f"  python {(script_dir / '02_train_rff_mlp.py').as_posix()}")
        if not local_bundle.exists():
            raise SystemExit(f"Local fallback bundle not found at: {local_bundle}")

    if has_hf_hub:
        try:
            emu = Emulator.from_pretrained(
                repo_id,
                cache_dir=script_dir / ".emuspec_cache",
            )
            source = f"Hugging Face ({repo_id})"
        except Exception as exc:
            print(f"Hugging Face load failed ({exc}).")
            print("Using local fallback bundle.")
            if not local_bundle.exists():
                print("To create it, run:")
                print(
                    f"  python {(script_dir / '01_generate_rff_dataset.py').as_posix()}"
                )
                print(f"  python {(script_dir / '02_train_rff_mlp.py').as_posix()}")
                raise
            emu = Emulator.from_bundle(local_bundle)
            source = f"local fallback bundle ({local_bundle})"
    else:
        emu = Emulator.from_bundle(local_bundle)
        source = f"local fallback bundle ({local_bundle})"

    x = np.load(script_dir / "data/rff.npy")[:256, :3].astype("float32")
    y_pred = emu.predict({"parameters": x})["predictions"]

    print("Loaded emulator from:", source)
    print("Predictions shape:", y_pred.shape)
    print("First prediction:", y_pred[0].tolist())


if __name__ == "__main__":
    main()
