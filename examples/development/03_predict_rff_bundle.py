# examples/02_plot_predictions.py
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator

script_dir = Path(__file__).parent.resolve()

# 2) Use the bundle from above to run predictions.
bundle_dir = script_dir / "runs/rff_mlp/bundle"
emu = Emulator.from_bundle(bundle_dir)

# use generated RFF samples
data = np.load(script_dir / "data/rff.npy")
x = data[:256, :3].astype("float32")
y_true = data[:256, 3:].astype("float32")
y_pred = emu.predict({"parameters": x})["predictions"]

print("Predictions shape:", y_pred.shape)
print("First prediction:", y_pred[0].tolist())
print("First true value:", y_true[0].tolist())
