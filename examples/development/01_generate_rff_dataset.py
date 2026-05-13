# examples/00_generate_rff_datset.py
from pathlib import Path

import numpy as np

from astro_emulators_toolkit.data.toy_utils import generate_rff_dataset

script_dir = Path(__file__).parent.resolve()
out_path = script_dir / "./data/rff.npy"
out_path.parent.mkdir(parents=True, exist_ok=True)

x, y = generate_rff_dataset(
    n_samples=2000,
    x_dim=3,
    y_dim=2,
    n_features=3,
    freq_scale=2.0,
    noise_std=0.01,
    x_dist="uniform",
    seed=0,
)

data = np.concatenate([x, y], axis=1).astype("float32")
np.save(out_path, data)

print(f"Saved RFF dataset to: {out_path}")
print(f"Shape: {data.shape} (x_dim=3, y_dim=2)")
