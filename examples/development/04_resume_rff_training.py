"""Resume internal RFF training state from development/02 checkpoints.

Data: examples/development/data/rff.npy produced by development/01.
Creates: examples/development/runs/rff_mlp/bundle_resumed.
Runtime: a few seconds on CPU after development/02 has run.
Notes: resume=True uses trainer checkpoint/run-management state from run_config.json
and checkpoints/, not the portable bundle sharing contract.
"""

from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import NpyTableConfig, load_config
from astro_emulators_toolkit.data import NpyTableDataset
from astro_emulators_toolkit.training import ProgressBarLogger

script_dir = Path(__file__).parent.resolve()

cfg = load_config(script_dir / "runs/rff_mlp/run_config.json")
data_cfg = NpyTableConfig(
    path=str(script_dir / "data/rff.npy"),
    inputs=(0, 1, 2),
    targets=(3, 4),
    memmap=True,
    dtype="float32",
)

ds = NpyTableDataset.from_config(data_cfg)
train_ds, val_ds = ds.train_val_split(cfg.training.val_fraction, seed=cfg.seed)

emu = Emulator.from_config(cfg).configure_training()
history = emu.fit(
    train_ds,
    validation_dataset=val_ds,
    callbacks=[ProgressBarLogger(total_steps=cfg.training.num_steps)],
    resume=True,
)

emu.save_bundle(script_dir / "runs/rff_mlp/bundle_resumed")
print("Resumed step count:", len(history.logs.get("training_loss", [])))
