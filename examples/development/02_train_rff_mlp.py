from pathlib import Path

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    NpyTableConfig,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import NpyTableDataset
from astro_emulators_toolkit.training import (
    CSVLogger,
    ModelCheckpoint,
    ProgressBarLogger,
)

script_dir = Path(__file__).parent.resolve()

data_cfg = NpyTableConfig(
    path=str(script_dir / "data/rff.npy"),
    inputs=(0, 1, 2),
    targets=(3, 4),
    memmap=True,
    dtype="float32",
)

cfg = RootConfig(
    seed=0,
    model=ModelSpec(
        name="mlp",
        params={"hidden_sizes": (32, 32), "activation": "gelu", "dtype": "float32"},
    ),
    task=TaskSpec(name="regression", params={"loss": "mse", "metrics": ["mse", "mae"]}),
    optim=OptimConfig(name="adamw", lr=3e-4, weight_decay=1e-4),
    training=TrainConfig(
        workdir=str(script_dir / "runs/rff_mlp"),
        batch_size=256,
        num_steps=2000,
        val_fraction=0.1,
        logging_interval_steps=25,
        evaluation_interval_steps=200,
        checkpoint_interval_steps=200,
        max_saved_checkpoints=3,
    ),
)

ds = NpyTableDataset.from_config(data_cfg)
train_ds, val_ds = ds.train_val_split(cfg.training.val_fraction, seed=cfg.seed)

emu = Emulator.from_config(cfg).configure_training()

history = emu.fit(
    train_ds,
    validation_dataset=val_ds,
    callbacks=[
        ProgressBarLogger(total_steps=cfg.training.num_steps),
        CSVLogger(Path(cfg.training.workdir) / "history_train.csv", split="train"),
        CSVLogger(Path(cfg.training.workdir) / "history_val.csv", split="val"),
        ModelCheckpoint(
            every_n_steps=cfg.training.checkpoint_interval_steps,
            explicit_steps=cfg.training.checkpoint_steps,
        ),
    ],
    resume=False,
    max_steps=275,
)

print("First run step count (capped):", len(history.logs.get("training_loss", [])))
print("Saved bundle to:", emu.save_bundle())
