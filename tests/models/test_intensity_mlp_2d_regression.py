from __future__ import annotations

import pytest

from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.data.toy import ToyIntensityDataset
from astro_emulators_toolkit.emulator import Emulator


def test_mlp_2d_regression_predict_shape_with_intensity_implicit_wavelengths(tmp_path):
    ds = ToyIntensityDataset(n_samples=48, x_dim=6, y_dim=24, seed=2)
    train_ds = TreeArrayDataset(
        x={"parameters": ds.x[:36]},
        y={"predictions": ds.y[:36]},
    )
    val_ds = TreeArrayDataset(
        x={"parameters": ds.x[36:48]},
        y={"predictions": ds.y[36:48]},
    )

    cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="experimental/mlp_2d_regression",
            params={
                "hidden_sizes": (32, 32),
                "activation": "gelu",
                "dtype": "float32",
                "channels": 2,
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run_intensity_implicit"),
            batch_size=12,
            num_steps=8,
            steps_per_epoch=2,
            evaluation_interval_steps=4,
            logging_interval_steps=2,
            checkpoint_interval_steps=50,
        ),
        io=IOSpec(),
    )

    emu = Emulator.from_config(cfg)
    history = emu.fit(train_ds, validation_dataset=val_ds)
    pred = emu.predict({"parameters": ds.x[:5]})["predictions"]

    assert pred.shape == (5, 24, 2)
    with pytest.raises(ValueError, match="canonical dict-tree inputs"):
        emu.predict(ds.x[:5])
    assert history.logs["training_step"] == [2.0, 4.0, 6.0, 8.0]
