from __future__ import annotations

import numpy as np
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
from astro_emulators_toolkit.data.toy import ToyBinaryClassificationDataset
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.resolver import build_task_from_name


def test_binary_classification_task_outputs_bce_and_accuracy():
    task = build_task_from_name(
        "experimental/binary_classification", {"decision_threshold": 0.5}
    )
    pred = np.array([[0.0], [4.0], [-4.0]], dtype=np.float32)
    batch = {"y": np.array([[0.0], [1.0], [0.0]], dtype=np.float32)}

    loss, metrics = task.loss_and_metrics(pred=pred, batch=batch)

    assert float(loss) > 0.0
    assert "bce" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0


def test_emulator_classification_predict_end_to_end(tmp_path):
    ds = ToyBinaryClassificationDataset(
        n_samples=128, x_dim=4, n_features=8, amplitude=1.5, seed=3
    )

    train_ds = TreeArrayDataset(
        x={"parameters": ds.x[:96]},
        y={"predictions": ds.y[:96]},
    )
    val_ds = TreeArrayDataset(
        x={"parameters": ds.x[96:128]},
        y={"predictions": ds.y[96:128]},
    )

    cfg = RootConfig(
        seed=0,
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (32, 32), "activation": "gelu", "dtype": "float32"},
        ),
        task=TaskSpec(
            name="experimental/binary_classification",
            params={"decision_threshold": 0.5},
        ),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            batch_size=32,
            num_steps=6,
            steps_per_epoch=3,
            shuffle=True,
            shuffle_seed=0,
            evaluation_interval_steps=2,
            logging_interval_steps=1,
            checkpoint_interval_steps=1000,
        ),
        io=IOSpec(),
    )

    emu = Emulator.from_config(cfg)
    emu.configure_training()
    emu.fit(train_ds, validation_dataset=val_ds, callbacks=[])

    probs = emu.predict({"parameters": ds.x[:16]})
    assert tuple(probs.keys()) == ("predictions",)
    assert probs["predictions"].shape == (16, 1)
    assert np.all(probs["predictions"] >= 0.0)
    assert np.all(probs["predictions"] <= 1.0)

    preds = emu.predict({"parameters": ds.x[:16]})
    np.testing.assert_allclose(preds["predictions"], probs["predictions"], atol=1e-6)


def test_classification_column_vector_sample_weight_matches_vector_weight():
    task = build_task_from_name(
        "experimental/binary_classification", {"decision_threshold": 0.5}
    )
    pred = np.array([[0.0], [4.0], [-4.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)

    w_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w_col = w_vec[:, None]

    loss_vec, metrics_vec = task.loss_and_metrics(
        pred=pred, batch={"y": y, "sample_weight": w_vec}
    )
    loss_col, metrics_col = task.loss_and_metrics(
        pred=pred, batch={"y": y, "sample_weight": w_col}
    )

    np.testing.assert_allclose(float(loss_col), float(loss_vec), rtol=1e-6)
    np.testing.assert_allclose(
        float(metrics_col["accuracy"]), float(metrics_vec["accuracy"]), rtol=1e-6
    )


def test_classification_rejects_invalid_sample_weight_shape():
    task = build_task_from_name(
        "experimental/binary_classification", {"decision_threshold": 0.5}
    )
    pred = np.array([[0.0], [4.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="sample_weight must have shape"):
        task.loss_and_metrics(
            pred=pred,
            batch={"y": y, "sample_weight": np.ones((2, 2), dtype=np.float32)},
        )
