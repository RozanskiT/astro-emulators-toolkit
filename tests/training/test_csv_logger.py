from __future__ import annotations

import csv

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from astro_emulators_toolkit.config import RootConfig, TrainConfig
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.callbacks import CSVLogger, ProgressBarLogger
from astro_emulators_toolkit.training.state import TrainState


def _read_rows(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_logger_supports_optional_train_subsampling(tmp_path):
    train_path = tmp_path / "history_train.csv"
    val_path = tmp_path / "history_val.csv"

    train_logger = CSVLogger(train_path, every_n_steps=2, split="train")
    val_logger = CSVLogger(val_path, split="val")

    for logger in (train_logger, val_logger):
        logger.on_train_begin()

    train_logger.on_train_batch_end(1, {"loss": 2.0})
    train_logger.on_train_batch_end(2, {"loss": 1.0, "mse": 1.0})
    train_logger.on_eval_end(2, {"loss": 0.8})

    val_logger.on_train_batch_end(2, {"loss": 1.0})
    val_logger.on_eval_end(2, {"loss": 0.8, "mse": 0.8})

    for logger in (train_logger, val_logger):
        logger.on_train_end()

    train_rows = _read_rows(train_path)
    val_rows = _read_rows(val_path)

    assert len(train_rows) == 1
    assert train_rows[0]["step"] == "2"
    assert train_rows[0]["loss"] == "1.0"

    assert len(val_rows) == 1
    assert val_rows[0]["step"] == "2"
    assert val_rows[0]["validation_loss"] == "0.8"
    assert val_rows[0]["validation_mse"] == "0.8"


def test_csv_logger_split_both_writes_two_files(tmp_path):
    path = tmp_path / "history.csv"
    logger = CSVLogger(path, split="both")
    logger.on_train_begin()

    logger.on_train_batch_end(1, {"loss": 1.0, "mse": 1.0})
    logger.on_eval_end(1, {"loss": 0.9, "mse": 0.9})
    logger.on_train_end()

    train_rows = _read_rows(tmp_path / "history_train.csv")
    val_rows = _read_rows(tmp_path / "history_val.csv")
    assert len(train_rows) == 1
    assert train_rows[0]["loss"] == "1.0"
    assert len(val_rows) == 1
    assert val_rows[0]["validation_loss"] == "0.9"


def test_progress_bar_logger_defaults_to_every_emitted_event(capsys):
    logger = ProgressBarLogger(total_steps=4)
    logger.on_train_begin()
    logger.on_train_batch_end(2, {"loss": 1.0})
    logger.on_eval_end(3, {"loss": 0.8})
    logger.on_train_end()

    captured = capsys.readouterr().out
    assert "step 2/4  loss=1" in captured
    assert "val @ step 3: loss=0.8" in captured


def test_progress_bar_logger_can_optionally_subsample_events(capsys):
    logger = ProgressBarLogger(total_steps=4, every_n_steps=2)
    logger.on_train_begin()
    logger.on_train_batch_end(1, {"loss": 2.0})
    logger.on_train_batch_end(2, {"loss": 1.0})
    logger.on_eval_end(3, {"loss": 0.7})
    logger.on_eval_end(4, {"loss": 0.6})
    logger.on_train_end()

    captured = capsys.readouterr().out
    assert "step 1/4" not in captured
    assert "step 2/4  loss=1" in captured
    assert "val @ step 3:" not in captured
    assert "val @ step 4: loss=0.6" in captured


def test_loggers_reject_non_positive_intervals(tmp_path):
    from astro_emulators_toolkit.training.callbacks import ModelCheckpoint

    with pytest.raises(ValueError, match="> 0"):
        CSVLogger(tmp_path / "history.csv", every_n_steps=0)

    with pytest.raises(ValueError, match="> 0"):
        ProgressBarLogger(every_n_steps=0)

    with pytest.raises(ValueError, match="> 0"):
        ModelCheckpoint(every_n_steps=0)


def test_csv_logger_writes_only_trainer_emitted_scheduled_steps(monkeypatch, tmp_path):
    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32), (None, None)

        return _forward

    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    class DummyDataset:
        def __init__(self, n: int):
            self.n = n

        def __len__(self):
            return self.n

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((idx.shape[0], 2), dtype=np.float32)
            y = np.zeros((idx.shape[0], 1), dtype=np.float32)
            return {
                "x": x,
                "y": y,
                "valid_mask": np.ones((idx.shape[0],), dtype=np.float32),
            }

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            steps_per_epoch=4,
            batch_size=2,
            logging_interval_steps=4,
            logging_steps=(2, 4),
            evaluation_interval_steps=4,
            evaluation_steps=(3,),
            checkpoint_interval_steps=0,
        )
    )

    tx = optax.sgd(1e-3)
    params = {"w": jnp.ones((2, 1), dtype=jnp.float32)}
    init_state = TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=jax.random.key(0),
        params=params,
        model_state={},
        opt_state=tx.init(params),
    )

    class DummyTask:
        def loss_and_metrics(self, pred, batch):
            return jnp.array(0.0), {}

    logger = CSVLogger(tmp_path / "history.csv", split="both")
    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=DummyDataset(4),
        val_dataset=DummyDataset(2),
        callbacks=[logger],
        resume=False,
    )

    train_rows = _read_rows(tmp_path / "history_train.csv")
    val_rows = _read_rows(tmp_path / "history_val.csv")

    assert [row["step"] for row in train_rows] == ["2", "4"]
    assert [row["step"] for row in val_rows] == ["3", "4"]
