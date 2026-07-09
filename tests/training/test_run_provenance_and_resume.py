"""Tests for run provenance records and resume-safety guards."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from astro_emulators_toolkit.config import (
    IOSpec,
    OptimConfig,
    RootConfig,
    TrainConfig,
)
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.callbacks import CSVLogger
from astro_emulators_toolkit.training.paths import (
    EFFECTIVE_RUNS_FILENAME,
    RUN_CONFIG_FILENAME,
)
from astro_emulators_toolkit.training.state import TrainState
from astro_emulators_toolkit.training.trainer import _record_run_config


def test_record_run_config_preserves_earlier_segments(tmp_path):
    cfg_a = RootConfig(training=TrainConfig(workdir=str(tmp_path), num_steps=10))
    cfg_b = RootConfig(training=TrainConfig(workdir=str(tmp_path), num_steps=20))

    _record_run_config(cfg_a, tmp_path)
    original = (tmp_path / RUN_CONFIG_FILENAME).read_text()

    # Same config again: nothing new recorded.
    _record_run_config(cfg_a, tmp_path)
    assert not (tmp_path / "run_config.resume-1.json").exists()

    # Changed config: original stays intact, change recorded alongside it.
    _record_run_config(cfg_b, tmp_path)
    assert (tmp_path / RUN_CONFIG_FILENAME).read_text() == original
    resumed = tmp_path / "run_config.resume-1.json"
    assert resumed.exists()
    assert json.loads(resumed.read_text())["training"]["num_steps"] == 20

    # The same changed config again: deduplicated, no resume-2.
    _record_run_config(cfg_b, tmp_path)
    assert not (tmp_path / "run_config.resume-2.json").exists()


def _dummy_training_pieces(tmp_path, monkeypatch, *, num_steps=2):
    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle=True, seed=0):
            self.batch_size = int(batch_size)

        def train_batch(self, step):
            return {
                "x": np.zeros((self.batch_size, 2), dtype=np.float32),
                "y": np.zeros((self.batch_size, 1), dtype=np.float32),
            }

        def iter_eval_batches(self):
            return iter(())

    monkeypatch.setattr(trainer, "DataLoader", DummyLoader)

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32), (None, None)

        return _forward

    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=num_steps,
            steps_per_epoch=2,
            batch_size=2,
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

    return cfg, tx, init_state, DummyTask()


def test_resume_without_checkpoint_warns_before_training_from_scratch(
    tmp_path, monkeypatch
):
    cfg, tx, init_state, task = _dummy_training_pieces(tmp_path, monkeypatch)
    monkeypatch.setattr(trainer.ckpt, "restore_latest", lambda _mngr, target: None)

    with pytest.warns(UserWarning, match="no checkpoint was found"):
        trainer.fit(
            cfg=cfg,
            graphdef=None,
            init_state=init_state,
            task=task,
            tx=tx,
            train_dataset=[0, 1],
            callbacks=[],
            resume=True,
        )


def test_resume_warns_when_optimizer_fingerprint_differs(tmp_path, monkeypatch):
    cfg, tx, init_state, task = _dummy_training_pieces(tmp_path, monkeypatch)
    monkeypatch.setattr(
        trainer.ckpt,
        "restore_latest",
        lambda _mngr, target: target.replace(
            step=jnp.array(1, dtype=target.step.dtype)
        ),
    )
    monkeypatch.setattr(
        trainer.ckpt,
        "latest_custom_metadata",
        lambda _mngr: {
            "step": 1,
            "optimizer_fingerprint": {
                "optim": {"name": "adamw", "lr": 999.0},
                "num_steps": 999,
                "external_optimizer": False,
            },
        },
    )

    with pytest.warns(UserWarning, match="optimizer/schedule configuration"):
        trainer.fit(
            cfg=cfg,
            graphdef=None,
            init_state=init_state,
            task=task,
            tx=tx,
            train_dataset=[0, 1],
            callbacks=[],
            resume=True,
        )


def test_csv_logger_appends_instead_of_truncating_on_resume(tmp_path):
    path = tmp_path / "history.csv"
    logger = CSVLogger(path, split="train")

    logger.on_train_begin({"resume": False, "start_step": 0})
    logger.on_train_batch_end(1, {"loss": 1.0})
    logger.on_train_batch_end(2, {"loss": 0.5})
    logger.on_train_end()

    resumed = CSVLogger(path, split="train")
    resumed.on_train_begin({"resume": True, "start_step": 2})
    resumed.on_train_batch_end(3, {"loss": 0.25})
    resumed.on_train_end()

    lines = path.read_text().strip().splitlines()
    assert lines[0] == "step,loss"
    assert len(lines) == 4  # one header + three data rows, nothing truncated
    assert lines[1].startswith("1,")
    assert lines[3].startswith("3,")


def test_csv_logger_fresh_run_still_truncates(tmp_path):
    path = tmp_path / "history.csv"
    logger = CSVLogger(path, split="train")
    logger.on_train_begin({"resume": False, "start_step": 0})
    logger.on_train_batch_end(1, {"loss": 1.0})
    logger.on_train_end()

    fresh = CSVLogger(path, split="train")
    fresh.on_train_begin({"resume": False, "start_step": 0})
    fresh.on_train_batch_end(1, {"loss": 2.0})
    fresh.on_train_end()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_emulator_fit_appends_effective_run_record(monkeypatch, tmp_path):
    def _fake_fit_backend(**kwargs):
        class _Result:
            history = {}
            params = kwargs["init_state"].params
            model_state = kwargs["init_state"].model_state
            method = "gradient"

        return _Result()

    class _Solver:
        name = "gradient"

        @staticmethod
        def run(**kwargs):
            return _fake_fit_backend(**kwargs)

    monkeypatch.setattr(
        "astro_emulators_toolkit.emulator.resolve_solver",
        lambda *_args, **_kwargs: _Solver(),
    )

    workdir = tmp_path / "run"
    cfg = RootConfig(
        io=IOSpec(),
        optim=OptimConfig(name="adamw", lr=3e-4),
        training=TrainConfig(workdir=str(workdir), num_steps=4),
    )
    emu = Emulator.from_config(cfg).configure_training()

    class _Dataset:
        def __len__(self):
            return 4

        def get_batch(self, idx):
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": {"parameters": x}, "y": {"predictions": y}}

    emu.fit(_Dataset(), callbacks=[])
    emu.fit(_Dataset(), callbacks=[])

    records = [
        json.loads(line)
        for line in (workdir / EFFECTIVE_RUNS_FILENAME).read_text().splitlines()
    ]
    assert len(records) == 2
    first = records[0]
    assert first["resolved_solver"] == "gradient"
    assert first["optimizer_source"] == "config"
    assert first["optim"]["lr"] == pytest.approx(3e-4)
    assert first["num_steps"] == 4
    assert first["toolkit_version"]


def test_emulator_fit_records_external_optimizer_override(monkeypatch, tmp_path):
    def _fake_fit_backend(**kwargs):
        assert kwargs["external_optimizer"] is True

        class _Result:
            history = {}
            params = kwargs["init_state"].params
            model_state = kwargs["init_state"].model_state
            method = "gradient"

        return _Result()

    class _Solver:
        name = "gradient"

        @staticmethod
        def run(**kwargs):
            return _fake_fit_backend(**kwargs)

    monkeypatch.setattr(
        "astro_emulators_toolkit.emulator.resolve_solver",
        lambda *_args, **_kwargs: _Solver(),
    )

    workdir = tmp_path / "run"
    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(workdir=str(workdir), num_steps=4),
    )
    emu = Emulator.from_config(cfg).configure_training(optimizer=optax.sgd(1e-2))

    class _Dataset:
        def __len__(self):
            return 4

        def get_batch(self, idx):
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": {"parameters": x}, "y": {"predictions": y}}

    emu.fit(_Dataset(), callbacks=[])

    record = json.loads(
        (workdir / EFFECTIVE_RUNS_FILENAME).read_text().splitlines()[0]
    )
    assert record["optimizer_source"] == "external_override"


def test_resolved_bundle_config_includes_optim():
    from astro_emulators_toolkit.bundle.metadata import build_resolved_bundle_config

    cfg = RootConfig(optim=OptimConfig(name="adamw", lr=2e-3, weight_decay=1e-5))

    resolved = build_resolved_bundle_config(cfg)

    assert resolved["optim"]["name"] == "adamw"
    assert resolved["optim"]["lr"] == pytest.approx(2e-3)
    assert resolved["optim"]["weight_decay"] == pytest.approx(1e-5)
