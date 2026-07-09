"""Tests for best-model checkpoint tracking (checkpoints_best/)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.training import (
    BestModelCheckpoint,
    build_callbacks_from_config,
    checkpoint as ckpt_module,
    trainer,
)
from astro_emulators_toolkit.training.state import TrainState


class _TinyDataset:
    def __init__(self, n: int = 6):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(n, 2)).astype(np.float32)
        self.y = rng.normal(size=(n, 1)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def get_batch(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class _ScheduledEvalTask:
    """Task whose eval metric follows a fixed schedule (streaming protocol)."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def loss_and_metrics(self, pred, batch):
        return jnp.mean(jnp.square(pred)), {}

    def init_eval_state(self):
        return None

    def update_eval_state(self, eval_state, pred, batch):
        return eval_state

    def finalize_eval(self, eval_state):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {"loss": float(value)}


def _fake_call(state_tuple):
    _graphdef, full_state = state_tuple

    def _forward(x, train=False, rngs=None):
        return x @ full_state["w"], (None, None)

    return _forward


def _run_fit(tmp_path, monkeypatch, *, callbacks, task, num_steps: int = 3):
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)
    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=num_steps,
            steps_per_epoch=4,
            batch_size=2,
            logging_interval_steps=None,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=None,
        ),
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
    dataset = _TinyDataset()
    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=task,
        tx=tx,
        train_dataset=dataset,
        val_dataset=dataset,
        callbacks=callbacks,
        resume=False,
    )
    return cfg


def test_best_checkpoint_keeps_the_best_eval_step_not_the_last(
    tmp_path, monkeypatch
):
    task = _ScheduledEvalTask([1.0, 0.5, 0.8])
    cfg = _run_fit(
        tmp_path,
        monkeypatch,
        callbacks=[BestModelCheckpoint(monitor="loss", mode="min")],
        task=task,
    )

    mngr = ckpt_module.create_best_manager(cfg.training.workdir, cfg=cfg)
    try:
        assert mngr.latest_step() == 2  # the eval step where loss=0.5
        meta = ckpt_module.latest_custom_metadata(mngr)
    finally:
        mngr.close()
    assert meta is not None
    assert meta["monitor"] == "loss"
    assert meta["best_value"] == pytest.approx(0.5)


def test_best_checkpoint_max_mode_tracks_maxima(tmp_path, monkeypatch):
    task = _ScheduledEvalTask([0.1, 0.7, 0.3])
    cfg = _run_fit(
        tmp_path,
        monkeypatch,
        callbacks=[BestModelCheckpoint(monitor="loss", mode="max")],
        task=task,
    )

    mngr = ckpt_module.create_best_manager(cfg.training.workdir, cfg=cfg)
    try:
        assert mngr.latest_step() == 2
    finally:
        mngr.close()


def test_best_checkpoint_warns_once_for_unknown_monitor(tmp_path, monkeypatch):
    task = _ScheduledEvalTask([1.0, 0.5, 0.8])
    with pytest.warns(UserWarning, match="not among"):
        _run_fit(
            tmp_path,
            monkeypatch,
            callbacks=[BestModelCheckpoint(monitor="does_not_exist")],
            task=task,
        )


def test_multiple_best_checkpoint_callbacks_are_rejected(tmp_path, monkeypatch):
    task = _ScheduledEvalTask([1.0])
    with pytest.raises(ValueError, match="At most one BestModelCheckpoint"):
        _run_fit(
            tmp_path,
            monkeypatch,
            callbacks=[BestModelCheckpoint(), BestModelCheckpoint()],
            task=task,
        )


def test_build_callbacks_from_config_includes_best_checkpoint(tmp_path):
    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path),
            checkpoint_best_metric="loss",
            checkpoint_best_mode="min",
        )
    )
    callbacks = build_callbacks_from_config(cfg)
    best = [c for c in callbacks if isinstance(c, BestModelCheckpoint)]
    assert len(best) == 1
    assert best[0].monitor == "loss"
    assert best[0].mode == "min"


def test_emulator_fit_then_load_best_checkpoint_roundtrip(tmp_path):
    x = np.random.default_rng(1).normal(size=(16, 3)).astype(np.float32)
    y = (x @ np.array([[0.5], [0.2], [-0.1]], dtype=np.float32)).astype(np.float32)

    from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset

    ds = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="mlp", params={"hidden_sizes": (8,)}),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-2, schedule="constant"),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            batch_size=8,
            num_steps=3,
            logging_interval_steps=None,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=None,
            checkpoint_best_metric="loss",
        ),
    )
    emu = Emulator.from_config(cfg)
    emu.fit(ds, validation_dataset=ds)

    params_after_fit = jax.tree_util.tree_map(np.asarray, emu.params)
    emu.load_best_checkpoint()
    assert emu.params is not None
    # Best checkpoint restores a valid parameter tree with the same structure.
    assert jax.tree_util.tree_structure(
        emu.params
    ) == jax.tree_util.tree_structure(params_after_fit)


def test_load_best_checkpoint_errors_when_never_tracked(tmp_path):
    x = np.zeros((8, 2), dtype=np.float32)
    y = np.zeros((8, 1), dtype=np.float32)

    from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset

    ds = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="mlp", params={"hidden_sizes": (4,)}),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            batch_size=8,
            num_steps=1,
            logging_interval_steps=None,
            evaluation_interval_steps=None,
            checkpoint_interval_steps=None,
        ),
    )
    emu = Emulator.from_config(cfg)
    emu.fit(ds, callbacks=[])

    with pytest.raises(FileNotFoundError, match="No best checkpoint"):
        emu.load_best_checkpoint()

def test_best_model_checkpoint_accepts_validation_prefixed_monitor():
    # History keys are "validation_"-prefixed while eval metrics are bare;
    # both spellings must work.
    assert BestModelCheckpoint(monitor="validation_loss").monitor == "loss"
    assert BestModelCheckpoint(monitor="loss").monitor == "loss"
