from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit.config import RootConfig, TrainConfig
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState


def test_model_checkpoint_uses_current_state_step_with_deferred_logging(
    tmp_path, monkeypatch
):
    class TinyDataset:
        def __len__(self):
            return 4

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": x, "y": y}

    def _fake_merge_state(params, model_state):
        return {"params": params, "model_state": model_state}

    def _fake_split_state(full_state, *_filters):
        return full_state["params"], full_state["model_state"]

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            pred = jnp.zeros((x.shape[0], 1), dtype=jnp.float32)
            return pred, (
                None,
                {
                    "params": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
                    "model_state": {},
                },
            )

        return _forward

    monkeypatch.setattr(trainer.nnx, "merge_state", _fake_merge_state)
    monkeypatch.setattr(trainer.nnx, "split_state", _fake_split_state)
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    calls: list[tuple[int, int]] = []

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )

    def _save(_mngr, step, train_state, *, custom_metadata=None):
        calls.append((int(step), int(jax.device_get(train_state.step))))
        return str(tmp_path / f"ckpt-{int(step)}")

    monkeypatch.setattr(trainer.ckpt, "save", _save)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=2,
            steps_per_epoch=2,
            batch_size=2,
            shuffle=False,
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
            return jnp.array(0.0), {"mae": jnp.array(0.0)}

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=TinyDataset(),
        callbacks=[trainer.ModelCheckpoint(every_n_steps=1, save_on_train_end=False)],
        resume=False,
    )

    assert calls == [(1, 1), (2, 2)]


def test_model_checkpoint_saves_final_state_on_train_end(tmp_path, monkeypatch):
    class TinyDataset:
        def __len__(self):
            return 4

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": x, "y": y}

    def _fake_merge_state(params, model_state):
        return {"params": params, "model_state": model_state}

    def _fake_split_state(full_state, *_filters):
        return full_state["params"], full_state["model_state"]

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            pred = jnp.zeros((x.shape[0], 1), dtype=jnp.float32)
            return pred, (
                None,
                {
                    "params": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
                    "model_state": {},
                },
            )

        return _forward

    monkeypatch.setattr(trainer.nnx, "merge_state", _fake_merge_state)
    monkeypatch.setattr(trainer.nnx, "split_state", _fake_split_state)
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    calls: list[tuple[int, int]] = []

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )

    def _save(_mngr, step, train_state, *, custom_metadata=None):
        calls.append((int(step), int(jax.device_get(train_state.step))))
        return str(tmp_path / f"ckpt-{int(step)}")

    monkeypatch.setattr(trainer.ckpt, "save", _save)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=2,
            steps_per_epoch=2,
            batch_size=2,
            shuffle=False,
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
            return jnp.array(0.0), {"mae": jnp.array(0.0)}

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=TinyDataset(),
        callbacks=[trainer.ModelCheckpoint(every_n_steps=100, save_on_train_end=True)],
        resume=False,
    )

    assert calls == [(2, 2)]


def test_model_checkpoint_supports_explicit_steps_without_periodic_interval(
    tmp_path, monkeypatch
):
    class TinyDataset:
        def __len__(self):
            return 8

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": x, "y": y}

    def _fake_merge_state(params, model_state):
        return {"params": params, "model_state": model_state}

    def _fake_split_state(full_state, *_filters):
        return full_state["params"], full_state["model_state"]

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            pred = jnp.zeros((x.shape[0], 1), dtype=jnp.float32)
            return pred, (
                None,
                {
                    "params": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
                    "model_state": {},
                },
            )

        return _forward

    monkeypatch.setattr(trainer.nnx, "merge_state", _fake_merge_state)
    monkeypatch.setattr(trainer.nnx, "split_state", _fake_split_state)
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    calls: list[tuple[int, int]] = []

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )

    def _save(_mngr, step, train_state, *, custom_metadata=None):
        calls.append((int(step), int(jax.device_get(train_state.step))))
        return str(tmp_path / f"ckpt-{int(step)}")

    monkeypatch.setattr(trainer.ckpt, "save", _save)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            steps_per_epoch=4,
            batch_size=2,
            shuffle=False,
            checkpoint_interval_steps=None,
            checkpoint_steps=(2, 4),
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
            return jnp.array(0.0), {"mae": jnp.array(0.0)}

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=TinyDataset(),
        callbacks=[
            trainer.ModelCheckpoint(
                every_n_steps=None,
                explicit_steps=(2, 4),
                save_on_train_end=False,
            )
        ],
        resume=False,
    )

    assert calls == [(2, 2), (4, 4)]
