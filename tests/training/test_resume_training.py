from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from astro_emulators_toolkit.config import RootConfig, TrainConfig
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState


def test_resume_uses_total_target_steps_and_epoch_seed_offset(tmp_path, monkeypatch):
    seen_steps: list[int] = []

    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle=True, seed=0):
            self.batch_size = int(batch_size)

        def train_batch(self, step):
            seen_steps.append(int(step))
            return {
                "x": np.zeros((self.batch_size, 2), dtype=np.float32),
                "y": np.zeros((self.batch_size, 1), dtype=np.float32),
                "valid_mask": np.ones((self.batch_size,), dtype=np.float32),
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

    restored_step = 4

    def _restore_latest(_mngr, target):
        return target.replace(step=jnp.array(restored_step, dtype=target.step.dtype))

    monkeypatch.setattr(trainer.ckpt, "restore_latest", _restore_latest)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=6,
            steps_per_epoch=2,
            batch_size=2,
            shuffle=True,
            shuffle_seed=17,
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

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        callbacks=[],
        resume=True,
    )

    assert int(jax.device_get(result.state.step)) == 6
    assert seen_steps == [4, 5]


def test_resume_noops_when_target_steps_already_reached(tmp_path, monkeypatch):
    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle=True, seed=0):
            pass

        def train_batch(self, step):
            raise AssertionError(
                "No training step should run when target steps are already reached"
            )

    monkeypatch.setattr(trainer, "DataLoader", DummyLoader)

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )

    restored_step = 6

    def _restore_latest(_mngr, target):
        return target.replace(step=jnp.array(restored_step, dtype=target.step.dtype))

    monkeypatch.setattr(trainer.ckpt, "restore_latest", _restore_latest)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=6,
            steps_per_epoch=2,
            batch_size=2,
            shuffle=True,
            shuffle_seed=17,
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

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        callbacks=[],
        resume=True,
    )

    assert int(jax.device_get(result.state.step)) == 6


def test_resume_max_steps_limits_continuation_without_changing_num_steps(
    tmp_path, monkeypatch
):
    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle=True, seed=0):
            self.batch_size = int(batch_size)

        def train_batch(self, step):
            return {
                "x": np.zeros((self.batch_size, 2), dtype=np.float32),
                "y": np.zeros((self.batch_size, 1), dtype=np.float32),
                "valid_mask": np.ones((self.batch_size,), dtype=np.float32),
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
    monkeypatch.setattr(
        trainer.ckpt,
        "restore_latest",
        lambda _mngr, target: target.replace(
            step=jnp.array(4, dtype=target.step.dtype)
        ),
    )

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=10,
            steps_per_epoch=2,
            batch_size=2,
            shuffle=True,
            shuffle_seed=17,
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

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        callbacks=[],
        resume=True,
        max_steps=2,
    )

    assert int(jax.device_get(result.state.step)) == 6


def test_resume_max_steps_validates_positive_value(tmp_path):
    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=10,
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

    with pytest.raises(ValueError, match="max_steps must be > 0"):
        trainer.fit(
            cfg=cfg,
            graphdef=None,
            init_state=init_state,
            task=DummyTask(),
            tx=tx,
            train_dataset=[0, 1, 2, 3],
            callbacks=[],
            resume=False,
            max_steps=0,
        )


def test_resume_matches_uninterrupted_indices_and_rng(tmp_path, monkeypatch):
    class RecordingDataset:
        def __init__(self, n):
            self.n = n
            self.requests: list[tuple[int, ...]] = []

        def __len__(self):
            return self.n

        def get_batch(self, idx):
            idx = np.asarray(idx)
            self.requests.append(tuple(int(i) for i in idx.tolist()))
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": x, "y": y}

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32), (None, None)

        return _forward

    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    saved_states: dict[int, TrainState] = {}

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
        saved_states[int(step)] = train_state
        return str(tmp_path / f"ckpt-{int(step)}")

    monkeypatch.setattr(trainer.ckpt, "save", _save)

    def _restore_latest(_mngr, target):
        if not saved_states:
            return None
        step = max(saved_states)
        return saved_states[step]

    monkeypatch.setattr(trainer.ckpt, "restore_latest", _restore_latest)

    total_steps = 6
    split_steps = 3

    def _cfg(num_steps: int):
        return RootConfig(
            training=TrainConfig(
                workdir=str(tmp_path / "run"),
                num_steps=num_steps,
                steps_per_epoch=4,
                batch_size=2,
                shuffle=True,
                shuffle_seed=13,
            )
        )

    tx = optax.sgd(1e-3)
    params = {"w": jnp.ones((2, 1), dtype=jnp.float32)}

    def _init_state():
        return TrainState(
            step=jnp.array(0, dtype=jnp.int32),
            rng_key=jax.random.key(7),
            params=params,
            model_state={},
            opt_state=tx.init(params),
        )

    class DummyTask:
        def loss_and_metrics(self, pred, batch):
            return jnp.array(0.0), {}

    full_dataset = RecordingDataset(8)
    full_result = trainer.fit(
        cfg=_cfg(total_steps),
        graphdef=None,
        init_state=_init_state(),
        task=DummyTask(),
        tx=tx,
        train_dataset=full_dataset,
        callbacks=[],
        resume=False,
    )

    initial_run_dataset = RecordingDataset(8)
    trainer.fit(
        cfg=_cfg(split_steps),
        graphdef=None,
        init_state=_init_state(),
        task=DummyTask(),
        tx=tx,
        train_dataset=initial_run_dataset,
        callbacks=[trainer.ModelCheckpoint(every_n_steps=1, save_on_train_end=False)],
        resume=False,
    )

    resumed_run_dataset = RecordingDataset(8)
    resumed_result = trainer.fit(
        cfg=_cfg(total_steps),
        graphdef=None,
        init_state=_init_state(),
        task=DummyTask(),
        tx=tx,
        train_dataset=resumed_run_dataset,
        callbacks=[],
        resume=True,
    )

    assert resumed_run_dataset.requests == full_dataset.requests[split_steps:]

    resumed_sequence = initial_run_dataset.requests + resumed_run_dataset.requests
    assert resumed_sequence == full_dataset.requests
    assert np.array_equal(
        jax.random.key_data(resumed_result.state.rng_key),
        jax.random.key_data(full_result.state.rng_key),
    )


def test_resume_with_explicit_schedules_only_emits_remaining_absolute_steps(
    tmp_path, monkeypatch
):
    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle=True, seed=0):
            self.batch_size = int(batch_size)

        def train_batch(self, step):
            return {
                "x": np.zeros((self.batch_size, 2), dtype=np.float32),
                "y": np.zeros((self.batch_size, 1), dtype=np.float32),
                "valid_mask": np.ones((self.batch_size,), dtype=np.float32),
            }

        def iter_eval_batches(self):
            yield {
                "x": np.zeros((self.batch_size, 2), dtype=np.float32),
                "y": np.zeros((self.batch_size, 1), dtype=np.float32),
                "valid_mask": np.ones((self.batch_size,), dtype=np.float32),
            }

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
    monkeypatch.setattr(
        trainer.ckpt,
        "restore_latest",
        lambda _mngr, target: target.replace(
            step=jnp.array(4, dtype=target.step.dtype)
        ),
    )

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=8,
            steps_per_epoch=4,
            batch_size=2,
            logging_interval_steps=4,
            logging_steps=(2, 5, 8),
            evaluation_interval_steps=4,
            evaluation_steps=(3, 6, 8),
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

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        val_dataset=[0, 1],
        callbacks=[],
        resume=True,
    )

    assert result.history.logs["training_step"] == [5.0, 8.0]
    assert result.history.logs["validation_step"] == [6.0, 8.0]


def test_resume_with_explicit_checkpoint_steps_only_saves_remaining_absolute_steps(
    tmp_path, monkeypatch
):
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

    saved_steps: list[int] = []

    class DummyManager:
        directory = tmp_path / "ckpts"

        def wait_until_finished(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        trainer.ckpt, "create_manager", lambda *args, **kwargs: DummyManager()
    )
    monkeypatch.setattr(
        trainer.ckpt,
        "restore_latest",
        lambda _mngr, target: target.replace(
            step=jnp.array(4, dtype=target.step.dtype)
        ),
    )

    def _save(_mngr, step, train_state, *, custom_metadata=None):
        del train_state, custom_metadata
        saved_steps.append(int(step))
        return str(tmp_path / f"ckpt-{int(step)}")

    monkeypatch.setattr(trainer.ckpt, "save", _save)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=8,
            steps_per_epoch=4,
            batch_size=2,
            checkpoint_interval_steps=None,
            checkpoint_steps=(2, 5, 8),
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

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        callbacks=[
            trainer.ModelCheckpoint(
                every_n_steps=None,
                explicit_steps=(2, 5, 8),
                save_on_train_end=False,
            )
        ],
        resume=True,
    )

    assert saved_steps == [5, 8]
