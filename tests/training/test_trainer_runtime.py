from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit.config import IOSpec, RootConfig, TrainConfig, load_config
from astro_emulators_toolkit.training.callbacks import Callback
from astro_emulators_toolkit.training.paths import RUN_CONFIG_FILENAME
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState


def test_trainer_uses_step_based_sampling_and_skips_checkpoint_manager_without_callback(
    monkeypatch, tmp_path
):
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

    create_manager_called = False

    def _create_manager(*args, **kwargs):
        nonlocal create_manager_called
        create_manager_called = True
        raise AssertionError(
            "Checkpoint manager should not be created without resume or ModelCheckpoint callback"
        )

    monkeypatch.setattr(trainer.ckpt, "create_manager", _create_manager)

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

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        val_dataset=None,
        callbacks=[],
        resume=False,
    )

    assert seen_steps == [0, 1, 2, 3, 4, 5]
    assert create_manager_called is False


def test_trainer_threads_model_state_and_handles_small_validation_set(
    tmp_path, monkeypatch
):
    class TinyDataset:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

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
            if train:
                updated = {"counter": jnp.array(1, dtype=jnp.int32)}
            else:
                updated = {"counter": jnp.array(0, dtype=jnp.int32)}
            return pred, (
                None,
                {
                    "params": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
                    "model_state": updated,
                },
            )

        return _forward

    monkeypatch.setattr(trainer.nnx, "merge_state", _fake_merge_state)
    monkeypatch.setattr(trainer.nnx, "split_state", _fake_split_state)
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=1,
            steps_per_epoch=1,
            batch_size=4,
            evaluation_interval_steps=1,
            shuffle=False,
        )
    )

    tx = optax.sgd(1e-3)
    params = {"w": jnp.ones((2, 1), dtype=jnp.float32)}
    init_state = TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=jax.random.key(0),
        params=params,
        model_state={"counter": jnp.array(0, dtype=jnp.int32)},
        opt_state=tx.init(params),
    )

    class DummyTask:
        def loss_and_metrics(self, pred, batch):
            return jnp.array(0.0), {"mae": jnp.array(0.0)}

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=TinyDataset(4),
        val_dataset=TinyDataset(1),
        callbacks=[],
        resume=False,
    )

    assert int(result.state.model_state["counter"]) == 1


def test_recursive_batch_to_jax_conversion_supports_nested_pytree():
    batch = {
        "x": np.ones((2, 1), dtype=np.float32),
        "meta": {
            "aux": [
                np.ones((2, 2), dtype=np.float32),
                (np.zeros((2, 1), dtype=np.float32),),
            ],
        },
    }

    converted = trainer._to_jax_batch(batch)
    assert isinstance(converted["x"], jax.Array)
    assert isinstance(converted["meta"]["aux"][0], jax.Array)
    assert isinstance(converted["meta"]["aux"][1][0], jax.Array)


def test_trainer_preserves_metadata_when_batch_transform_drops_keys(
    monkeypatch, tmp_path
):
    from astro_emulators_toolkit.training import trainer as trainer_module

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32), (None, None)

        return _forward

    monkeypatch.setattr(trainer_module.nnx, "call", _fake_call)

    class DummyTask:
        seen_train_has_sample_weight = False
        seen_eval_valid_mask = None

        def loss_and_metrics(self, pred, batch):
            self.seen_train_has_sample_weight = "sample_weight" in batch
            return jnp.array(0.0), {}

        def init_eval_state(self):
            return 0

        def update_eval_state(self, state, pred, batch):
            self.seen_eval_valid_mask = np.asarray(batch.get("valid_mask"))
            return state + 1

        def finalize_eval(self, state):
            return {"loss": jnp.array(float(state))}

    class DummyDataset:
        def __len__(self):
            return 3

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((idx.shape[0], 2), dtype=np.float32)
            y = np.zeros((idx.shape[0], 1), dtype=np.float32)
            sample_weight = np.ones((idx.shape[0],), dtype=np.float32)
            return {"x": x, "y": y, "sample_weight": sample_weight}

    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=2,
            batch_size=2,
            logging_interval_steps=1,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=0,
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

    def drop_metadata_transform(batch, *, rng, train):
        del rng, train
        return {"x": batch["x"], "y": batch["y"]}

    task = DummyTask()
    trainer_module.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=task,
        tx=tx,
        train_dataset=DummyDataset(),
        val_dataset=DummyDataset(),
        callbacks=[],
        resume=False,
        device_batch_transform=drop_metadata_transform,
    )

    assert task.seen_train_has_sample_weight is True
    assert task.seen_eval_valid_mask is not None
    assert task.seen_eval_valid_mask.shape == (2,)


def test_trainer_applies_device_batch_transform_with_train_and_eval_flags(
    monkeypatch, tmp_path
):
    from astro_emulators_toolkit.training import trainer as trainer_module

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32), (None, None)

        return _forward

    monkeypatch.setattr(trainer_module.nnx, "call", _fake_call)

    class DummyDataset:
        def __len__(self):
            return 2

        def get_batch(self, idx):
            idx = np.asarray(idx)
            x = np.zeros((idx.shape[0], 2), dtype=np.float32)
            y = np.zeros((idx.shape[0], 1), dtype=np.float32)
            return {"x": x, "y": y}

    class FlagAwareTransform:
        def __call__(self, batch, *, rng, train):
            del rng
            shift = 1.0 if train else 2.0
            return {
                "x": batch["x"] + shift,
                "y": batch["y"],
            }

    class DummyTask:
        def loss_and_metrics(self, pred, batch):
            del pred
            x_mean = jnp.mean(batch["x"])
            return x_mean, {"x_mean": x_mean}

        def init_eval_state(self):
            return 0.0

        def update_eval_state(self, state, pred, batch):
            del pred
            return state + float(np.asarray(batch["x"]).mean())

        def finalize_eval(self, state):
            return {"x_mean": jnp.array(state, dtype=jnp.float32)}

    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=1,
            batch_size=2,
            logging_interval_steps=1,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=0,
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

    result = trainer_module.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=DummyDataset(),
        val_dataset=DummyDataset(),
        callbacks=[],
        resume=False,
        device_batch_transform=FlagAwareTransform(),
    )

    assert result.history.logs["training_x_mean"] == [1.0]
    assert result.history.logs["validation_x_mean"] == [2.0]


def test_trainer_supports_explicit_logging_and_evaluation_steps(monkeypatch, tmp_path):
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

    class Recorder(Callback):
        def __init__(self):
            self.train_steps: list[int] = []
            self.eval_steps: list[int] = []

        def on_train_batch_end(self, step: int, logs):
            self.train_steps.append(int(step))

        def on_eval_end(self, step: int, logs):
            self.eval_steps.append(int(step))

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=6,
            steps_per_epoch=6,
            batch_size=2,
            logging_interval_steps=4,
            logging_steps=(2, 5),
            evaluation_interval_steps=4,
            evaluation_steps=(3, 6),
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

    recorder = Recorder()
    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1, 2, 3],
        val_dataset=[0, 1],
        callbacks=[recorder],
        resume=False,
    )

    assert result.history.logs["training_step"] == [2.0, 4.0, 5.0]
    assert result.history.logs["validation_step"] == [3.0, 4.0, 6.0]
    assert recorder.train_steps == [2, 4, 5]
    assert recorder.eval_steps == [3, 4, 6]


def test_trainer_ignores_explicit_steps_beyond_target_num_steps(monkeypatch, tmp_path):
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

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            steps_per_epoch=4,
            batch_size=2,
            logging_interval_steps=None,
            logging_steps=(2, 5, 8),
            evaluation_interval_steps=None,
            evaluation_steps=(4, 9),
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
        resume=False,
    )

    assert result.history.logs["training_step"] == [2.0]
    assert result.history.logs["validation_step"] == [4.0]


def test_trainer_allows_disabling_periodic_logging_and_validation_with_none(
    monkeypatch, tmp_path
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

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            steps_per_epoch=4,
            batch_size=2,
            logging_interval_steps=None,
            logging_steps=None,
            evaluation_interval_steps=None,
            evaluation_steps=None,
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
        resume=False,
    )

    assert "training_step" not in result.history.logs
    assert "validation_step" not in result.history.logs


def test_trainer_writes_run_config_json_at_training_start(monkeypatch, tmp_path):
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

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=1,
            steps_per_epoch=1,
            batch_size=2,
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

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=DummyTask(),
        tx=tx,
        train_dataset=[0, 1],
        callbacks=[],
        resume=False,
    )

    run_cfg_path = (tmp_path / "run").resolve() / RUN_CONFIG_FILENAME
    assert run_cfg_path.exists()
    loaded = load_config(run_cfg_path)
    assert loaded.training.workdir == str(tmp_path / "run")
