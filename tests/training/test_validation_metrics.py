from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit.config import RootConfig, TaskSpec, TrainConfig
from astro_emulators_toolkit.tasks import build_task
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState


class IndexedDataset:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def get_batch(self, idx):
        idx = np.asarray(idx, dtype=np.int32)
        x = {"parameters": np.zeros((len(idx), 2), dtype=np.float32)}
        y = {"predictions": idx.astype(np.float32).reshape((-1, 1))}
        sample_weight = np.where((idx % 2) == 0, 4.0, 1.0).astype(np.float32)
        return {"x": x, "y": y, "sample_weight": sample_weight}


def test_validation_metrics_use_task_aggregation_semantics(tmp_path, monkeypatch):
    def _fake_merge_state(params, model_state):
        return {"params": params, "model_state": model_state}

    def _fake_split_state(full_state, *_filters):
        return full_state["params"], full_state["model_state"]

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            pred = {
                "predictions": jnp.zeros(
                    (x["parameters"].shape[0], 1), dtype=jnp.float32
                )
            }
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

    task = build_task(
        "regression",
        {
            "loss": "weighted_mse",
            "metrics": ["mae", "weighted_mae"],
        },
    )

    cfg = RootConfig(
        task=TaskSpec(
            name="regression",
            params={"loss": "weighted_mse", "metrics": ["mae", "weighted_mae"]},
        ),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=1,
            steps_per_epoch=1,
            batch_size=4,
            evaluation_interval_steps=1,
            shuffle=False,
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

    result = trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=task,
        tx=tx,
        train_dataset=IndexedDataset(4),
        val_dataset=IndexedDataset(5),
        callbacks=[],
        resume=False,
    )

    assert result.history.logs["validation_mae"][-1] == 2.0
    # Weighted mean of absolute values [0,1,2,3,4] with weights [4,1,4,1,4] => 28/14.
    np.testing.assert_allclose(
        result.history.logs["validation_weighted_mae"][-1], 28.0 / 14.0
    )
    # Weighted mean of squared values => 90/14.
    np.testing.assert_allclose(
        result.history.logs["validation_weighted_mse"][-1], 90.0 / 14.0
    )
    np.testing.assert_allclose(result.history.logs["validation_loss"][-1], 90.0 / 14.0)
