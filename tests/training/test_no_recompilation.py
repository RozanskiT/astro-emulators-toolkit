"""Guard: train/eval batches keep static shapes, so nothing retraces.

The trainer builds exactly two jitted callables per fit() — train_step and
eval_predict — and the DataLoader guarantees fixed batch shapes (train
batches wrap around modularly; the final eval batch is padded and masked).
This test counts forward-trace executions across a run whose dataset sizes
do NOT divide the batch size and asserts each callable compiles exactly
once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit.config import RootConfig, TrainConfig
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.state import TrainState


class _OddSizedDataset:
    """Deterministic dataset whose length does not divide the batch size."""

    def __init__(self, n: int):
        rng = np.random.default_rng(n)
        self.x = rng.normal(size=(n, 2)).astype(np.float32)
        self.y = rng.normal(size=(n, 1)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def get_batch(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class _MSETask:
    def loss_and_metrics(self, pred, batch):
        return jnp.mean(jnp.square(pred - batch["y"])), {}


def test_train_and_eval_each_compile_exactly_once(tmp_path, monkeypatch):
    forward_traces = {"count": 0}

    def _fake_call(state_tuple):
        _graphdef, full_state = state_tuple

        def _forward(x, train=False, rngs=None):
            # Executes only while a jitted callable is being traced, so this
            # counts compilations, not steps.
            forward_traces["count"] += 1
            return x @ full_state["w"], (None, None)

        return _forward

    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    cfg = RootConfig(
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=7,  # crosses shuffle-cycle boundaries with n=5, batch=2
            steps_per_epoch=3,
            batch_size=2,
            logging_interval_steps=None,
            evaluation_interval_steps=1,  # eval every step, incl. padded batch
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

    trainer.fit(
        cfg=cfg,
        graphdef=None,
        init_state=init_state,
        task=_MSETask(),
        tx=tx,
        train_dataset=_OddSizedDataset(5),
        val_dataset=_OddSizedDataset(3),  # final eval batch is padded, not reshaped
        callbacks=[],
        resume=False,
    )

    # One trace for train_step, one for eval_predict; any additional trace
    # means a batch changed shape/dtype mid-run and triggered recompilation.
    assert forward_traces["count"] == 2
