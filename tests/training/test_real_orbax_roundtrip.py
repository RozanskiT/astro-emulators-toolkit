"""Real orbax save/restore round-trip tests (no checkpoint mocking).

These are the only tests that exercise actual orbax serialization of the
full TrainState, including optimizer internals: adamw (optax counters and
moments inside a warmup+cosine schedule) and the vendored SOAP (whose state
carries a custom registered pytree in its preconditioners).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit.config import (
    OptimConfig,
    RootConfig,
    TrainConfig,
)
from astro_emulators_toolkit.optimizers import make_tx
from astro_emulators_toolkit.training import trainer
from astro_emulators_toolkit.training.callbacks import ModelCheckpoint
from astro_emulators_toolkit.training.state import TrainState


class _LinearDataset:
    def __init__(self, n: int = 8):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(n, 2)).astype(np.float32)
        self.y = rng.normal(size=(n, 1)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def get_batch(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class _MSETask:
    def loss_and_metrics(self, pred, batch):
        loss = jnp.mean(jnp.square(pred - batch["y"]))
        return loss, {}


def _fake_call(state_tuple):
    _graphdef, full_state = state_tuple

    def _forward(x, train=False, rngs=None):
        return x @ full_state["w"], (None, None)

    return _forward


def _cfg(workdir, *, optim_name: str, num_steps: int) -> RootConfig:
    return RootConfig(
        optim=OptimConfig(name=optim_name, lr=1e-2),
        training=TrainConfig(
            workdir=str(workdir),
            num_steps=num_steps,
            steps_per_epoch=4,
            batch_size=4,
            shuffle=True,
            shuffle_seed=5,
            logging_interval_steps=None,
            evaluation_interval_steps=None,
            checkpoint_interval_steps=None,
        ),
    )


def _init_state(tx) -> TrainState:
    params = {"w": jnp.full((2, 1), 0.5, dtype=jnp.float32)}
    return TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=jax.random.key(11),
        params=params,
        model_state={},
        opt_state=tx.init(params),
    )


@pytest.mark.parametrize("optim_name", ["adamw", "soap"])
def test_real_orbax_split_run_matches_uninterrupted_bitwise(
    tmp_path, monkeypatch, optim_name
):
    monkeypatch.setattr(trainer.nnx, "call", _fake_call)

    total_steps = 6
    split_steps = 3
    dataset = _LinearDataset()
    task = _MSETask()

    # Uninterrupted reference run: no checkpointing at all.
    full_cfg = _cfg(tmp_path / "full", optim_name=optim_name, num_steps=total_steps)
    full_tx = make_tx(full_cfg)
    full = trainer.fit(
        cfg=full_cfg,
        graphdef=None,
        init_state=_init_state(full_tx),
        task=task,
        tx=full_tx,
        train_dataset=dataset,
        callbacks=[],
        resume=False,
    )

    # Split run: same num_steps (so the schedule shape is identical — the
    # supported way to chunk a run), capped at 3 steps via max_steps, with a
    # real orbax checkpoint at the segment end; then resume to completion.
    split_dir = tmp_path / "split"
    split_cfg = _cfg(split_dir, optim_name=optim_name, num_steps=total_steps)
    first_tx = make_tx(split_cfg)
    trainer.fit(
        cfg=split_cfg,
        graphdef=None,
        init_state=_init_state(first_tx),
        task=task,
        tx=first_tx,
        train_dataset=dataset,
        callbacks=[ModelCheckpoint(every_n_steps=None, save_on_train_end=True)],
        resume=False,
        max_steps=split_steps,
    )

    resumed_tx = make_tx(split_cfg)
    resumed = trainer.fit(
        cfg=split_cfg,
        graphdef=None,
        init_state=_init_state(resumed_tx),
        task=task,
        tx=resumed_tx,
        train_dataset=dataset,
        callbacks=[],
        resume=True,
    )

    assert int(jax.device_get(resumed.state.step)) == total_steps
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(resumed.state.params["w"])),
        np.asarray(jax.device_get(full.state.params["w"])),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(resumed.state.rng_key),
        jax.random.key_data(full.state.rng_key),
    )
