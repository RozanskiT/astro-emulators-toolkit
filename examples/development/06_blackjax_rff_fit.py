"""Fit RFF emulator inputs with BlackJAX after a short synthetic-data training run.

Data: deterministic in-memory RFF toy dataset.
Creates: runs/blackjax_example.
Runtime: minutes on CPU.
Requires: `uv sync --extra blackjax`.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    ModelSpec,
    OptimConfig,
    RootConfig,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.data.toy_utils import generate_rff_dataset
from astro_emulators_toolkit.training import ProgressBarLogger


def main() -> None:
    if importlib.util.find_spec("blackjax") is None:
        raise SystemExit("This example requires `uv sync --extra blackjax`.")
    import blackjax

    # Train a tiny emulator on deterministic synthetic data.
    x_train, y_train = generate_rff_dataset(
        n_samples=384,
        x_dim=3,
        y_dim=1,
        n_features=24,
        freq_scale=1.25,
        noise_std=0.01,
        x_dist="normal",
        seed=11,
    )

    ds = TreeArrayDataset(x={"parameters": x_train}, y={"predictions": y_train})

    cfg = RootConfig(
        seed=11,
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (32, 32), "activation": "gelu"}
        ),
        optim=OptimConfig(name="adam", lr=5e-3),
        training=TrainConfig(
            workdir="./runs/blackjax_example",
            batch_size=64,
            num_steps=1000,
            logging_interval_steps=100,
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()
    callbacks = [
        ProgressBarLogger(total_steps=cfg.training.num_steps),
    ]
    emu.fit(ds, callbacks=callbacks)

    # Freeze weights for inference (recreate this callable if emu.params/model_state change).
    apply = emu.make_frozen_apply(postprocess=True, jit=False)

    theta_true = jnp.array([0.2, -0.3, 0.5], dtype=jnp.float32)
    y_obs = jnp.squeeze(
        apply({"parameters": theta_true[None, :]})["predictions"], axis=0
    )
    noise_sigma = jnp.array([0.03], dtype=jnp.float32)

    def logprior(theta):
        return jstats.norm.logpdf(theta, 0.0, 1.0).sum()

    def loglik(theta):
        y_pred = jnp.squeeze(
            apply({"parameters": theta[None, :]})["predictions"], axis=0
        )
        return jstats.norm.logpdf(y_obs, y_pred, noise_sigma).sum()

    def logdensity(theta):
        return logprior(theta) + loglik(theta)

    rng_key = jax.random.key(0)
    initial_position = jnp.zeros((3,), dtype=jnp.float32)

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (initial_state, tuned_parameters), _ = warmup.run(
        rng_key, initial_position, num_steps=200
    )

    kernel = jax.jit(blackjax.nuts(logdensity, **tuned_parameters).step)

    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, _info = kernel(subkey, state)
        return (state, key), state.position

    (_, _), samples = jax.lax.scan(
        one_step, (initial_state, rng_key), xs=None, length=400
    )

    print("Posterior mean:", jnp.mean(samples, axis=0))
    print("Posterior std:", jnp.std(samples, axis=0))
    print("True theta:", theta_true)


if __name__ == "__main__":
    main()
