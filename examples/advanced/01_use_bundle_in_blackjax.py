"""Run a tiny BlackJAX NUTS step using a frozen bundle callable.

Data: shipped reference flux bundle + synthetic observations.
Creates: nothing.
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from astro_emulators_toolkit import Emulator, normalize_tree

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_bundle import require_reference_bundle


def main() -> None:
    bundle = require_reference_bundle()
    emu = Emulator.from_bundle(bundle)
    # Freeze once, then jit the sampler init/transition around the full log-density.
    apply_fn = emu.make_frozen_apply(jit=False)

    ref = emu.reference_scaling_inputs or {}
    x = normalize_tree(
        {"parameters": np.array([[5600.0, 4.2, -0.1]], dtype=np.float32)},
        ref["min_tree"],
        ref["max_tree"],
    )

    def model_flux(mu):
        return apply_fn(x)["flux"] + mu

    compiled_model_flux = jax.jit(model_flux)
    y_obs = compiled_model_flux(jnp.asarray([0.0], dtype=jnp.float32))
    y_err = jnp.asarray(np.full_like(np.asarray(y_obs), 0.05, dtype=np.float32))

    def logprob(mu):
        model = model_flux(mu)
        resid = (y_obs - model) / y_err
        return -0.5 * jnp.sum(resid**2) - 0.5 * jnp.sum((mu / 5.0) ** 2)

    nuts = blackjax.nuts(logprob, step_size=1e-2, inverse_mass_matrix=jnp.ones((1,)))
    init = jax.jit(nuts.init)
    kernel = jax.jit(nuts.step)
    state = init(jnp.array([0.0], dtype=jnp.float32))
    rng_key = jax.random.PRNGKey(0)
    state, _ = kernel(rng_key, state)
    print("Sampled mu:", float(state.position[0]))


if __name__ == "__main__":
    main()
