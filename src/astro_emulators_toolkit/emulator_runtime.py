from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils.tree import to_jax_tree, to_numpy_tree


def validate_model_state_dict(model_state):
    if not isinstance(model_state, dict):
        if isinstance(model_state, np.ndarray) and model_state.size == 0:
            return {}
        raise TypeError(
            f"model_state must be a nested dict pytree, got {type(model_state).__name__}."
        )
    return model_state


def to_numpy_pytree(tree):
    return to_numpy_tree(tree)


def build_init_state(*, params, model_state, tx, seed: int, train_state_cls):
    opt_state = tx.init(params)
    return train_state_cls(
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=jax.random.key(seed),
        params=params,
        model_state=model_state,
        opt_state=opt_state,
    )


def apply_jax_runtime(
    *,
    graphdef,
    params,
    model_state,
    task,
    x,
    rng: jax.Array | None,
    postprocess: bool,
    train: bool,
):
    xj = to_jax_tree(x)
    full_state = nnx.merge_state(params, model_state)
    rngs = None if rng is None else nnx.Rngs(dropout=rng)
    y, _ = nnx.call((graphdef, full_state))(xj, train=train, rngs=rngs)
    if postprocess and hasattr(task, "postprocess_pred"):
        y = task.postprocess_pred(y)
    return y


def make_frozen_apply_runtime(
    *, graphdef, params, model_state, post_fn, jit: bool = False
):
    full_state = nnx.merge_state(params, model_state)
    call = nnx.call((graphdef, full_state))

    def apply(x, *, rng: jax.Array | None = None):
        xj = to_jax_tree(x)
        rngs = None if rng is None else nnx.Rngs(dropout=rng)
        y, _ = call(xj, train=False, rngs=rngs)
        return post_fn(y) if post_fn is not None else y

    return jax.jit(apply) if jit else apply
