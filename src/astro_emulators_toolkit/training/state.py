# src/astro_emulators_toolkit/training/state.py
from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any, cast

from flax import struct
import jax

from .._typing import PytreeDict


@struct.dataclass
class TrainState:
    step: jax.Array
    rng_key: jax.Array

    # Params and model_state are stored as pure nested dict pytrees.
    params: PytreeDict
    model_state: PytreeDict

    opt_state: object  # optax OptState (pytree)

    def replace(self, **updates: object) -> "TrainState":
        return dataclass_replace(self, **cast(dict[str, Any], updates))

    def next_rng(self) -> tuple["TrainState", jax.Array]:
        k1, k2 = jax.random.split(self.rng_key)
        return self.replace(rng_key=k1), k2
