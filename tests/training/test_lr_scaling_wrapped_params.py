"""Regression tests for LR-group labeling on wrapper-nested parameter trees.

Canonical wrappers nest the trainable model under ``core_model``, which is the
layout every ``Emulator.from_config(...)`` emulator trains with. These tests
pin that layout end-to-end so ``optim.lr_scaling`` can never silently degrade
to unscaled optimization again.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.optimizers import make_tx
from astro_emulators_toolkit.optimizers import (
    _collect_label_counts,
    _label_scaled_params,
)


jax.config.update("jax_enable_x64", True)


def _transformer_core_params():
    return {
        "param_embedding": {
            "w0": jnp.ones((2, 8), dtype=jnp.float32),
            "w1": jnp.ones((8, 16), dtype=jnp.float32),
            "b0": jnp.ones((8,), dtype=jnp.float32),
        },
        "attn_layers": {
            0: {
                "wq": jnp.ones((8, 8), dtype=jnp.float32),
                "wo": jnp.ones((8, 8), dtype=jnp.float32),
            }
        },
        "ff_layers": {0: {"in_proj": {"w": jnp.ones((8, 32), dtype=jnp.float32)}}},
        "head": {
            "proj0": {
                "w": jnp.ones((8, 8), dtype=jnp.float32),
                "b": jnp.ones((8,), dtype=jnp.float32),
            }
        },
    }


def _transformer_mup_cfg():
    return RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={"dim": 8, "reference_width": 4, "no_layers": 2},
        ),
        optim=OptimConfig(
            name="sgd",
            lr=1.0,
            b1=0.0,
            lr_scaling="mup",
            scale_embedding_lr=0.5,
            schedule="constant",
        ),
    )


def test_transformer_payne_mup_labels_match_between_bare_and_wrapped_trees():
    cfg = _transformer_mup_cfg()
    bare = _transformer_core_params()
    wrapped = {"core_model": bare}

    bare_counts = _collect_label_counts(_label_scaled_params(cfg, bare))
    wrapped_counts = _collect_label_counts(_label_scaled_params(cfg, wrapped))

    assert bare_counts == wrapped_counts
    assert wrapped_counts == {
        "emb_in": 1,
        "emb_out": 1,
        "bias": 2,
        "attn": 2,
        "ff": 1,
        "head": 1,
    }


def test_make_tx_transformer_payne_mup_scales_wrapped_parameter_groups():
    cfg = _transformer_mup_cfg()
    params = {"core_model": _transformer_core_params()}
    grads = jax.tree_util.tree_map(jnp.ones_like, params)

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    core_updates = updates["core_model"]
    assert float(core_updates["param_embedding"]["w0"][0, 0]) == pytest.approx(-0.5)
    assert float(core_updates["param_embedding"]["w1"][0, 0]) == pytest.approx(-0.25)
    assert float(core_updates["param_embedding"]["b0"][0]) == pytest.approx(-1.0)
    assert float(core_updates["attn_layers"][0]["wq"][0, 0]) == pytest.approx(-0.5)
    assert float(core_updates["ff_layers"][0]["in_proj"]["w"][0, 0]) == pytest.approx(
        -0.5
    )
    assert float(core_updates["head"]["proj0"]["w"][0, 0]) == pytest.approx(-0.5)


def test_make_tx_mlp_mup_scales_wrapped_kernels():
    params = {
        "core_model": {
            "layers": {
                0: {
                    "kernel": jnp.ones((3, 8), dtype=jnp.float32),
                    "bias": jnp.ones((8,), dtype=jnp.float32),
                }
            }
        }
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (8,), "reference_width": 4}
        ),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup", schedule="constant"),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["core_model"]["layers"][0]["kernel"][0, 0]) == pytest.approx(
        -0.5
    )
    assert float(updates["core_model"]["layers"][0]["bias"][0]) == pytest.approx(-1.0)


def test_make_tx_fails_loud_when_lr_scaling_matches_no_parameters():
    params = {"unrecognized_module": {"w": jnp.ones((4, 4), dtype=jnp.float32)}}
    cfg = RootConfig(
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (8,), "reference_width": 4}
        ),
        optim=OptimConfig(name="sgd", lr=1.0, lr_scaling="mup"),
    )

    with pytest.raises(ValueError, match="matched no parameters"):
        make_tx(cfg, params=params)


def test_emulator_from_config_mlp_mup_produces_scaled_groups(tmp_path):
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (8, 8), "activation": "tanh", "reference_width": 4},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup", schedule="constant"),
        training=TrainConfig(workdir=str(tmp_path / "mlp"), batch_size=4, num_steps=1),
    )
    emu = Emulator.from_config(cfg)
    emu.initialize(
        inputs={"parameters": np.zeros((1, 3), dtype=np.float32)},
        outputs={"predictions": np.zeros((1, 2), dtype=np.float32)},
    )
    assert emu.params is not None

    counts = _collect_label_counts(_label_scaled_params(cfg, emu.params))
    assert counts.get("mlp_kernel", 0) >= 3

    tx = make_tx(cfg, params=emu.params)
    grads = jax.tree_util.tree_map(jnp.ones_like, emu.params)
    updates, _ = tx.update(grads, tx.init(emu.params), emu.params)
    kernel_update = updates["core_model"]["layers"][0]["kernel"]
    bias_update = updates["core_model"]["layers"][0]["bias"]
    assert float(kernel_update[0, 0]) == pytest.approx(-0.5)
    assert float(bias_update[0]) == pytest.approx(-1.0)


def test_emulator_from_config_transformer_payne_mup_produces_scaled_groups(tmp_path):
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 1,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 2,
                "reference_width": 8,
                "dtype": "float64",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(
            name="sgd",
            lr=1.0,
            b1=0.0,
            lr_scaling="mup",
            scale_embedding_lr=0.5,
            schedule="constant",
        ),
        training=TrainConfig(
            workdir=str(tmp_path / "transformer"), batch_size=4, num_steps=1
        ),
    )
    rng = np.random.default_rng(0)
    params_x = rng.normal(size=(2, 3)).astype(np.float64)
    waves = np.sort(rng.uniform(0.2, 2.0, size=(2, 6)), axis=1)
    flux = rng.normal(size=(2, 6)).astype(np.float64)

    emu = Emulator.from_config(cfg)
    emu.initialize(inputs=(params_x, waves), outputs=flux)
    assert emu.params is not None
    assert "core_model" in emu.params

    counts = _collect_label_counts(_label_scaled_params(cfg, emu.params))
    for group in ("emb_in", "emb_out", "attn", "ff", "head"):
        assert counts.get(group, 0) >= 1, counts
    # This is the exact regression: the wrapped tree used to label everything
    # "default", silently reducing mup to unscaled optimization.
    tx = make_tx(cfg, params=emu.params)
    grads = jax.tree_util.tree_map(jnp.ones_like, emu.params)
    updates, _ = tx.update(grads, tx.init(emu.params), emu.params)
    update_magnitudes = {
        float(jnp.max(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(updates)
    }
    assert len(update_magnitudes) > 1
