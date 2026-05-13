from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from astro_emulators_toolkit.config import (
    ModelSpec,
    OptimConfig,
    RootConfig,
    TrainConfig,
)
from astro_emulators_toolkit.optimizers import make_learning_rate, make_tx
from astro_emulators_toolkit.training.callbacks import History


def _schedule_value(schedule, step: int) -> float:
    if callable(schedule):
        return float(schedule(step))
    return float(schedule)


@pytest.mark.parametrize(
    "schedule_name,warmup_steps,decay_steps",
    [
        ("cosine", 2, 0),
        ("wsd", 2, 3),
        ("wsd", 2, 0),
    ],
)
def test_decay_schedules_reach_zero_on_last_step(
    schedule_name: str, warmup_steps: int, decay_steps: int
):
    num_steps = 10
    cfg = RootConfig(
        optim=OptimConfig(
            schedule=schedule_name,
            lr=1e-3,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        ),
        training=TrainConfig(num_steps=num_steps),
    )
    schedule = make_learning_rate(cfg)

    assert _schedule_value(schedule, num_steps - 1) == pytest.approx(0.0)


def test_history_records_train_and_validation_steps_separately():
    history = History()

    history.add_train(1, {"loss": 1.0, "mae": 0.5})
    history.add_train(2, {"loss": 0.8})
    history.add_eval(2, {"loss": 0.9})

    assert history.logs["training_step"] == [1.0, 2.0]
    assert history.logs["training_loss"] == [1.0, 0.8]
    assert history.logs["training_mae"] == [0.5]
    assert history.logs["validation_step"] == [2.0]
    assert history.logs["validation_loss"] == [0.9]


def test_make_learning_rate_constant_is_default_scalar():
    cfg = RootConfig()

    lr = make_learning_rate(cfg)

    assert isinstance(lr, float)
    assert lr == cfg.optim.lr


def test_make_learning_rate_cosine_with_warmup_shape():
    cfg = RootConfig(
        optim=OptimConfig(lr=1.0, schedule="cosine", warmup_steps=2),
        training=TrainConfig(num_steps=10),
    )

    lr = make_learning_rate(cfg)

    assert callable(lr)
    assert float(lr(0)) == pytest.approx(0.0)
    assert float(lr(2)) == pytest.approx(1.0)
    assert float(lr(9)) < 0.1


def test_make_learning_rate_wsd_warmup_stable_decay_shape():
    cfg = RootConfig(
        optim=OptimConfig(lr=1.0, schedule="wsd", warmup_steps=2, decay_steps=3),
        training=TrainConfig(num_steps=10),
    )

    lr = make_learning_rate(cfg)

    assert callable(lr)
    assert float(lr(0)) == pytest.approx(0.0)
    assert float(lr(2)) == pytest.approx(1.0)
    assert float(lr(5)) == pytest.approx(1.0)
    assert float(lr(8)) < 0.8
    assert float(lr(9)) < 0.5
    assert float(lr(10)) == pytest.approx(0.0)


def test_make_tx_accepts_schedule_for_all_builtin_optimizers():
    params = {"w": jnp.ones((2, 1), dtype=jnp.float32)}

    for optim_name in ("adam", "adamw", "sgd"):
        cfg = RootConfig(
            optim=OptimConfig(
                name=optim_name, lr=1e-3, schedule="cosine", warmup_steps=1
            ),
            training=TrainConfig(num_steps=8),
        )
        tx = make_tx(cfg)
        state = tx.init(params)
        grads = jax.tree_util.tree_map(jnp.ones_like, params)
        updates, _ = tx.update(grads, state, params)
        assert updates["w"].shape == params["w"].shape


def test_make_tx_applies_global_grad_clip_before_optimizer():
    params = {"w": jnp.zeros((2,), dtype=jnp.float32)}
    grads = {"w": jnp.asarray([3.0, 4.0], dtype=jnp.float32)}
    cfg = RootConfig(optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, grad_clip=1.0))

    tx = make_tx(cfg)
    updates, _ = tx.update(grads, tx.init(params), params)

    np_updates = jnp.asarray(updates["w"])
    assert float(np_updates[0]) == pytest.approx(-0.6)
    assert float(np_updates[1]) == pytest.approx(-0.8)


def test_make_tx_lr_scaling_none_keeps_single_shared_lr_without_params():
    params = {
        "param_embedding": {
            "w0": jnp.ones((2, 4), dtype=jnp.float32),
            "w1": jnp.ones((4, 8), dtype=jnp.float32),
        },
        "head": {"proj0": {"w": jnp.ones((4, 4), dtype=jnp.float32)}},
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(optim=OptimConfig(name="sgd", lr=0.25, b1=0.0))

    tx = make_tx(cfg)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["param_embedding"]["w0"][0, 0]) == pytest.approx(-0.25)
    assert float(updates["param_embedding"]["w1"][0, 0]) == pytest.approx(-0.25)
    assert float(updates["head"]["proj0"]["w"][0, 0]) == pytest.approx(-0.25)


def test_make_tx_transformer_payne_mup_scales_parameter_groups():
    params = {
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
        "ff_layers": {
            0: {
                "in_proj": {"w": jnp.ones((8, 32), dtype=jnp.float32)},
                "out_proj": {"w": jnp.ones((32, 8), dtype=jnp.float32)},
            }
        },
        "head": {
            "proj0": {
                "w": jnp.ones((8, 8), dtype=jnp.float32),
                "b": jnp.ones((8,), dtype=jnp.float32),
            },
            "proj1": {"w": jnp.ones((8, 1), dtype=jnp.float32)},
        },
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
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
        ),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["param_embedding"]["w0"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["param_embedding"]["w1"][0, 0]) == pytest.approx(-0.25)
    assert float(updates["param_embedding"]["b0"][0]) == pytest.approx(-1.0)
    assert float(updates["attn_layers"][0]["wq"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["ff_layers"][0]["in_proj"]["w"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["head"]["proj0"]["w"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["head"]["proj0"]["b"][0]) == pytest.approx(-1.0)


def test_make_tx_transformer_payne_mup_depth_scales_attention_and_ff_by_depth():
    params = {
        "attn_layers": {0: {"wq": jnp.ones((8, 8), dtype=jnp.float32)}},
        "ff_layers": {0: {"out_proj": {"w": jnp.ones((32, 8), dtype=jnp.float32)}}},
        "head": {"proj1": {"w": jnp.ones((8, 1), dtype=jnp.float32)}},
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={
                "dim": 8,
                "reference_width": 4,
                "no_layers": 8,
                "reference_depth": 2,
            },
        ),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup_depth"),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["attn_layers"][0]["wq"][0, 0]) == pytest.approx(-0.25)
    assert float(updates["ff_layers"][0]["out_proj"]["w"][0, 0]) == pytest.approx(-0.25)
    assert float(updates["head"]["proj1"]["w"][0, 0]) == pytest.approx(-0.5)


def test_make_tx_transformer_payne_mup_uses_model_config_defaults():
    params = {"head": {"proj1": {"w": jnp.ones((8, 1), dtype=jnp.float32)}}}
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
        model=ModelSpec(name="transformer_payne", params={"reference_width": 64}),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup"),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["head"]["proj1"]["w"][0, 0]) == pytest.approx(-0.5)


def test_make_tx_transformer_payne_mup_requires_initialized_params():
    cfg = RootConfig(
        model=ModelSpec(
            name="transformer_payne",
            params={"dim": 8, "reference_width": 4, "no_layers": 2},
        ),
        optim=OptimConfig(lr_scaling="mup"),
    )

    with pytest.raises(ValueError, match="requires initialized params"):
        make_tx(cfg)


def test_make_tx_transformer_payne_mup_requires_reference_width():
    params = {"head": {"proj1": {"w": jnp.ones((8, 1), dtype=jnp.float32)}}}
    cfg = RootConfig(
        model=ModelSpec(name="transformer_payne", params={"dim": 8, "no_layers": 2}),
        optim=OptimConfig(lr_scaling="mup"),
    )

    with pytest.raises(ValueError, match="model.params.reference_width"):
        make_tx(cfg, params=params)


def test_make_tx_mlp_mup_scales_kernels_and_leaves_biases_at_base_lr():
    params = {
        "layers": {
            0: {
                "kernel": jnp.ones((3, 8), dtype=jnp.float32),
                "bias": jnp.ones((8,), dtype=jnp.float32),
            },
            1: {
                "kernel": jnp.ones((8, 8), dtype=jnp.float32),
                "bias": jnp.ones((8,), dtype=jnp.float32),
            },
            2: {
                "kernel": jnp.ones((8, 2), dtype=jnp.float32),
                "bias": jnp.ones((2,), dtype=jnp.float32),
            },
        }
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (8, 8), "reference_width": 4},
        ),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup"),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["layers"][0]["kernel"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["layers"][1]["kernel"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["layers"][2]["kernel"][0, 0]) == pytest.approx(-0.5)
    assert float(updates["layers"][0]["bias"][0]) == pytest.approx(-1.0)
    assert float(updates["layers"][2]["bias"][0]) == pytest.approx(-1.0)


def test_make_tx_mlp_mup_uses_default_hidden_width():
    params = {"layers": {0: {"kernel": jnp.ones((3, 2), dtype=jnp.float32)}}}
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    cfg = RootConfig(
        model=ModelSpec(name="mlp", params={"reference_width": 128}),
        optim=OptimConfig(name="sgd", lr=1.0, b1=0.0, lr_scaling="mup"),
    )

    tx = make_tx(cfg, params=params)
    updates, _ = tx.update(grads, tx.init(params), params)

    assert float(updates["layers"][0]["kernel"][0, 0]) == pytest.approx(-0.5)


def test_make_tx_mlp_mup_requires_reference_width():
    params = {"layers": {0: {"kernel": jnp.ones((3, 2), dtype=jnp.float32)}}}
    cfg = RootConfig(
        model=ModelSpec(name="mlp", params={"hidden_sizes": (8, 8)}),
        optim=OptimConfig(lr_scaling="mup"),
    )

    with pytest.raises(ValueError, match="model.params.reference_width"):
        make_tx(cfg, params=params)


def test_make_tx_mlp_rejects_mup_depth():
    params = {"layers": {0: {"kernel": jnp.ones((3, 2), dtype=jnp.float32)}}}
    cfg = RootConfig(
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (8, 8), "reference_width": 4}
        ),
        optim=OptimConfig(lr_scaling="mup_depth"),
    )

    with pytest.raises(ValueError, match="mup_depth"):
        make_tx(cfg, params=params)


def test_make_tx_builds_soap_with_config(monkeypatch):
    calls = {}

    def _soap(**kwargs):
        calls.update(kwargs)
        return "soap-tx"

    monkeypatch.setattr("astro_emulators_toolkit.optimizers._soap.soap", _soap)

    cfg = RootConfig(
        optim=OptimConfig(
            name="soap",
            lr=3e-3,
            b1=0.95,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.01,
            precondition_frequency=5,
            precondition_1d=True,
        )
    )

    tx = make_tx(cfg)

    assert tx == "soap-tx"
    assert calls == {
        "learning_rate": 3e-3,
        "b1": 0.95,
        "b2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "precondition_frequency": 5,
        "precondition_1d": True,
    }


def test_internal_vendored_soap_runs_one_update_step():
    params = {"w": jnp.ones((2, 1), dtype=jnp.float32)}
    grads = {"w": jnp.full((2, 1), 0.5, dtype=jnp.float32)}
    cfg = RootConfig(
        optim=OptimConfig(
            name="soap",
            lr=1e-3,
            schedule="constant",
            precondition_frequency=2,
            precondition_1d=True,
        ),
        training=TrainConfig(num_steps=4),
    )

    tx = make_tx(cfg)
    state = tx.init(params)

    updates, state = tx.update(grads, state, params)
    assert updates["w"].shape == params["w"].shape

    updates, _ = tx.update(grads, state, params)
    assert updates["w"].shape == params["w"].shape
