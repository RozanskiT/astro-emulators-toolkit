from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optax


def make_learning_rate(cfg):
    lr = float(cfg.optim.lr)
    schedule_name = str(cfg.optim.schedule).lower()
    warmup_steps = int(cfg.optim.warmup_steps)
    decay_steps = int(cfg.optim.decay_steps)
    total_steps = int(cfg.training.num_steps)

    if warmup_steps < 0:
        raise ValueError("optim.warmup_steps must be >= 0.")
    if decay_steps < 0:
        raise ValueError("optim.decay_steps must be >= 0.")

    if schedule_name == "constant":
        return lr

    if total_steps <= 0:
        raise ValueError("training.num_steps must be > 0.")

    def _zero_ending_linear(init_value: float, end_value: float, num_updates: int):
        if num_updates <= 1:
            return optax.constant_schedule(end_value)
        return optax.linear_schedule(
            init_value=init_value,
            end_value=end_value,
            transition_steps=num_updates - 1,
        )

    effective_warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    if schedule_name == "cosine":
        cosine_updates = max(1, total_steps - effective_warmup_steps)
        cosine = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=max(1, cosine_updates - 1),
        )
        if effective_warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=effective_warmup_steps,
            )
            return optax.join_schedules(
                [warmup, cosine], boundaries=[effective_warmup_steps]
            )
        return cosine

    if schedule_name == "wsd":
        schedules = []
        boundaries = []
        elapsed = 0

        if effective_warmup_steps > 0:
            schedules.append(
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=lr,
                    transition_steps=effective_warmup_steps,
                )
            )
            elapsed += effective_warmup_steps

        updates_after_warmup = total_steps - effective_warmup_steps
        requested_decay_updates = decay_steps if decay_steps > 0 else 1
        decay_updates = min(max(1, requested_decay_updates), updates_after_warmup)
        stable_steps = updates_after_warmup - decay_updates

        if stable_steps > 0:
            if schedules:
                boundaries.append(elapsed)
            schedules.append(optax.constant_schedule(lr))
            elapsed += stable_steps

        if schedules:
            boundaries.append(elapsed)
        schedules.append(
            _zero_ending_linear(init_value=lr, end_value=0.0, num_updates=decay_updates)
        )

        if len(schedules) == 1:
            return schedules[0]
        return optax.join_schedules(schedules, boundaries=boundaries)

    raise ValueError(f"Unknown optimizer schedule '{cfg.optim.schedule}'.")


def _scale_learning_rate(lr: Any, factor: float):
    factor = float(factor)
    if callable(lr):

        def _schedule(step):
            return lr(step) * factor

        return _schedule
    return float(lr) * factor


def _make_base_tx(cfg, learning_rate) -> optax.GradientTransformation:
    name = cfg.optim.name.lower()
    wd = float(cfg.optim.weight_decay)

    if name == "adam":
        return optax.adam(
            learning_rate,
            b1=cfg.optim.b1,
            b2=cfg.optim.b2,
            eps=cfg.optim.eps,
        )
    if name == "adamw":
        return optax.adamw(
            learning_rate,
            b1=cfg.optim.b1,
            b2=cfg.optim.b2,
            eps=cfg.optim.eps,
            weight_decay=wd,
        )
    if name == "sgd":
        return optax.sgd(learning_rate, momentum=cfg.optim.b1)
    if name == "soap":
        from ._soap import soap

        return soap(
            learning_rate=learning_rate,
            b1=cfg.optim.b1,
            b2=cfg.optim.b2,
            eps=cfg.optim.eps,
            weight_decay=wd,
            precondition_frequency=cfg.optim.precondition_frequency,
            precondition_1d=cfg.optim.precondition_1d,
        )
    raise ValueError(f"Unknown optimizer '{cfg.optim.name}'.")


def _with_grad_clip(
    cfg, tx: optax.GradientTransformation
) -> optax.GradientTransformation:
    grad_clip = float(cfg.optim.grad_clip)
    if grad_clip <= 0.0:
        return tx
    return optax.chain(optax.clip_by_global_norm(grad_clip), tx)


def _transformer_payne_lr_factors(cfg) -> dict[str, float]:
    lr_scaling = cfg.optim.lr_scaling
    if lr_scaling not in {"mup", "mup_depth"}:
        raise ValueError("Only 'mup' and 'mup_depth' LR scaling are supported.")

    from ..models.transformer_payne import TransformerPayneConfig

    model_cfg = TransformerPayneConfig.from_dict(dict(cfg.model.params))
    if model_cfg.reference_width is None:
        raise ValueError(
            f"optim.lr_scaling={lr_scaling!r} requires model.params.reference_width."
        )
    width_scaling = float(model_cfg.dim) / float(model_cfg.reference_width)

    depth_scaling = 1.0
    if lr_scaling == "mup_depth":
        if model_cfg.reference_depth is None:
            raise ValueError(
                "optim.lr_scaling='mup_depth' requires model.params.reference_depth."
            )
        depth_scaling = (model_cfg.no_layers / float(model_cfg.reference_depth)) ** 0.5

    scale_embedding_lr = float(cfg.optim.scale_embedding_lr)
    scaled_width_lr = 1.0 / width_scaling
    scaled_width_depth_lr = scaled_width_lr / depth_scaling
    return {
        "emb_in": scale_embedding_lr,
        "emb_out": scale_embedding_lr * scaled_width_lr,
        "attn": scaled_width_depth_lr,
        "ff": scaled_width_depth_lr,
        "head": scaled_width_lr,
        "bias": 1.0,
        "default": 1.0,
    }


def _mlp_lr_factors(cfg) -> dict[str, float]:
    if cfg.optim.lr_scaling == "mup_depth":
        raise ValueError(
            "optim.lr_scaling='mup_depth' is only supported for transformer_payne."
        )
    if cfg.optim.lr_scaling != "mup":
        raise ValueError("Only 'mup' LR scaling is supported for mlp.")

    from ..models.mlp import MLPConfig

    model_cfg = MLPConfig.from_dict(dict(cfg.model.params))
    if model_cfg.reference_width is None:
        raise ValueError(
            "optim.lr_scaling='mup' requires model.params.reference_width."
        )
    if not model_cfg.hidden_sizes:
        raise ValueError(
            "optim.lr_scaling='mup' requires non-empty model.params.hidden_sizes."
        )

    width_scaling = float(model_cfg.hidden_sizes[0]) / float(model_cfg.reference_width)
    return {
        "mlp_kernel": 1.0 / width_scaling,
        "bias": 1.0,
        "default": 1.0,
    }


def _lr_scaling_factors(cfg) -> dict[str, float]:
    if cfg.model.name == "transformer_payne":
        return _transformer_payne_lr_factors(cfg)
    if cfg.model.name == "mlp":
        return _mlp_lr_factors(cfg)
    raise ValueError(
        "optim.lr_scaling is currently supported for mlp and transformer_payne."
    )


_BIAS_PARAM_NAMES = {
    "b",
    "b0",
    "b1",
    "bq",
    "bk",
    "bv",
    "bo",
    "be0",
    "be1",
    "b_q",
    "b_k",
    "b_v",
    "b_o",
    "bfi",
    "bfo",
    "bp0",
    "bp1",
    "bias",
}


def _label_transformer_payne_param_path(path: tuple[Any, ...]) -> str:
    if not path:
        return "default"
    root = path[0]
    leaf = str(path[-1])

    if leaf in _BIAS_PARAM_NAMES:
        return "bias"
    if root == "param_embedding":
        if leaf == "w0":
            return "emb_in"
        if leaf == "w1":
            return "emb_out"
    if root == "attn_layers" and leaf in {"wq", "wk", "wv", "wo"}:
        return "attn"
    if root == "ff_layers" and leaf == "w":
        return "ff"
    if root == "head":
        return "head"
    return "default"


def _label_transformer_payne_params(params: Any, path: tuple[Any, ...] = ()) -> Any:
    if isinstance(params, Mapping):
        return {
            key: _label_transformer_payne_params(value, (*path, key))
            for key, value in params.items()
        }
    if isinstance(params, list):
        return [
            _label_transformer_payne_params(value, (*path, idx))
            for idx, value in enumerate(params)
        ]
    if isinstance(params, tuple):
        return tuple(
            _label_transformer_payne_params(value, (*path, idx))
            for idx, value in enumerate(params)
        )
    return _label_transformer_payne_param_path(path)


def _label_mlp_param_path(path: tuple[Any, ...]) -> str:
    if not path:
        return "default"
    leaf = str(path[-1])
    if leaf in _BIAS_PARAM_NAMES:
        return "bias"
    if path[0] == "layers" and leaf == "kernel":
        return "mlp_kernel"
    return "default"


def _label_mlp_params(params: Any, path: tuple[Any, ...] = ()) -> Any:
    if isinstance(params, Mapping):
        return {
            key: _label_mlp_params(value, (*path, key)) for key, value in params.items()
        }
    if isinstance(params, list):
        return [
            _label_mlp_params(value, (*path, idx)) for idx, value in enumerate(params)
        ]
    if isinstance(params, tuple):
        return tuple(
            _label_mlp_params(value, (*path, idx)) for idx, value in enumerate(params)
        )
    return _label_mlp_param_path(path)


def _label_scaled_params(cfg, params: Any) -> Any:
    if cfg.model.name == "transformer_payne":
        return _label_transformer_payne_params(params)
    if cfg.model.name == "mlp":
        return _label_mlp_params(params)
    raise ValueError(
        "optim.lr_scaling is currently supported for mlp and transformer_payne."
    )


def make_tx(cfg, params: Any | None = None) -> optax.GradientTransformation:
    lr = make_learning_rate(cfg)

    if cfg.optim.lr_scaling is None:
        return _with_grad_clip(cfg, _make_base_tx(cfg, lr))

    if params is None:
        raise ValueError(
            "make_tx requires initialized params when optim.lr_scaling is set."
        )

    factors = _lr_scaling_factors(cfg)
    transforms = {
        label: _make_base_tx(cfg, _scale_learning_rate(lr, factor))
        for label, factor in factors.items()
    }
    labels = _label_scaled_params(cfg, params)
    return _with_grad_clip(cfg, optax.multi_transform(transforms, labels))


__all__ = ["make_learning_rate", "make_tx"]
