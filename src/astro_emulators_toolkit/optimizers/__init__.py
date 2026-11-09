from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optax


def make_learning_rate(cfg):
    lr = float(cfg.optim.lr)
    schedule_name = str(cfg.optim.schedule).lower()
    decay_steps = int(cfg.optim.decay_steps)
    total_steps = int(cfg.training.num_steps)

    if decay_steps < 0:
        raise ValueError("optim.decay_steps must be >= 0.")

    if schedule_name == "constant":
        return lr

    if total_steps <= 0:
        raise ValueError("training.num_steps must be > 0.")

    if cfg.optim.warmup_steps is None:
        # Automatic linear warmup over the first 10% of training.
        warmup_steps = int(round(0.1 * total_steps))
    else:
        warmup_steps = int(cfg.optim.warmup_steps)
    if warmup_steps < 0:
        raise ValueError("optim.warmup_steps must be >= 0.")

    # Endpoint convention: optax evaluates schedule(count) with count starting
    # at 0 for the first applied update, so trainer step k uses schedule(k-1).
    # Every applied step (counts 0 .. num_steps-1) gets a strictly positive LR;
    # decay reaches zero only at count == num_steps, which is never applied.
    # Warmup ramps lr/warmup, 2*lr/warmup, ..., lr over the first warmup steps.

    def _warmup_schedule(num_updates: int):
        if num_updates <= 1:
            return optax.constant_schedule(lr)
        return optax.linear_schedule(
            init_value=lr / num_updates,
            end_value=lr,
            transition_steps=num_updates - 1,
        )

    def _decay_to_zero_linear(num_updates: int):
        # Reaches 0 one count past the last applied update.
        return optax.linear_schedule(
            init_value=lr,
            end_value=0.0,
            transition_steps=max(1, num_updates),
        )

    effective_warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    if schedule_name == "cosine":
        cosine_updates = max(1, total_steps - effective_warmup_steps)
        cosine = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=cosine_updates,
        )
        if effective_warmup_steps > 0:
            return optax.join_schedules(
                [_warmup_schedule(effective_warmup_steps), cosine],
                boundaries=[effective_warmup_steps],
            )
        return cosine

    if schedule_name == "wsd":
        schedules = []
        boundaries = []
        elapsed = 0

        if effective_warmup_steps > 0:
            schedules.append(_warmup_schedule(effective_warmup_steps))
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
        schedules.append(_decay_to_zero_linear(decay_updates))

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
    if lr_scaling not in {"si", "mup", "mup_depth"}:
        raise ValueError("Only 'si', 'mup', and 'mup_depth' LR scaling are supported.")

    if lr_scaling == "si":
        # Legacy standard parameterization: embedding groups train at
        # lr * scale_embedding_lr (the output embedding is NOT width-divided,
        # unlike mup); all other groups keep the base learning rate.
        scale_embedding_lr = float(cfg.optim.scale_embedding_lr)
        return {
            "emb_in": scale_embedding_lr,
            "emb_out": scale_embedding_lr,
            "attn": 1.0,
            "ff": 1.0,
            "head": 1.0,
            "bias": 1.0,
            "default": 1.0,
        }

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
    if cfg.optim.lr_scaling in {"si", "mup_depth"}:
        raise ValueError(
            f"optim.lr_scaling={cfg.optim.lr_scaling!r} is only supported for "
            "transformer_payne."
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

    hidden_sizes = tuple(int(size) for size in model_cfg.hidden_sizes)
    if len(set(hidden_sizes)) != 1:
        raise ValueError(
            "optim.lr_scaling='mup' for mlp requires uniform hidden_sizes "
            f"(got {hidden_sizes}); the width factor is derived from a single "
            "hidden width."
        )
    # Faithful to the original recipe: ALL dense kernels are width-divided,
    # including the input and output layers (nonstandard MuP, deliberate).
    width_scaling = float(hidden_sizes[0]) / float(model_cfg.reference_width)
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


# Bias/vector leaf names used by aemu model families (kept at base LR).
_BIAS_PARAM_NAMES = {
    "b",
    "b0",
    "b1",
    "bq",
    "bk",
    "bv",
    "bo",
    "bias",
}

# Canonical wrappers (models/canonical_wrappers.py) store the trainable model
# under this attribute, so param trees built through Emulator.from_config are
# rooted at "core_model" rather than at the raw model's own modules.
_WRAPPER_ROOT_KEYS = frozenset({"core_model"})


def _strip_wrapper_prefix(path: tuple[Any, ...]) -> tuple[Any, ...]:
    idx = 0
    while idx < len(path) - 1 and str(path[idx]) in _WRAPPER_ROOT_KEYS:
        idx += 1
    return path[idx:]


def _label_transformer_payne_param_path(path: tuple[Any, ...]) -> str:
    path = _strip_wrapper_prefix(path)
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
    path = _strip_wrapper_prefix(path)
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


def _collect_label_counts(labels: Any) -> dict[str, int]:
    counts: dict[str, int] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)
        else:
            counts[str(node)] = counts.get(str(node), 0) + 1

    _walk(labels)
    return counts


def _validate_scaled_labels(cfg, factors: Mapping[str, float], labels: Any) -> None:
    counts = _collect_label_counts(labels)
    required = set(factors) - {"bias", "default"}
    missing = sorted(name for name in required if counts.get(name, 0) == 0)
    if missing:
        raise ValueError(
            f"optim.lr_scaling={cfg.optim.lr_scaling!r} for model "
            f"'{cfg.model.name}' matched no parameters for LR groups {missing} "
            f"(observed label counts: {counts}). The parameter tree layout is "
            "not recognized by the LR-group labeler, so scaled learning rates "
            "would silently not be applied."
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
    _validate_scaled_labels(cfg, factors, labels)
    return _with_grad_clip(cfg, optax.multi_transform(transforms, labels))


__all__ = ["make_learning_rate", "make_tx"]
