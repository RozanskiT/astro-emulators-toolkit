from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
from jax import tree_util

from ...io_trees import iter_leaf_paths
from ...tasks.common import (
    effective_sample_weight,
    reduce_over_non_batch_axes,
    sample_weight_from_batch,
    valid_mask_from_batch,
)


def _mean_with_weight(values, weight):
    values = jnp.asarray(values)
    if weight is None:
        return jnp.mean(values)
    w = jnp.asarray(weight, dtype=values.dtype)
    return jnp.sum(values * w) / jnp.maximum(jnp.sum(w), 1e-12)


def _extract_single_prediction_leaf(value, *, field_name: str):
    if isinstance(value, dict):
        leaves = list(iter_leaf_paths(value))
        if len(leaves) != 1:
            raise ValueError(
                f"{field_name} must contain exactly one leaf, found {len(leaves)}."
            )
        return jnp.asarray(leaves[0][1])
    return jnp.asarray(value)


@dataclass(frozen=True)
class _ClassificationEvalState:
    weighted_sums: dict[str, Any]
    weights: dict[str, jnp.ndarray]


@dataclass(frozen=True)
class BinaryClassificationTaskConfig:
    positive_class_weight: float = 1.0
    decision_threshold: float = 0.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BinaryClassificationTaskConfig":
        allowed = {"positive_class_weight", "decision_threshold"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown classification task params: {unknown}.")
        return cls(
            positive_class_weight=float(d.get("positive_class_weight", 1.0)),
            decision_threshold=float(d.get("decision_threshold", 0.5)),
        )


class BinaryClassificationTask:
    name = "binary_classification"
    prediction_kind = "probability"

    def __init__(self, cfg: BinaryClassificationTaskConfig):
        self.cfg = cfg

    def loss_and_metrics(self, pred, batch):
        y = _extract_single_prediction_leaf(batch["y"], field_name="batch['y']")
        logits = _extract_single_prediction_leaf(pred, field_name="pred")

        if logits.shape != y.shape:
            raise ValueError(
                f"BinaryClassificationTask expected pred/y shape match, got {logits.shape} vs {y.shape}."
            )

        valid_mask = valid_mask_from_batch(batch, batch_size=y.shape[0])
        sample_weight = sample_weight_from_batch(batch, batch_size=y.shape[0])
        eff_weight = effective_sample_weight(
            sample_weight=sample_weight, valid_mask=valid_mask
        )

        pos_weight = float(self.cfg.positive_class_weight)
        per_elem_weight = 1.0 + (pos_weight - 1.0) * y
        per_elem_bce = optax_sigmoid_binary_cross_entropy(logits, y)
        per_sample_bce = reduce_over_non_batch_axes(per_elem_bce * per_elem_weight)
        loss = _mean_with_weight(per_sample_bce, eff_weight)

        probs = jnn.sigmoid(logits)
        preds = (probs >= self.cfg.decision_threshold).astype(y.dtype)
        per_sample_acc = reduce_over_non_batch_axes((preds == y).astype(jnp.float32))
        acc = _mean_with_weight(per_sample_acc, eff_weight)
        return loss, {"bce": loss, "accuracy": acc}

    def init_eval_state(self):
        return _ClassificationEvalState(weighted_sums={}, weights={})

    def update_eval_state(self, state: _ClassificationEvalState, pred, batch):
        loss, metrics = self.loss_and_metrics(pred, batch)
        metrics = {"loss": loss, **metrics}
        valid_mask = valid_mask_from_batch(batch)
        sample_weight = sample_weight_from_batch(batch, batch_size=valid_mask.shape[0])
        unweighted = jnp.sum(valid_mask)
        weighted = jnp.sum(
            effective_sample_weight(sample_weight=sample_weight, valid_mask=valid_mask)
        )

        weighted_sums = dict(state.weighted_sums)
        weights = dict(state.weights)
        for name, value in metrics.items():
            denom = weighted if name in {"loss", "bce", "accuracy"} else unweighted
            v = jnp.asarray(value)
            d = jnp.asarray(denom, dtype=v.dtype)
            weighted_sums[name] = weighted_sums.get(name, jnp.zeros_like(v)) + v * d
            weights[name] = weights.get(name, jnp.array(0.0, dtype=v.dtype)) + d
        return _ClassificationEvalState(weighted_sums=weighted_sums, weights=weights)

    def finalize_eval(self, state: _ClassificationEvalState):
        out = {}
        for name, value in state.weighted_sums.items():
            out[name] = value / jnp.maximum(state.weights[name], 1e-12)
        return out

    def postprocess_pred(self, pred):
        return tree_util.tree_map(lambda leaf: jnn.sigmoid(jnp.asarray(leaf)), pred)


def optax_sigmoid_binary_cross_entropy(logits, labels):
    x = logits
    z = labels
    return jnp.maximum(x, 0) - x * z + jnp.log1p(jnp.exp(-jnp.abs(x)))
