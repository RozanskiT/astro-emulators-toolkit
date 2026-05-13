from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from ..io_trees import iter_leaf_paths
from .common import (
    effective_sample_weight,
    reduce_over_non_batch_axes,
    sample_weight_from_batch,
    valid_mask_from_batch,
    weighted_masked_mean,
)


def _normalize_axis_tuple(rank: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        ax = int(axis)
        if ax < 0:
            ax = rank + ax
        if ax < 0 or ax >= rank:
            raise ValueError(f"metric axis {axis} is out of bounds for rank {rank}.")
        normalized.append(ax)
    return tuple(sorted(set(normalized)))


def _flatten_metric(name: str, value: Any) -> dict[str, Any]:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return {name: arr}
    flat = arr.reshape((-1,))
    return {f"{name}_{i}": flat[i] for i in range(flat.shape[0])}


def _reduce_metric(
    source, reduce_axes: tuple[int, ...], sample_weight, valid_mask, *, weighted: bool
):
    non_batch_reduce_axes = tuple(ax for ax in reduce_axes if ax != 0)
    reduced = (
        jnp.mean(source, axis=non_batch_reduce_axes)
        if non_batch_reduce_axes
        else source
    )
    if weighted:
        return weighted_masked_mean(
            reduced, sample_weight=sample_weight, valid_mask=valid_mask
        )
    return weighted_masked_mean(reduced, sample_weight=None, valid_mask=valid_mask)


def _flatten_leaf_arrays(tree: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(tree, dict):
        raise ValueError(f"{field_name} must be a nested dict of prediction leaves.")
    flat = {path: value for path, value in iter_leaf_paths(tree)}
    if not flat:
        raise ValueError(f"{field_name} must contain at least one array leaf.")
    return flat


def _validate_matching_shapes(pred, y, *, where: str) -> None:
    pred_shape = tuple(jnp.shape(pred))
    y_shape = tuple(jnp.shape(y))
    if pred_shape != y_shape:
        raise ValueError(
            f"{where} shape mismatch: pred.shape={pred_shape} vs y.shape={y_shape}."
        )


def _regression_metric_set(
    pred, y, sample_weight, valid_mask, metric_names, metric_axes
):
    abs_err = jnp.abs(pred - y)
    err2 = (pred - y) ** 2
    metric_sources = {
        "mse": (err2, False),
        "mae": (abs_err, False),
        "weighted_mse": (err2, True),
        "weighted_mae": (abs_err, True),
    }
    out: dict[str, Any] = {}
    for metric_name in metric_names:
        source, weighted = metric_sources[metric_name]
        for suffix, axes in metric_axes:
            reduce_axes = (0,) + axes
            value = _reduce_metric(
                source, reduce_axes, sample_weight, valid_mask, weighted=weighted
            )
            base_name = metric_name if suffix == "global" else f"{metric_name}_{suffix}"
            out.update(_flatten_metric(base_name, value))
    return out


def _leaf_name(path: str) -> str:
    return path.split("/")[-1]


def _resolve_loss_weights(
    loss_weights: Mapping[str, float] | None, leaf_paths: tuple[str, ...]
) -> dict[str, float]:
    if loss_weights is None:
        return {path: 1.0 for path in leaf_paths}

    leaf_name_matches: dict[str, list[str]] = {}
    for path in leaf_paths:
        leaf_name_matches.setdefault(_leaf_name(path), []).append(path)

    resolved: dict[str, float] = {}
    for raw_key, raw_weight in loss_weights.items():
        key = str(raw_key)
        if key in leaf_paths:
            resolved_path = key
        else:
            matches = leaf_name_matches.get(key, [])
            if not matches:
                raise ValueError(
                    f"loss_weights key '{key}' does not match any prediction leaf."
                )
            if len(matches) != 1:
                raise ValueError(
                    f"loss_weights key '{key}' is ambiguous; use one of {sorted(matches)}."
                )
            resolved_path = matches[0]
        if resolved_path in resolved:
            raise ValueError(
                f"loss_weights keys must resolve uniquely; duplicate mapping for '{resolved_path}'."
            )
        resolved[resolved_path] = float(raw_weight)

    for path in leaf_paths:
        resolved.setdefault(path, 1.0)
    return resolved


def _weighted_leaf_mean(
    values_by_path: Mapping[str, Any], leaf_weights: Mapping[str, float]
):
    weighted_sum = None
    total_weight = None
    for path, value in values_by_path.items():
        arr = jnp.asarray(value)
        weight = jnp.asarray(float(leaf_weights[path]), dtype=arr.dtype)
        weighted_sum = (
            arr * weight if weighted_sum is None else weighted_sum + arr * weight
        )
        total_weight = weight if total_weight is None else total_weight + weight
    if weighted_sum is None or total_weight is None:
        raise ValueError("Expected at least one leaf metric to aggregate.")
    return weighted_sum / jnp.maximum(
        total_weight, jnp.asarray(1e-12, dtype=weighted_sum.dtype)
    )


@dataclass(frozen=True)
class _RegressionEvalState:
    weighted_sums: dict[str, Any]
    weights: dict[str, Any]


@dataclass(frozen=True)
class RegressionTaskConfig:
    loss: str = "mse"
    loss_weights: Mapping[str, float] | None = None
    metrics: tuple[str, ...] = ("mse", "mae")
    metric_axes: Mapping[str, str | int | list[int] | tuple[int, ...]] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegressionTaskConfig":
        allowed = {"loss", "loss_weights", "metrics", "metric_axes"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown regression task params: {unknown}.")
        lw = d.get("loss_weights", None)
        metrics = tuple(str(m) for m in d.get("metrics", ("mse", "mae")))
        ma = d.get("metric_axes", None)
        return cls(
            loss=str(d.get("loss", "mse")),
            loss_weights=lw,
            metrics=metrics,
            metric_axes=ma,
        )


class RegressionTask:
    name = "regression"
    prediction_kind = "value"

    def __init__(self, cfg: RegressionTaskConfig):
        self.cfg = cfg
        allowed_metrics = {"mse", "mae", "weighted_mse", "weighted_mae"}
        if cfg.loss not in allowed_metrics:
            raise ValueError(f"loss={cfg.loss!r} not implemented")
        unknown = [m for m in cfg.metrics if m not in allowed_metrics]
        if unknown:
            raise ValueError(
                f"Unsupported regression metrics: {unknown}. Allowed: {sorted(allowed_metrics)}"
            )
        self._metrics = tuple(cfg.metrics)

        raw_metric_axes = (
            cfg.metric_axes if cfg.metric_axes is not None else {"global": "all"}
        )
        if not raw_metric_axes:
            raise ValueError("metric_axes must not be empty.")
        parsed_axes: list[tuple[str, tuple[int, ...] | None]] = []
        for name, axis_spec in raw_metric_axes.items():
            suffix = str(name)
            if suffix == "":
                raise ValueError("metric_axes keys must be non-empty strings.")
            if isinstance(axis_spec, str):
                if axis_spec != "all":
                    raise ValueError("metric axis string must be 'all'.")
                parsed_axes.append((suffix, None))
                continue
            axis_tuple: tuple[int, ...]
            if isinstance(axis_spec, int):
                axis_tuple = (int(axis_spec),)
            else:
                axis_tuple = tuple(int(ax) for ax in axis_spec)
            parsed_axes.append((suffix, axis_tuple))
        self._raw_metric_axes = tuple(parsed_axes)

        self._loss_weights = None
        if cfg.loss_weights is not None:
            self._loss_weights = {
                str(key): float(value) for key, value in cfg.loss_weights.items()
            }

    def _metric_axes_for_rank(self, rank: int):
        non_batch_rank = max(rank - 1, 0)
        resolved = []
        for suffix, axes in self._raw_metric_axes:
            if axes is None:
                resolved_axes = tuple(range(1, rank))
            else:
                normalized_non_batch = _normalize_axis_tuple(non_batch_rank, axes)
                resolved_axes = tuple(ax + 1 for ax in normalized_non_batch)
            resolved.append((suffix, resolved_axes))
        return tuple(resolved)

    def _loss_value(self, pred, y, sample_weight, valid_mask):
        if self.cfg.loss in {"mse", "weighted_mse"}:
            per_sample = reduce_over_non_batch_axes((pred - y) ** 2)
        else:
            per_sample = reduce_over_non_batch_axes(jnp.abs(pred - y))
        use_w = sample_weight if self.cfg.loss.startswith("weighted_") else None
        return weighted_masked_mean(
            per_sample, sample_weight=use_w, valid_mask=valid_mask
        )

    def loss_and_metrics(self, pred, batch):
        y = batch["y"]
        valid_mask = valid_mask_from_batch(batch)
        w = sample_weight_from_batch(batch, batch_size=valid_mask.shape[0])

        if not isinstance(pred, dict):
            if isinstance(y, dict):
                raise ValueError(
                    "If pred is array-like, batch['y'] must also be array-like."
                )
            _validate_matching_shapes(pred, y, where="regression")
            scalar_metrics = _regression_metric_set(
                pred,
                y,
                w,
                valid_mask,
                self._metrics,
                self._metric_axes_for_rank(pred.ndim),
            )
            loss = self._loss_value(pred, y, w, valid_mask)
            scalar_metrics[self.cfg.loss] = loss
            return loss, scalar_metrics

        if not isinstance(y, dict):
            raise ValueError(
                "If pred is dict, batch['y'] must be dict with matching keys."
            )
        pred_leaves = _flatten_leaf_arrays(pred, field_name="pred")
        y_leaves = _flatten_leaf_arrays(y, field_name="batch['y']")
        pred_paths = tuple(pred_leaves)
        if set(pred_paths) != set(y_leaves):
            raise ValueError("pred and batch['y'] must have identical leaf paths.")

        leaf_weights = _resolve_loss_weights(self._loss_weights, pred_paths)

        leaf_losses: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        global_metrics_by_name: dict[str, dict[str, Any]] = {}
        for path in pred_paths:
            pred_leaf = pred_leaves[path]
            y_leaf = y_leaves[path]
            _validate_matching_shapes(pred_leaf, y_leaf, where=f"regression['{path}']")

            leaf_loss = self._loss_value(pred_leaf, y_leaf, w, valid_mask)
            leaf_losses[path] = leaf_loss

            leaf_metrics = _regression_metric_set(
                pred_leaf,
                y_leaf,
                w,
                valid_mask,
                self._metrics,
                self._metric_axes_for_rank(jnp.asarray(pred_leaf).ndim),
            )
            for name, value in leaf_metrics.items():
                metrics[f"{name}/{path}"] = value
                global_metrics_by_name.setdefault(name, {})[path] = value

        for name, values_by_path in global_metrics_by_name.items():
            metrics[name] = _weighted_leaf_mean(values_by_path, leaf_weights)

        loss = _weighted_leaf_mean(leaf_losses, leaf_weights)
        metrics[self.cfg.loss] = loss
        return loss, metrics

    def init_eval_state(self):
        return _RegressionEvalState(weighted_sums={}, weights={})

    def update_eval_state(self, state: _RegressionEvalState, pred, batch):
        loss, metrics = self.loss_and_metrics(pred, batch)
        metrics = {"loss": loss, **metrics}
        valid_mask = valid_mask_from_batch(batch)
        sample_weight = sample_weight_from_batch(batch, batch_size=valid_mask.shape[0])
        weighted = jnp.sum(
            effective_sample_weight(sample_weight=sample_weight, valid_mask=valid_mask)
        )
        unweighted = jnp.sum(valid_mask)

        weighted_sums = dict(state.weighted_sums)
        weights = dict(state.weights)
        for name, value in metrics.items():
            is_weighted = name.startswith("weighted_") or (
                name == "loss" and self.cfg.loss.startswith("weighted_")
            )
            denom = weighted if is_weighted else unweighted
            v = jnp.asarray(value)
            d = jnp.asarray(denom, dtype=v.dtype)
            weighted_sums[name] = weighted_sums.get(name, jnp.zeros_like(v)) + v * d
            weights[name] = weights.get(name, jnp.array(0.0, dtype=v.dtype)) + d
        return _RegressionEvalState(weighted_sums=weighted_sums, weights=weights)

    def finalize_eval(self, state: _RegressionEvalState):
        return {
            k: v / jnp.maximum(state.weights[k], 1e-12)
            for k, v in state.weighted_sums.items()
        }

    def postprocess_pred(self, pred):
        return pred
