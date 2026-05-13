from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .._typing import PytreeDict
from ..data.loader import DataLoader
from ..data.protocols import (
    DeviceBatchTransformLike,
    IdentityDeviceBatchTransform,
)
from ..io_trees import iter_leaf_paths, set_leaf_by_path
from ..tasks.common import sample_weight_from_batch, valid_mask_from_batch
from .callbacks import History


@dataclass
class BackendFitResult:
    params: PytreeDict
    model_state: PytreeDict
    history: History
    method: str
    metadata: dict[str, Any] | None = None


def fit_with_gradient_backend(
    *,
    cfg,
    graphdef,
    init_state,
    task,
    tx,
    train_dataset,
    validation_dataset,
    callbacks,
    resume: bool,
    max_steps: int | None,
    device_batch_transform: DeviceBatchTransformLike | None,
) -> BackendFitResult:
    from .trainer import fit as gradient_fit

    result = gradient_fit(
        cfg=cfg,
        graphdef=graphdef,
        init_state=init_state,
        task=task,
        tx=tx,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        callbacks=callbacks,
        resume=resume,
        max_steps=max_steps,
        device_batch_transform=device_batch_transform,
    )
    return BackendFitResult(
        params=result.state.params,
        model_state=result.state.model_state,
        history=result.history,
        method="gradient",
    )


def _iterate_dataset_batches(dataset, *, batch_size: int):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    yield from loader.iter_eval_batches()


def _step_matches_schedule(
    *,
    step: int,
    interval_steps: int | None,
    explicit_steps: frozenset[int] | None,
) -> bool:
    if interval_steps is not None and step % int(interval_steps) == 0:
        return True
    if explicit_steps is not None and step in explicit_steps:
        return True
    return False


def _extract_single_array_leaf(
    value: Any, *, field_name: str
) -> tuple[str | None, jnp.ndarray]:
    if isinstance(value, dict):
        leaves = list(iter_leaf_paths(value))
        if len(leaves) != 1:
            raise ValueError(
                f"{field_name} must contain exactly one leaf for closed_form_linear, found {len(leaves)}."
            )
        path, leaf = leaves[0]
        return path, jnp.asarray(leaf)
    return None, jnp.asarray(value)


def _wrap_single_leaf(path: str | None, value: Any) -> Any:
    if path is None:
        return value
    tree: dict[str, Any] = {}
    set_leaf_by_path(tree, path, value)
    return tree


def _replace_single_leaf(
    tree: dict[str, Any], value: Any, *, field_name: str
) -> dict[str, Any]:
    leaves = list(iter_leaf_paths(tree))
    if len(leaves) != 1:
        raise ValueError(
            f"{field_name} must contain exactly one leaf to use closed_form_linear, found {len(leaves)}."
        )
    path, _ = leaves[0]
    updated = deepcopy(tree)
    set_leaf_by_path(updated, path, value)
    return updated


def _as_closed_form_target_matrix(y: Any, *, field_name: str) -> jnp.ndarray:
    arr = jnp.asarray(y)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            "closed_form_linear expects the target leaf to have shape (N,) or (N, C); "
            f"got {tuple(arr.shape)} for {field_name}."
        )
    return arr


def _replace_batch_y(
    batch: dict[str, Any], *, path: str | None, value: Any
) -> dict[str, Any]:
    updated = dict(batch)
    updated["y"] = _wrap_single_leaf(path, value)
    return updated


def fit_with_closed_form_linear_backend(
    *,
    cfg,
    params: dict[str, Any],
    task,
    train_dataset,
    validation_dataset,
    callbacks,
    design_matrix_fn: Callable[..., jnp.ndarray],
    design_matrix_kwargs: dict[str, Any] | None = None,
    ridge: float = 0.0,
    unregularized_columns: tuple[int, ...] = (),
    resume: bool = False,
    max_steps: int | None = None,
    device_batch_transform: DeviceBatchTransformLike | None = None,
) -> BackendFitResult:
    if resume:
        raise ValueError("closed_form_linear solver does not support resume=True.")
    if device_batch_transform is not None and not isinstance(
        device_batch_transform, IdentityDeviceBatchTransform
    ):
        raise ValueError(
            "closed_form_linear solver only supports device_batch_transform=None "
            "or IdentityDeviceBatchTransform()."
        )
    if max_steps is not None:
        raise ValueError("closed_form_linear solver does not support max_steps.")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value):
        raise ValueError("closed_form_linear solver ridge must be finite.")
    if ridge_value < 0.0:
        raise ValueError("closed_form_linear solver ridge must be >= 0.")

    history = History()
    callback_list = callbacks or []
    dm_kwargs = dict(design_matrix_kwargs or {})
    logging_steps = (
        None
        if cfg.training.logging_steps is None
        else frozenset(int(step) for step in cfg.training.logging_steps)
    )
    evaluation_steps = (
        None
        if cfg.training.evaluation_steps is None
        else frozenset(int(step) for step in cfg.training.evaluation_steps)
    )

    for cb in callback_list:
        cb.on_train_begin({})

    if len(train_dataset) <= 0:
        raise ValueError(
            "closed_form_linear solver requires a non-empty train dataset."
        )

    accum_A: jnp.ndarray | None = None
    accum_B: jnp.ndarray | None = None
    batch_size = max(1, min(1024, max(1, int(len(train_dataset)) // 2)))

    for train_batch in _iterate_dataset_batches(train_dataset, batch_size=batch_size):
        _, x_train = _extract_single_array_leaf(
            train_batch["x"], field_name="train_batch['x']"
        )
        _, y_train = _extract_single_array_leaf(
            train_batch["y"], field_name="train_batch['y']"
        )
        y_train = _as_closed_form_target_matrix(y_train, field_name="train_batch['y']")
        valid_mask = valid_mask_from_batch(train_batch, batch_size=x_train.shape[0])
        sample_weight = sample_weight_from_batch(
            train_batch, batch_size=x_train.shape[0]
        )
        eff_weight = valid_mask if sample_weight is None else sample_weight * valid_mask

        design = design_matrix_fn(x_train, **dm_kwargs)
        w_sqrt_col = jnp.sqrt(jnp.clip(eff_weight, min=0.0))[:, None]
        weighted_design = design * w_sqrt_col
        weighted_targets = y_train * w_sqrt_col

        batch_A = weighted_design.T @ weighted_design
        batch_B = weighted_design.T @ weighted_targets

        if accum_A is None:
            accum_A = batch_A
            accum_B = batch_B
        else:
            assert accum_B is not None
            accum_A = accum_A + batch_A
            accum_B = accum_B + batch_B

    if accum_A is None or accum_B is None:
        raise ValueError(
            "closed_form_linear solver received no batches from train dataset."
        )

    system_matrix = accum_A
    exempt_columns = tuple(sorted({int(i) for i in unregularized_columns}))
    for idx in exempt_columns:
        if idx < 0 or idx >= int(system_matrix.shape[0]):
            raise ValueError(
                "closed_form_linear received an out-of-range unregularized column "
                f"index {idx} for feature dimension {int(system_matrix.shape[0])}."
            )

    if ridge_value > 0.0:
        ridge_diag = jnp.ones((system_matrix.shape[0],), dtype=system_matrix.dtype)
        if exempt_columns:
            ridge_diag = ridge_diag.at[jnp.asarray(exempt_columns)].set(0.0)
        system_matrix = system_matrix + jnp.asarray(
            ridge_value, dtype=system_matrix.dtype
        ) * jnp.diag(ridge_diag)

    raw_cond = float(jnp.asarray(jnp.linalg.cond(system_matrix)))
    condition_number = raw_cond if np.isfinite(raw_cond) else None
    solve_backend = "solve"
    if not np.isfinite(raw_cond) or raw_cond > 1e8:
        coeff = jnp.linalg.lstsq(system_matrix, accum_B, rcond=1e-7)[0]
        solve_backend = "lstsq"
    else:
        try:
            coeff = jnp.linalg.solve(system_matrix, accum_B)
        except Exception:
            coeff = jnp.linalg.lstsq(system_matrix, accum_B, rcond=1e-7)[0]
            solve_backend = "lstsq"
    updated_params = _replace_single_leaf(params, coeff, field_name="params")

    train_eval_state = (
        task.init_eval_state()
        if all(
            hasattr(task, n)
            for n in ("init_eval_state", "update_eval_state", "finalize_eval")
        )
        else None
    )
    train_sums: dict[str, float] = {}
    train_denom = 0.0

    for train_batch in _iterate_dataset_batches(train_dataset, batch_size=batch_size):
        _, x_train = _extract_single_array_leaf(
            train_batch["x"], field_name="train_batch['x']"
        )
        y_path, y_train = _extract_single_array_leaf(
            train_batch["y"], field_name="train_batch['y']"
        )
        y_train = _as_closed_form_target_matrix(y_train, field_name="train_batch['y']")
        eval_batch = _replace_batch_y(train_batch, path=y_path, value=y_train)
        pred_train = _wrap_single_leaf(
            y_path, design_matrix_fn(x_train, **dm_kwargs) @ coeff
        )
        if train_eval_state is not None:
            train_eval_state = task.update_eval_state(
                train_eval_state, pred_train, eval_batch
            )
            continue

        loss, metrics = task.loss_and_metrics(pred_train, eval_batch)
        logs_eval = {"loss": loss, **metrics}
        w = float(jnp.asarray(valid_mask_from_batch(train_batch)).sum())
        train_denom += w
        for k, v in logs_eval.items():
            train_sums[k] = train_sums.get(k, 0.0) + float(v) * w

    if train_eval_state is not None:
        train_logs = {
            k: float(v) for k, v in task.finalize_eval(train_eval_state).items()
        }
    else:
        train_logs = {k: v / max(train_denom, 1e-12) for k, v in train_sums.items()}

    step = 1
    if _step_matches_schedule(
        step=step,
        interval_steps=cfg.training.logging_interval_steps,
        explicit_steps=logging_steps,
    ):
        history.add_train(step, train_logs)
        for cb in callback_list:
            cb.on_train_batch_end(step, train_logs)

    should_run_validation = _step_matches_schedule(
        step=step,
        interval_steps=cfg.training.evaluation_interval_steps,
        explicit_steps=evaluation_steps,
    )
    if (
        validation_dataset is not None
        and len(validation_dataset) > 0
        and should_run_validation
    ):
        val_batch_size = max(1, min(1024, max(1, int(len(validation_dataset)) // 2)))
        if all(
            hasattr(task, n)
            for n in ("init_eval_state", "update_eval_state", "finalize_eval")
        ):
            eval_state = task.init_eval_state()
            for val_batch in _iterate_dataset_batches(
                validation_dataset, batch_size=val_batch_size
            ):
                _, x_val = _extract_single_array_leaf(
                    val_batch["x"], field_name="val_batch['x']"
                )
                y_path, y_val = _extract_single_array_leaf(
                    val_batch["y"], field_name="val_batch['y']"
                )
                y_val = _as_closed_form_target_matrix(
                    y_val, field_name="val_batch['y']"
                )
                eval_batch = _replace_batch_y(val_batch, path=y_path, value=y_val)
                pred_val = _wrap_single_leaf(
                    y_path, design_matrix_fn(x_val, **dm_kwargs) @ coeff
                )
                eval_state = task.update_eval_state(eval_state, pred_val, eval_batch)
            val_logs = {k: float(v) for k, v in task.finalize_eval(eval_state).items()}
        else:
            sums: dict[str, float] = {}
            denom = 0.0
            for val_batch in _iterate_dataset_batches(
                validation_dataset, batch_size=val_batch_size
            ):
                _, x_val = _extract_single_array_leaf(
                    val_batch["x"], field_name="val_batch['x']"
                )
                y_path, y_val = _extract_single_array_leaf(
                    val_batch["y"], field_name="val_batch['y']"
                )
                y_val = _as_closed_form_target_matrix(
                    y_val, field_name="val_batch['y']"
                )
                eval_batch = _replace_batch_y(val_batch, path=y_path, value=y_val)
                pred_val = _wrap_single_leaf(
                    y_path, design_matrix_fn(x_val, **dm_kwargs) @ coeff
                )
                loss, metrics = task.loss_and_metrics(pred_val, eval_batch)
                logs_eval = {"loss": loss, **metrics}
                w = float(jnp.asarray(valid_mask_from_batch(val_batch)).sum())
                denom += w
                for k, v in logs_eval.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * w
            val_logs = {k: v / max(denom, 1e-12) for k, v in sums.items()}

        history.add_eval(step, val_logs)
        for cb in callback_list:
            cb.on_eval_end(step, val_logs)

    for cb in callback_list:
        cb.on_train_end({})

    return BackendFitResult(
        params=updated_params,
        model_state={},
        history=history,
        method="closed_form_linear",
        metadata={
            "diagnostics": {
                "condition_number": condition_number,
                "solution_backend": solve_backend,
                "num_train_examples": int(len(train_dataset)),
                "feature_dim": int(system_matrix.shape[0]),
                "target_dim": int(accum_B.shape[1]),
            }
        },
    )
