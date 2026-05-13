from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ..config.io import save_config
from ..data.loader import DataLoader
from ..data.protocols import (
    Batch,
    DeviceBatchTransformLike,
    call_device_batch_transform,
)
from ..utils.tree import logs_device_to_python, to_jax_tree
from ..training.state import TrainState
from .callbacks import Callback, History, ModelCheckpoint, _HistoryCallback
from .paths import RUN_CONFIG_FILENAME


@dataclass
class FitResult:
    """Training result containing metric history and the final train state."""

    history: History
    state: TrainState


class _CheckpointProxy:
    def create_manager(self, *args, **kwargs):
        from ..training import checkpoint as ckpt_module

        return ckpt_module.create_manager(*args, **kwargs)

    def save(self, *args, **kwargs):
        from ..training import checkpoint as ckpt_module

        return ckpt_module.save(*args, **kwargs)

    def restore_latest(self, *args, **kwargs):
        from ..training import checkpoint as ckpt_module

        return ckpt_module.restore_latest(*args, **kwargs)


ckpt = _CheckpointProxy()


def _to_jax_batch(batch: Batch) -> Batch:
    return cast(Batch, to_jax_tree(batch))


def _maybe_copy_to_host_async(tree: Any) -> Any:
    def _request_copy(x: Any) -> Any:
        if hasattr(x, "copy_to_host_async"):
            try:
                x.copy_to_host_async()
            except Exception:
                pass
        return x

    return jax.tree_util.tree_map(_request_copy, tree)


def _merge_batch_metadata(original_batch: Batch, transformed_batch: Batch) -> Batch:
    merged = dict(transformed_batch)
    for key in ("sample_weight", "valid_mask"):
        if key in original_batch and key not in merged:
            merged[key] = original_batch[key]
    return merged


def _build_eval_predict(*, graphdef, device_batch_transform):
    eval_transform_key = jax.random.key(0)

    @jax.jit
    def eval_predict(state: TrainState, batch: Batch):
        if device_batch_transform is not None:
            original_batch = batch
            transformed = call_device_batch_transform(
                device_batch_transform,
                batch,
                rng=eval_transform_key,
                train=False,
            )
            batch = _merge_batch_metadata(original_batch, transformed)

        full_state = nnx.merge_state(state.params, state.model_state)
        pred, _ = nnx.call((graphdef, full_state))(batch["x"], train=False, rngs=None)
        return pred, batch

    return eval_predict


def _build_step_selector(
    *,
    interval_steps: int | None,
    explicit_steps: tuple[int, ...] | None,
):
    step_set = (
        None
        if explicit_steps is None
        else frozenset(int(step) for step in explicit_steps)
    )
    interval = None if interval_steps is None else int(interval_steps)

    if interval is None and step_set is None:
        return lambda step: False

    def _should_run(step: int) -> bool:
        if interval is not None and step % interval == 0:
            return True
        if step_set is not None and step in step_set:
            return True
        return False

    return _should_run


def fit(
    *,
    cfg,
    graphdef,
    init_state: TrainState,
    task,
    tx: optax.GradientTransformation,
    train_dataset,
    val_dataset=None,
    callbacks: list[Callback] | None = None,
    resume: bool = False,
    max_steps: int | None = None,
    device_batch_transform: DeviceBatchTransformLike | None = None,
) -> FitResult:
    """Run the gradient-training loop and return the recorded history and state."""
    workdir = Path(cfg.training.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, workdir / RUN_CONFIG_FILENAME)

    history = History()
    cb: list[Callback] = [_HistoryCallback(history)]
    if callbacks:
        cb.extend(callbacks)

    model_checkpoint_callbacks = [c for c in cb if isinstance(c, ModelCheckpoint)]
    log_callbacks = [c for c in cb if not isinstance(c, ModelCheckpoint)]

    mngr = None
    if resume or model_checkpoint_callbacks:
        mngr = ckpt.create_manager(workdir, cfg=cfg)

    state = init_state
    if resume:
        if mngr is None:
            raise RuntimeError("Checkpoint manager is required when resume=True.")
        restored = ckpt.restore_latest(mngr, target=init_state)
        if restored is not None:
            state = restored

    @jax.jit
    def train_step(state: TrainState, batch: Batch):
        state, step_key = state.next_rng()
        model_key = step_key
        if device_batch_transform is not None:
            transform_key, model_key = jax.random.split(step_key)
            original_batch = batch
            transformed = call_device_batch_transform(
                device_batch_transform,
                batch,
                rng=transform_key,
                train=True,
            )
            batch = _merge_batch_metadata(original_batch, transformed)

        rngs = nnx.Rngs(dropout=model_key)

        def loss_fn(params, model_state):
            full_state = nnx.merge_state(params, model_state)
            pred, (_, new_full_state) = nnx.call((graphdef, full_state))(
                batch["x"], train=True, rngs=rngs
            )
            _, new_model_state = (
                nnx.split_state(new_full_state, nnx.Param, ...)
                if new_full_state is not None
                else (None, model_state)
            )
            loss, metrics = task.loss_and_metrics(pred, batch)
            return loss, (metrics, new_model_state)

        (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(state.params, state.model_state)
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + jnp.array(1, dtype=state.step.dtype),
            params=new_params,
            model_state=new_model_state,
            opt_state=new_opt_state,
        )
        return new_state, {"loss": loss, **metrics}

    eval_predict = _build_eval_predict(
        graphdef=graphdef, device_batch_transform=device_batch_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        seed=cfg.training.shuffle_seed,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    inferred_steps_per_epoch = max(
        1, int(np.ceil(len(train_dataset) / int(cfg.training.batch_size)))
    )
    steps_per_epoch = (
        inferred_steps_per_epoch
        if cfg.training.steps_per_epoch is None
        else int(cfg.training.steps_per_epoch)
    )
    if steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0.")

    num_steps = int(cfg.training.num_steps)
    if num_steps <= 0:
        raise ValueError("training.num_steps must be > 0.")

    should_emit_train_logs = _build_step_selector(
        interval_steps=cfg.training.logging_interval_steps,
        explicit_steps=cfg.training.logging_steps,
    )
    should_run_eval = _build_step_selector(
        interval_steps=cfg.training.evaluation_interval_steps,
        explicit_steps=cfg.training.evaluation_steps,
    )
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be > 0 when provided.")

    for c in cb:
        c.on_train_begin({})

    host_step = int(jax.device_get(state.step))
    effective_target_steps = (
        num_steps if max_steps is None else min(num_steps, host_step + int(max_steps))
    )
    completed_steps = host_step
    pending_train: tuple[int, dict[str, Any]] | None = None

    def _flush_pending_train(step: int, logs: dict[str, Any]) -> None:
        if not should_emit_train_logs(step):
            return
        logs_python = logs_device_to_python(logs)
        for callback in log_callbacks:
            callback.on_train_batch_end(step, logs_python)

    while completed_steps < effective_target_steps:
        epoch = completed_steps // steps_per_epoch
        for c in cb:
            c.on_epoch_begin(epoch, {})

        epoch_end = min((epoch + 1) * steps_per_epoch, effective_target_steps)
        while completed_steps < epoch_end:
            batch = _to_jax_batch(train_loader.train_batch(completed_steps))
            state, logs = train_step(state, batch)
            host_step += 1
            logs = _maybe_copy_to_host_async(logs)

            for c in model_checkpoint_callbacks:
                if mngr is None or not c.should_save(host_step):
                    continue
                path = ckpt.save(
                    mngr, host_step, state, custom_metadata={"step": int(host_step)}
                )
                if path is not None:
                    c.on_checkpoint_end(host_step, path)

            if pending_train is not None:
                prev_step, prev_logs = pending_train
                _flush_pending_train(prev_step, prev_logs)
            pending_train = (host_step, logs)

            if val_loader is not None and should_run_eval(host_step):
                if all(
                    hasattr(task, n)
                    for n in ("init_eval_state", "update_eval_state", "finalize_eval")
                ):
                    eval_state = task.init_eval_state()
                    for vb_np in val_loader.iter_eval_batches():
                        vb = _to_jax_batch(vb_np)
                        pred, vb = eval_predict(state, vb)
                        eval_state = task.update_eval_state(eval_state, pred, vb)
                    agg = logs_device_to_python(
                        jax.device_get(task.finalize_eval(eval_state))
                    )
                else:
                    sums: dict[str, float] = {}
                    denom = 0.0
                    for vb_np in val_loader.iter_eval_batches():
                        vb = _to_jax_batch(vb_np)
                        pred, vb = eval_predict(state, vb)
                        loss, metrics = task.loss_and_metrics(pred, vb)
                        logs_eval = {"loss": loss, **metrics}
                        w = float(np.asarray(vb_np["valid_mask"]).sum())
                        denom += w
                        h = logs_device_to_python(logs_eval)
                        for k, v in h.items():
                            sums[k] = sums.get(k, 0.0) + float(v) * w
                    agg = {k: v / max(denom, 1e-12) for k, v in sums.items()}
                for c in cb:
                    c.on_eval_end(host_step, agg)

            completed_steps += 1

        if pending_train is not None:
            prev_step, prev_logs = pending_train
            _flush_pending_train(prev_step, prev_logs)
            pending_train = None

        for c in cb:
            c.on_epoch_end(epoch, {})

    final_step = int(jax.device_get(state.step))
    for c in cb:
        c.on_train_end({"step": final_step})

    if mngr is not None:
        for c in model_checkpoint_callbacks:
            if c.save_on_train_end:
                path = ckpt.save(
                    mngr, final_step, state, custom_metadata={"step": int(final_step)}
                )
                if path is not None:
                    c.on_checkpoint_end(final_step, path)
        mngr.wait_until_finished()
        mngr.close()

    return FitResult(history=history, state=state)
