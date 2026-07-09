from __future__ import annotations

from pathlib import Path
from typing import Any

import jax

from ..config.schema import RootConfig


def _require_orbax():
    import orbax.checkpoint as ocp

    return ocp


def checkpoints_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "checkpoints"


def create_manager(workdir: str | Path, *, cfg: RootConfig):
    ocp = _require_orbax()
    ckpt_dir = checkpoints_dir(workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=(
            None
            if cfg.training.max_saved_checkpoints is None
            else int(cfg.training.max_saved_checkpoints)
        ),
        create=True,
        enable_async_checkpointing=bool(cfg.training.async_checkpointing),
    )
    metadata = {
        "tool": "astro_emulators_toolkit",
        "schema_version": int(cfg.schema_version),
    }
    return ocp.CheckpointManager(ckpt_dir, options=options, metadata=metadata)


def save(
    mngr, step: int, train_state, *, custom_metadata: dict[str, Any] | None = None
) -> str | None:
    """Request a checkpoint save; return the step path, or None if declined.

    Orbax declines saves it considers redundant (e.g. a step at or before the
    latest saved step). Saves may complete asynchronously after this returns.
    """
    ocp = _require_orbax()
    state_for_save = train_state.replace(
        rng_key=jax.random.key_data(train_state.rng_key)
    )
    saved = mngr.save(
        int(step),
        args=ocp.args.StandardSave(state_for_save),
        custom_metadata=custom_metadata,
    )
    if saved is False:
        return None
    return str(mngr.directory / str(step))


def latest_step(mngr) -> int | None:
    step = mngr.latest_step()
    return None if step is None else int(step)


def latest_custom_metadata(mngr) -> dict[str, Any] | None:
    """Return the custom metadata saved with the latest checkpoint, if any.

    Defensive by design: checkpoint managers used in tests (and older orbax
    versions) may not expose step metadata, in which case ``None`` is returned
    rather than failing the resume path.
    """
    try:
        step = mngr.latest_step()
        if step is None:
            return None
        meta = mngr.metadata(int(step))
    except Exception:
        return None
    custom = getattr(meta, "custom_metadata", None)
    if isinstance(custom, dict):
        return dict(custom)
    return None


def _restore_step(mngr, step: int, target):
    ocp = _require_orbax()
    target_for_restore = target.replace(rng_key=jax.random.key_data(target.rng_key))
    restored = mngr.restore(
        int(step), args=ocp.args.StandardRestore(target_for_restore)
    )
    return restored.replace(rng_key=jax.random.wrap_key_data(restored.rng_key))


def restore(mngr, step: int, target):
    return _restore_step(mngr, step, target)


def restore_latest(mngr, target):
    step = mngr.latest_step()
    if step is None:
        return None
    return _restore_step(mngr, step, target)
