# src/astro_emulators_toolkit/training/callbacks.py
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

if TYPE_CHECKING:
    from ..config.schema import RootConfig


class Callback:
    """Base interface for training lifecycle callbacks."""

    # Keras-ish lifecycle
    def on_train_begin(self, logs: dict[str, Any] | None = None):
        """Called once before any training steps run."""
        ...

    def on_train_end(self, logs: dict[str, Any] | None = None):
        """Called once after training finishes."""
        ...

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called when a new logical epoch begins."""
        ...

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called when a logical epoch finishes."""
        ...

    def on_train_batch_end(self, step: int, logs: dict[str, Any]):
        """Called after a training step emits host-side logs."""
        ...

    def on_eval_end(self, step: int, logs: dict[str, Any]):
        """Called after validation metrics are aggregated for a step."""
        ...

    def on_checkpoint_end(self, step: int, path: str):
        """Called after trainer-managed checkpoint saving completes."""
        ...


@dataclass
class History:
    """In-memory record of scalar training and validation metrics."""

    logs: dict[str, list[float]]

    def __init__(self):
        self.logs = {}

    def _append(self, key: str, value: Any):
        try:
            fv = float(value)
        except Exception:
            return
        self.logs.setdefault(key, []).append(fv)

    def add_train(self, step: int, logs: dict[str, Any]):
        """Append a training-step log record."""
        self.logs.setdefault("training_step", []).append(float(step))
        for k, v in logs.items():
            if v is None:
                continue
            name = k if k.startswith("training_") else f"training_{k}"
            self._append(name, v)

    def add_eval(self, step: int, logs: dict[str, Any]):
        """Append an evaluation-step log record."""
        self.logs.setdefault("validation_step", []).append(float(step))
        for k, v in logs.items():
            if v is None:
                continue
            name = k if k.startswith("validation_") else f"validation_{k}"
            self._append(name, v)


class _HistoryCallback(Callback):
    def __init__(self, history: History):
        self.history = history

    def on_train_batch_end(self, step: int, logs: dict[str, Any]):
        self.history.add_train(step, logs)

    def on_eval_end(self, step: int, logs: dict[str, Any]):
        self.history.add_eval(step, logs)


class _SingleCSVLogger(Callback):
    def __init__(
        self,
        path: str | Path,
        *,
        every_n_steps: int | None = None,
        split: Literal["train", "val"],
    ):
        self.path = Path(path)
        self.every_n_steps = _normalize_optional_interval(
            every_n_steps, field_name="CSVLogger every_n_steps"
        )
        self.split = split
        self._writer: csv.DictWriter[str] | None = None
        self._file: TextIO | None = None

    def on_train_begin(self, logs=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = None

    def _ensure_writer(self, row: dict[str, Any]):
        if self._writer is not None:
            return
        if self._file is None:
            raise RuntimeError(
                "CSVLogger file handle is not initialized. Call on_train_begin() first."
            )
        fieldnames = list(row.keys())
        self._writer = csv.DictWriter(
            self._file, fieldnames=fieldnames, extrasaction="ignore"
        )
        self._writer.writeheader()

    def _write(self, row: dict[str, Any]):
        if self._file is None:
            raise RuntimeError(
                "CSVLogger file handle is not initialized. Call on_train_begin() first."
            )
        self._ensure_writer(row)
        if self._writer is None:
            raise RuntimeError("CSVLogger writer is not initialized.")
        self._writer.writerow(row)
        self._file.flush()

    def on_train_batch_end(self, step: int, logs: dict[str, Any]):
        if self.split != "train":
            return
        if self.every_n_steps is not None and step % self.every_n_steps != 0:
            return
        row = {
            "step": int(step),
            **{k: float(v) for k, v in logs.items() if _is_number(v)},
        }
        self._write(row)

    def on_eval_end(self, step: int, logs: dict[str, Any]):
        if self.split != "val":
            return
        if self.every_n_steps is not None and step % self.every_n_steps != 0:
            return
        row = {
            "step": int(step),
            **{f"validation_{k}": float(v) for k, v in logs.items() if _is_number(v)},
        }
        self._write(row)

    def on_train_end(self, logs=None):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class CSVLogger(Callback):
    """Write scalar training and/or validation metrics to CSV files."""

    def __init__(
        self,
        path: str | Path,
        every_n_steps: int | None = None,
        split: Literal["both", "train", "val"] = "both",
    ):
        self.path = Path(path)
        self.every_n_steps = _normalize_optional_interval(
            every_n_steps, field_name="CSVLogger every_n_steps"
        )
        if split not in {"both", "train", "val"}:
            raise ValueError("CSVLogger split must be one of: both, train, val")
        self.split = split
        self._delegates: list[_SingleCSVLogger] = []

        if split == "both":
            base = self.path
            stem = base.stem
            suffix = base.suffix if base.suffix else ".csv"
            train_path = base.with_name(f"{stem}_train{suffix}")
            val_path = base.with_name(f"{stem}_val{suffix}")
            self._delegates = [
                _SingleCSVLogger(
                    train_path, every_n_steps=self.every_n_steps, split="train"
                ),
                _SingleCSVLogger(
                    val_path, every_n_steps=self.every_n_steps, split="val"
                ),
            ]
        else:
            self._delegates = [
                _SingleCSVLogger(
                    self.path, every_n_steps=self.every_n_steps, split=split
                )
            ]

    def on_train_begin(self, logs=None):
        for delegate in self._delegates:
            delegate.on_train_begin(logs)

    def on_train_batch_end(self, step: int, logs: dict[str, Any]):
        for delegate in self._delegates:
            delegate.on_train_batch_end(step, logs)

    def on_eval_end(self, step: int, logs: dict[str, Any]):
        for delegate in self._delegates:
            delegate.on_eval_end(step, logs)

    def on_train_end(self, logs=None):
        for delegate in self._delegates:
            delegate.on_train_end(logs)


class ProgressBarLogger(Callback):
    """Print lightweight training progress and validation summaries to stdout."""

    def __init__(
        self, total_steps: int | None = None, every_n_steps: int | None = None
    ):
        self.total_steps = total_steps
        self.every_n_steps = _normalize_optional_interval(
            every_n_steps, field_name="ProgressBarLogger every_n_steps"
        )
        self._t0: float | None = None
        self._last_print = 0

    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        self._last_print = 0

    def on_train_batch_end(self, step: int, logs: dict[str, Any]):
        if self.every_n_steps is not None and step % self.every_n_steps != 0:
            return
        if self._t0 is None:
            self._t0 = time.time()
        dt = time.time() - self._t0
        loss = logs.get("loss", None)
        msg = f"\rstep {step}"
        if self.total_steps is not None:
            msg += f"/{self.total_steps}"
        if loss is not None and _is_number(loss):
            msg += f"  loss={float(loss):.5g}"
        msg += f"  t={dt:.1f}s"
        print(msg, end="", flush=True)
        self._last_print = step

    def on_eval_end(self, step: int, logs: dict[str, Any]):
        if self.every_n_steps is not None and step % self.every_n_steps != 0:
            return
        parts = [f"{k}={float(v):.5g}" for k, v in logs.items() if _is_number(v)]
        print(f"\nval @ step {step}: " + ", ".join(parts), flush=True)

    def on_train_end(self, logs=None):
        print("", flush=True)


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _normalize_optional_interval(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer step interval or None.")
    interval = int(value)
    if interval <= 0:
        raise ValueError(f"{field_name} must be > 0 or None.")
    return interval


def _normalize_explicit_steps(
    value: tuple[int, ...] | list[int] | None, *, field_name: str
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of integer steps or None.")

    normalized: list[int] = []
    seen: set[int] = set()
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{field_name}[{idx}] must be an integer step.")
        step = int(item)
        if step <= 0:
            raise ValueError(f"{field_name}[{idx}] must be > 0.")
        if step in seen:
            continue
        normalized.append(step)
        seen.add(step)
    return tuple(normalized)


class ModelCheckpoint(Callback):
    """Checkpoint policy callback used by trainer-managed checkpoint saving."""

    def __init__(
        self,
        every_n_steps: int | None = 500,
        *,
        explicit_steps: tuple[int, ...] | list[int] | None = None,
        save_on_train_end: bool = True,
    ):
        self.every_n_steps = _normalize_optional_interval(
            every_n_steps, field_name="ModelCheckpoint every_n_steps"
        )
        self.explicit_steps = _normalize_explicit_steps(
            explicit_steps, field_name="ModelCheckpoint explicit_steps"
        )
        self._explicit_step_set = (
            None
            if self.explicit_steps is None
            else frozenset(int(step) for step in self.explicit_steps)
        )
        self.save_on_train_end = bool(save_on_train_end)

    def should_save(self, step: int) -> bool:
        """Return ``True`` when the configured schedule matches ``step``."""
        if self.every_n_steps is not None and step % self.every_n_steps == 0:
            return True
        if self._explicit_step_set is not None and int(step) in self._explicit_step_set:
            return True
        return False


def build_callbacks_from_config(cfg: "RootConfig") -> list[Callback]:
    """Build default progress and checkpoint callbacks from a root config."""
    callbacks: list[Callback] = []
    if (
        cfg.training.logging_interval_steps is not None
        or cfg.training.logging_steps is not None
    ):
        callbacks.append(
            ProgressBarLogger(
                total_steps=int(cfg.training.num_steps),
            )
        )
    if (
        cfg.training.checkpoint_interval_steps is not None
        or cfg.training.checkpoint_steps is not None
    ):
        callbacks.append(
            ModelCheckpoint(
                every_n_steps=cfg.training.checkpoint_interval_steps,
                explicit_steps=cfg.training.checkpoint_steps,
            )
        )
    return callbacks
