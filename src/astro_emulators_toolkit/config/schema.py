# src/astro_emulators_toolkit/config/schema.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..io_trees import (
    validate_metadata_tree_like,
    validate_minmax_values,
    validate_structure_tree,
)


SchemaDict = dict[str, Any]
CONFIG_SCHEMA_VERSION = 1


def _normalize_registry_name(name: str) -> str:
    return str(name).strip().lower()


def _canonicalize_registry_name(
    name: str, *, legacy_aliases: Mapping[str, str] | None = None
) -> str:
    normalized = _normalize_registry_name(name)
    if legacy_aliases is None:
        return normalized
    return _normalize_registry_name(legacy_aliases.get(normalized, normalized))


_LEGACY_MODEL_NAME_ALIASES: dict[str, str] = {}
_LEGACY_TASK_NAME_ALIASES: dict[str, str] = {}
_LEGACY_SOLVER_NAME_ALIASES: dict[str, str] = {}
_LEGACY_OPTIMIZER_NAME_ALIASES: dict[str, str] = {}
_LEGACY_SCHEDULE_NAME_ALIASES: dict[str, str] = {}


def _normalize_mapping_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_mapping_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_normalize_mapping_tree(v) for v in value)
    if isinstance(value, list):
        return [_normalize_mapping_tree(v) for v in value]
    return value


def _validate_optional_string_leaf(value: Any, path: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"metadata leaf '{path}' must be a string or None.")


def _validate_optional_string_sequence_leaf(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(
            f"metadata leaf '{path}' must be None or a sequence of strings."
        )


def _coerce_iotree_spec(
    value: "IOTreeSpec | Mapping[str, Any] | None", *, field_name: str
) -> "IOTreeSpec | None":
    if value is None:
        return None
    if isinstance(value, IOTreeSpec):
        return value
    if isinstance(value, Mapping):
        return IOTreeSpec(**dict(value))
    raise TypeError(f"{field_name} must be an IOTreeSpec or mapping.")


def _coerce_minmax_tree_spec(
    value: "MinMaxTreeSpec | Mapping[str, Any] | None",
    *,
    field_name: str,
    require_positive_span: bool = False,
) -> "MinMaxTreeSpec | None":
    if value is None:
        return None
    if isinstance(value, MinMaxTreeSpec):
        out = value
    elif isinstance(value, Mapping):
        out = MinMaxTreeSpec(**dict(value))
    else:
        raise TypeError(f"{field_name} must be a MinMaxTreeSpec or mapping.")
    validate_minmax_values(
        out.min_tree,
        out.max_tree,
        field_name=field_name,
        require_positive_span=require_positive_span,
    )
    return out


def _normalize_step_schedule(
    value: tuple[int, ...] | list[int] | None, *, field_name: str
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of integer steps or None.")

    normalized: list[int] = []
    seen: set[int] = set()
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{field_name}[{idx}] must be an integer step.")
        step = int(item)
        if step <= 0:
            raise ValueError(f"{field_name}[{idx}] must be > 0.")
        if step in seen:
            continue
        normalized.append(step)
        seen.add(step)
    return tuple(normalized)


def _normalize_optional_step_interval(
    value: int | None, *, field_name: str
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer step interval or None.")
    interval = int(value)
    if interval < 0:
        raise ValueError(f"{field_name} must be >= 0 or None.")
    if interval == 0:
        return None
    return interval


def _normalize_optional_nonnegative_int(
    value: int | None, *, field_name: str
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer or None.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be >= 0 or None.")
    return normalized


def _normalize_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer.")
    return int(value)


def _normalize_finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite float.")
    if isinstance(value, str):
        try:
            normalized = float(value.strip())
        except ValueError as exc:
            raise TypeError(f"{field_name} must be a finite float.") from exc
    elif isinstance(value, (int, float, np.integer, np.floating)):
        normalized = float(value)
    else:
        raise TypeError(f"{field_name} must be a finite float.")
    if not np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _normalize_table_selector(value: Any, *, field_name: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, str)):
        raise TypeError(
            f"{field_name} must be an integer column index or string column name."
        )
    if isinstance(value, str):
        return value
    return int(value)


@dataclass(frozen=True)
class NpyTableConfig:
    """Configuration for loading a supervised dataset from a single `.npy` table."""

    path: str
    inputs: tuple[int | str, ...]
    targets: tuple[int | str, ...]
    columns: tuple[str, ...] | None = None  # optional for 2D arrays
    memmap: bool = True
    dtype: str = "float32"

    def __post_init__(self):
        inputs = tuple(
            _normalize_table_selector(item, field_name=f"NpyTableConfig.inputs[{idx}]")
            for idx, item in enumerate(self.inputs)
        )
        targets = tuple(
            _normalize_table_selector(item, field_name=f"NpyTableConfig.targets[{idx}]")
            for idx, item in enumerate(self.targets)
        )
        if not inputs:
            raise ValueError("NpyTableConfig.inputs must be non-empty.")
        if not targets:
            raise ValueError("NpyTableConfig.targets must be non-empty.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)

        if self.columns is not None:
            columns = tuple(self.columns)
            if any(not isinstance(column, str) for column in columns):
                raise TypeError("NpyTableConfig.columns must be a sequence of strings.")
            if len(set(columns)) != len(columns):
                raise ValueError("NpyTableConfig.columns must be unique.")
            object.__setattr__(self, "columns", columns)
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(
                f"NpyTableConfig.dtype is not a valid NumPy dtype: {self.dtype!r}."
            ) from exc
        object.__setattr__(self, "dtype", dtype.name)


@dataclass(frozen=True)
class ModelSpec:
    """Registry key and primitive parameter mapping for the model family."""

    name: str  # e.g. "mlp"
    params: Mapping[str, Any] = field(default_factory=dict)
    init_hints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _canonicalize_registry_name(
                self.name, legacy_aliases=_LEGACY_MODEL_NAME_ALIASES
            ),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Registry key and primitive parameter mapping for the training task."""

    name: str  # e.g. "regression"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _canonicalize_registry_name(
                self.name, legacy_aliases=_LEGACY_TASK_NAME_ALIASES
            ),
        )


@dataclass(frozen=True)
class SolverConfig:
    """Training solver selection used by `Emulator.fit(...)`."""

    name: str = "auto"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        name = _canonicalize_registry_name(
            self.name, legacy_aliases=_LEGACY_SOLVER_NAME_ALIASES
        )
        object.__setattr__(self, "name", name)
        params = dict(self.params)
        if name == "closed_form_linear" and "ridge" in params:
            ridge = _normalize_finite_float(
                params["ridge"], field_name="solver.params.ridge"
            )
            if ridge < 0.0:
                raise ValueError("solver.params.ridge must be >= 0.")
            params["ridge"] = ridge
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and learning-rate schedule settings for gradient training."""

    name: str = "adamw"  # "adam" | "adamw" | "sgd" | "soap"
    lr: float = 1e-3
    lr_scaling: str | None = None  # None | "mup" | "mup_depth"
    scale_embedding_lr: float = 1.0
    schedule: str = "constant"  # "constant" | "cosine" | "wsd"
    warmup_steps: int = 0
    decay_steps: int = 0  # used by "wsd" (warmup-stable-decay)
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    precondition_frequency: int = 10
    precondition_1d: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _canonicalize_registry_name(
                self.name, legacy_aliases=_LEGACY_OPTIMIZER_NAME_ALIASES
            ),
        )
        lr = _normalize_finite_float(self.lr, field_name="optim.lr")
        if lr < 0.0:
            raise ValueError("optim.lr must be >= 0.")
        object.__setattr__(self, "lr", lr)

        lr_scaling = self.lr_scaling
        if lr_scaling is not None:
            lr_scaling = _normalize_registry_name(str(lr_scaling))
            if lr_scaling in {"", "none", "standard", "si"}:
                lr_scaling = None
            elif lr_scaling not in {"mup", "mup_depth"}:
                raise ValueError(
                    "optim.lr_scaling must be None, 'mup', or 'mup_depth'."
                )
        object.__setattr__(self, "lr_scaling", lr_scaling)

        scale_embedding_lr = _normalize_finite_float(
            self.scale_embedding_lr, field_name="optim.scale_embedding_lr"
        )
        if scale_embedding_lr < 0.0:
            raise ValueError("optim.scale_embedding_lr must be >= 0.")
        object.__setattr__(self, "scale_embedding_lr", scale_embedding_lr)

        weight_decay = _normalize_finite_float(
            self.weight_decay, field_name="optim.weight_decay"
        )
        if weight_decay < 0.0:
            raise ValueError("optim.weight_decay must be >= 0.")
        object.__setattr__(self, "weight_decay", weight_decay)

        grad_clip = _normalize_finite_float(
            self.grad_clip, field_name="optim.grad_clip"
        )
        if grad_clip < 0.0:
            raise ValueError("optim.grad_clip must be >= 0.")
        object.__setattr__(self, "grad_clip", grad_clip)

        b1 = _normalize_finite_float(self.b1, field_name="optim.b1")
        if b1 < 0.0 or b1 >= 1.0:
            raise ValueError("optim.b1 must satisfy 0 <= b1 < 1.")
        object.__setattr__(self, "b1", b1)

        b2 = _normalize_finite_float(self.b2, field_name="optim.b2")
        if b2 < 0.0 or b2 >= 1.0:
            raise ValueError("optim.b2 must satisfy 0 <= b2 < 1.")
        object.__setattr__(self, "b2", b2)

        eps = _normalize_finite_float(self.eps, field_name="optim.eps")
        if eps <= 0.0:
            raise ValueError("optim.eps must be > 0.")
        object.__setattr__(self, "eps", eps)

        precondition_frequency = _normalize_int(
            self.precondition_frequency,
            field_name="optim.precondition_frequency",
        )
        if precondition_frequency <= 0:
            raise ValueError("optim.precondition_frequency must be > 0.")
        object.__setattr__(
            self,
            "precondition_frequency",
            precondition_frequency,
        )
        object.__setattr__(
            self,
            "schedule",
            _canonicalize_registry_name(
                self.schedule, legacy_aliases=_LEGACY_SCHEDULE_NAME_ALIASES
            ),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Step-based training-loop settings, logging, evaluation, and checkpointing."""

    workdir: str = "./runs/astroemu_run"
    batch_size: int = 1024
    # total target optimization steps; resume=True trains until state.step reaches num_steps
    num_steps: int = 1000
    val_fraction: float = 0.1

    shuffle: bool = True
    shuffle_seed: int = 0

    logging_interval_steps: int | None = 50
    # optional absolute optimization steps to emit training-loss/history callbacks
    # when set, these steps are unioned with periodic logging_interval_steps
    logging_steps: tuple[int, ...] | None = None
    evaluation_interval_steps: int | None = 500
    # optional absolute optimization steps to run validation on; unioned with evaluation_interval_steps
    evaluation_steps: tuple[int, ...] | None = None
    checkpoint_interval_steps: int | None = 500
    # optional absolute optimization steps to checkpoint on; unioned with checkpoint_interval_steps
    checkpoint_steps: tuple[int, ...] | None = None
    # None means preserve all saved checkpoints
    max_saved_checkpoints: int | None = 5

    # if None: inferred from dataset length (ceil(len / batch_size))
    # used to infer epoch boundaries/metrics while training is step-based
    steps_per_epoch: int | None = None

    def __post_init__(self):
        batch_size = _normalize_int(self.batch_size, field_name="training.batch_size")
        if batch_size <= 0:
            raise ValueError("training.batch_size must be > 0.")
        object.__setattr__(self, "batch_size", batch_size)

        num_steps = _normalize_int(self.num_steps, field_name="training.num_steps")
        if num_steps < 0:
            raise ValueError("training.num_steps must be >= 0.")
        object.__setattr__(self, "num_steps", num_steps)

        val_fraction = _normalize_finite_float(
            self.val_fraction,
            field_name="training.val_fraction",
        )
        if val_fraction < 0.0 or val_fraction > 1.0:
            raise ValueError(
                "training.val_fraction must satisfy 0 <= val_fraction <= 1."
            )
        object.__setattr__(self, "val_fraction", val_fraction)

        shuffle_seed = _normalize_int(
            self.shuffle_seed,
            field_name="training.shuffle_seed",
        )
        object.__setattr__(self, "shuffle_seed", shuffle_seed)

        if self.steps_per_epoch is not None:
            steps_per_epoch = _normalize_int(
                self.steps_per_epoch,
                field_name="training.steps_per_epoch",
            )
            if steps_per_epoch <= 0:
                raise ValueError("training.steps_per_epoch must be > 0 or None.")
            object.__setattr__(self, "steps_per_epoch", steps_per_epoch)

        object.__setattr__(
            self,
            "logging_interval_steps",
            _normalize_optional_step_interval(
                self.logging_interval_steps,
                field_name="training.logging_interval_steps",
            ),
        )
        object.__setattr__(
            self,
            "logging_steps",
            _normalize_step_schedule(
                self.logging_steps, field_name="training.logging_steps"
            ),
        )
        object.__setattr__(
            self,
            "evaluation_interval_steps",
            _normalize_optional_step_interval(
                self.evaluation_interval_steps,
                field_name="training.evaluation_interval_steps",
            ),
        )
        object.__setattr__(
            self,
            "evaluation_steps",
            _normalize_step_schedule(
                self.evaluation_steps, field_name="training.evaluation_steps"
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_interval_steps",
            _normalize_optional_step_interval(
                self.checkpoint_interval_steps,
                field_name="training.checkpoint_interval_steps",
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_steps",
            _normalize_step_schedule(
                self.checkpoint_steps, field_name="training.checkpoint_steps"
            ),
        )
        object.__setattr__(
            self,
            "max_saved_checkpoints",
            _normalize_optional_nonnegative_int(
                self.max_saved_checkpoints,
                field_name="training.max_saved_checkpoints",
            ),
        )


@dataclass(frozen=True)
class BundleConfig:
    """Default bundle save location settings for `Emulator.save_bundle(...)`."""

    # where Emulator.save_bundle() writes by default (relative to workdir)
    bundle_subdir: str = "bundle"


@dataclass(frozen=True)
class IOTreeSpec:
    """Tree-structured input or output spec with optional metadata sidecars."""

    structure_tree: dict[str, Any]
    channel_names_tree: dict[str, Any] | None = None
    leaf_units_tree: dict[str, Any] | None = None
    channel_units_tree: dict[str, Any] | None = None
    leaf_meanings_tree: dict[str, Any] | None = None
    channel_meanings_tree: dict[str, Any] | None = None

    def __post_init__(self):
        structure_tree = _normalize_mapping_tree(self.structure_tree)
        validate_structure_tree(structure_tree, field_name="io.structure_tree")
        object.__setattr__(self, "structure_tree", structure_tree)

        for field_name, leaf_validator in (
            ("channel_names_tree", _validate_optional_string_sequence_leaf),
            ("leaf_units_tree", _validate_optional_string_leaf),
            ("channel_units_tree", _validate_optional_string_sequence_leaf),
            ("leaf_meanings_tree", _validate_optional_string_leaf),
            ("channel_meanings_tree", _validate_optional_string_sequence_leaf),
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            normalized = _normalize_mapping_tree(raw_value)
            validate_metadata_tree_like(
                normalized,
                structure_tree,
                field_name=f"io.{field_name}",
                allow_sequences=True,
                leaf_validator=leaf_validator,
            )
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class MinMaxTreeSpec:
    """Min/max tree pair used for explicit scaling or input-domain metadata."""

    min_tree: dict[str, Any]
    max_tree: dict[str, Any]

    def __post_init__(self):
        min_tree = _normalize_mapping_tree(self.min_tree)
        max_tree = _normalize_mapping_tree(self.max_tree)
        validate_structure_tree(min_tree, field_name="io.min_tree")
        validate_structure_tree(max_tree, field_name="io.max_tree")
        validate_minmax_values(
            min_tree,
            max_tree,
            field_name="minmax_tree",
            require_positive_span=False,
        )
        object.__setattr__(self, "min_tree", min_tree)
        object.__setattr__(self, "max_tree", max_tree)


@dataclass(frozen=True)
class IOSpec:
    """Canonical input/output spec plus optional scaling and domain metadata."""

    inputs: IOTreeSpec | None = None
    outputs: IOTreeSpec | None = None
    reference_scaling_inputs: MinMaxTreeSpec | None = None
    reference_scaling_outputs: MinMaxTreeSpec | None = None
    input_domain: MinMaxTreeSpec | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "inputs", _coerce_iotree_spec(self.inputs, field_name="io.inputs")
        )
        object.__setattr__(
            self, "outputs", _coerce_iotree_spec(self.outputs, field_name="io.outputs")
        )
        object.__setattr__(
            self,
            "reference_scaling_inputs",
            _coerce_minmax_tree_spec(
                self.reference_scaling_inputs,
                field_name="io.reference_scaling_inputs",
                require_positive_span=True,
            ),
        )
        object.__setattr__(
            self,
            "reference_scaling_outputs",
            _coerce_minmax_tree_spec(
                self.reference_scaling_outputs,
                field_name="io.reference_scaling_outputs",
                require_positive_span=True,
            ),
        )
        object.__setattr__(
            self,
            "input_domain",
            _coerce_minmax_tree_spec(
                self.input_domain,
                field_name="io.input_domain",
                require_positive_span=False,
            ),
        )


@dataclass(frozen=True)
class HubConfig:
    """Optional metadata for download-only pretrained bundle lookup."""

    # optional metadata for download-only from_pretrained workflows
    repo_id: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class RootConfig:
    """Top-level JSON/YAML-friendly config for model, task, training, and I/O."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    seed: int = 0

    model: ModelSpec = field(default_factory=lambda: ModelSpec("mlp", {}))
    task: TaskSpec = field(default_factory=lambda: TaskSpec("regression", {}))
    solver: SolverConfig = field(default_factory=SolverConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    bundle: BundleConfig = field(default_factory=BundleConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    io: IOSpec = field(default_factory=IOSpec)

    def with_updates(self, **kwargs) -> "RootConfig":
        """Return a copy with selected top-level fields replaced."""
        return canonicalize_config_names(replace(self, **kwargs))


def canonicalize_config_names(cfg: RootConfig) -> RootConfig:
    return replace(
        cfg,
        model=replace(
            cfg.model,
            name=_canonicalize_registry_name(
                cfg.model.name, legacy_aliases=_LEGACY_MODEL_NAME_ALIASES
            ),
        ),
        task=replace(
            cfg.task,
            name=_canonicalize_registry_name(
                cfg.task.name, legacy_aliases=_LEGACY_TASK_NAME_ALIASES
            ),
        ),
        solver=replace(
            cfg.solver,
            name=_canonicalize_registry_name(
                cfg.solver.name, legacy_aliases=_LEGACY_SOLVER_NAME_ALIASES
            ),
        ),
        optim=replace(
            cfg.optim,
            name=_canonicalize_registry_name(
                cfg.optim.name, legacy_aliases=_LEGACY_OPTIMIZER_NAME_ALIASES
            ),
            schedule=_canonicalize_registry_name(
                cfg.optim.schedule, legacy_aliases=_LEGACY_SCHEDULE_NAME_ALIASES
            ),
        ),
    )
