from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import numpy as np
from flax import nnx

from ._typing import PytreeDict
from .bundle.artifact import load_bundle_artifact, save_bundle_artifact
from .bundle.hub import get_cache_dir, snapshot_download
from .bundle.metadata import get_bundle_model_init
from .bundle.versions import WEIGHTS_LAYOUT
from .config.schema import RootConfig, canonicalize_config_names
from .emulator_runtime import (
    apply_jax_runtime,
    build_init_state,
    make_frozen_apply_runtime,
    to_numpy_pytree,
    validate_model_state_dict,
)
from .resolver import (
    build_model_from_name,
    build_task_from_name,
    get_model_entry_from_name,
    get_stable_model_entry_from_name,
    resolve_model_init_context,
    validate_model_io_compatibility,
)
from .io_trees import get_leaf_by_path, iter_leaf_paths, set_leaf_by_path
from .spec import materialize_effective_spec
from .data.protocols import (
    DeviceBatchTransformLike,
    init_batch_via_device_transform,
)
from .training.callbacks import History

if TYPE_CHECKING:
    import optax


def resolve_solver(*args, **kwargs):
    from .training.solvers import resolve_solver as _resolve_solver

    return _resolve_solver(*args, **kwargs)


def make_tx(cfg, **kwargs):
    from .optimizers import make_tx as _make_tx_impl

    return _make_tx_impl(cfg, **kwargs)


def _validate_bundle_weights_payload(
    metadata: dict[str, Any], params_pure: dict[str, Any]
) -> None:
    layout = metadata.get("weights_layout")
    if layout != WEIGHTS_LAYOUT:
        raise ValueError(f"Unsupported bundle weights_layout '{layout}'.")
    required_keys = ("params", "model_state")
    missing = [key for key in required_keys if key not in params_pure]
    if missing:
        raise ValueError(
            f"weights_layout '{layout}' is missing required weights keys: {missing}."
        )


def _display_tree_path(path: str) -> str:
    return path or "<root>"


def _normalize_tree_keys_for_bundle_validation(
    tree: Any, *, tree_name: str, path: str = ""
) -> Any:
    if not isinstance(tree, dict):
        return tree
    normalized: dict[str, Any] = {}
    for raw_key, value in tree.items():
        key = str(raw_key)
        if not key:
            raise ValueError(
                f"{tree_name} contains an empty key at '{_display_tree_path(path)}'."
            )
        if "/" in key:
            raise ValueError(
                f"{tree_name} contains key {raw_key!r} with '/' at '{_display_tree_path(path)}'."
            )
        if key in normalized:
            raise ValueError(
                f"{tree_name} key collision after string normalization at '{_display_tree_path(path)}': {raw_key!r}."
            )
        next_path = f"{path}/{key}" if path else key
        normalized[key] = _normalize_tree_keys_for_bundle_validation(
            value, tree_name=tree_name, path=next_path
        )
    return normalized


def _validate_loaded_tree_matches_initialized(
    *,
    initialized_tree: Any,
    loaded_tree: Any,
    tree_name: str,
    path: str = "",
) -> None:
    if isinstance(initialized_tree, dict):
        if not isinstance(loaded_tree, dict):
            raise ValueError(
                f"Loaded {tree_name} tree has a leaf at '{_display_tree_path(path)}' "
                f"where the initialized {tree_name} tree has a subtree."
            )
        initialized_keys = set(initialized_tree)
        loaded_keys = set(loaded_tree)
        if initialized_keys != loaded_keys:
            missing = sorted(initialized_keys - loaded_keys)
            extra = sorted(loaded_keys - initialized_keys)
            detail: list[str] = []
            if missing:
                detail.append(f"missing keys={missing}")
            if extra:
                detail.append(f"extra keys={extra}")
            raise ValueError(
                f"Loaded {tree_name} tree structure mismatch at '{_display_tree_path(path)}': {', '.join(detail)}."
            )
        for key in sorted(initialized_keys):
            next_path = f"{path}/{key}" if path else key
            _validate_loaded_tree_matches_initialized(
                initialized_tree=initialized_tree[key],
                loaded_tree=loaded_tree[key],
                tree_name=tree_name,
                path=next_path,
            )
        return
    if isinstance(loaded_tree, dict):
        raise ValueError(
            f"Loaded {tree_name} tree has a subtree at '{_display_tree_path(path)}' "
            f"where the initialized {tree_name} tree has a leaf."
        )

    initialized_arr = np.asarray(initialized_tree)
    loaded_arr = np.asarray(loaded_tree)
    initialized_shape = tuple(int(dim) for dim in initialized_arr.shape)
    loaded_shape = tuple(int(dim) for dim in loaded_arr.shape)
    if initialized_shape != loaded_shape:
        raise ValueError(
            f"Loaded {tree_name} leaf '{_display_tree_path(path)}' has shape "
            f"{loaded_shape}, expected {initialized_shape}."
        )
    initialized_dtype = np.dtype(initialized_arr.dtype)
    loaded_dtype = np.dtype(loaded_arr.dtype)
    if initialized_dtype != loaded_dtype:
        raise ValueError(
            f"Loaded {tree_name} leaf '{_display_tree_path(path)}' has dtype "
            f"{loaded_dtype}, expected {initialized_dtype}."
        )


def _validate_loaded_runtime_trees_match_initialization(
    *,
    initialized_params: PytreeDict,
    loaded_params: PytreeDict,
    initialized_model_state: PytreeDict,
    loaded_model_state: PytreeDict,
) -> None:
    normalized_initialized_params = _normalize_tree_keys_for_bundle_validation(
        initialized_params, tree_name="initialized parameter tree"
    )
    normalized_loaded_params = _normalize_tree_keys_for_bundle_validation(
        loaded_params, tree_name="loaded parameter tree"
    )
    _validate_loaded_tree_matches_initialized(
        initialized_tree=normalized_initialized_params,
        loaded_tree=normalized_loaded_params,
        tree_name="parameter",
    )

    normalized_initialized_model_state = _normalize_tree_keys_for_bundle_validation(
        initialized_model_state, tree_name="initialized model_state tree"
    )
    normalized_loaded_model_state = _normalize_tree_keys_for_bundle_validation(
        loaded_model_state, tree_name="loaded model_state tree"
    )
    _validate_loaded_tree_matches_initialized(
        initialized_tree=normalized_initialized_model_state,
        loaded_tree=normalized_loaded_model_state,
        tree_name="model_state",
    )


def _extract_init_example(init_example: dict[str, Any]) -> tuple[Any, Any | None]:
    if not isinstance(init_example, dict):
        raise TypeError(
            "init_example must be a mapping with 'inputs'/'outputs' or 'x'/'y'."
        )
    if "inputs" in init_example or "outputs" in init_example:
        return init_example.get("inputs"), init_example.get("outputs")
    return init_example.get("x"), init_example.get("y")


def _expand_scalar_target_leaf_for_init(outputs: Any) -> Any:
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        leaves = list(iter_leaf_paths(outputs))
        if len(leaves) != 1:
            return outputs
        path, leaf = leaves[0]
        arr = np.asarray(leaf)
        if arr.ndim != 1:
            return outputs
        updated: dict[str, Any] = {}
        set_leaf_by_path(updated, path, arr[:, None])
        return updated

    arr = np.asarray(outputs)
    if arr.ndim == 1:
        return arr[:, None]
    return outputs


def _validate_runtime_contract_metadata(
    metadata: dict[str, Any], *, cfg: RootConfig, model_init: dict[str, Any]
) -> None:
    runtime_contract = metadata.get("runtime_contract")
    if runtime_contract is None:
        return
    if not isinstance(runtime_contract, dict):
        raise ValueError(
            "Bundle metadata field 'runtime_contract' must be a dictionary when present."
        )
    new_required_keys = ("surface", "role_paths", "affine_leaf_specs")
    missing_new = [key for key in new_required_keys if key not in runtime_contract]
    if missing_new:
        raise ValueError(
            f"Bundle metadata field 'runtime_contract' is missing required keys: {missing_new}."
        )
    if not isinstance(runtime_contract.get("role_paths"), dict):
        raise ValueError(
            "Bundle metadata field 'runtime_contract.role_paths' must be a dictionary."
        )
    if "affine_leaf_specs" in runtime_contract and not isinstance(
        runtime_contract.get("affine_leaf_specs"), dict
    ):
        raise ValueError(
            "Bundle metadata field 'runtime_contract.affine_leaf_specs' must be a dictionary."
        )

    entry = get_model_entry_from_name(cfg.model.name)
    if entry is None or entry.runtime is None:
        return
    expected = entry.runtime.describe_runtime(
        cfg=cfg, spec=metadata["spec"], model_init=model_init
    )
    if runtime_contract != expected:
        raise ValueError(
            "Bundle metadata runtime_contract does not match the resolved model family runtime contract."
        )


def _spec_block(spec: dict[str, Any], field_name: str) -> dict[str, Any] | None:
    block = spec.get(field_name)
    return block if isinstance(block, dict) else None


def _describe_spec_metadata_presence(section: dict[str, Any] | None) -> str:
    if section is None:
        return "none"

    present: list[str] = []
    if isinstance(section.get("channel_names_tree"), dict):
        present.append("names")
    if any(
        isinstance(section.get(field_name), dict)
        for field_name in ("leaf_units_tree", "channel_units_tree")
    ):
        present.append("units")
    if any(
        isinstance(section.get(field_name), dict)
        for field_name in ("leaf_meanings_tree", "channel_meanings_tree")
    ):
        present.append("meanings")
    return ", ".join(present) if present else "none"


def _metadata_leaf(
    section: dict[str, Any] | None,
    metadata_tree_name: str,
    path: str,
) -> Any:
    if section is None:
        return None
    metadata_tree = section.get(metadata_tree_name)
    if not isinstance(metadata_tree, dict):
        return None
    try:
        return get_leaf_by_path(metadata_tree, path)
    except KeyError:
        return None


def _format_domain_number(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        return str(value)
    scalar = float(arr.reshape(-1)[0])
    return f"{scalar:g}"


def _format_domain_interval(min_value: Any, max_value: Any) -> str:
    return f"[{_format_domain_number(min_value)}, {_format_domain_number(max_value)}]"


def _format_domain_unit(unit: Any) -> str:
    if unit is None:
        return ""
    return f" {unit}"


def _domain_channel_units(
    input_spec: dict[str, Any] | None,
    path: str,
    index: int,
) -> str:
    channel_units = _metadata_leaf(input_spec, "channel_units_tree", path)
    if isinstance(channel_units, (list, tuple)) and index < len(channel_units):
        return _format_domain_unit(channel_units[index])
    return _format_domain_unit(_metadata_leaf(input_spec, "leaf_units_tree", path))


def _domain_leaf_lines(
    path: str,
    min_value: Any,
    max_value: Any,
    *,
    input_spec: dict[str, Any] | None,
) -> list[str]:
    channel_names = _metadata_leaf(input_spec, "channel_names_tree", path)
    min_arr = np.asarray(min_value)
    max_arr = np.asarray(max_value)
    if (
        isinstance(channel_names, (list, tuple))
        and min_arr.shape == max_arr.shape
        and min_arr.ndim >= 1
        and min_arr.shape[-1] == len(channel_names)
    ):
        min_channels = min_arr.reshape(-1, min_arr.shape[-1])[0]
        max_channels = max_arr.reshape(-1, max_arr.shape[-1])[0]
        lines = [f"  {path}:"]
        for idx, name in enumerate(channel_names):
            interval = _format_domain_interval(min_channels[idx], max_channels[idx])
            lines.append(
                f"    {name}: {interval}{_domain_channel_units(input_spec, path, idx)}"
            )
        return lines

    unit = _format_domain_unit(_metadata_leaf(input_spec, "leaf_units_tree", path))
    if min_arr.shape == max_arr.shape and min_arr.ndim == 1 and min_arr.size > 1:
        lines = [f"  {path}:"]
        for idx, (min_item, max_item) in enumerate(zip(min_arr, max_arr, strict=True)):
            lines.append(
                f"    [{idx}]: {_format_domain_interval(min_item, max_item)}{unit}"
            )
        return lines
    return [f"  {path}: {_format_domain_interval(min_value, max_value)}{unit}"]


def _validate_public_batched_input_tree(x: dict[str, Any], *, field_name: str) -> None:
    for path, value in iter_leaf_paths(x):
        raw_shape = getattr(value, "shape", None)
        shape = (
            tuple(int(dim) for dim in raw_shape)
            if raw_shape is not None
            else tuple(int(dim) for dim in np.asarray(value).shape)
        )
        if len(shape) == 0:
            raise ValueError(
                f"{field_name} leaf '{path}' must include an explicit leading batch "
                "axis and at least one non-batch axis, "
                f"got shape {shape}. For a single scalar example, use shape (1, 1)."
            )
        if len(shape) == 1:
            raise ValueError(
                f"{field_name} leaf '{path}' must include an explicit leading batch "
                "axis and at least one non-batch axis, "
                f"got shape {shape}. For a single example, use shape (1, {shape[0]}) instead of {shape}."
            )


class Emulator:
    """High-level facade for bundle inference, training, and portable save/load.

    The canonical user flows are:
    - load a saved bundle with :meth:`from_bundle` or :meth:`from_pretrained`
    - build from a :class:`~astro_emulators_toolkit.config.schema.RootConfig`
      with :meth:`from_config`
    - train with :meth:`fit`
    - run inference with :meth:`predict`, :meth:`apply_jax`, or
      :meth:`make_frozen_apply`
    """

    def __init__(self, cfg: RootConfig):
        """Create an emulator shell from a resolved root config."""
        self.cfg = canonicalize_config_names(cfg)

        self.graphdef: Any | None = None
        self.params: PytreeDict | None = None
        self.model_state: PytreeDict | None = None

        self.task: Any | None = None
        self.tx: optax.GradientTransformation | None = None
        self.last_fit_method: str | None = None
        self.last_fit_metadata: dict[str, Any] | None = None
        self.bundle_metadata: dict[str, Any] | None = None
        self.model_init: dict[str, Any] | None = None

        self._model_entry = get_model_entry_from_name(cfg.model.name)
        self._stable_model_entry = get_stable_model_entry_from_name(cfg.model.name)
        self.model_family_id: str | None = (
            self._model_entry.family_id if self._model_entry is not None else None
        )

    def _initialize_from_resolved_spec(
        self,
        resolved_spec: dict[str, Any],
        *,
        inputs: Any | None = None,
        outputs: Any | None = None,
        init_hints: dict[str, Any] | None = None,
    ) -> "Emulator":
        model_init = resolve_model_init_context(
            self.cfg,
            spec=resolved_spec,
            inputs=inputs,
            outputs=outputs,
            init_hints=init_hints,
        )
        validate_model_io_compatibility(
            self.cfg.model.name, dict(self.cfg.model.params), init_context=model_init
        )
        if self._model_entry is not None and self._model_entry.runtime is not None:
            self._model_entry.runtime.validate_io_spec(cfg=self.cfg, spec=resolved_spec)
        model = build_model_from_name(
            self.cfg.model.name,
            dict(self.cfg.model.params),
            init_context=model_init,
            rngs=nnx.Rngs(self.cfg.seed),
            cfg=self.cfg,
            spec=resolved_spec,
        )
        self.graphdef, params_v, model_state_v = cast(
            tuple[Any, Any, Any], nnx.split(model, nnx.Param, ...)
        )
        self.params = nnx.to_pure_dict(params_v)
        self.model_state = validate_model_state_dict(nnx.to_pure_dict(model_state_v))
        self.model_init = model_init
        self.last_fit_method = None
        self.last_fit_metadata = None
        return self

    @classmethod
    def from_config(
        cls, cfg: RootConfig, *, init_example: dict[str, Any] | None = None
    ) -> "Emulator":
        """Build an emulator from config and optionally initialize it immediately.

        When ``init_example`` or ``cfg.model.init_hints`` is provided, the model
        graph and state are initialized up front. Otherwise the returned emulator
        stays uninitialized until :meth:`initialize`, :meth:`fit`, or bundle
        loading provides enough shape information.
        """
        resolved_spec = materialize_effective_spec(cfg)
        emu = cls(cfg)
        if init_example is not None:
            inputs, outputs = _extract_init_example(init_example)
            return emu._initialize_from_resolved_spec(
                resolved_spec, inputs=inputs, outputs=outputs
            )
        if dict(cfg.model.init_hints):
            return emu._initialize_from_resolved_spec(
                resolved_spec, init_hints=dict(cfg.model.init_hints)
            )
        validate_model_io_compatibility(cfg.model.name, dict(cfg.model.params))
        return emu

    def initialize(
        self,
        *,
        inputs: Any,
        outputs: Any | None = None,
        init_hints: dict[str, Any] | None = None,
    ) -> "Emulator":
        """Initialize model parameters and state from example canonical I/O."""
        resolved_spec = materialize_effective_spec(self.cfg)
        return self._initialize_from_resolved_spec(
            resolved_spec,
            inputs=inputs,
            outputs=outputs,
            init_hints=init_hints,
        )

    def configure_training(
        self,
        optimizer: "optax.GradientTransformation" | None = None,
        task: Any | None = None,
    ) -> "Emulator":
        """Attach training components before calling :meth:`fit`.

        This stores an optimizer override and/or resolved task object for later
        training. It does not call :func:`jax.jit`, trigger XLA compilation, or
        initialize optimizer state.
        """
        if optimizer is not None:
            self.tx = optimizer
        self.task = (
            task
            if task is not None
            else build_task_from_name(self.cfg.task.name, dict(self.cfg.task.params))
        )
        return self

    def _ensure_task(self):
        if self.task is None:
            self.task = build_task_from_name(
                self.cfg.task.name, dict(self.cfg.task.params)
            )
        return self.task

    def _init_train_state(self):
        from .training.state import TrainState

        if self.params is None or self.model_state is None or self.tx is None:
            raise RuntimeError(
                "Emulator is not initialized for training; call initialize(...), "
                "pass init_example=... to from_config(...), or fit(...) on a "
                "non-empty dataset first."
            )
        return build_init_state(
            params=self.params,
            model_state=self.model_state,
            tx=self.tx,
            seed=self.cfg.seed,
            train_state_cls=TrainState,
        )

    def _lazy_initialize_from_dataset(
        self,
        dataset,
        *,
        device_batch_transform: DeviceBatchTransformLike | None = None,
        scalar_target_leaf_as_single_channel: bool = False,
    ) -> None:
        if (
            self.graphdef is not None
            and self.params is not None
            and self.model_state is not None
        ):
            return
        if len(dataset) <= 0:
            raise ValueError("Cannot lazily initialize from an empty dataset.")
        batch = dataset.get_batch(np.asarray([0], dtype=np.int64))
        if device_batch_transform is not None:
            batch = init_batch_via_device_transform(device_batch_transform, batch)
        outputs = batch.get("y")
        if scalar_target_leaf_as_single_channel:
            outputs = _expand_scalar_target_leaf_for_init(outputs)
        self.initialize(inputs=batch["x"], outputs=outputs)

    def fit(
        self,
        train_dataset,
        *,
        validation_dataset=None,
        callbacks=None,
        resume: bool = False,
        max_steps: int | None = None,
        device_batch_transform: DeviceBatchTransformLike | None = None,
        method: str | None = None,
    ) -> History:
        """Train or solve the emulator on a dataset and return metric history.

        The emulator lazily initializes itself from the first training batch when
        necessary. Gradient-based runs use the configured optimizer unless an
        override was attached with :meth:`configure_training`.
        """
        if self.task is None:
            self.task = build_task_from_name(
                self.cfg.task.name, dict(self.cfg.task.params)
            )

        callbacks = [] if callbacks is None else list(callbacks)

        model_name = self.cfg.model.name.lower()
        resolved_method = self.cfg.solver.name if method is None else method
        try:
            solver = resolve_solver(
                resolved_method,
                model_name=model_name,
                task_name=self.cfg.task.name,
                task_params=dict(self.cfg.task.params),
            )
        except ImportError as exc:
            raise RuntimeError(
                "Required training dependencies are unavailable in this environment. "
                "Reinstall `astro-emulators-toolkit` or verify that its core training packages import correctly."
            ) from exc

        self._lazy_initialize_from_dataset(
            train_dataset,
            device_batch_transform=device_batch_transform,
            scalar_target_leaf_as_single_channel=solver.name == "closed_form_linear",
        )

        if solver.name == "gradient" and self.tx is None:
            try:
                self.tx = make_tx(self.cfg, params=self.params)
            except ImportError as exc:
                raise RuntimeError(
                    "Required training dependencies are unavailable in this environment. "
                    "Reinstall `astro-emulators-toolkit` or verify that its core training packages import correctly."
                ) from exc

        result = solver.run(
            cfg=self.cfg,
            model_name=model_name,
            graphdef=self.graphdef,
            init_state=self._init_train_state()
            if (solver.name == "gradient")
            else None,
            params=self.params,
            task=self._ensure_task(),
            tx=self.tx if solver.name == "gradient" else None,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            callbacks=callbacks,
            resume=resume,
            max_steps=max_steps,
            device_batch_transform=device_batch_transform,
        )

        self.params = result.params
        self.model_state = result.model_state
        self.last_fit_method = result.method
        metadata = getattr(result, "metadata", None)
        self.last_fit_metadata = dict(metadata) if isinstance(metadata, dict) else None
        return cast(History, result.history)

    def apply_jax(
        self,
        x: dict[str, Any] | Any,
        *,
        rng: jax.Array | None = None,
        postprocess: bool = True,
        train: bool = False,
    ) -> Any:
        """Run the predictive mapping in JAX and return a JAX pytree output."""
        if self._model_entry is not None and self._model_entry.runtime is not None:
            if not isinstance(x, dict):
                raise ValueError("Public apply_jax expects canonical dict-tree inputs.")
            _validate_public_batched_input_tree(x, field_name="apply_jax inputs")
        if self.graphdef is None or self.params is None or self.model_state is None:
            raise RuntimeError(
                "Emulator is not initialized; call initialize(...), pass "
                "init_example=... to from_config(...), fit(...), or load from a "
                "bundle first."
            )
        task = self._ensure_task() if postprocess else None
        return apply_jax_runtime(
            graphdef=self.graphdef,
            params=self.params,
            model_state=self.model_state,
            task=task,
            x=x,
            rng=rng,
            postprocess=postprocess,
            train=train,
        )

    def make_frozen_apply(
        self, *, postprocess: bool = True, jit: bool = False
    ) -> Callable[..., Any]:
        """Return a stateless callable closed over the current params and state.

        Recreate the returned callable after any training step or parameter
        mutation so it stays synchronized with the current emulator state.
        """
        if self.graphdef is None or self.params is None or self.model_state is None:
            raise RuntimeError(
                "Emulator is not initialized; call initialize(...), pass "
                "init_example=... to from_config(...), fit(...), or load from a "
                "bundle first."
            )
        task = self._ensure_task() if postprocess else None
        post_fn = (
            task.postprocess_pred
            if (postprocess and task is not None and hasattr(task, "postprocess_pred"))
            else None
        )
        apply = cast(
            Callable[..., Any],
            make_frozen_apply_runtime(
                graphdef=self.graphdef,
                params=self.params,
                model_state=self.model_state,
                post_fn=post_fn,
                jit=jit,
            ),
        )
        if self._model_entry is None or self._model_entry.runtime is None:
            return apply

        def canonical_apply(x, *, rng: jax.Array | None = None):
            if not isinstance(x, dict):
                raise ValueError("Frozen apply expects canonical dict-tree inputs.")
            _validate_public_batched_input_tree(x, field_name="frozen apply inputs")
            return apply(x, rng=rng)

        return canonical_apply

    def make_device_batch_transform(self, **kwargs: Any) -> DeviceBatchTransformLike:
        """Build a runtime-specific device-side batch transform object for this model family."""
        if self._model_entry is None or self._model_entry.runtime is None:
            raise NotImplementedError(
                "make_device_batch_transform is only available for models with a runtime adapter."
            )
        runtime = self._model_entry.runtime
        if not hasattr(runtime, "make_device_batch_transform"):
            raise NotImplementedError(
                f"Model family '{self.cfg.model.name}' does not provide device batch transform helpers."
            )
        return cast(
            DeviceBatchTransformLike,
            runtime.make_device_batch_transform(cfg=self.cfg, spec=self.spec, **kwargs),
        )

    def predict(self, x: dict[str, Any] | Any) -> dict[str, Any] | Any:
        """Run host-side inference and return NumPy-backed outputs."""
        if self._model_entry is not None and self._model_entry.runtime is not None:
            if not isinstance(x, dict):
                raise ValueError("Public predict expects canonical dict-tree inputs.")
            _validate_public_batched_input_tree(x, field_name="predict inputs")
        y = self.apply_jax(x, postprocess=True, train=False)
        return to_numpy_pytree(y)

    def describe_domain(self) -> str:
        """Return a compact human-readable summary of input-domain metadata."""
        domain = self.input_domain
        if domain is None:
            return "input_domain: not provided"
        min_tree = domain.get("min_tree")
        max_tree = domain.get("max_tree")
        if isinstance(min_tree, dict) and isinstance(max_tree, dict):
            lines = [
                "input_domain:",
                f"  kind: {domain.get('kind')}",
            ]
            value_space = domain.get("value_space")
            if value_space is not None:
                lines.append(f"  value_space: {value_space}")
            for path, min_value in iter_leaf_paths(min_tree):
                try:
                    max_value = get_leaf_by_path(max_tree, path)
                except KeyError:
                    lines.append(f"  {path}: max_tree missing")
                    continue
                lines.extend(
                    _domain_leaf_lines(
                        path,
                        min_value,
                        max_value,
                        input_spec=self.input_spec,
                    )
                )
            return "\n".join(lines)
        return f"input_domain: kind={domain.get('kind')} (sidecar descriptor)"

    def describe_bundle(self) -> str:
        """Summarize the currently loaded bundle metadata for display or logging."""
        spec = self.spec
        metadata = self.bundle_metadata or {}
        provenance = metadata.get("provenance", {})
        runtime_contract = metadata.get("runtime_contract") or {}
        solver_metadata = metadata.get("solver_metadata") or {}
        release = metadata.get("release")
        release_label = "unreleased"
        if isinstance(release, dict):
            release_name = release.get("name")
            release_version = release.get("version")
            release_status = release.get("status")
            if isinstance(release_name, str) and isinstance(release_version, str):
                release_label = f"{release_name}@{release_version}"
            elif isinstance(release_version, str):
                release_label = release_version
            if (
                release_label != "unreleased"
                and isinstance(release_status, str)
                and release_status
            ):
                release_label = f"{release_label} ({release_status})"
        lines = [
            "Loaded emulator bundle",
            f"model={self.cfg.model.name}",
            f"bundle_format_version={metadata.get('bundle_format_version', 'unknown')}",
            f"bundle_id={metadata.get('bundle_id', 'unknown')}",
            f"release={release_label}",
            f"config_schema_version={metadata.get('config_schema_version', 'unknown')}",
            f"spec_version={spec.get('spec_version', 'unknown')}",
            f"weights_layout={metadata.get('weights_layout', 'unknown')}",
            f"model_family_id={metadata.get('model_family_id', self.model_family_id)}",
            f"fingerprint_evaluation={'present' if metadata.get('fingerprint_evaluation') is not None else 'absent'}",
            f"task={self.cfg.task.name}",
            f"fit_method={metadata.get('fit_method', 'unknown')}",
            f"solver_params={solver_metadata.get('params', {}) if isinstance(solver_metadata, dict) else {}}",
            f"solver_diagnostics={solver_metadata.get('diagnostics', {}) if isinstance(solver_metadata, dict) else {}}",
            "solver_design_matrix="
            f"{solver_metadata.get('design_matrix', {}) if isinstance(solver_metadata, dict) else {}}",
            f"role_paths={runtime_contract.get('role_paths', {})}",
            f"reference_scaling_inputs={'present' if spec.get('reference_scaling_inputs') is not None else 'absent'}",
            f"reference_scaling_outputs={'present' if spec.get('reference_scaling_outputs') is not None else 'absent'}",
            f"input_metadata={_describe_spec_metadata_presence(self.input_spec)}",
            f"output_metadata={_describe_spec_metadata_presence(self.output_spec)}",
            self.describe_domain(),
            f"provenance.toolkit_version={provenance.get('toolkit_version', 'unknown')}",
            f"provenance.python_version={provenance.get('python_version', 'unknown')}",
            f"provenance.created_at={provenance.get('created_at', 'unknown')}",
            f"provenance.git_commit={provenance.get('git_commit', 'unknown')}",
        ]
        return "\n".join(lines)

    @property
    def spec(self) -> dict[str, Any]:
        """Resolved portability spec for the current emulator or loaded bundle."""
        if self.bundle_metadata is None:
            return materialize_effective_spec(self.cfg)
        if "spec" not in self.bundle_metadata:
            raise ValueError(
                "Bundle metadata is missing required 'spec' entry. Re-save the bundle with current toolkit version."
            )
        return cast(dict[str, Any], self.bundle_metadata["spec"])

    @property
    def reference_scaling_inputs(self) -> dict[str, Any] | None:
        """Explicit input reference-scaling metadata from :attr:`spec`, if present."""
        return _spec_block(self.spec, "reference_scaling_inputs")

    @property
    def reference_scaling_outputs(self) -> dict[str, Any] | None:
        """Explicit output reference-scaling metadata from :attr:`spec`, if present."""
        return _spec_block(self.spec, "reference_scaling_outputs")

    @property
    def input_domain(self) -> dict[str, Any] | None:
        """Explicit input-domain bounds from :attr:`spec`, if present."""
        return _spec_block(self.spec, "input_domain")

    @property
    def bundle_extras(self) -> dict[str, Any]:
        """Bundle ``extras`` metadata as a dictionary."""
        extras = (self.bundle_metadata or {}).get("extras")
        return extras if isinstance(extras, dict) else {}

    @property
    def input_spec(self) -> dict[str, Any] | None:
        """Canonical input block from :attr:`spec`, if present."""
        return _spec_block(self.spec, "inputs")

    @property
    def output_spec(self) -> dict[str, Any] | None:
        """Canonical output block from :attr:`spec`, if present."""
        return _spec_block(self.spec, "outputs")

    @property
    def input_channel_names_tree(self) -> dict[str, Any] | None:
        """Input channel names keyed by canonical input tree path, if present."""
        inputs = self.input_spec
        if inputs is None:
            return None
        channel_names_tree = inputs.get("channel_names_tree")
        return channel_names_tree if isinstance(channel_names_tree, dict) else None

    @property
    def output_channel_names_tree(self) -> dict[str, Any] | None:
        """Output channel names keyed by canonical output tree path, if present."""
        outputs = self.output_spec
        if outputs is None:
            return None
        channel_names_tree = outputs.get("channel_names_tree")
        return channel_names_tree if isinstance(channel_names_tree, dict) else None

    def save_bundle(
        self,
        dirpath: str | Path | None = None,
        *,
        spec: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Save a portable inference bundle and return the bundle directory path."""
        path, metadata = save_bundle_artifact(
            cfg=self.cfg,
            graphdef=self.graphdef,
            params=self.params,
            model_state=self.model_state,
            model_init={} if self.model_init is None else dict(self.model_init),
            last_fit_method=self.last_fit_method,
            last_fit_metadata=self.last_fit_metadata,
            dirpath=dirpath,
            spec=spec,
            extras=extras,
        )
        self.bundle_metadata = metadata
        self.model_family_id = metadata.get("model_family_id", self.model_family_id)
        return path

    @classmethod
    def from_bundle(cls, dirpath: str | Path, *, verbose: bool = False) -> "Emulator":
        """Load a portable bundle from a local directory."""
        bundle_dir, loaded = load_bundle_artifact(dirpath)
        resolved_cfg = loaded.cfg
        model_init = get_bundle_model_init(loaded.metadata, cfg=resolved_cfg)
        _validate_bundle_weights_payload(loaded.metadata, loaded.params_pure)
        _validate_runtime_contract_metadata(
            loaded.metadata, cfg=resolved_cfg, model_init=model_init
        )

        expected_family_id = None
        model_entry = get_model_entry_from_name(resolved_cfg.model.name)
        if model_entry is not None:
            expected_family_id = model_entry.family_id
        bundled_family_id = loaded.metadata.get("model_family_id")
        if expected_family_id is not None and bundled_family_id != expected_family_id:
            raise ValueError(
                f"Bundle model_family_id '{bundled_family_id}' does not match "
                f"resolved model family '{expected_family_id}'."
            )

        emu = cls(resolved_cfg)
        emu._initialize_from_resolved_spec(
            loaded.metadata["spec"], init_hints=model_init
        )
        loaded_model_state = validate_model_state_dict(
            loaded.params_pure.get("model_state", {})
        )
        _validate_loaded_runtime_trees_match_initialization(
            initialized_params=cast(PytreeDict, emu.params),
            loaded_params=cast(PytreeDict, loaded.params_pure["params"]),
            initialized_model_state=cast(PytreeDict, emu.model_state),
            loaded_model_state=loaded_model_state,
        )
        emu.params = loaded.params_pure["params"]
        emu.model_state = loaded_model_state
        emu.last_fit_method = loaded.metadata.get("fit_method")
        solver_metadata = loaded.metadata.get("solver_metadata")
        emu.last_fit_metadata = (
            dict(solver_metadata) if isinstance(solver_metadata, dict) else None
        )
        emu.model_family_id = loaded.metadata.get(
            "model_family_id", emu.model_family_id
        )
        emu.bundle_metadata = loaded.metadata
        if verbose:
            print(emu.describe_bundle())
        return emu

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        verbose: bool = False,
    ) -> "Emulator":
        """Download a pretrained bundle snapshot and load it as an emulator."""
        local = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache_dir or str(get_cache_dir() / "hub"),
        )
        return cls.from_bundle(local, verbose=verbose)
