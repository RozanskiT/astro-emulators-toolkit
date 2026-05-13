from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..config.parsing import parse_bool
from ..models.cannon import cannon_design_matrix
from .fit_backends import (
    BackendFitResult,
    fit_with_closed_form_linear_backend,
    fit_with_gradient_backend,
)


@dataclass(frozen=True)
class ClosedFormLinearSolverConfig:
    ridge: float = 1e-4
    regularize_intercept: bool = False

    def __post_init__(self) -> None:
        ridge = float(self.ridge)
        if not math.isfinite(ridge):
            raise ValueError("closed_form_linear solver ridge must be finite.")
        if ridge < 0.0:
            raise ValueError("closed_form_linear solver ridge must be >= 0.")
        object.__setattr__(self, "ridge", ridge)

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> "ClosedFormLinearSolverConfig":
        allowed = {"ridge", "regularize_intercept"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"Unknown closed_form_linear solver params: {unknown}.")
        return cls(
            ridge=float(d.get("ridge", cls.ridge)),
            regularize_intercept=parse_bool(
                d.get("regularize_intercept", cls.regularize_intercept),
                field_name="regularize_intercept",
            ),
        )


@dataclass(frozen=True)
class SolverSpec:
    name: str
    supports_model: Callable[[str], bool]
    run: Callable[..., BackendFitResult]


def _run_gradient_solver(
    *,
    cfg,
    graphdef,
    init_state,
    task,
    tx,
    train_dataset,
    validation_dataset,
    callbacks,
    resume,
    max_steps,
    device_batch_transform,
    **_unused,
) -> BackendFitResult:
    return fit_with_gradient_backend(
        cfg=cfg,
        graphdef=graphdef,
        init_state=init_state,
        task=task,
        tx=tx,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        callbacks=callbacks,
        resume=resume,
        max_steps=max_steps,
        device_batch_transform=device_batch_transform,
    )


def _supports_all_models(_model_name: str) -> bool:
    return True


def _supports_cannon_only(model_name: str) -> bool:
    return model_name == "cannon"


def _run_closed_form_linear_solver(
    *,
    model_name: str,
    params,
    task,
    train_dataset,
    validation_dataset,
    callbacks,
    cfg,
    resume=False,
    max_steps=None,
    device_batch_transform=None,
    **_unused,
) -> BackendFitResult:
    if model_name != "cannon":
        raise ValueError(
            f"closed_form_linear solver is not implemented for model '{model_name}'."
        )
    include_bias = parse_bool(
        cfg.model.params.get("include_bias", True),
        field_name="model.params.include_bias",
    )
    solver_cfg = ClosedFormLinearSolverConfig.from_mapping(dict(cfg.solver.params))
    result = fit_with_closed_form_linear_backend(
        cfg=cfg,
        params=params,
        task=task,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        callbacks=callbacks,
        design_matrix_fn=cannon_design_matrix,
        design_matrix_kwargs={"include_bias": include_bias},
        ridge=solver_cfg.ridge,
        unregularized_columns=()
        if (not include_bias or solver_cfg.regularize_intercept)
        else (0,),
        resume=resume,
        max_steps=max_steps,
        device_batch_transform=device_batch_transform,
    )
    diagnostics = {}
    if isinstance(result.metadata, dict) and isinstance(
        result.metadata.get("diagnostics"), dict
    ):
        diagnostics = dict(result.metadata["diagnostics"])
    result.metadata = {
        "name": "closed_form_linear",
        "params": {
            "ridge": solver_cfg.ridge,
            "regularize_intercept": solver_cfg.regularize_intercept,
        },
        "diagnostics": diagnostics,
        "design_matrix": {
            "kind": "cannon_quadratic_v1",
            "include_bias": include_bias,
            "intercept_column_index": 0 if include_bias else None,
        },
    }
    return result


_SOLVER_REGISTRY: dict[str, SolverSpec] = {
    "gradient": SolverSpec(
        name="gradient", supports_model=_supports_all_models, run=_run_gradient_solver
    ),
    "closed_form_linear": SolverSpec(
        name="closed_form_linear",
        supports_model=_supports_cannon_only,
        run=_run_closed_form_linear_solver,
    ),
}


def _is_closed_form_task_compatible(
    task_name: str, task_params: dict[str, Any]
) -> bool:
    if str(task_name).lower() != "regression":
        return False
    return str(task_params.get("loss", "mse")).lower() in {"mse", "weighted_mse"}


def default_solver_for_model(
    model_name: str,
    *,
    task_name: str = "regression",
    task_params: dict[str, Any] | None = None,
) -> str:
    if model_name.lower() == "cannon":
        params = {} if task_params is None else dict(task_params)
        return (
            "closed_form_linear"
            if _is_closed_form_task_compatible(task_name, params)
            else "gradient"
        )
    return "gradient"


def available_solver_names() -> tuple[str, ...]:
    return tuple(_SOLVER_REGISTRY)


def resolve_solver(
    method: str,
    *,
    model_name: str,
    task_name: str = "regression",
    task_params: dict[str, Any] | None = None,
) -> SolverSpec:
    model_name = model_name.lower()
    resolved_name = (
        default_solver_for_model(
            model_name, task_name=task_name, task_params=task_params
        )
        if method == "auto"
        else str(method).lower()
    )
    spec = _SOLVER_REGISTRY.get(resolved_name)
    if spec is None:
        raise ValueError(
            f"Unknown solver '{resolved_name}'. Available solvers: {available_solver_names()}."
        )
    if resolved_name == "closed_form_linear" and not _is_closed_form_task_compatible(
        task_name, {} if task_params is None else dict(task_params)
    ):
        raise ValueError(
            "closed_form_linear solver is only valid for regression tasks with "
            "squared loss (loss in {'mse', 'weighted_mse'})."
        )
    if not spec.supports_model(model_name):
        supported = tuple(
            name
            for name, candidate in _SOLVER_REGISTRY.items()
            if candidate.supports_model(model_name)
        )
        raise ValueError(
            f"Model '{model_name}' does not support solver '{resolved_name}'. Supported solvers: {supported}."
        )
    return spec
