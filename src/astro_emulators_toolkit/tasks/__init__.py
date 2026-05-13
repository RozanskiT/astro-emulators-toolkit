from __future__ import annotations

from typing import Any

from .._typing import SupportsFromDict
from .regression import RegressionTask, RegressionTaskConfig

_STABLE_TASKS: dict[str, tuple[SupportsFromDict, Any]] = {
    "regression": (RegressionTaskConfig, RegressionTask),
}


def get_stable_task_registry() -> dict[str, tuple[SupportsFromDict, Any]]:
    return dict(_STABLE_TASKS)


def build_task(name: str, params: dict[str, Any]):
    key = name.lower()
    if key not in _STABLE_TASKS:
        raise KeyError(f"Unknown task '{name}'. Available={list(_STABLE_TASKS)}")
    cfg_cls, task_cls = _STABLE_TASKS[key]
    cfg = cfg_cls.from_dict(params)
    return task_cls(cfg)
