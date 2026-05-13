from __future__ import annotations

from typing import Any

from ..._typing import SupportsFromDict
from .classification import BinaryClassificationTask, BinaryClassificationTaskConfig

_EXPERIMENTAL_TASKS: dict[str, tuple[SupportsFromDict, Any]] = {
    "binary_classification": (BinaryClassificationTaskConfig, BinaryClassificationTask),
}


def get_experimental_task_registry() -> dict[str, tuple[SupportsFromDict, Any]]:
    return dict(_EXPERIMENTAL_TASKS)


def build_experimental_task(name: str, params: dict[str, Any]):
    key = name.lower()
    if key not in _EXPERIMENTAL_TASKS:
        raise KeyError(
            f"Unknown experimental task '{name}'. Available={list(_EXPERIMENTAL_TASKS)}"
        )
    cfg_cls, task_cls = _EXPERIMENTAL_TASKS[key]
    cfg = cfg_cls.from_dict(params)
    return task_cls(cfg)
