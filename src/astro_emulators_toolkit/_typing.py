from __future__ import annotations

from typing import Any, Protocol, TypeAlias


PytreeDict: TypeAlias = dict[str, Any]
AffineLeafSpec: TypeAlias = dict[str, Any]
AffineLeafSpecs: TypeAlias = dict[str, AffineLeafSpec]


class SupportsFromDict(Protocol):
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any: ...
