from __future__ import annotations

from typing import Any


def parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


__all__ = ["parse_bool"]
