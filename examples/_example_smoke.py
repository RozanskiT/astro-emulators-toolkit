from __future__ import annotations

import os
from typing import TypeVar


T = TypeVar("T")

_FALSEY = {"", "0", "false", "False", "no", "No"}


def example_smoke_enabled() -> bool:
    return os.environ.get("ASTRO_EMU_EXAMPLE_SMOKE", "").strip() not in _FALSEY


def smoke_value(default: T, *, smoke: T) -> T:
    return smoke if example_smoke_enabled() else default
