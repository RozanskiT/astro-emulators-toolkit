from __future__ import annotations

from ..config.schema import CONFIG_SCHEMA_VERSION
from ..spec import SPEC_VERSION


BUNDLE_FORMAT_VERSION = 1
WEIGHTS_LAYOUT = "params_plus_model_state_v1"

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "SPEC_VERSION",
    "WEIGHTS_LAYOUT",
]
