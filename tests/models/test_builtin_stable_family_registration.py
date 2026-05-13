from __future__ import annotations

from astro_emulators_toolkit.models import get_stable_model_entry
from astro_emulators_toolkit.models.runtime_adapters import (
    CannonRuntimeAdapter,
    MLPRuntimeAdapter,
    TransformerPayneRuntimeAdapter,
)


def test_builtin_stable_families_register_family_specific_runtime_adapters():
    assert isinstance(get_stable_model_entry("mlp").runtime, MLPRuntimeAdapter)
    assert isinstance(get_stable_model_entry("cannon").runtime, CannonRuntimeAdapter)
    assert isinstance(
        get_stable_model_entry("transformer_payne").runtime,
        TransformerPayneRuntimeAdapter,
    )
