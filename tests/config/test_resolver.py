from __future__ import annotations

import subprocess
import sys

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.resolver import (
    get_supported_stable_model_names,
    get_supported_stable_task_names,
)


def test_resolver_import_does_not_import_experimental_registries():
    code = """
import sys
import astro_emulators_toolkit.resolver

loaded = [
    name
    for name in (
        "astro_emulators_toolkit.experimental.models",
        "astro_emulators_toolkit.experimental.tasks",
    )
    if name in sys.modules
]
if loaded:
    raise SystemExit(f"Experimental registries imported eagerly: {loaded}")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_stable_model_registry_contains_only_stable_models():
    assert get_supported_stable_model_names() == ("mlp", "transformer_payne", "cannon")


def test_stable_task_registry_contains_only_regression():
    assert get_supported_stable_task_names() == ("regression",)


def test_experimental_model_namespaced_resolution():
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp", params={"channels": 1}
        ),
    )
    emu = Emulator.from_config(cfg)
    assert emu.cfg.model.name == "experimental/explicit_wavelength_mlp"


def test_experimental_task_namespaced_resolution(tmp_path):
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="mlp", params={"hidden_sizes": (8,)}),
        task=TaskSpec(
            name="experimental/binary_classification",
            params={"decision_threshold": 0.5},
        ),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=1,
            batch_size=4,
            checkpoint_interval_steps=0,
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()
    assert emu.task.name == "binary_classification"
