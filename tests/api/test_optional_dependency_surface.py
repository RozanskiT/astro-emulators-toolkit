from __future__ import annotations

import builtins
import tomllib
from pathlib import Path

import numpy as np

from astro_emulators_toolkit import Emulator, __version__, normalize_tree
from astro_emulators_toolkit._version import __version__ as package_source_version
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data.toy import ToyNormalizedFluxDataset


class BlockImports:
    def __init__(self, blocked: set[str]):
        self._blocked = blocked
        self._orig = builtins.__import__

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in self._blocked:
            raise ImportError(f"blocked import: {name}")
        return self._orig(name, globals, locals, fromlist, level)


def test_core_distribution_promotes_training_and_hub_to_base_and_exposes_viz_extra():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.11"

    dependencies = set(pyproject["project"]["dependencies"])
    assert "optax>=0.2.5" in dependencies
    assert "orbax-checkpoint>=0.11.5" in dependencies
    assert "huggingface_hub>=1.0.0" in dependencies

    optional_deps = pyproject["project"]["optional-dependencies"]
    assert set(optional_deps) == {"blackjax", "viz"}
    assert optional_deps["blackjax"] == ["blackjax>=1.2.5"]
    assert optional_deps["viz"] == ["matplotlib>=3.9"]

    dependency_groups = pyproject["dependency-groups"]
    assert "docs" not in dependency_groups


def test_core_workflows_do_not_require_blackjax(monkeypatch, tmp_path):
    monkeypatch.setattr(builtins, "__import__", BlockImports({"blackjax"}))

    bundle = Path("examples/assets/reference_bundle_release")
    weights = bundle / "weights" / "weights.safetensors"
    if not weights.exists():
        raise AssertionError(
            "Missing shipped reference bundle asset at "
            f"{weights}. Update it intentionally with "
            "`python examples/assets/build_reference_bundle.py`."
        )

    monkeypatch.setattr(
        "astro_emulators_toolkit.emulator.snapshot_download",
        lambda *args, **kwargs: bundle,
    )

    emu = Emulator.from_pretrained("dummy/repo", revision="main")
    ref = emu.reference_scaling_inputs
    x = {"parameters": np.asarray([[4500.0, 2.0, -0.2]], dtype=np.float32)}
    y = emu.predict(normalize_tree(x, ref["min_tree"], ref["max_tree"]))
    assert np.asarray(y["flux"]).shape == (1, 500)

    cfg = RootConfig(
        model=ModelSpec(name="mlp", params={"hidden_sizes": [8], "activation": "relu"}),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "train_run"),
            num_steps=1,
            batch_size=4,
            logging_interval_steps=1,
            checkpoint_interval_steps=0,
            evaluation_interval_steps=1,
        ),
    )
    train_ds = ToyNormalizedFluxDataset(n_samples=8, x_dim=3, y_dim=8, seed=0)

    trained = Emulator.from_config(cfg)
    history = trained.fit(train_ds, validation_dataset=train_ds, callbacks=[])

    assert history.logs["training_loss"]


def test_version_uses_single_packaging_source():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["dynamic"] == ["version"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "astro_emulators_toolkit._version.__version__"
    )
    assert __version__ == package_source_version
