"""Regenerate the old-toolkit bundles used by test_legacy_bundles.py.

These bundles must be written by the old toolkit code itself, so run this
with an export of the matching git revision on ``PYTHONPATH``:

    # toolkit 0.1.0 -> fixtures/v0_1_0/
    git archive v0.1.0 | tar -x -C /tmp/aet-0.1.0
    JAX_PLATFORMS=cpu PYTHONPATH=/tmp/aet-0.1.0/src \
        uv run python tests/bundle/fixtures/make_fixtures.py

    # toolkit 0.2.0.dev0 (stamped config schema v1) -> fixtures/v0_2_0_dev0/
    git archive 65feb0b | tar -x -C /tmp/aet-0.2.0.dev0
    JAX_PLATFORMS=cpu PYTHONPATH=/tmp/aet-0.2.0.dev0/src \
        uv run python tests/bundle/fixtures/make_fixtures.py

Each bundle uses settings whose handling differs between toolkit versions
(see config/migrations.py). Next to the bundles it writes
``probe_inputs.npy`` and ``<bundle>.predictions.npy`` with the predictions
the old toolkit itself made, which the tests compare against.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import astro_emulators_toolkit as aet
from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    SolverConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset

HERE = Path(__file__).resolve().parent

BUNDLES_BY_VERSION = {
    "0.1.0": {
        # Three settings 0.1.0 silently ignored, which 0.2.0 rejects.
        "mlp_ignored_settings": dict(
            model={"hidden_sizes": (8,), "activation": "tanh"},
            optim=dict(
                lr_scaling=None,
                scale_embedding_lr=0.5,
                schedule="cosine",
                decay_steps=3,
            ),
            solver=dict(name="gradient", params={"foo": 1}),
        ),
        # bfloat16 compute with float32 parameter storage (the 0.1.0
        # behavior), plus 'auto' solver params that resolve to 'gradient'.
        "mlp_bfloat16_auto_solver_params": dict(
            model={"hidden_sizes": (8,), "activation": "tanh", "dtype": "bfloat16"},
            optim=dict(),
            solver=dict(name="auto", params={"foo": 1}),
        ),
    },
    "0.2.0.dev0": {
        # Stamped config schema v1 but with v2 meaning: parameters are
        # stored in bfloat16, not float32 as under 0.1.x.
        "mlp_bfloat16": dict(
            model={"hidden_sizes": (8,), "activation": "tanh", "dtype": "bfloat16"},
            optim=dict(),
            solver=dict(name="auto"),
        ),
    },
}


def main() -> None:
    version = aet.__version__
    if version not in BUNDLES_BY_VERSION:
        raise SystemExit(
            f"Run this with one of {sorted(BUNDLES_BY_VERSION)} on sys.path "
            f"(got {version})."
        )
    out_dir = HERE / ("v" + version.replace(".", "_"))
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    y = (x @ np.array([[0.5], [-0.2], [0.1]], dtype=np.float32)).astype(np.float32)
    dataset = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})
    probe = rng.normal(size=(4, 3)).astype(np.float32)
    np.save(out_dir / "probe_inputs.npy", probe)

    for name, spec in BUNDLES_BY_VERSION[version].items():
        with tempfile.TemporaryDirectory() as workdir:
            cfg = RootConfig(
                io=IOSpec(),
                model=ModelSpec(name="mlp", params=spec["model"]),
                task=TaskSpec(name="regression", params={"loss": "mse"}),
                solver=SolverConfig(**spec["solver"]),
                optim=OptimConfig(name="adam", lr=1e-2, **spec["optim"]),
                training=TrainConfig(
                    workdir=workdir,
                    batch_size=8,
                    num_steps=5,
                    evaluation_interval_steps=1,
                ),
            )
            emu = Emulator.from_config(cfg)
            emu.fit(dataset, validation_dataset=dataset, callbacks=[])
            emu.save_bundle(out_dir / name)
        predictions = Emulator.from_bundle(out_dir / name).predict(
            {"parameters": probe}
        )
        np.save(
            out_dir / f"{name}.predictions.npy",
            np.asarray(predictions["predictions"], dtype=np.float32),
        )
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
