from __future__ import annotations

import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.resolver import (
    get_supported_experimental_model_names,
)


def _cfg(workdir: str) -> RootConfig:
    return RootConfig(
        seed=0,
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"predictions": None}),
        ),
        model=ModelSpec(
            name="experimental/siren",
            params={
                "hidden_sizes": (16, 16),
                "omega0_first": 12.0,
                "omega0_hidden": 1.0,
            },
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=workdir,
            batch_size=8,
            num_steps=2,
            evaluation_interval_steps=1,
            checkpoint_interval_steps=0,
            max_saved_checkpoints=0,
        ),
    )


def test_experimental_siren_is_registered():
    assert "experimental/siren" in get_supported_experimental_model_names()


def test_siren_predict_and_bundle_roundtrip(tmp_path):
    cfg = _cfg(str(tmp_path / "run"))
    emu = Emulator.from_config(cfg)

    x = np.random.default_rng(0).uniform(-1.0, 1.0, size=(24, 2)).astype(np.float32)
    y = np.sin(4.0 * x[:, :1]).astype(np.float32)
    ds = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})

    emu.fit(ds, validation_dataset=ds, callbacks=[])
    pred = emu.predict({"parameters": x[:3]})
    assert pred["predictions"].shape == (3, 1)

    bundle_dir = emu.save_bundle(tmp_path / "bundle")
    loaded = Emulator.from_bundle(bundle_dir)
    loaded_pred = loaded.predict({"parameters": x[:3]})

    np.testing.assert_allclose(
        loaded_pred["predictions"], pred["predictions"], rtol=1e-6, atol=1e-6
    )
    assert loaded.model_family_id == "experimental_siren_v1"
