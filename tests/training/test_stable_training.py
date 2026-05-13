from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset
from astro_emulators_toolkit.presets import (
    cannon_flux,
    isochrone_mlp,
    payne_flux_mlp,
    transformer_payne_flux,
    transformer_payne_intensity,
)
from astro_emulators_toolkit.training.solvers import default_solver_for_model


jax.config.update("jax_enable_x64", True)


def _canonical_xy(x, y):
    return {"parameters": x}, {"predictions": y}


class TinyExplicitWavelengthDataset:
    def __init__(self, *, n: int, x_dim: int, n_wave: int, seed: int):
        rng = np.random.default_rng(seed)
        self.params = rng.normal(size=(n, x_dim)).astype(np.float64)
        self.wavelengths = rng.uniform(0.2, 2.0, size=(n, n_wave)).astype(np.float64)
        self.wavelengths.sort(axis=1)
        self.y = (np.sin(self.wavelengths * (1.0 + 0.1 * self.params[:, 0:1]))).astype(
            np.float64
        )

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def get_batch(self, idx):
        return {"x": (self.params[idx], self.wavelengths[idx]), "y": self.y[idx]}


def test_fit_without_compile_for_stable_models(tmp_path):
    x = np.random.default_rng(0).normal(size=(24, 3)).astype(np.float32)
    y = (x @ np.array([[0.5], [0.2], [-0.1]], dtype=np.float32)).astype(np.float32)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    mlp_cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (16,), "activation": "tanh"}
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "mlp"),
            batch_size=8,
            num_steps=2,
            evaluation_interval_steps=1,
        ),
    )
    mlp = Emulator.from_config(mlp_cfg)
    mlp.fit(ds, validation_dataset=ds, callbacks=[])
    assert mlp.last_fit_method == "gradient"

    cannon_cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        training=TrainConfig(
            workdir=str(tmp_path / "cannon"), batch_size=8, num_steps=1
        ),
    )
    cannon = Emulator.from_config(cannon_cfg)
    cannon.fit(ds, validation_dataset=ds, callbacks=[])
    assert cannon.last_fit_method == "closed_form_linear"

    tds = TinyExplicitWavelengthDataset(n=12, x_dim=3, n_wave=6, seed=2)
    transformer_cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="transformer_payne",
            params={
                "channels": 1,
                "dim": 16,
                "dim_head": 8,
                "no_layers": 1,
                "no_tokens": 2,
                "reference_width": 8,
                "dtype": "float64",
            },
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3, lr_scaling="mup"),
        training=TrainConfig(
            workdir=str(tmp_path / "transformer"),
            batch_size=6,
            num_steps=1,
            evaluation_interval_steps=1,
        ),
    )
    transformer = Emulator.from_config(transformer_cfg)
    transformer.fit(tds, validation_dataset=tds, callbacks=[])
    assert transformer.last_fit_method == "gradient"


def test_configure_training_still_overrides_optimizer_when_used(monkeypatch, tmp_path):
    called = {"count": 0}

    def _unexpected_make_tx(_cfg):
        called["count"] += 1
        raise AssertionError(
            "make_tx should not be called when optimizer is provided via configure_training()."
        )

    monkeypatch.setattr("astro_emulators_toolkit.emulator.make_tx", _unexpected_make_tx)

    x = np.random.default_rng(3).normal(size=(16, 2)).astype(np.float32)
    y = (0.3 * x[:, :1]).astype(np.float32)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (8,), "activation": "tanh"}
        ),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            batch_size=8,
            num_steps=2,
            evaluation_interval_steps=1,
        ),
    )

    custom_tx = optax.sgd(learning_rate=1e-2)
    emu = Emulator.from_config(cfg).configure_training(optimizer=custom_tx)
    emu.fit(ds, validation_dataset=ds, callbacks=[])

    assert emu.tx is custom_tx
    assert called["count"] == 0


def test_configure_training_does_not_eagerly_create_optimizer_for_closed_form_solver(
    monkeypatch,
):
    cfg = RootConfig(io=IOSpec())
    emu = Emulator.from_config(cfg)

    called = {"count": 0}

    def _unexpected_make_tx(_cfg):
        called["count"] += 1
        raise AssertionError("make_tx should not be called from configure_training()")

    monkeypatch.setattr("astro_emulators_toolkit.emulator.make_tx", _unexpected_make_tx)

    emu.configure_training()

    assert emu.tx is None
    assert called["count"] == 0


def test_auto_solver_for_cannon(tmp_path):
    assert (
        default_solver_for_model(
            "cannon", task_name="regression", task_params={"loss": "mse"}
        )
        == "closed_form_linear"
    )

    x = np.random.default_rng(11).normal(size=(24, 2)).astype(np.float32)
    y = (x @ np.array([[0.4], [0.1]], dtype=np.float32)).astype(np.float32)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        training=TrainConfig(workdir=str(tmp_path / "run"), batch_size=8, num_steps=1),
    )
    emu = Emulator.from_config(cfg)
    emu.fit(ds, validation_dataset=ds, callbacks=[])
    assert emu.last_fit_method == "closed_form_linear"


def test_fit_uses_device_batch_transform_for_lazy_init(tmp_path):
    class RawArrayDataset:
        def __init__(self):
            self.x = np.random.default_rng(21).normal(size=(12, 2)).astype(np.float32)
            self.y = self.x[:, :1] * np.float32(0.5)

        def __len__(self):
            return int(self.x.shape[0])

        def get_batch(self, idx):
            idx = np.asarray(idx)
            return {"x": self.x[idx], "y": self.y[idx]}

    class ExpandingTransform:
        @staticmethod
        def _transform(batch, *, train: bool):
            x = jnp.asarray(batch["x"])
            y = jnp.asarray(batch["y"])
            offset = jnp.full((x.shape[0], 1), 1.0 if train else 2.0, dtype=x.dtype)
            return {
                "x": jnp.concatenate([x, offset], axis=-1),
                "y": jnp.concatenate([y, y + 1.0], axis=-1),
            }

        def for_init(self, batch):
            return self._transform(batch, train=False)

        def __call__(self, batch, *, train: bool, rng):
            del rng
            return self._transform(batch, train=train)

    ds = RawArrayDataset()
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp", params={"hidden_sizes": (8,), "activation": "tanh"}
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            batch_size=4,
            num_steps=1,
            evaluation_interval_steps=1,
        ),
    )

    emu = Emulator.from_config(cfg)
    emu.fit(
        ds,
        validation_dataset=ds,
        device_batch_transform=ExpandingTransform(),
        callbacks=[],
    )

    assert emu.model_init == {"input_last_axis": 3, "output_last_axis": 2}
    assert emu.last_fit_method == "gradient"


def test_stable_presets_build_valid_configs():
    cfgs = [
        payne_flux_mlp(),
        isochrone_mlp(),
        transformer_payne_flux(channels=1),
        transformer_payne_intensity(channels=2),
        cannon_flux(),
    ]

    names = [cfg.model.name for cfg in cfgs]
    assert names == ["mlp", "mlp", "transformer_payne", "transformer_payne", "cannon"]

    intensity_cfg = cfgs[3]
    assert intensity_cfg.io.outputs.channel_names_tree["flux"] == (
        "normalized_intensity",
        "log10_continuum_minmax",
    )

    smoke_cfg = payne_flux_mlp(profile="smoke")
    assert smoke_cfg.training.num_steps < cfgs[0].training.num_steps

    for cfg in cfgs:
        emu = Emulator.from_config(cfg)
        assert emu.cfg.model.name == cfg.model.name
