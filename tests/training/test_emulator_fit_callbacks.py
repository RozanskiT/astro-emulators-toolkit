from __future__ import annotations

import numpy as np

from astro_emulators_toolkit.config import IOSpec, RootConfig, TrainConfig
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.training import build_callbacks_from_config


def test_build_callbacks_from_config_includes_progress_and_checkpoint_callbacks(
    tmp_path,
):
    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            logging_interval_steps=2,
            checkpoint_interval_steps=3,
        ),
    )

    callbacks = build_callbacks_from_config(cfg)

    assert [callback.__class__.__name__ for callback in callbacks] == [
        "ProgressBarLogger",
        "ModelCheckpoint",
    ]


def test_build_callbacks_from_config_supports_explicit_step_schedules(tmp_path):
    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            logging_interval_steps=None,
            logging_steps=(2, 4),
            checkpoint_interval_steps=None,
            checkpoint_steps=(2, 4),
        ),
    )

    callbacks = build_callbacks_from_config(cfg)

    assert [callback.__class__.__name__ for callback in callbacks] == [
        "ProgressBarLogger",
        "ModelCheckpoint",
    ]


def test_build_callbacks_from_config_returns_empty_list_when_visible_callbacks_disabled(
    tmp_path,
):
    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            logging_interval_steps=None,
            logging_steps=None,
            checkpoint_interval_steps=None,
            checkpoint_steps=None,
        ),
    )

    assert build_callbacks_from_config(cfg) == []


def test_emulator_fit_treats_callbacks_none_the_same_as_empty_list(
    monkeypatch, tmp_path
):
    observed_callbacks = None

    def _fake_fit_backend(**kwargs):
        nonlocal observed_callbacks
        observed_callbacks = kwargs["callbacks"]

        class _Result:
            history = {}
            params = kwargs["init_state"].params
            model_state = kwargs["init_state"].model_state
            method = "gradient"

        return _Result()

    class _Solver:
        name = "gradient"

        @staticmethod
        def run(**kwargs):
            return _fake_fit_backend(**kwargs)

    monkeypatch.setattr(
        "astro_emulators_toolkit.emulator.resolve_solver",
        lambda *_args, **_kwargs: _Solver(),
    )

    cfg = RootConfig(
        io=IOSpec(),
        training=TrainConfig(
            workdir=str(tmp_path / "run"),
            num_steps=4,
            logging_interval_steps=2,
            checkpoint_interval_steps=3,
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()

    class _Dataset:
        def __len__(self):
            return 4

        def get_batch(self, idx):
            x = np.zeros((len(idx), 2), dtype=np.float32)
            y = np.zeros((len(idx), 1), dtype=np.float32)
            return {"x": {"parameters": x}, "y": {"predictions": y}}

    emu.fit(_Dataset(), callbacks=None)

    assert observed_callbacks == []
