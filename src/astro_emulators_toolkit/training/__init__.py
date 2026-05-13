from .state import TrainState
from .callbacks import (
    Callback,
    History,
    CSVLogger,
    ProgressBarLogger,
    ModelCheckpoint,
    build_callbacks_from_config,
)

__all__ = [
    "TrainState",
    "fit",
    "Callback",
    "History",
    "CSVLogger",
    "ProgressBarLogger",
    "ModelCheckpoint",
    "build_callbacks_from_config",
]


def __getattr__(name: str):
    if name == "fit":
        from .trainer import fit

        return fit
    raise AttributeError(name)
