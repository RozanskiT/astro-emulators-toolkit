from ._version import __version__
from .config.io import load_config, save_config
from .config.schema import RootConfig
from .io_trees import denormalize_tree, normalize_tree


__all__ = [
    "Emulator",
    "RootConfig",
    "load_config",
    "save_config",
    "normalize_tree",
    "denormalize_tree",
    "__version__",
]


def __getattr__(name: str):
    if name == "Emulator":
        from .emulator import Emulator

        return Emulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
