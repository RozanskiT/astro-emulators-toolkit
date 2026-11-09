from __future__ import annotations

import importlib.resources

from astro_emulators_toolkit import RootConfig
from astro_emulators_toolkit.config.schema import IOSpec, IOTreeSpec


def test_top_level_package_exposes_core_config_api():
    # Note: `import astro_emulators_toolkit` is NOT lightweight — the
    # top-level normalize_tree/denormalize_tree exports pull in JAX. Only the
    # Emulator/flax import is deferred via module __getattr__.
    import astro_emulators_toolkit as aet

    assert hasattr(aet, "RootConfig")
    assert hasattr(aet, "load_config")
    assert hasattr(aet, "save_config")


def test_top_level_all_exports_are_resolvable():
    import astro_emulators_toolkit as aet

    for name in aet.__all__:
        assert getattr(aet, name) is not None, name


def test_training_callback_import_path_is_public():
    from astro_emulators_toolkit.training import (
        CSVLogger,
        ModelCheckpoint,
        ProgressBarLogger,
    )

    assert CSVLogger.__name__ == "CSVLogger"
    assert ModelCheckpoint.__name__ == "ModelCheckpoint"
    assert ProgressBarLogger.__name__ == "ProgressBarLogger"


def test_iospec_preserves_tree_metadata_for_family_runtime_validation():
    spec = IOSpec(
        inputs=IOTreeSpec(
            structure_tree={"stellar": {"labels": None}},
            channel_names_tree={"stellar": {"labels": ("x0", "x1")}},
        ),
        outputs=IOTreeSpec(
            structure_tree={"spectra": {"flux": None}},
            channel_names_tree={"spectra": {"flux": ("y0", "y1", "y2")}},
        ),
    )
    assert spec.inputs.channel_names_tree == {"stellar": {"labels": ("x0", "x1")}}
    assert spec.outputs.channel_names_tree == {"spectra": {"flux": ("y0", "y1", "y2")}}


def test_root_config_default_still_constructs():
    cfg = RootConfig()
    assert cfg.io.inputs is None
    assert cfg.io.outputs is None


def test_package_declares_typed_marker():
    marker = importlib.resources.files("astro_emulators_toolkit").joinpath("py.typed")
    assert marker.is_file()
