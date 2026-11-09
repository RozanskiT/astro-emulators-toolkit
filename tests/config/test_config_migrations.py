"""Config schema v1 (toolkit 0.1.x) documents load under v1 meaning.

v1 settings that 0.1.x accepted but ignored become warnings plus the value
that was actually in effect; the same settings in a v2 document are errors.
"""

from __future__ import annotations

import json
import warnings

import pytest

from astro_emulators_toolkit.config import load_config
from astro_emulators_toolkit.config.io import load_config_with_migration_notes
from astro_emulators_toolkit.config.migrations import (
    SUPPORTED_CONFIG_SCHEMA_VERSIONS,
    LegacyConfigWarning,
    wrote_v2_meaning,
)
from astro_emulators_toolkit.config.schema import CONFIG_SCHEMA_VERSION


def _write(tmp_path, schema_version, **sections):
    path = tmp_path / f"config_v{schema_version}.json"
    raw = {"schema_version": schema_version, "model": {"name": "mlp", "params": {}}}
    raw.update(sections)
    path.write_text(json.dumps(raw))
    return path


def _load_strict_no_legacy(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyConfigWarning)
        return load_config(path)


IGNORED_IN_V1 = [
    pytest.param(
        {"optim": {"scale_embedding_lr": 0.5}},
        "scale_embedding_lr has no effect",
        lambda cfg: cfg.optim.scale_embedding_lr == 1.0,
        id="scale_embedding_lr_without_scaling",
    ),
    pytest.param(
        {"optim": {"schedule": "cosine", "decay_steps": 3}},
        "decay_steps is only used by the 'wsd' schedule",
        lambda cfg: cfg.optim.decay_steps == 0,
        id="cosine_decay_steps",
    ),
    pytest.param(
        {"solver": {"name": "gradient", "params": {"foo": 1}}},
        "solver.params is not used by the 'gradient' solver",
        lambda cfg: dict(cfg.solver.params) == {},
        id="gradient_solver_params",
    ),
    pytest.param(
        {"optim": {"lr_scaling": "standard"}},
        "lr_scaling='standard' is ambiguous",
        lambda cfg: cfg.optim.lr_scaling is None,
        id="lr_scaling_standard",
    ),
]


@pytest.mark.parametrize("sections, v2_error, check_v1", IGNORED_IN_V1)
def test_setting_ignored_by_v1_warns_in_v1_and_errors_in_v2(
    tmp_path, sections, v2_error, check_v1
):
    with pytest.warns(LegacyConfigWarning, match="config schema v1") as record:
        cfg = load_config(_write(tmp_path, 1, **sections))
    assert len(record) == 1
    assert cfg.schema_version == CONFIG_SCHEMA_VERSION
    assert check_v1(cfg)

    with pytest.raises(ValueError, match=v2_error):
        load_config(_write(tmp_path, 2, **sections))


def test_v1_si_alias_keeps_its_v1_meaning(tmp_path):
    # In v1, "si" was an alias for None; in v2 it is the legacy SI mode.
    sections = {"optim": {"lr_scaling": "si", "scale_embedding_lr": 0.1}}

    with pytest.warns(LegacyConfigWarning) as record:
        cfg_v1 = load_config(_write(tmp_path, 1, **sections))
    message = str(record[0].message)
    assert "lr_scaling='si' was an alias for None" in message
    assert "scale_embedding_lr=0.1" in message
    assert cfg_v1.optim.lr_scaling is None
    assert cfg_v1.optim.scale_embedding_lr == 1.0

    cfg_v2 = _load_strict_no_legacy(_write(tmp_path, 2, **sections))
    assert cfg_v2.optim.lr_scaling == "si"
    assert cfg_v2.optim.scale_embedding_lr == 0.1


def test_absent_v1_keys_get_v1_defaults_silently(tmp_path):
    cfg_v1 = _load_strict_no_legacy(_write(tmp_path, 1, optim={"name": "adam"}))
    assert cfg_v1.optim.schedule == "constant"
    assert cfg_v1.optim.warmup_steps == 0

    cfg_no_optim = _load_strict_no_legacy(_write(tmp_path, 1))
    assert cfg_no_optim.optim.schedule == "constant"

    cfg_v2 = _load_strict_no_legacy(_write(tmp_path, 2, optim={"name": "adam"}))
    assert cfg_v2.optim.schedule == "cosine"
    assert cfg_v2.optim.warmup_steps is None


def test_v1_non_float32_mlp_keeps_float32_parameter_storage(tmp_path):
    model = {"name": "mlp", "params": {"dtype": "bfloat16"}}

    cfg_v1 = _load_strict_no_legacy(_write(tmp_path, 1, model=model))
    assert cfg_v1.model.params["param_dtype"] == "float32"

    cfg_v2 = _load_strict_no_legacy(_write(tmp_path, 2, model=model))
    assert "param_dtype" not in cfg_v2.model.params


def test_unsupported_schema_version_names_supported_versions(tmp_path):
    with pytest.raises(ValueError, match=r"supported: \[1, 2\]"):
        load_config(_write(tmp_path, 99))
    assert SUPPORTED_CONFIG_SCHEMA_VERSIONS == {1, CONFIG_SCHEMA_VERSION}


@pytest.mark.parametrize(
    "toolkit_version, expected", [("0.1.0", False), (None, False), ("0.2.0.dev0", True)]
)
def test_only_020_prereleases_wrote_v2_meaning_under_a_v1_stamp(
    toolkit_version, expected
):
    assert wrote_v2_meaning(toolkit_version) is expected


def test_prerelease_v1_stamped_config_keeps_si_without_legacy_warning(tmp_path):
    # 0.2.0.dev0 stamped v1 while already writing v2 meaning (a real
    # lr_scaling="si"); bundles carry the toolkit version that says so.
    path = _write(tmp_path, 1, optim={"lr_scaling": "si", "scale_embedding_lr": 0.1})
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyConfigWarning)
        cfg, notes, version = load_config_with_migration_notes(
            path, toolkit_version="0.2.0.dev0"
        )
    assert (notes, version) == ([], 1)
    assert cfg.schema_version == CONFIG_SCHEMA_VERSION
    assert (cfg.optim.lr_scaling, cfg.optim.scale_embedding_lr) == ("si", 0.1)


@pytest.mark.parametrize("toolkit_version", [None, "0.2.0.dev0"])
def test_v1_val_fraction_is_dropped_silently(tmp_path, toolkit_version):
    path = _write(tmp_path, 1, training={"val_fraction": 0.2, "num_steps": 7})
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyConfigWarning)
        cfg, notes, _ = load_config_with_migration_notes(
            path, toolkit_version=toolkit_version
        )
    assert notes == []
    assert cfg.training.num_steps == 7
    assert not hasattr(cfg.training, "val_fraction")


def test_v2_val_fraction_is_an_unknown_key(tmp_path):
    path = _write(tmp_path, 2, training={"val_fraction": 0.2})
    with pytest.raises(
        ValueError, match=r"Unknown training config keys: \['val_fraction'\]"
    ):
        load_config(path)
