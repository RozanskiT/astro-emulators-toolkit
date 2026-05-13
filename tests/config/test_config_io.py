from __future__ import annotations

import json

import numpy as np
import pytest

from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    MinMaxTreeSpec,
    ModelSpec,
    NpyTableConfig,
    OptimConfig,
    RootConfig,
    SolverConfig,
    TaskSpec,
    TrainConfig,
    load_config,
    save_config,
)
from astro_emulators_toolkit.config.schema import CONFIG_SCHEMA_VERSION
from astro_emulators_toolkit.spec import (
    SPEC_VERSION,
    materialize_effective_spec,
    validate_spec,
)


def _example_inputs() -> IOTreeSpec:
    return IOTreeSpec(
        structure_tree={"stellar_labels": None, "observation": {"wavelengths": None}},
        channel_names_tree={
            "stellar_labels": ["teff", "logg"],
            "observation": {"wavelengths": None},
        },
        leaf_units_tree={
            "stellar_labels": "dex",
            "observation": {"wavelengths": "angstrom"},
        },
        channel_units_tree={
            "stellar_labels": ["K", "dex"],
            "observation": {"wavelengths": None},
        },
        leaf_meanings_tree={
            "stellar_labels": "stellar label vector",
            "observation": {"wavelengths": "grid"},
        },
        channel_meanings_tree={
            "stellar_labels": ["effective_temperature", "surface_gravity"],
            "observation": {"wavelengths": None},
        },
    )


def _example_outputs() -> IOTreeSpec:
    return IOTreeSpec(
        structure_tree={"spectra": {"flux": None}},
        channel_names_tree={"spectra": {"flux": ["line", "continuum"]}},
        leaf_units_tree={"spectra": {"flux": "normalized"}},
        channel_units_tree={"spectra": {"flux": ["relative", "relative"]}},
        leaf_meanings_tree={"spectra": {"flux": "predicted flux"}},
        channel_meanings_tree={"spectra": {"flux": ["line_flux", "continuum_flux"]}},
    )


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    path = tmp_path / "bad.json"
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "seed": 0,
        "model": {"name": "mlp", "params": {}},
        "solvre": {"name": "gradient"},
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Unknown top-level config keys"):
        load_config(path)


@pytest.mark.parametrize("filename", ["config.json", "config.yaml"])
def test_config_roundtrip_preserves_tree_based_io_sections(tmp_path, filename):
    cfg = RootConfig(
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp", params={"channels": 2}
        ),
        training=TrainConfig(
            logging_interval_steps=None,
            evaluation_interval_steps=25,
            logging_steps=(10, 30, 100),
            evaluation_steps=(30, 300),
            checkpoint_interval_steps=None,
            checkpoint_steps=(40, 400),
            max_saved_checkpoints=None,
        ),
        io=IOSpec(
            inputs=_example_inputs(),
            outputs=_example_outputs(),
            reference_scaling_inputs=MinMaxTreeSpec(
                min_tree={
                    "stellar_labels": [0.0, 0.0],
                    "observation": {"wavelengths": 4000.0},
                },
                max_tree={
                    "stellar_labels": [1.0, 1.0],
                    "observation": {"wavelengths": 8000.0},
                },
            ),
            reference_scaling_outputs=MinMaxTreeSpec(
                min_tree={
                    "spectra": {"flux": 0.0},
                },
                max_tree={
                    "spectra": {"flux": 1.0},
                },
            ),
            input_domain=MinMaxTreeSpec(
                min_tree={
                    "stellar_labels": [3000.0, 0.0],
                    "observation": {"wavelengths": 4000.0},
                },
                max_tree={
                    "stellar_labels": [8000.0, 5.0],
                    "observation": {"wavelengths": 8000.0},
                },
            ),
        ),
    )
    path = tmp_path / filename

    save_config(cfg, path)
    loaded = load_config(path)

    assert loaded.schema_version == CONFIG_SCHEMA_VERSION
    assert isinstance(loaded.io.inputs, IOTreeSpec)
    assert isinstance(loaded.io.outputs, IOTreeSpec)
    assert isinstance(loaded.io.reference_scaling_inputs, MinMaxTreeSpec)
    assert isinstance(loaded.io.reference_scaling_outputs, MinMaxTreeSpec)
    assert loaded.model.init_hints == {}
    assert loaded.training.logging_interval_steps is None
    assert loaded.training.evaluation_interval_steps == 25
    assert loaded.training.logging_steps == (10, 30, 100)
    assert loaded.training.evaluation_steps == (30, 300)
    assert loaded.training.checkpoint_interval_steps is None
    assert loaded.training.checkpoint_steps == (40, 400)
    assert loaded.training.max_saved_checkpoints is None
    assert loaded.io.inputs.structure_tree == {
        "stellar_labels": None,
        "observation": {"wavelengths": None},
    }
    assert loaded.io.outputs.channel_names_tree == {
        "spectra": {"flux": ["line", "continuum"]}
    }
    assert loaded.io.reference_scaling_inputs.min_tree["stellar_labels"] == [0.0, 0.0]
    assert loaded.io.reference_scaling_outputs.max_tree["spectra"]["flux"] == 1.0
    assert loaded.io.input_domain.max_tree["stellar_labels"] == [8000.0, 5.0]
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        assert raw["training"]["logging_interval_steps"] is None
        assert raw["training"]["evaluation_interval_steps"] == 25
        assert raw["training"]["logging_steps"] == [10, 30, 100]
        assert raw["training"]["evaluation_steps"] == [30, 300]
        assert raw["training"]["checkpoint_interval_steps"] is None
        assert raw["training"]["checkpoint_steps"] == [40, 400]
        assert raw["training"]["max_saved_checkpoints"] is None
        assert "x_dim" not in raw["io"]
        assert "y_dim" not in raw["io"]
    else:
        text = path.read_text()
        assert "schema_version:" in text
        assert '"schema_version"' not in text
        assert "logging_steps:" in text
        assert "evaluation_steps:" in text
        assert "checkpoint_steps:" in text
        assert "x_dim" not in text
        assert "y_dim" not in text


def test_materialize_effective_spec_uses_inputs_outputs_sections():
    cfg = RootConfig(
        model=ModelSpec(
            name="experimental/explicit_wavelength_mlp", params={"channels": 2}
        ),
        io=IOSpec(inputs=_example_inputs(), outputs=_example_outputs()),
    )

    spec = materialize_effective_spec(cfg)

    assert spec["spec_version"] == SPEC_VERSION
    assert "x" not in spec
    assert "y" not in spec
    assert spec["inputs"]["structure_tree"] == {
        "stellar_labels": None,
        "observation": {"wavelengths": None},
    }
    assert spec["outputs"]["channel_names_tree"] == {
        "spectra": {"flux": ["line", "continuum"]}
    }


def test_validate_spec_rejects_legacy_x_y_fields():
    cfg = RootConfig(io=IOSpec())
    spec = {
        "spec_version": SPEC_VERSION,
        "x": {"names": ["teff"]},
        "inputs": {"structure_tree": {"stellar_labels": None}},
        "outputs": {"structure_tree": {"flux": None}},
    }

    with pytest.raises(ValueError, match="Legacy spec keys are not supported"):
        validate_spec(spec, cfg)


@pytest.mark.parametrize(
    "structure_tree",
    [
        {"": None},
        {"bad/key": None},
        {"nested": {"": None}},
    ],
)
def test_iotree_spec_rejects_invalid_keys(structure_tree):
    with pytest.raises(ValueError, match="keys must"):
        IOTreeSpec(structure_tree=structure_tree)


def test_minmax_tree_spec_rejects_non_numeric_leaf_values():
    with pytest.raises(ValueError, match="must be numeric"):
        MinMaxTreeSpec(
            min_tree={"stellar_labels": "low"}, max_tree={"stellar_labels": "high"}
        )


def test_minmax_tree_spec_rejects_nonfinite_and_reversed_values():
    with pytest.raises(ValueError, match="must be finite"):
        MinMaxTreeSpec(
            min_tree={"stellar_labels": [0.0, np.nan]},
            max_tree={"stellar_labels": [1.0, 2.0]},
        )
    with pytest.raises(ValueError, match="must have max >= min"):
        MinMaxTreeSpec(
            min_tree={"stellar_labels": [1.0, 0.0]},
            max_tree={"stellar_labels": [0.0, 2.0]},
        )


def test_iospec_reference_scaling_requires_positive_span():
    with pytest.raises(
        ValueError,
        match="io.reference_scaling_inputs leaf 'parameters' must have max > min",
    ):
        IOSpec(
            reference_scaling_inputs=MinMaxTreeSpec(
                min_tree={"parameters": [0.0, 1.0]},
                max_tree={"parameters": [1.0, 1.0]},
            )
        )


def test_iospec_input_domain_allows_zero_width_leaf():
    io = IOSpec(
        input_domain=MinMaxTreeSpec(
            min_tree={"parameters": [3500.0, 0.0]},
            max_tree={"parameters": [3500.0, 5.0]},
        )
    )

    assert io.input_domain is not None
    assert io.input_domain.max_tree["parameters"][0] == 3500.0


def test_validate_spec_rejects_unknown_public_keys():
    cfg = RootConfig(io=IOSpec())
    spec = {
        "spec_version": SPEC_VERSION,
        "inputs": {"structure_tree": {"stellar_labels": None}},
        "outputs": {"structure_tree": {"flux": None}},
        "extras": {"wavelength_angstrom": [5000.0]},
    }

    with pytest.raises(ValueError, match="Unknown spec keys"):
        validate_spec(spec, cfg)


def test_load_config_preserves_json_channel_name_lists_and_model_init_hints(tmp_path):
    raw = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "io": {
            "inputs": {
                "structure_tree": {"parameters": None},
                "channel_names_tree": {"parameters": ["x0", "x1"]},
            },
            "outputs": {
                "structure_tree": {"target": None},
                "channel_names_tree": {"target": ["y0"]},
            },
        },
        "model": {
            "name": "mlp",
            "params": {},
            "init_hints": {"input_last_axis": 2, "output_last_axis": 1},
        },
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))

    cfg = load_config(path)

    assert cfg.io.inputs.channel_names_tree["parameters"] == ["x0", "x1"]
    assert cfg.io.outputs.channel_names_tree["target"] == ["y0"]
    assert cfg.model.init_hints == {"input_last_axis": 2, "output_last_axis": 1}


def test_root_config_canonicalizes_registry_names():
    cfg = RootConfig(
        model=ModelSpec(name=" MLP ", params={}),
        task=TaskSpec(name=" Regression ", params={}),
        solver=SolverConfig(name=" AUTO ", params={}),
        optim=OptimConfig(name=" ADAMW ", schedule=" COSINE ", lr_scaling=" NONE "),
    )

    assert cfg.model.name == "mlp"
    assert cfg.task.name == "regression"
    assert cfg.solver.name == "auto"
    assert cfg.optim.name == "adamw"
    assert cfg.optim.schedule == "cosine"
    assert cfg.optim.lr_scaling is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1.0}, "optim.lr must be >= 0"),
        ({"weight_decay": -1.0}, "optim.weight_decay must be >= 0"),
        ({"grad_clip": -1.0}, "optim.grad_clip must be >= 0"),
        ({"b1": 1.0}, "optim.b1 must satisfy"),
        ({"b2": -0.1}, "optim.b2 must satisfy"),
        ({"eps": 0.0}, "optim.eps must be > 0"),
        ({"precondition_frequency": 0}, "optim.precondition_frequency must be > 0"),
        ({"scale_embedding_lr": -1.0}, "optim.scale_embedding_lr must be >= 0"),
        ({"lr_scaling": "wide"}, "optim.lr_scaling must be"),
    ],
)
def test_optim_config_rejects_invalid_numeric_constraints(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OptimConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"batch_size": 0}, "training.batch_size must be > 0"),
        ({"num_steps": -1}, "training.num_steps must be >= 0"),
        ({"val_fraction": 1.5}, "training.val_fraction must satisfy"),
        ({"steps_per_epoch": 0}, "training.steps_per_epoch must be > 0 or None"),
    ],
)
def test_train_config_rejects_invalid_numeric_constraints(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**kwargs)


def test_train_config_rejects_non_integer_shuffle_seed():
    with pytest.raises(TypeError, match="training.shuffle_seed must be an integer"):
        TrainConfig(shuffle_seed=1.5)


def test_train_config_allows_zero_max_saved_checkpoints():
    assert TrainConfig(max_saved_checkpoints=0).max_saved_checkpoints == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"inputs": ()}, "inputs must be non-empty"),
        ({"targets": ()}, "targets must be non-empty"),
        ({"dtype": "not-a-dtype"}, "valid NumPy dtype"),
        ({"columns": ("teff", "teff")}, "columns must be unique"),
    ],
)
def test_npy_table_config_rejects_invalid_metadata(kwargs, match):
    base = {"path": "table.npy", "inputs": ("teff",), "targets": ("flux",)}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        NpyTableConfig(**base)


def test_solver_config_rejects_negative_closed_form_ridge():
    with pytest.raises(ValueError, match="solver.params.ridge must be >= 0"):
        SolverConfig(name="closed_form_linear", params={"ridge": -1e-4})


def test_load_config_rejects_legacy_io_dim_keys(tmp_path):
    raw = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "io": {
            "x_dim": 2,
            "inputs": {"structure_tree": {"parameters": None}},
            "outputs": {"structure_tree": {"target": None}},
        },
        "model": {"name": "mlp", "params": {}},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="Unknown io config keys"):
        load_config(path)


def test_load_config_allows_missing_optional_sections(tmp_path):
    raw = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "model": {"name": "mlp", "params": {}},
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(raw))

    cfg = load_config(path)

    assert cfg.task.name == "regression"
    assert cfg.optim.name == "adamw"
    assert cfg.io.inputs is None
    assert cfg.io.outputs is None
    assert cfg.model.init_hints == {}


def test_load_config_yaml_scientific_notation_values_are_coerced(tmp_path):
    path = tmp_path / "grid.yaml"
    path.write_text(
        """
schema_version: 1
seed: 7
model:
  name: mlp
  params:
    hidden_sizes: [64, 64]
task:
  name: regression
optim:
  name: adamw
  lr: 1e-4
  weight_decay: 3e-4
training:
  workdir: ./runs/yaml_grid
  num_steps: 12
  batch_size: 16
io:
  inputs:
    structure_tree:
      parameters: null
  outputs:
    structure_tree:
      flux: null
  reference_scaling_inputs:
    min_tree:
      parameters: [1e-4, 3e-4]
    max_tree:
      parameters: [1e-2, 3e-2]
  reference_scaling_outputs:
    min_tree:
      flux: 0.0
    max_tree:
      flux: 1.0
"""
    )

    cfg = load_config(path)

    assert cfg.seed == 7
    assert cfg.training.num_steps == 12
    assert cfg.training.batch_size == 16
    assert cfg.optim.lr == pytest.approx(1e-4)
    assert cfg.optim.weight_decay == pytest.approx(3e-4)
    assert cfg.io.reference_scaling_inputs.min_tree["parameters"] == pytest.approx(
        [1e-4, 3e-4]
    )
    assert cfg.io.reference_scaling_inputs.max_tree["parameters"] == pytest.approx(
        [1e-2, 3e-2]
    )
    assert cfg.io.reference_scaling_outputs.max_tree["flux"] == pytest.approx(1.0)


def test_train_config_normalizes_explicit_step_schedules():
    cfg = RootConfig(
        training=TrainConfig(
            logging_interval_steps=0,
            evaluation_interval_steps=None,
            checkpoint_interval_steps=0,
            logging_steps=[10, 30, 30, 100],
            evaluation_steps=(30, 300),
            checkpoint_steps=[40, 80, 80],
            max_saved_checkpoints=None,
        )
    )

    assert cfg.training.logging_interval_steps is None
    assert cfg.training.evaluation_interval_steps is None
    assert cfg.training.checkpoint_interval_steps is None
    assert cfg.training.logging_steps == (10, 30, 100)
    assert cfg.training.evaluation_steps == (30, 300)
    assert cfg.training.checkpoint_steps == (40, 80)
    assert cfg.training.max_saved_checkpoints is None


def test_train_config_rejects_negative_intervals():
    with pytest.raises(
        ValueError, match="training.logging_interval_steps must be >= 0 or None"
    ):
        RootConfig(training=TrainConfig(logging_interval_steps=-1))
    with pytest.raises(
        ValueError, match="training.checkpoint_interval_steps must be >= 0 or None"
    ):
        RootConfig(training=TrainConfig(checkpoint_interval_steps=-1))
    with pytest.raises(
        ValueError, match="training.max_saved_checkpoints must be >= 0 or None"
    ):
        RootConfig(training=TrainConfig(max_saved_checkpoints=-1))


def test_train_config_rejects_non_positive_explicit_step_schedules():
    with pytest.raises(ValueError, match="training.logging_steps\\[1\\] must be > 0"):
        RootConfig(training=TrainConfig(logging_steps=[10, 0]))
    with pytest.raises(
        ValueError, match="training.checkpoint_steps\\[1\\] must be > 0"
    ):
        RootConfig(training=TrainConfig(checkpoint_steps=[10, 0]))
