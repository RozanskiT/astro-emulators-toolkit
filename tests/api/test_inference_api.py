from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    MinMaxTreeSpec,
    ModelSpec,
    RootConfig,
)


def _make_emulator(*, seed: int = 0) -> Emulator:
    cfg = RootConfig(
        seed=seed,
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"stellar": {"labels": None}}),
            outputs=IOTreeSpec(structure_tree={"spectra": {"flux": None}}),
        ),
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": [8], "activation": "tanh"},
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
    )
    return Emulator.from_config(cfg).configure_training()


def _offset_params(params, delta: float):
    return jax.tree_util.tree_map(
        lambda x: x + delta if hasattr(x, "dtype") else x, params
    )


def test_apply_jax_matches_predict_numpy_output_for_canonical_dict_trees():
    emu = _make_emulator()
    x = {"stellar": {"labels": np.array([[0.1, -0.3], [1.2, 0.2]], dtype=np.float32)}}

    y_jax = emu.apply_jax(x)
    y_np = emu.predict(x)

    assert isinstance(y_jax, dict)
    assert isinstance(y_jax["spectra"]["flux"], jax.Array)
    np.testing.assert_allclose(
        np.asarray(y_jax["spectra"]["flux"]),
        y_np["spectra"]["flux"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_stable_inference_rejects_non_dict_public_inputs():
    emu = _make_emulator()
    with pytest.raises(ValueError, match="canonical dict-tree inputs"):
        emu.apply_jax(np.array([[0.1, 0.2]], dtype=np.float32))
    with pytest.raises(ValueError, match="canonical dict-tree inputs"):
        emu.predict(np.array([[0.1, 0.2]], dtype=np.float32))


def test_initialize_rejects_unbatched_1d_input_leaves():
    cfg = RootConfig(io=IOSpec(), model=ModelSpec(name="mlp", params={}))
    emu = Emulator.from_config(cfg)

    with pytest.raises(ValueError, match="explicit leading batch axis"):
        emu.initialize(inputs={"parameters": np.ones(3, dtype=np.float32)})


def test_stable_inference_rejects_unbatched_1d_input_leaves():
    emu = _make_emulator()

    with pytest.raises(
        ValueError,
        match="predict inputs leaf 'stellar/labels' must include an explicit leading batch axis",
    ):
        emu.predict({"stellar": {"labels": np.ones(2, dtype=np.float32)}})

    with pytest.raises(
        ValueError,
        match="apply_jax inputs leaf 'stellar/labels' must include an explicit leading batch axis",
    ):
        emu.apply_jax({"stellar": {"labels": np.ones(2, dtype=np.float32)}})


def test_make_frozen_apply_captures_parameter_snapshot_with_canonical_io():
    emu = _make_emulator(seed=7)
    x = {"stellar": {"labels": jnp.array([[0.5, -0.25]], dtype=jnp.float32)}}

    frozen = emu.make_frozen_apply(jit=False)
    before = frozen(x)

    emu.params = _offset_params(emu.params, 0.25)

    after_frozen = frozen(x)
    after_live = emu.apply_jax(x)

    np.testing.assert_allclose(
        np.asarray(after_frozen["spectra"]["flux"]),
        np.asarray(before["spectra"]["flux"]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(
        np.asarray(after_live["spectra"]["flux"]),
        np.asarray(before["spectra"]["flux"]),
    )


def test_make_frozen_apply_respects_jit_toggle_with_canonical_io():
    emu = _make_emulator(seed=123)
    x = {"stellar": {"labels": jnp.array([[0.0, 0.1], [0.2, 0.3]], dtype=jnp.float32)}}

    apply_eager = emu.make_frozen_apply(jit=False)
    apply_jitted = emu.make_frozen_apply(jit=True)

    y_eager = apply_eager(x)
    y_jitted = apply_jitted(x)

    np.testing.assert_allclose(
        np.asarray(y_eager["spectra"]["flux"]),
        np.asarray(y_jitted["spectra"]["flux"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_validate_model_state_dict_rejects_non_dict():
    from astro_emulators_toolkit.emulator_runtime import validate_model_state_dict

    with pytest.raises(TypeError, match="model_state must be a nested dict pytree"):
        validate_model_state_dict(("not", "a", "dict"))


def test_validate_model_state_dict_accepts_empty_array_sentinel():
    from astro_emulators_toolkit.emulator_runtime import validate_model_state_dict

    restored = validate_model_state_dict(np.array([], dtype=np.float32))
    assert restored == {}


def test_emulator_from_config_validates_model_io_compatibility():
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="mlp", params={"channels": 0}),
    )

    with pytest.raises(ValueError, match="requires channels > 0"):
        Emulator.from_config(cfg)


def test_from_config_uses_model_init_hints():
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 2, "output_last_axis": 1},
        ),
    )
    emu = Emulator.from_config(cfg)

    pred = emu.predict({"parameters": np.zeros((2, 2), dtype=np.float32)})

    assert pred["predictions"].shape == (2, 1)


def test_metadata_accessors_expose_live_config_spec_blocks():
    cfg = RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(
                structure_tree={"stellar": {"labels": None}},
                channel_names_tree={"stellar": {"labels": ["teff", "logg"]}},
                channel_units_tree={"stellar": {"labels": ["K", "dex"]}},
                leaf_meanings_tree={"stellar": {"labels": "stellar labels"}},
            ),
            outputs=IOTreeSpec(
                structure_tree={"spectra": {"flux": None}},
                channel_names_tree={"spectra": {"flux": ["blue", "green", "red"]}},
                leaf_units_tree={"spectra": {"flux": "normalized"}},
                channel_meanings_tree={
                    "spectra": {"flux": ["blue band", "green band", "red band"]}
                },
            ),
            reference_scaling_inputs=MinMaxTreeSpec(
                min_tree={
                    "stellar": {"labels": [0.0, 1.0]},
                },
                max_tree={
                    "stellar": {"labels": [2.0, 3.0]},
                },
            ),
            reference_scaling_outputs=MinMaxTreeSpec(
                min_tree={
                    "spectra": {"flux": [4.0, 5.0, 6.0]},
                },
                max_tree={
                    "spectra": {"flux": [7.0, 8.0, 9.0]},
                },
            ),
            input_domain=MinMaxTreeSpec(
                min_tree={"stellar": {"labels": [3500.0, 0.0]}},
                max_tree={"stellar": {"labels": [8000.0, 5.0]}},
            ),
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={"input_last_axis": 2, "output_last_axis": 3},
        ),
    )
    emu = Emulator.from_config(cfg)

    assert emu.reference_scaling_inputs == emu.spec["reference_scaling_inputs"]
    assert emu.reference_scaling_outputs == emu.spec["reference_scaling_outputs"]
    assert emu.input_domain == emu.spec["input_domain"]
    assert emu.bundle_extras == {}
    assert emu.input_spec == emu.spec["inputs"]
    assert emu.output_spec == emu.spec["outputs"]
    assert emu.input_channel_names_tree == {"stellar": {"labels": ["teff", "logg"]}}
    assert emu.output_channel_names_tree == {
        "spectra": {"flux": ["blue", "green", "red"]}
    }
    assert emu.input_spec["channel_units_tree"]["stellar"]["labels"] == ["K", "dex"]
    assert emu.output_spec["leaf_units_tree"]["spectra"]["flux"] == "normalized"
    assert emu.output_spec["channel_meanings_tree"]["spectra"]["flux"][0] == "blue band"


def test_predict_returns_numpy_pytree_for_dict_outputs(monkeypatch):
    cfg = RootConfig(io=IOSpec(inputs=IOTreeSpec(structure_tree={"query": None})))
    emu = Emulator(cfg)
    emu.graphdef = object()
    emu.params = {}
    emu.model_state = {}

    def _fake_call(_state_tuple):
        def _forward(x, train=False, rngs=None):
            assert train is False
            assert rngs is None
            return {
                "mean": jnp.ones((x["query"].shape[0], 1), dtype=jnp.float32),
                "logvar": jnp.zeros((x["query"].shape[0], 1), dtype=jnp.float32),
            }, (None, None)

        return _forward

    monkeypatch.setattr("astro_emulators_toolkit.emulator.nnx.call", _fake_call)

    pred = emu.predict({"query": np.zeros((3, 1), dtype=np.float32)})

    assert isinstance(pred, dict)
    assert set(pred) == {"mean", "logvar"}
    assert all(isinstance(v, np.ndarray) for v in pred.values())
    assert pred["mean"].shape == (3, 1)
    assert pred["logvar"].shape == (3, 1)


def test_from_config_accepts_init_example():
    cfg = RootConfig(io=IOSpec())

    emu = Emulator.from_config(
        cfg,
        init_example={
            "inputs": {"parameters": np.zeros((4, 2), dtype=np.float32)},
            "outputs": {"predictions": np.zeros((4, 1), dtype=np.float32)},
        },
    )

    pred = emu.predict({"parameters": np.zeros((2, 2), dtype=np.float32)})

    assert pred["predictions"].shape == (2, 1)


def test_from_config_rejects_legacy_example_arguments():
    cfg = RootConfig(io=IOSpec())

    with pytest.raises(TypeError):
        Emulator.from_config(
            cfg,
            example_x=np.zeros((4, 2), dtype=np.float32),
            example_y=np.zeros((4, 1), dtype=np.float32),
        )


def test_predict_raises_runtime_error_when_emulator_not_initialized():
    cfg = RootConfig(io=IOSpec())
    emu = Emulator(cfg)

    with pytest.raises(RuntimeError, match="initialize"):
        emu.predict({"parameters": np.zeros((2, 1), dtype=np.float32)})
