from __future__ import annotations

from flax import nnx
import jax.numpy as jnp
import pytest

from astro_emulators_toolkit.experimental.models.explicit_wavelength_mlp import (
    ExplicitWavelengthMLPConfig,
)
from astro_emulators_toolkit.experimental.models.mlp_2d_regression import (
    MLP2DRegressionConfig,
)
from astro_emulators_toolkit.experimental.models.siren import SirenConfig
from astro_emulators_toolkit.models.cannon import CannonConfig
from astro_emulators_toolkit.models.mlp import MLP, MLPConfig
from astro_emulators_toolkit.models.transformer_payne import TransformerPayneConfig


@pytest.mark.parametrize(
    ("config_cls", "payload", "expected"),
    [
        (MLPConfig, {"use_bias": "off"}, {"use_bias": False}),
        (CannonConfig, {"include_bias": "yes"}, {"include_bias": True}),
        (
            TransformerPayneConfig,
            {
                "bias_dense": "on",
                "bias_parameter_embedding": "off",
                "bias_feed_forward": "1",
                "bias_output_head": "yes",
                "bias_attention": "0",
            },
            {
                "bias_dense": True,
                "bias_parameter_embedding": False,
                "bias_feed_forward": True,
                "bias_output_head": True,
                "bias_attention": False,
            },
        ),
        (
            ExplicitWavelengthMLPConfig,
            {"use_bias": "no"},
            {"use_bias": False},
        ),
        (
            MLP2DRegressionConfig,
            {"use_bias": "1"},
            {"use_bias": True},
        ),
        (
            SirenConfig,
            {"use_bias": "false"},
            {"use_bias": False},
        ),
    ],
)
def test_model_config_from_dict_parses_boolean_strings(config_cls, payload, expected):
    cfg = config_cls.from_dict(payload)

    for field_name, expected_value in expected.items():
        assert getattr(cfg, field_name) is expected_value


@pytest.mark.parametrize(
    ("config_cls", "payload", "field_name"),
    [
        (MLPConfig, {"use_bias": "maybe"}, "use_bias"),
        (CannonConfig, {"include_bias": 2}, "include_bias"),
        (TransformerPayneConfig, {"bias_dense": []}, "bias_dense"),
        (
            TransformerPayneConfig,
            {"bias_parameter_embedding": []},
            "bias_parameter_embedding",
        ),
        (TransformerPayneConfig, {"bias_feed_forward": []}, "bias_feed_forward"),
        (TransformerPayneConfig, {"bias_output_head": []}, "bias_output_head"),
        (
            ExplicitWavelengthMLPConfig,
            {"use_bias": object()},
            "use_bias",
        ),
        (MLP2DRegressionConfig, {"use_bias": "sure"}, "use_bias"),
        (SirenConfig, {"use_bias": "sure"}, "use_bias"),
    ],
)
def test_model_config_from_dict_rejects_invalid_boolean_values(
    config_cls, payload, field_name
):
    with pytest.raises(ValueError, match=rf"{field_name} must be a boolean\."):
        config_cls.from_dict(payload)


def test_transformer_payne_fine_grained_biases_inherit_from_bias_dense():
    cfg = TransformerPayneConfig.from_dict(
        {"bias_dense": "on", "bias_feed_forward": "off"}
    )

    assert cfg.bias_dense is True
    assert cfg.bias_parameter_embedding is None
    assert cfg.bias_feed_forward is False
    assert cfg.bias_output_head is None
    assert cfg.use_parameter_embedding_bias is True
    assert cfg.use_feed_forward_bias is False
    assert cfg.use_output_head_bias is True


def test_mlp_config_parses_output_activation_and_reference_scaling_fields():
    cfg = MLPConfig.from_dict(
        {
            "hidden_sizes": [32, 32],
            "activation": "gelu",
            "output_activation": "sigmoid",
            "reference_width": 16,
            "reference_depth": 2,
        }
    )

    assert cfg.output_activation == "sigmoid"
    assert cfg.reference_width == 16
    assert cfg.reference_depth == 2


def test_mlp_output_activation_is_applied_to_final_layer():
    model = MLP(
        in_dim=2,
        out_dim=1,
        cfg=MLPConfig(hidden_sizes=(4,), output_activation="sigmoid"),
        rngs=nnx.Rngs(0),
    )

    y = model(jnp.asarray([[0.1, -0.2]], dtype=jnp.float32))

    assert y.shape == (1, 1)
    assert jnp.all(y >= 0.0)
    assert jnp.all(y <= 1.0)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"hidden_sizes": [32, 0]}, "hidden_sizes"),
        ({"activation": "unknown_activation"}, "Unknown activation"),
        ({"output_activation": "unknown_activation"}, "Unknown activation"),
        ({"reference_width": 0}, "reference_width must be > 0"),
        ({"reference_depth": 0}, "reference_depth must be > 0"),
        ({"dtype": "not-a-dtype"}, "valid JAX dtype"),
    ],
)
def test_mlp_config_from_dict_rejects_invalid_numeric_and_symbolic_values(
    payload, match
):
    with pytest.raises(ValueError, match=match):
        MLPConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"hidden_sizes": [32, 0]}, "hidden_sizes"),
        ({"hidden_sizes": []}, "hidden_sizes"),
        ({"omega0_first": 0.0}, "omega0_first"),
        ({"omega0_hidden": float("inf")}, "omega0_hidden"),
        ({"dtype": "not-a-dtype"}, "valid JAX dtype"),
    ],
)
def test_siren_config_from_dict_rejects_invalid_values(payload, match):
    with pytest.raises((TypeError, ValueError), match=match):
        SirenConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"dim": 0}, "dim must be > 0"),
        ({"dim_ff_multiplier": 0}, "dim_ff_multiplier must be > 0"),
        ({"no_tokens": 0}, "no_tokens must be > 0"),
        ({"no_layers": 0}, "no_layers must be > 0"),
        ({"dim_head": 0}, "dim_head must be > 0"),
        ({"channels": 0}, "channels must be > 0"),
        ({"min_period": 0.0}, "min_period must be > 0"),
        ({"min_period": 2.0, "max_period": 1.0}, "max_period must be >= min_period"),
        ({"sigma": 0.0}, "sigma must be > 0"),
        ({"alpha_emb": float("nan")}, "alpha_emb must be finite"),
        ({"alpha_att": float("inf")}, "alpha_att must be finite"),
        ({"reference_depth": 0}, "reference_depth must be > 0"),
        ({"reference_width": 0}, "reference_width must be > 0"),
        ({"activation": "not_an_activation"}, "Unknown activation"),
        ({"emb_init": "random"}, "emb_init must be one of"),
        ({"dtype": "int32"}, "cfg.dtype must be 'float32' or 'float64'"),
    ],
)
def test_transformer_payne_config_from_dict_rejects_invalid_values(payload, match):
    with pytest.raises(ValueError, match=match):
        TransformerPayneConfig.from_dict(payload)
