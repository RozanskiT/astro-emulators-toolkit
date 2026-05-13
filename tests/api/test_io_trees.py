from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit.io_trees import (
    denormalize_tree,
    get_leaf_by_path,
    iter_leaf_paths,
    normalize_tree,
    set_leaf_by_path,
    validate_metadata_tree_like,
    validate_minmax_values,
    validate_same_structure,
    validate_structure_tree,
)


def test_normalize_denormalize_roundtrip_nested():
    value = {
        "inputs": {"parameters": jnp.array([[2.0, 5.0]])},
        "outputs": {"flux": jnp.array([[0.2, 0.8]])},
    }
    min_tree = {
        "inputs": {"parameters": jnp.array([0.0, 0.0])},
        "outputs": {"flux": 0.0},
    }
    max_tree = {
        "inputs": {"parameters": jnp.array([10.0, 10.0])},
        "outputs": {"flux": 1.0},
    }
    norm = normalize_tree(value, min_tree, max_tree)
    back = denormalize_tree(norm, min_tree, max_tree)
    np.testing.assert_allclose(
        np.asarray(back["inputs"]["parameters"]),
        np.asarray(value["inputs"]["parameters"]),
    )


def test_iter_leaf_paths_and_path_helpers():
    tree = {
        "inputs": {"stellar_labels": [1.0, 2.0]},
        "outputs": {"spectra": {"flux": 0.5}},
    }

    assert list(iter_leaf_paths(tree)) == [
        ("inputs/stellar_labels", [1.0, 2.0]),
        ("outputs/spectra/flux", 0.5),
    ]
    assert get_leaf_by_path(tree, "outputs/spectra/flux") == 0.5

    set_leaf_by_path(tree, "outputs/spectra/ivar", 2.0)
    assert get_leaf_by_path(tree, "outputs/spectra/ivar") == 2.0


@pytest.mark.parametrize(
    "bad_tree", [{"": None}, {"bad/key": None}, {"nested": {"": None}}]
)
def test_validate_structure_tree_rejects_invalid_keys(bad_tree):
    with pytest.raises(ValueError, match="keys must"):
        validate_structure_tree(bad_tree)


def test_validate_structure_tree_rejects_object_leaf():
    with pytest.raises(ValueError, match="array-like value or scalar metadata"):
        validate_structure_tree({"flux": object()})


def test_validate_metadata_tree_like_requires_matching_structure():
    structure_tree = {"spectra": {"flux": None}}

    with pytest.raises(ValueError, match="structure mismatch"):
        validate_metadata_tree_like({"spectra": {"ivar": "relative"}}, structure_tree)


def test_validate_metadata_tree_like_accepts_channel_metadata_sequences():
    structure_tree = {"spectra": {"flux": None}, "stellar_labels": None}
    metadata_tree = {
        "spectra": {"flux": ["line", "continuum"]},
        "stellar_labels": ["teff", "logg"],
    }

    validate_metadata_tree_like(metadata_tree, structure_tree)


def test_structure_mismatch_error():
    with pytest.raises(ValueError):
        validate_same_structure({"a": {"b": 1}}, {"a": 1})


def test_validate_minmax_values_rejects_nonfinite_leaf():
    with pytest.raises(ValueError, match="must be finite"):
        validate_minmax_values(
            {"x": [0.0, np.nan]},
            {"x": [1.0, 2.0]},
            field_name="reference_scaling_inputs",
        )


def test_validate_minmax_values_rejects_nonpositive_span_when_required():
    with pytest.raises(ValueError, match="must have max > min"):
        validate_minmax_values(
            {"x": [0.0, 1.0]},
            {"x": [1.0, 1.0]},
            field_name="reference_scaling_inputs",
            require_positive_span=True,
        )


def test_normalize_tree_rejects_zero_width_bounds():
    with pytest.raises(ValueError, match="must have max > min"):
        normalize_tree({"x": 1.0}, {"x": 0.0}, {"x": 0.0})


def test_normalize_tree_no_longer_accepts_eps_keyword():
    with pytest.raises(TypeError):
        normalize_tree({"x": 1.0}, {"x": 0.0}, {"x": 1.0}, eps=1e-6)


def test_grad_and_jit():
    def loss(x):
        tree = {"x": x}
        out = normalize_tree(tree, {"x": 0.0}, {"x": 2.0})
        return jnp.sum(out["x"])

    grad = jax.grad(loss)(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(grad), np.array([0.5, 0.5], dtype=np.float32))
    jit_loss = jax.jit(loss)
    assert float(jit_loss(jnp.array([1.0, 2.0]))) == pytest.approx(1.5)
