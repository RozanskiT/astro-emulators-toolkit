from __future__ import annotations

import numpy as np
import pytest

from astro_emulators_toolkit.tasks import build_task


def test_regression_task_rejects_unimplemented_loss():
    with pytest.raises(ValueError, match="loss='huber' not implemented"):
        build_task("regression", {"loss": "huber"})


def test_regression_task_defaults_to_mse_and_mae_only_for_array_path():
    task = build_task("regression", {"loss": "mse"})

    pred = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    y = np.array([[1.2, 1.9], [0.0, 2.0]], dtype=np.float32)

    loss, metrics = task.loss_and_metrics(pred=pred, batch={"y": y})

    assert float(loss) >= 0.0
    assert set(metrics.keys()) == {"mse", "mae"}


def test_regression_task_supports_metric_axis_groups_for_array_outputs():
    task = build_task(
        "regression",
        {
            "loss": "mse",
            "metrics": ["mse", "mae"],
            "metric_axes": {"global": "all", "per_channel": [0]},
        },
    )

    pred = np.array(
        [
            [[1.0, 0.0], [3.0, 2.0]],
            [[2.0, 1.0], [4.0, 3.0]],
        ],
        dtype=np.float32,
    )
    y = np.zeros_like(pred)

    _, metrics = task.loss_and_metrics(pred=pred, batch={"y": y})

    assert metrics["mse"] == pytest.approx(5.5)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["mse_per_channel_0"] == pytest.approx(7.5)
    assert metrics["mse_per_channel_1"] == pytest.approx(3.5)
    assert metrics["mae_per_channel_0"] == pytest.approx(2.5)
    assert metrics["mae_per_channel_1"] == pytest.approx(1.5)


def test_regression_task_emits_global_and_per_leaf_metrics_for_nested_dict_outputs():
    task = build_task(
        "regression",
        {
            "loss": "mse",
            "metrics": ["mse", "mae"],
            "loss_weights": {"normalized_flux": 2.0, "labels/teff": 1.0},
        },
    )

    pred = {
        "spectra": {
            "normalized_flux": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        },
        "labels": {"teff": np.array([[0.0], [2.0]], dtype=np.float32)},
    }
    y = {
        "spectra": {"normalized_flux": np.zeros((2, 2), dtype=np.float32)},
        "labels": {"teff": np.zeros((2, 1), dtype=np.float32)},
    }

    loss, metrics = task.loss_and_metrics(pred=pred, batch={"y": y})

    assert float(metrics["mse/spectra/normalized_flux"]) == pytest.approx(7.5)
    assert float(metrics["mse/labels/teff"]) == pytest.approx(2.0)
    assert float(metrics["mae/spectra/normalized_flux"]) == pytest.approx(2.5)
    assert float(metrics["mae/labels/teff"]) == pytest.approx(1.0)
    assert float(metrics["mse"]) == pytest.approx((2.0 * 7.5 + 1.0 * 2.0) / 3.0)
    assert float(metrics["mae"]) == pytest.approx((2.0 * 2.5 + 1.0 * 1.0) / 3.0)
    assert float(loss) == pytest.approx(float(metrics["mse"]))


def test_regression_task_supports_per_dimension_metrics_for_dict_leaves():
    task = build_task(
        "regression",
        {
            "loss": "mse",
            "metrics": ["mse", "mae"],
            "metric_axes": {"global": "all", "per_dim": []},
        },
    )

    pred = {"outputs": {"flux": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}}
    y = {"outputs": {"flux": np.zeros((2, 2), dtype=np.float32)}}

    _, metrics = task.loss_and_metrics(pred=pred, batch={"y": y})

    assert float(metrics["mse"]) == pytest.approx(7.5)
    assert float(metrics["mae"]) == pytest.approx(2.5)
    assert float(metrics["mse/outputs/flux"]) == pytest.approx(7.5)
    assert float(metrics["mae/outputs/flux"]) == pytest.approx(2.5)
    assert float(metrics["mse_per_dim_0"]) == pytest.approx(5.0)
    assert float(metrics["mse_per_dim_1"]) == pytest.approx(10.0)
    assert float(metrics["mae_per_dim_0"]) == pytest.approx(2.0)
    assert float(metrics["mae_per_dim_1"]) == pytest.approx(3.0)
    assert float(metrics["mse_per_dim_0/outputs/flux"]) == pytest.approx(5.0)
    assert float(metrics["mse_per_dim_1/outputs/flux"]) == pytest.approx(10.0)


def test_regression_task_weighted_metrics_and_losses_for_array_path():
    pred = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    y = np.array([[1.0, 1.0], [4.0, 10.0]], dtype=np.float32)
    sample_weight = np.array([1.0, 3.0], dtype=np.float32)

    task = build_task(
        "regression",
        {
            "loss": "weighted_mse",
            "metrics": ["mse", "weighted_mse", "mae", "weighted_mae"],
        },
    )

    loss, metrics = task.loss_and_metrics(
        pred=pred, batch={"y": y, "sample_weight": sample_weight}
    )

    np.testing.assert_allclose(float(metrics["mse"]), 4.5, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["weighted_mse"]), 6.25, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["mae"]), 1.5, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["weighted_mae"]), 1.75, rtol=1e-6)
    np.testing.assert_allclose(float(loss), float(metrics["weighted_mse"]), rtol=1e-6)


def test_regression_task_rejects_ambiguous_leaf_name_in_loss_weights():
    task = build_task("regression", {"loss": "mse", "loss_weights": {"flux": 1.0}})

    pred = {
        "a": {"flux": np.array([[1.0]], dtype=np.float32)},
        "b": {"flux": np.array([[1.0]], dtype=np.float32)},
    }
    y = {
        "a": {"flux": np.array([[1.0]], dtype=np.float32)},
        "b": {"flux": np.array([[1.0]], dtype=np.float32)},
    }

    with pytest.raises(ValueError, match="ambiguous"):
        task.loss_and_metrics(pred=pred, batch={"y": y})


def test_regression_task_rejects_unknown_leaf_name_in_loss_weights():
    task = build_task("regression", {"loss": "mse", "loss_weights": {"missing": 1.0}})
    pred = {"outputs": {"flux": np.array([[1.0]], dtype=np.float32)}}
    y = {"outputs": {"flux": np.array([[1.0]], dtype=np.float32)}}

    with pytest.raises(ValueError, match="does not match any prediction leaf"):
        task.loss_and_metrics(pred=pred, batch={"y": y})


def test_regression_task_rejects_mismatched_output_leaf_paths():
    task = build_task("regression", {"loss": "mse"})

    pred = {"a": {"flux": np.array([[1.0]], dtype=np.float32)}}
    y = {
        "a": {"flux": np.array([[1.0]], dtype=np.float32)},
        "b": {"flux": np.array([[1.0]], dtype=np.float32)},
    }

    with pytest.raises(ValueError, match="identical leaf paths"):
        task.loss_and_metrics(pred=pred, batch={"y": y})


def test_regression_task_rejects_broadcastable_shape_mismatch():
    task = build_task("regression", {"loss": "mse"})
    pred = np.array([[1.0], [2.0]], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        task.loss_and_metrics(pred=pred, batch={"y": y})


def test_regression_task_rejects_shape_mismatch_per_dict_leaf():
    task = build_task("regression", {"loss": "mse"})
    pred = {"outputs": {"flux": np.array([[1.0], [2.0]], dtype=np.float32)}}
    y = {"outputs": {"flux": np.array([1.0, 2.0], dtype=np.float32)}}

    with pytest.raises(ValueError, match="shape mismatch"):
        task.loss_and_metrics(pred=pred, batch={"y": y})


def test_regression_task_column_vector_sample_weight_matches_vector_weight():
    pred = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    y = np.array([[1.0, 1.0], [4.0, 10.0]], dtype=np.float32)
    w_vec = np.array([1.0, 3.0], dtype=np.float32)
    w_col = w_vec[:, None]

    task = build_task(
        "regression",
        {"loss": "weighted_mse", "metrics": ["weighted_mse", "weighted_mae"]},
    )

    loss_vec, metrics_vec = task.loss_and_metrics(
        pred=pred, batch={"y": y, "sample_weight": w_vec}
    )
    loss_col, metrics_col = task.loss_and_metrics(
        pred=pred, batch={"y": y, "sample_weight": w_col}
    )

    np.testing.assert_allclose(float(loss_col), float(loss_vec), rtol=1e-6)
    np.testing.assert_allclose(
        float(metrics_col["weighted_mse"]),
        float(metrics_vec["weighted_mse"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(metrics_col["weighted_mae"]),
        float(metrics_vec["weighted_mae"]),
        rtol=1e-6,
    )


def test_regression_task_rejects_invalid_sample_weight_shape():
    task = build_task("regression", {"loss": "weighted_mse"})
    pred = np.array([[1.0], [2.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="sample_weight must have shape"):
        task.loss_and_metrics(
            pred=pred,
            batch={"y": y, "sample_weight": np.ones((2, 2), dtype=np.float32)},
        )
