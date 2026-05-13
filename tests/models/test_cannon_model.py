from __future__ import annotations

import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle.bundle import Bundle
from astro_emulators_toolkit.config import (
    IOSpec,
    ModelSpec,
    RootConfig,
    SolverConfig,
    TrainConfig,
)
from astro_emulators_toolkit.data import IdentityDeviceBatchTransform
from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset
from astro_emulators_toolkit.io_trees import iter_leaf_paths
from astro_emulators_toolkit.training.solvers import (
    default_solver_for_model,
    resolve_solver,
)
from astro_emulators_toolkit.models.cannon import (
    cannon_design_matrix,
    cannon_feature_dim,
)


def _canonical_xy(x, y):
    return {"parameters": x}, {"predictions": y}


def _coefficients_from_params(params):
    leaves = list(iter_leaf_paths(params))
    assert len(leaves) == 1
    return np.asarray(leaves[0][1])


def _ridge_diagonal(
    feature_dim: int, ridge: float, *, regularize_intercept: bool
) -> np.ndarray:
    diag = np.full((feature_dim,), float(ridge), dtype=np.float32)
    if feature_dim > 0 and not regularize_intercept:
        diag[0] = 0.0
    return diag


def _make_quadratic_dataset(
    n: int = 96, in_dim: int = 3, out_dim: int = 4, seed: int = 0
):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, in_dim)).astype(np.float32)

    feat_dim = cannon_feature_dim(in_dim, include_bias=True)
    coeff_true = rng.normal(size=(feat_dim, out_dim)).astype(np.float32)

    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    y = phi @ coeff_true
    weights = np.linspace(0.5, 1.5, num=n, dtype=np.float32)
    return x, y.astype(np.float32), weights, coeff_true


def _manual_cannon_design_matrix(x: np.ndarray, *, include_bias: bool) -> np.ndarray:
    columns = [x]
    if include_bias:
        columns.insert(0, np.ones((x.shape[0], 1), dtype=x.dtype))

    quadratic_terms = []
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            quadratic_terms.append((x[:, i] * x[:, j])[:, None])
    if quadratic_terms:
        columns.append(np.concatenate(quadratic_terms, axis=1))
    return np.concatenate(columns, axis=1)


def test_solver_defaults_are_model_specific():
    assert default_solver_for_model("cannon") == "closed_form_linear"
    assert (
        default_solver_for_model(
            "cannon", task_name="regression", task_params={"loss": "mae"}
        )
        == "gradient"
    )
    assert default_solver_for_model("mlp") == "gradient"
    assert resolve_solver("gradient", model_name="cannon").name == "gradient"


def test_cannon_design_matrix_matches_manual_quadratic_expansion():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(7, 5)).astype(np.float32)

    expected = _manual_cannon_design_matrix(x, include_bias=True)
    actual = np.asarray(cannon_design_matrix(x, include_bias=True))

    np.testing.assert_array_equal(actual, expected)


def test_cannon_closed_form_rejected_for_incompatible_task_loss():
    with pytest.raises(
        ValueError, match="only valid for regression tasks with squared loss"
    ):
        resolve_solver(
            "closed_form_linear",
            model_name="cannon",
            task_name="regression",
            task_params={"loss": "mae"},
        )


def test_cannon_closed_form_fit_recovers_quadratic_mapping():
    x, y, weights, _ = _make_quadratic_dataset()
    x_tree, y_tree = _canonical_xy(x, y)
    train_ds = TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=weights)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
        training=TrainConfig(logging_interval_steps=1, evaluation_interval_steps=1),
    )
    emu = Emulator.from_config(cfg).configure_training()

    history = emu.fit(train_ds, validation_dataset=train_ds)
    pred = emu.predict({"parameters": x})["predictions"]

    np.testing.assert_allclose(pred, y, rtol=1e-4, atol=1e-4)
    assert "training_loss" in history.logs
    assert "validation_loss" in history.logs
    assert history.logs["training_loss"][-1] < 1e-8
    assert emu.last_fit_method == "closed_form_linear"


def test_cannon_closed_form_parses_string_include_bias_false():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    phi = _manual_cannon_design_matrix(x, include_bias=False)
    coeff_true = rng.normal(size=(phi.shape[1], 2)).astype(np.float32)
    y = (phi @ coeff_true).astype(np.float32)

    x_tree, y_tree = _canonical_xy(x, y)
    train_ds = TreeArrayDataset(x=x_tree, y=y_tree)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": "false"}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
        training=TrainConfig(logging_interval_steps=1, evaluation_interval_steps=1),
    )
    emu = Emulator.from_config(cfg).configure_training()

    emu.fit(train_ds, validation_dataset=train_ds)

    np.testing.assert_allclose(
        emu.predict({"parameters": x})["predictions"], y, rtol=1e-4, atol=1e-4
    )
    assert _coefficients_from_params(emu.params).shape == coeff_true.shape


def test_cannon_closed_form_respects_explicit_eval_schedule():
    x, y, weights, _ = _make_quadratic_dataset()
    x_tree, y_tree = _canonical_xy(x, y)
    train_ds = TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=weights)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
        training=TrainConfig(
            logging_interval_steps=None,
            logging_steps=(),
            evaluation_interval_steps=None,
            evaluation_steps=(1,),
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()

    history = emu.fit(train_ds, validation_dataset=train_ds)

    assert "training_loss" not in history.logs
    assert history.logs["validation_step"] == [1.0]


def test_cannon_gradient_override_runs_and_is_recorded():
    x, y, _, _ = _make_quadratic_dataset(n=64, out_dim=2, seed=7)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        training=TrainConfig(
            batch_size=32,
            num_steps=40,
            logging_interval_steps=10,
            evaluation_interval_steps=20,
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()

    history = emu.fit(ds, validation_dataset=ds, method="gradient")

    assert "training_loss" in history.logs
    assert emu.last_fit_method == "gradient"


def test_cannon_bundle_round_trip_preserves_predictions_and_fit_method(tmp_path):
    x, y, _, _ = _make_quadratic_dataset(n=64, seed=11)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        training=TrainConfig(workdir=str(tmp_path / "run")),
    )
    emu = Emulator.from_config(cfg).configure_training()
    emu.fit(ds)

    before = emu.predict({"parameters": x[:8]})["predictions"]

    bundle_dir = emu.save_bundle(tmp_path / "bundle")
    metadata = Bundle.load(bundle_dir).metadata
    assert metadata["fit_method"] == "closed_form_linear"
    assert metadata["solver_metadata"]["name"] == "closed_form_linear"
    assert metadata["solver_metadata"]["params"]["ridge"] > 0.0
    assert metadata["solver_metadata"]["params"]["regularize_intercept"] is False
    assert metadata["solver_metadata"]["design_matrix"] == {
        "kind": "cannon_quadratic_v1",
        "include_bias": True,
        "intercept_column_index": 0,
    }
    assert metadata["solver_metadata"]["diagnostics"]["solution_backend"] in {
        "solve",
        "lstsq",
    }
    assert metadata["solver_metadata"]["diagnostics"]["condition_number"] is not None

    restored = Emulator.from_bundle(bundle_dir).configure_training()
    after = restored.predict({"parameters": x[:8]})["predictions"]

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


def test_mlp_rejects_closed_form_method():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    y = rng.normal(size=(16, 2)).astype(np.float32)
    x_tree, y_tree = _canonical_xy(x, y)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="mlp", params={"hidden_sizes": [8]}),
    )
    emu = Emulator.from_config(cfg).configure_training()

    with pytest.raises(ValueError, match="does not support solver"):
        emu.fit(TreeArrayDataset(x=x_tree, y=y_tree), method="closed_form_linear")


def test_cannon_closed_form_solver_streams_batches():
    class SpyArrayDataset(TreeArrayDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def get_batch(self, idx):
            idx = np.asarray(idx)
            self.calls.append(idx.copy())
            return super().get_batch(idx)

    x, y, weights, _ = _make_quadratic_dataset(n=17)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = SpyArrayDataset(x=x_tree, y=y_tree, sample_weight=weights)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
        training=TrainConfig(batch_size=4),
    )
    emu = Emulator.from_config(cfg).configure_training()
    emu.fit(ds, validation_dataset=None)

    assert len(ds.calls) >= 2
    assert all(call.shape[0] <= 17 for call in ds.calls)
    assert not any(
        call.shape == (17,) and np.array_equal(call, np.arange(17, dtype=np.int64))
        for call in ds.calls
    )


def test_cannon_streamed_and_dense_closed_form_agree():
    x, y, weights, _ = _make_quadratic_dataset(n=48, in_dim=3, out_dim=2, seed=19)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=weights)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.2}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    emu.fit(ds)

    coeff_streamed = _coefficients_from_params(emu.params)
    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    w = np.sqrt(np.clip(weights, 0.0, None))[:, None]
    phi_w = phi * w
    y_w = y * w
    ata = phi_w.T @ phi_w + np.diag(
        _ridge_diagonal(
            phi_w.shape[1],
            0.2,
            regularize_intercept=False,
        )
    )
    atb = phi_w.T @ y_w
    coeff_dense = np.linalg.solve(ata, atb)

    np.testing.assert_allclose(coeff_streamed, coeff_dense, rtol=1e-5, atol=1e-5)


def test_cannon_closed_form_accepts_identity_device_batch_transform():
    x, y, weights, _ = _make_quadratic_dataset(n=48, seed=13)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=weights)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()

    emu.fit(
        ds,
        validation_dataset=ds,
        device_batch_transform=IdentityDeviceBatchTransform(),
    )

    assert emu.last_fit_method == "closed_form_linear"


def test_cannon_closed_form_rejects_non_identity_device_batch_transform():
    class NoopTransform:
        def for_init(self, batch):
            return batch

        def __call__(self, batch, *, train: bool, rng):
            del train, rng
            return batch

    x, y, _, _ = _make_quadratic_dataset(n=32, seed=17)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()

    with pytest.raises(ValueError, match="IdentityDeviceBatchTransform"):
        emu.fit(
            ds,
            validation_dataset=ds,
            device_batch_transform=NoopTransform(),
        )
