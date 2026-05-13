from __future__ import annotations

import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import IOSpec, ModelSpec, RootConfig, SolverConfig
from astro_emulators_toolkit.data.array_dataset import TreeArrayDataset
from astro_emulators_toolkit.io_trees import iter_leaf_paths
from astro_emulators_toolkit.models.cannon import cannon_design_matrix
from astro_emulators_toolkit.presets import cannon_flux
from astro_emulators_toolkit.training.solvers import ClosedFormLinearSolverConfig


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


class MaskedDataset(TreeArrayDataset):
    def __init__(self, *args, valid_mask, **kwargs):
        super().__init__(*args, **kwargs)
        self._valid_mask = np.asarray(valid_mask, dtype=np.float32)

    def get_batch(self, idx):
        out = super().get_batch(idx)
        idx = np.asarray(idx, dtype=np.int64)
        out["valid_mask"] = self._valid_mask[idx]
        return out


def _dataset(seed: int = 0, n: int = 64):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff = rng.normal(size=(phi.shape[1], 2)).astype(np.float32)
    y = (phi @ coeff).astype(np.float32)
    return x, y


def test_closed_form_linear_matches_one_shot_solution_on_well_conditioned_problem():
    x, y = _dataset(seed=3, n=50)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))
    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff = np.linalg.solve(phi.T @ phi, phi.T @ y)
    np.testing.assert_allclose(
        _coefficients_from_params(emu.params), coeff, rtol=1e-5, atol=1e-5
    )


def test_closed_form_linear_handles_scalar_target_leaf_as_single_output_channel():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff_true = rng.normal(size=(phi.shape[1],)).astype(np.float32)
    y = (phi @ coeff_true).astype(np.float32).reshape(-1)

    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="cannon",
            params={"include_bias": True},
        ),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))

    fitted = _coefficients_from_params(emu.params)
    assert emu.model_init == {"input_last_axis": x.shape[1], "output_last_axis": 1}
    assert fitted.shape == (phi.shape[1], 1)
    np.testing.assert_allclose(fitted, coeff_true[:, None], rtol=1e-5, atol=1e-5)
    pred = emu.predict({"parameters": x})["predictions"]
    assert pred.shape == (x.shape[0], 1)
    np.testing.assert_allclose(pred[:, 0], y, rtol=1e-5, atol=1e-5)


def test_closed_form_linear_rejects_non_matrix_target_leaf():
    x, y = _dataset(seed=5, n=24)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y[:, :, None])

    with pytest.raises(ValueError, match=r"shape \(N,\) or \(N, C\)"):
        emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))


def test_closed_form_linear_handles_collinear_features_with_ridge():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(80, 1)).astype(np.float32)
    x = np.concatenate(
        [base, 2.0 * base, rng.normal(size=(80, 1)).astype(np.float32)], axis=1
    )
    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff_true = rng.normal(size=(phi.shape[1], 1)).astype(np.float32)
    y = (phi @ coeff_true).astype(np.float32)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 1e-3}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))
    pred = emu.predict({"parameters": x})["predictions"]
    assert float(np.mean((pred - y) ** 2)) < 1e-5


def test_closed_form_linear_streaming_accumulation_matches_full_accumulation():
    x, y = _dataset(seed=9, n=43)
    w = np.linspace(0.5, 1.5, num=x.shape[0], dtype=np.float32)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.2}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=w))

    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    ws = np.sqrt(np.clip(w, 0.0, None))[:, None]
    ridge = np.diag(_ridge_diagonal(phi.shape[1], 0.2, regularize_intercept=False))
    coeff_full = np.linalg.solve(
        (phi * ws).T @ (phi * ws) + ridge, (phi * ws).T @ (y * ws)
    )
    np.testing.assert_allclose(
        _coefficients_from_params(emu.params), coeff_full, rtol=1e-5, atol=1e-5
    )


def test_cannon_preset_uses_stable_regularization_default():
    cfg = cannon_flux()
    assert cfg.solver.name == "closed_form_linear"
    assert float(cfg.solver.params["ridge"]) > 0.0
    assert ClosedFormLinearSolverConfig.from_mapping({}).ridge > 0.0
    assert ClosedFormLinearSolverConfig.from_mapping({}).regularize_intercept is False


def test_closed_form_linear_solver_parses_regularize_intercept_boolean_strings():
    solver_cfg = ClosedFormLinearSolverConfig.from_mapping(
        {"regularize_intercept": "yes"}
    )
    assert solver_cfg.regularize_intercept is True


def test_closed_form_linear_solver_rejects_invalid_regularize_intercept_value():
    with pytest.raises(ValueError, match=r"regularize_intercept must be a boolean\."):
        ClosedFormLinearSolverConfig.from_mapping({"regularize_intercept": 2})


def test_closed_form_linear_solver_rejects_negative_ridge():
    with pytest.raises(ValueError, match="ridge must be >= 0"):
        ClosedFormLinearSolverConfig.from_mapping({"ridge": -1e-6})


def test_closed_form_linear_respects_valid_mask_and_sample_weight():
    x, y = _dataset(seed=7, n=32)
    weights = np.ones((32,), dtype=np.float32)
    valid_mask = np.ones((32,), dtype=np.float32)
    valid_mask[:8] = 0.0

    x_tree, y_tree = _canonical_xy(x, y)
    ds = MaskedDataset(x=x_tree, y=y_tree, sample_weight=weights, valid_mask=valid_mask)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 0.1}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    emu.fit(ds)

    phi = np.asarray(cannon_design_matrix(x[8:], include_bias=True))
    ridge = np.diag(_ridge_diagonal(phi.shape[1], 0.1, regularize_intercept=False))
    coeff_expected = np.linalg.solve(phi.T @ phi + ridge, phi.T @ y[8:])
    np.testing.assert_allclose(
        _coefficients_from_params(emu.params), coeff_expected, rtol=1e-4, atol=1e-4
    )


def test_closed_form_linear_skips_intercept_regularization_by_default():
    x, y = _dataset(seed=21, n=40)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(name="closed_form_linear", params={"ridge": 5.0}),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))

    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff_expected = np.linalg.solve(
        phi.T @ phi
        + np.diag(
            _ridge_diagonal(
                phi.shape[1],
                5.0,
                regularize_intercept=False,
            )
        ),
        phi.T @ y,
    )
    np.testing.assert_allclose(
        _coefficients_from_params(emu.params),
        coeff_expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_closed_form_linear_can_regularize_intercept_when_requested():
    x, y = _dataset(seed=22, n=40)
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(name="cannon", params={"include_bias": True}),
        solver=SolverConfig(
            name="closed_form_linear",
            params={"ridge": 5.0, "regularize_intercept": True},
        ),
    )
    emu = Emulator.from_config(cfg).configure_training()
    x_tree, y_tree = _canonical_xy(x, y)
    emu.fit(TreeArrayDataset(x=x_tree, y=y_tree))

    phi = np.asarray(cannon_design_matrix(x, include_bias=True))
    coeff_expected = np.linalg.solve(
        phi.T @ phi
        + np.diag(
            _ridge_diagonal(
                phi.shape[1],
                5.0,
                regularize_intercept=True,
            )
        ),
        phi.T @ y,
    )
    np.testing.assert_allclose(
        _coefficients_from_params(emu.params),
        coeff_expected,
        rtol=1e-5,
        atol=1e-5,
    )
