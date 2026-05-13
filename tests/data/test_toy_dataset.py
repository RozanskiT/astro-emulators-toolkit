import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit.data import (
    DataLoader,
    make_flux_batch_transform,
    make_intensity_batch_transform,
)
from astro_emulators_toolkit.data.toy import (
    ToyBinaryClassificationDataset,
    ToyIntensityDataset,
    ToyNormalizedFluxDataset,
    ToyRFFDataset,
)
from astro_emulators_toolkit.data.npy_table import SubsetDataset


def test_toy_rff_dataset_shapes_and_determinism():
    ds1 = ToyRFFDataset(n_samples=10, x_dim=3, y_dim=2, n_features=5, seed=42)
    ds2 = ToyRFFDataset(n_samples=10, x_dim=3, y_dim=2, n_features=5, seed=42)

    idx = np.arange(10)
    batch = ds1.get_batch(idx)
    assert batch["x"].shape == (10, 3)
    assert batch["y"].shape == (10, 2)
    np.testing.assert_allclose(batch["x"], ds2.get_batch(idx)["x"])
    np.testing.assert_allclose(batch["y"], ds2.get_batch(idx)["y"])


def test_toy_normalized_flux_dataset_shapes():
    ds = ToyNormalizedFluxDataset(n_samples=8, x_dim=5, y_dim=32, seed=0)
    batch = ds.get_batch(np.arange(4))
    assert batch["x"].shape == (4, 5)
    assert batch["y"].shape == (4, 32)


def test_toy_intensity_dataset_shapes():
    ds = ToyIntensityDataset(n_samples=8, x_dim=6, y_dim=24, seed=1)
    batch = ds.get_batch(np.arange(3))
    assert batch["x"].shape == (3, 6)
    assert batch["y"].shape == (3, 24, 2)


def test_toy_binary_classification_dataset_shapes_and_labels():
    ds = ToyBinaryClassificationDataset(n_samples=20, x_dim=4, n_features=6, seed=2)
    batch = ds.get_batch(np.arange(20))
    assert batch["x"].shape == (20, 4)
    assert batch["y"].shape == (20, 1)
    assert set(np.unique(batch["y"]).tolist()).issubset({0.0, 1.0})


def test_dataloader_handles_subset_without_special_cases():
    ds = ToyRFFDataset(n_samples=6, x_dim=2, y_dim=1, n_features=4, seed=3)
    subset = SubsetDataset(base=ds, indices=np.array([4, 1, 5, 0]))

    loader = DataLoader(dataset=subset, batch_size=2, shuffle=False)

    b0 = loader.train_batch(0)
    b1 = loader.train_batch(1)

    np.testing.assert_allclose(b0["x"], ds.x[[4, 1]])
    np.testing.assert_allclose(b1["x"], ds.x[[5, 0]])


def test_subset_loader_batches_do_not_repeat_first_batch():
    ds = ToyRFFDataset(n_samples=10, x_dim=2, y_dim=1, n_features=4, seed=21)
    train_like_subset = SubsetDataset(base=ds, indices=np.array([8, 2, 6, 1, 9, 0]))

    loader = DataLoader(dataset=train_like_subset, batch_size=2, shuffle=False)

    np.testing.assert_allclose(loader.train_batch(0)["x"], ds.x[[8, 2]])
    np.testing.assert_allclose(loader.train_batch(1)["x"], ds.x[[6, 1]])
    np.testing.assert_allclose(loader.train_batch(2)["x"], ds.x[[9, 0]])


def test_dataloader_step_sampling_is_deterministic_and_cycle_specific():
    ds = ToyRFFDataset(n_samples=8, x_dim=2, y_dim=1, n_features=4, seed=7)
    loader_a = DataLoader(dataset=ds, batch_size=4, shuffle=True, seed=11)
    loader_b = DataLoader(dataset=ds, batch_size=4, shuffle=True, seed=11)

    first_cycle_a = loader_a.train_batch(0)["x"].copy()
    first_cycle_b = loader_b.train_batch(0)["x"].copy()
    second_cycle = loader_a.train_batch(2)["x"].copy()

    np.testing.assert_allclose(first_cycle_a, first_cycle_b)
    assert not np.array_equal(first_cycle_a, second_cycle)


def test_flux_batch_transform_interpolates_single_grid_outputs():
    ds = ToyNormalizedFluxDataset(n_samples=8, x_dim=5, y_dim=32, seed=0)
    eval_grid = np.linspace(
        float(ds.wavelength_grid[0]),
        float(ds.wavelength_grid[-1]),
        num=9,
        dtype=np.float32,
    )
    transform = make_flux_batch_transform(
        wavelength_grid=ds.wavelength_grid,
        n_wavelength=9,
        eval_wavelength_grid=eval_grid,
    )
    loader = DataLoader(dataset=ds, batch_size=4, shuffle=False)

    batch = loader.train_batch(0)
    out = transform(
        {"x": jnp.asarray(batch["x"]), "y": jnp.asarray(batch["y"])},
        rng=jax.random.key(5),
        train=True,
    )
    x_params, x_wave = out["x"]

    assert x_params.shape == (4, 5)
    assert x_wave.shape == (4, 9)
    assert out["y"].shape == (4, 9)


def test_intensity_batch_transform_interpolates_two_reference_grids_to_one_query_grid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 6)).astype(np.float32)
    wave_lines = np.linspace(0.2, 1.2, 11, dtype=np.float32)
    wave_cont = np.linspace(0.25, 1.25, 7, dtype=np.float32)
    y_lines = np.sin(x[:, :1] + wave_lines[None, :]).astype(np.float32)
    y_cont = np.cos(x[:, :1] + wave_cont[None, :]).astype(np.float32)

    eval_grid = np.linspace(0.3, 1.1, num=7, dtype=np.float32)
    transform = make_intensity_batch_transform(
        common_waves={"lines": wave_lines, "continuum": wave_cont},
        n_wavelength=7,
        eval_wavelength_grid=eval_grid,
        output_order=("lines", "continuum"),
    )

    batch = {
        "x": jnp.asarray(x),
        "y": {"lines": jnp.asarray(y_lines), "continuum": jnp.asarray(y_cont)},
    }
    out = transform(batch, rng=jax.random.key(13), train=True)
    assert out["x"][0].shape == (3, 6)
    assert out["x"][1].shape == (3, 7)
    assert out["y"].shape == (3, 7, 2)


def test_flux_batch_transform_preserves_sample_weight_and_extra_keys():
    ds = ToyNormalizedFluxDataset(n_samples=6, x_dim=3, y_dim=16, seed=9)
    eval_grid = np.linspace(
        float(ds.wavelength_grid[0]),
        float(ds.wavelength_grid[-1]),
        num=5,
        dtype=np.float32,
    )
    transform = make_flux_batch_transform(
        wavelength_grid=ds.wavelength_grid,
        n_wavelength=5,
        eval_wavelength_grid=eval_grid,
    )

    batch = {
        "x": jnp.asarray(ds.x[:3]),
        "y": jnp.asarray(ds.y[:3]),
        "sample_weight": jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32),
        "meta": jnp.array([7, 8, 9], dtype=jnp.int32),
    }

    train_out = transform(batch, rng=jax.random.key(0), train=True)
    eval_out = transform(batch, rng=jax.random.key(1), train=False)

    np.testing.assert_allclose(
        np.asarray(train_out["sample_weight"]), np.asarray(batch["sample_weight"])
    )
    np.testing.assert_array_equal(
        np.asarray(train_out["meta"]), np.asarray(batch["meta"])
    )
    np.testing.assert_allclose(
        np.asarray(eval_out["sample_weight"]), np.asarray(batch["sample_weight"])
    )
    np.testing.assert_array_equal(
        np.asarray(eval_out["meta"]), np.asarray(batch["meta"])
    )


def test_flux_batch_transform_uses_eval_wavelength_grid_when_provided():
    ds = ToyNormalizedFluxDataset(n_samples=6, x_dim=3, y_dim=16, seed=9)
    eval_grid = np.linspace(
        float(ds.wavelength_grid[0]),
        float(ds.wavelength_grid[-1]),
        num=5,
        dtype=np.float32,
    )
    transform = make_flux_batch_transform(
        wavelength_grid=ds.wavelength_grid,
        n_wavelength=5,
        eval_wavelength_grid=eval_grid,
    )
    batch = {"x": jnp.asarray(ds.x[:2]), "y": jnp.asarray(ds.y[:2])}
    out = transform(batch, rng=jax.random.key(0), train=False)
    np.testing.assert_allclose(
        np.asarray(out["x"][1]), np.broadcast_to(eval_grid[None, :], (2, 5))
    )


def test_flux_batch_transform_for_init_matches_eval_path():
    ds = ToyNormalizedFluxDataset(n_samples=6, x_dim=3, y_dim=16, seed=9)
    eval_grid = np.linspace(
        float(ds.wavelength_grid[0]),
        float(ds.wavelength_grid[-1]),
        num=5,
        dtype=np.float32,
    )
    transform = make_flux_batch_transform(
        wavelength_grid=ds.wavelength_grid,
        n_wavelength=5,
        eval_wavelength_grid=eval_grid,
    )
    batch = {"x": jnp.asarray(ds.x[:2]), "y": jnp.asarray(ds.y[:2])}

    init_out = transform.for_init(batch)
    eval_out = transform(batch, rng=jax.random.key(0), train=False)

    np.testing.assert_allclose(
        np.asarray(init_out["x"][1]), np.asarray(eval_out["x"][1])
    )
    np.testing.assert_allclose(np.asarray(init_out["y"]), np.asarray(eval_out["y"]))


def test_flux_batch_transform_reuses_training_grid_when_lengths_match():
    ds = ToyNormalizedFluxDataset(n_samples=6, x_dim=3, y_dim=16, seed=9)
    transform = make_flux_batch_transform(
        wavelength_grid=ds.wavelength_grid, n_wavelength=ds.wavelength_grid.shape[0]
    )
    batch = {"x": jnp.asarray(ds.x[:2]), "y": jnp.asarray(ds.y[:2])}
    out = transform(batch, rng=jax.random.key(0), train=False)
    expected = np.broadcast_to(
        np.asarray(ds.wavelength_grid, dtype=np.float32)[None, :],
        (2, ds.wavelength_grid.shape[0]),
    )
    np.testing.assert_allclose(np.asarray(out["x"][1]), expected)


def test_flux_batch_transform_raises_when_eval_grid_required_but_missing():
    ds = ToyNormalizedFluxDataset(n_samples=6, x_dim=3, y_dim=16, seed=9)
    with pytest.raises(ValueError, match="Evaluation wavelength grid must be provided"):
        make_flux_batch_transform(wavelength_grid=ds.wavelength_grid, n_wavelength=5)


def test_flux_batch_transform_is_linear_in_flux_channel():
    wavelength_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    eval_grid = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    transform = make_flux_batch_transform(
        wavelength_grid=wavelength_grid,
        n_wavelength=eval_grid.shape[0],
        eval_wavelength_grid=eval_grid,
    )

    batch = {
        "x": jnp.zeros((2, 2), dtype=jnp.float32),
        "y": jnp.array(
            [
                2.0 * wavelength_grid + 1.0,
                -1.5 * wavelength_grid + 4.0,
            ],
            dtype=jnp.float32,
        ),
    }

    out = transform(batch, rng=jax.random.key(0), train=False)
    expected = np.array(
        [
            2.0 * eval_grid + 1.0,
            -1.5 * eval_grid + 4.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.asarray(out["y"]), expected, atol=1e-6)


def test_intensity_batch_transform_is_linear_in_each_channel():
    wave_lines = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    wave_continuum = np.array([0.25, 1.25, 2.25, 3.25], dtype=np.float32)
    eval_grid = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    transform = make_intensity_batch_transform(
        common_waves={"lines": wave_lines, "continuum": wave_continuum},
        n_wavelength=eval_grid.shape[0],
        eval_wavelength_grid=eval_grid,
        output_order=("lines", "continuum"),
    )

    batch = {
        "x": jnp.zeros((2, 2), dtype=jnp.float32),
        "y": {
            "lines": jnp.array(
                [
                    3.0 * wave_lines - 1.0,
                    -2.0 * wave_lines + 5.0,
                ],
                dtype=jnp.float32,
            ),
            "continuum": jnp.array(
                [
                    0.5 * wave_continuum + 2.0,
                    -1.25 * wave_continuum + 3.5,
                ],
                dtype=jnp.float32,
            ),
        },
    }

    out = transform(batch, rng=jax.random.key(1), train=False)
    expected = np.stack(
        [
            np.array(
                [
                    3.0 * eval_grid - 1.0,
                    -2.0 * eval_grid + 5.0,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.5 * eval_grid + 2.0,
                    -1.25 * eval_grid + 3.5,
                ],
                dtype=np.float32,
            ),
        ],
        axis=-1,
    )

    np.testing.assert_allclose(np.asarray(out["y"]), expected, atol=1e-6)


def test_intensity_batch_transform_requires_eval_wavelength_grid():
    with pytest.raises(TypeError):
        make_intensity_batch_transform(
            common_waves={
                "lines": np.linspace(0.0, 1.0, 5, dtype=np.float32),
                "continuum": np.linspace(0.0, 1.0, 6, dtype=np.float32),
            },
            n_wavelength=5,
        )
