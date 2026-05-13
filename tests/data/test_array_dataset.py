import numpy as np
import pytest

from astro_emulators_toolkit.data import (
    IdentityDeviceBatchTransform,
    MappedDataset,
    TreeArrayDataset,
    XYArrayDataset,
    pack_xy_as_tree,
    train_val_split,
)
from astro_emulators_toolkit.data.toy_utils import generate_rff_dataset


def _canonical_xy(x, y):
    return {"parameters": x}, {"predictions": y}


def test_arraydataset_basic():
    x, y = generate_rff_dataset(
        n_samples=12,
        x_dim=3,
        y_dim=2,
        n_features=8,
        freq_scale=2.0,
        noise_std=0.0,
        x_dist="uniform",
        seed=123,
    )
    sample_weight = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    x_tree, y_tree = _canonical_xy(x, y)

    ds = TreeArrayDataset(x=x_tree, y=y_tree, sample_weight=sample_weight)

    assert len(ds) == 12
    batch = ds.get_batch(np.array([1, 4, 7]))
    assert batch["x"]["parameters"].shape == (3, 3)
    assert batch["y"]["predictions"].shape == (3, 2)
    assert batch["sample_weight"].shape == (3,)
    assert batch["x"]["parameters"].dtype == x.dtype
    assert batch["y"]["predictions"].dtype == y.dtype
    assert batch["sample_weight"].dtype == sample_weight.dtype


def test_train_val_split_disjoint_and_sizes():
    x, y = generate_rff_dataset(
        n_samples=20,
        x_dim=4,
        y_dim=1,
        n_features=6,
        freq_scale=1.5,
        noise_std=0.0,
        x_dist="normal",
        seed=8,
    )
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    train_a, val_a = train_val_split(ds, val_fraction=0.25, seed=99)
    train_b, val_b = train_val_split(ds, val_fraction=0.25, seed=99)

    assert len(train_a) == 15
    assert len(val_a) == 5

    assert set(train_a.indices.tolist()).isdisjoint(set(val_a.indices.tolist()))
    np.testing.assert_array_equal(train_a.indices, train_b.indices)
    np.testing.assert_array_equal(val_a.indices, val_b.indices)


def test_train_val_split_keeps_both_splits_non_empty_for_tiny_dataset():
    x = np.arange(2, dtype=np.float32).reshape(2, 1)
    y = np.arange(2, dtype=np.float32).reshape(2, 1)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    train_ds, val_ds = train_val_split(ds, val_fraction=0.9, seed=0)
    assert len(train_ds) == 1
    assert len(val_ds) == 1


def test_train_val_split_rejects_singleton_dataset():
    x = np.arange(1, dtype=np.float32).reshape(1, 1)
    y = np.arange(1, dtype=np.float32).reshape(1, 1)
    x_tree, y_tree = _canonical_xy(x, y)
    ds = TreeArrayDataset(x=x_tree, y=y_tree)

    with np.testing.assert_raises_regex(ValueError, "at least 2 samples"):
        train_val_split(ds, val_fraction=0.5, seed=0)


def test_array_dataset_accepts_nested_dict_inputs_and_targets():
    x = {
        "stellar": {"labels": np.arange(12, dtype=np.float32).reshape(6, 2)},
        "context": {"aux": np.arange(18, dtype=np.float32).reshape(6, 3)},
    }
    y = {
        "spectra": {"lines": np.arange(30, dtype=np.float32).reshape(6, 5)},
        "photometry": {"continuum": np.arange(18, dtype=np.float32).reshape(6, 3)},
    }

    ds = TreeArrayDataset(x=x, y=y)
    batch = ds.get_batch(np.array([0, 5]))

    assert batch["x"]["stellar"]["labels"].shape == (2, 2)
    assert batch["x"]["context"]["aux"].shape == (2, 3)
    assert batch["y"]["spectra"]["lines"].shape == (2, 5)
    assert batch["y"]["photometry"]["continuum"].shape == (2, 3)


def test_array_dataset_rejects_non_dict_roots():
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = np.arange(4, dtype=np.float32).reshape(4, 1)

    with pytest.raises(ValueError, match="nested dict of arrays"):
        TreeArrayDataset(x=x, y={"predictions": y})
    with pytest.raises(ValueError, match="nested dict of arrays"):
        TreeArrayDataset(x={"parameters": x}, y=y)


def test_array_dataset_rejects_list_and_tuple_branches():
    x = {"parameters": [1.0, 2.0, 3.0]}
    y = {"predictions": np.arange(3, dtype=np.float32).reshape(3, 1)}

    with pytest.raises(ValueError, match="NumPy/JAX array"):
        TreeArrayDataset(x=x, y=y)


def test_array_dataset_rejects_invalid_tree_keys():
    x = {"bad/key": np.arange(8, dtype=np.float32).reshape(4, 2)}
    y = {"predictions": np.arange(4, dtype=np.float32).reshape(4, 1)}

    with pytest.raises(ValueError, match="must not contain '/'"):
        TreeArrayDataset(x=x, y=y)


def test_array_dataset_rejects_non_flat_sample_weight():
    x = {"parameters": np.arange(8, dtype=np.float32).reshape(4, 2)}
    y = {"predictions": np.arange(4, dtype=np.float32).reshape(4, 1)}
    sample_weight = np.ones((4, 1), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "flat 1D"):
        TreeArrayDataset(x=x, y=y, sample_weight=sample_weight)


def test_xy_array_dataset_returns_raw_array_batches():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(18, dtype=np.float32).reshape(6, 3)
    sample_weight = np.linspace(0.0, 1.0, num=6, dtype=np.float32)

    ds = XYArrayDataset(x=x, y=y, sample_weight=sample_weight)
    batch = ds.get_batch(np.array([1, 4]))

    assert batch["x"].shape == (2, 2)
    assert batch["y"].shape == (2, 3)
    np.testing.assert_array_equal(batch["sample_weight"], sample_weight[[1, 4]])


def test_xy_array_dataset_rejects_tree_roots_and_scalar_batches():
    x = {"parameters": np.arange(8, dtype=np.float32).reshape(4, 2)}
    y = np.arange(4, dtype=np.float32).reshape(4, 1)

    with pytest.raises(ValueError, match="must be a NumPy/JAX array"):
        XYArrayDataset(x=x, y=y)
    with pytest.raises(ValueError, match="at least 1D"):
        XYArrayDataset(x=np.array(1.0, dtype=np.float32), y=y)


def test_mapped_dataset_pack_xy_as_tree_wraps_raw_xy_dataset():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(30, dtype=np.float32).reshape(6, 5)

    ds = MappedDataset(
        XYArrayDataset(x=x, y=y),
        map_batch=pack_xy_as_tree(x_leaf="parameters", y_leaf="flux"),
    )
    batch = ds.get_batch(np.array([0, 5]))

    assert batch["x"]["parameters"].shape == (2, 2)
    assert batch["y"]["flux"].shape == (2, 5)


def test_identity_device_batch_transform_is_noop():
    transform = IdentityDeviceBatchTransform()
    batch = {
        "x": {"parameters": np.ones((2, 3), dtype=np.float32)},
        "y": {"predictions": np.zeros((2, 1), dtype=np.float32)},
    }

    assert transform.for_init(batch) is batch
    assert transform(batch, train=True, rng=None) is batch
