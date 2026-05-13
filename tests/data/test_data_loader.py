from __future__ import annotations

import numpy as np
import pytest

from astro_emulators_toolkit.config.schema import NpyTableConfig
from astro_emulators_toolkit.data import TreeArrayDataset, DataLoader, XYArrayDataset
from astro_emulators_toolkit.data.npy_table import NpyTableDataset
from astro_emulators_toolkit.data.protocols import DatasetProtocol
from astro_emulators_toolkit.data.subset import SubsetDataset


def _canonical_xy(x, y):
    return {"parameters": x}, {"predictions": y}


def test_train_loader_wraps_across_permutation_boundary_and_keeps_full_batch():
    x_tree, y_tree = _canonical_xy(np.arange(10)[:, None], np.arange(10)[:, None])
    ds = TreeArrayDataset(x=x_tree, y=y_tree)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    b2 = loader.train_batch(2)
    np.testing.assert_array_equal(
        b2["x"]["parameters"].reshape(-1), np.array([8, 9, 0, 1])
    )
    assert b2["valid_mask"].shape == (4,)


def test_train_loader_is_deterministic_from_global_step():
    x_tree, y_tree = _canonical_xy(np.arange(7)[:, None], np.arange(7)[:, None])
    ds = TreeArrayDataset(x=x_tree, y=y_tree)
    l1 = DataLoader(ds, batch_size=5, shuffle=True, seed=123)
    l2 = DataLoader(ds, batch_size=5, shuffle=True, seed=123)
    np.testing.assert_array_equal(
        l1.train_batch(3)["x"]["parameters"], l2.train_batch(3)["x"]["parameters"]
    )


def test_train_loader_reuses_cached_permutation_within_cycle(monkeypatch):
    x_tree, y_tree = _canonical_xy(np.arange(10)[:, None], np.arange(10)[:, None])
    ds = TreeArrayDataset(x=x_tree, y=y_tree)
    loader = DataLoader(ds, batch_size=3, shuffle=True, seed=17)

    calls = {"count": 0}
    original = np.random.default_rng

    def _counting_rng(seed):
        calls["count"] += 1
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", _counting_rng)

    loader.train_batch(0)
    loader.train_batch(1)
    loader.train_batch(2)
    assert calls["count"] == 1

    loader.train_batch(3)
    assert calls["count"] == 2


def test_train_loader_handles_dataset_smaller_than_batch_size():
    x_tree, y_tree = _canonical_xy(np.arange(3)[:, None], np.arange(3)[:, None])
    ds = TreeArrayDataset(x=x_tree, y=y_tree)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    b = loader.train_batch(0)
    assert b["x"]["parameters"].shape[0] == 8


def test_eval_loader_pads_last_batch_and_emits_valid_mask():
    x_tree, y_tree = _canonical_xy(np.arange(5)[:, None], np.arange(5)[:, None])
    ds = TreeArrayDataset(x=x_tree, y=y_tree)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batches = list(loader.iter_eval_batches())
    assert len(batches) == 2
    np.testing.assert_array_equal(
        batches[1]["valid_mask"], np.array([1, 0, 0, 0], dtype=np.float32)
    )


def test_array_dataset_subset_npytable_satisfy_dataset_protocol(tmp_path):
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    y = np.arange(3, dtype=np.float32).reshape(3, 1)
    x_tree, y_tree = _canonical_xy(x, y)
    arr_ds = TreeArrayDataset(x=x_tree, y=y_tree)
    raw_ds = XYArrayDataset(x=x, y=y)
    sub_ds = SubsetDataset(base=arr_ds, indices=np.array([0, 2]))

    table = np.concatenate([x, y], axis=1)
    path = tmp_path / "table.npy"
    np.save(path, table)
    npy_ds = NpyTableDataset.from_config(
        NpyTableConfig(
            path=str(path),
            memmap=False,
            inputs=(0, 1),
            targets=(2,),
            columns=None,
            dtype="float32",
        )
    )

    assert isinstance(arr_ds, DatasetProtocol)
    assert isinstance(raw_ds, DatasetProtocol)
    assert isinstance(sub_ds, DatasetProtocol)
    assert isinstance(npy_ds, DatasetProtocol)


class _MaskedDataset:
    def __init__(self):
        self.x = {"parameters": np.arange(5, dtype=np.float32)[:, None]}
        self.y = {"predictions": np.arange(5, dtype=np.float32)[:, None]}
        self.mask = np.array([1.0, 0.0, 1.0, 0.5, 1.0], dtype=np.float32)

    def __len__(self):
        return int(self.x["parameters"].shape[0])

    def get_batch(self, idx):
        return {
            "x": {"parameters": self.x["parameters"][idx]},
            "y": {"predictions": self.y["predictions"][idx]},
            "valid_mask": self.mask[idx],
        }


def test_train_loader_preserves_dataset_valid_mask():
    ds = _MaskedDataset()
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batch = loader.train_batch(0)
    np.testing.assert_array_equal(
        batch["valid_mask"], np.array([1.0, 0.0, 1.0], dtype=np.float32)
    )


def test_eval_loader_combines_dataset_valid_mask_with_padding_mask():
    ds = _MaskedDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batches = list(loader.iter_eval_batches())
    np.testing.assert_array_equal(
        batches[1]["valid_mask"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


def test_eval_loader_preserves_full_valid_mask_when_no_padding():
    ds = _MaskedDataset()
    loader = DataLoader(ds, batch_size=5, shuffle=False)
    only = list(loader.iter_eval_batches())[0]
    np.testing.assert_array_equal(only["valid_mask"], ds.mask)


def test_array_dataset_supports_pytree_targets():
    x = {"parameters": np.arange(12, dtype=np.float32).reshape(6, 2)}
    y = {
        "lines": np.arange(30, dtype=np.float32).reshape(6, 5),
        "continuum": np.arange(18, dtype=np.float32).reshape(6, 3),
    }
    ds = TreeArrayDataset(x=x, y=y)
    batch = ds.get_batch(np.array([1, 4]))
    assert batch["x"]["parameters"].shape == (2, 2)
    assert batch["y"]["lines"].shape == (2, 5)
    assert batch["y"]["continuum"].shape == (2, 3)


def test_array_dataset_rejects_mismatched_pytree_leading_dims():
    x = {"parameters": np.arange(8, dtype=np.float32).reshape(4, 2)}
    y = {
        "lines": np.arange(20, dtype=np.float32).reshape(4, 5),
        "continuum": np.arange(9, dtype=np.float32).reshape(3, 3),
    }
    with pytest.raises(ValueError, match="same first dimension"):
        TreeArrayDataset(x=x, y=y)


def test_dataloader_with_pytree_targets_preserves_structure():
    x = {"parameters": np.arange(12, dtype=np.float32).reshape(6, 2)}
    y = {
        "lines": np.arange(30, dtype=np.float32).reshape(6, 5),
        "continuum": np.arange(18, dtype=np.float32).reshape(6, 3),
    }
    ds = TreeArrayDataset(x=x, y=y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    train_batch = loader.train_batch(0)
    assert train_batch["x"]["parameters"].shape == (4, 2)
    assert train_batch["y"]["lines"].shape == (4, 5)
    assert train_batch["y"]["continuum"].shape == (4, 3)

    eval_batch = list(loader.iter_eval_batches())[1]
    assert eval_batch["y"]["lines"].shape == (4, 5)
    assert eval_batch["y"]["continuum"].shape == (4, 3)
    np.testing.assert_array_equal(
        eval_batch["valid_mask"], np.array([1, 1, 0, 0], dtype=np.float32)
    )
