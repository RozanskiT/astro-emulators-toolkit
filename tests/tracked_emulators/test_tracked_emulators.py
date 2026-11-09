"""Tracked pretrained emulators still load and predict as they did.

Opt-in, because it downloads every bundle in registry.toml from the Hugging
Face Hub (cached after the first run):

    AEMU_TRACKED_EMULATORS=1 uv run pytest tests/tracked_emulators -q

This is a portability/regression check against the toolkit version that
wrote each bundle. It is not scientific validation of the emulators.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.bundle import verify_bundle_fingerprint_evaluation
from astro_emulators_toolkit.config.migrations import LegacyConfigWarning

from _tracked import (
    flatten_tree,
    load_registry,
    read_reference,
    to_model_inputs,
)

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    os.environ.get("AEMU_TRACKED_EMULATORS") != "1",
    reason="opt-in: set AEMU_TRACKED_EMULATORS=1 (downloads from the HF Hub)",
)

ENTRIES = load_registry()


@pytest.fixture(scope="module", params=ENTRIES, ids=lambda entry: entry.slug)
def tracked(request):
    from huggingface_hub import snapshot_download

    entry = request.param
    local_dir = Path(snapshot_download(entry.repo_id, revision=entry.revision))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emu = Emulator.from_pretrained(entry.repo_id, revision=entry.revision)
    legacy = [
        str(w.message) for w in caught if issubclass(w.category, LegacyConfigWarning)
    ]
    return entry, local_dir, emu, legacy


def _assert_trees_close(actual, expected, *, rtol, atol, what):
    actual_flat = flatten_tree(jax.tree.map(np.asarray, actual), "out")
    expected_flat = flatten_tree(expected, "out")
    assert sorted(actual_flat) == sorted(expected_flat), what
    for path, want in expected_flat.items():
        got = actual_flat[path]
        assert got.shape == want.shape, f"{what}: {path} shape"
        err = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        np.testing.assert_allclose(
            got,
            want,
            rtol=rtol,
            atol=atol,
            err_msg=f"{what}: {path} (max abs error {err:.3g})",
        )


def test_loads_without_unexpected_legacy_warning(tracked):
    entry, _, emu, legacy = tracked
    if not entry.allow_legacy_warning:
        assert legacy == [], (
            f"{entry.slug} now loads with a LegacyConfigWarning; if that is "
            "intended, set allow_legacy_warning = true in registry.toml.\n"
            + "\n".join(legacy)
        )
    assert emu.describe_bundle()


def test_published_fingerprint_verifies(tracked):
    entry, local_dir, emu, _ = tracked
    if not (emu.bundle_metadata or {}).get("fingerprint_evaluation"):
        pytest.skip(f"{entry.slug} publishes no fingerprint")
    try:
        verify_bundle_fingerprint_evaluation(local_dir)
    except ValueError as exc:
        meta = (
            read_reference(entry.reference_path)[2]
            if entry.reference_path.exists()
            else {}
        )
        recorded = meta.get("fingerprint") or {}
        if recorded.get("passed") is False:
            pytest.xfail(
                f"Also fails with toolkit {meta['recorded_with_toolkit']} on "
                f"the recording machine; the fingerprint was computed on "
                f"{meta.get('published_on')}. Platform numerics, not a "
                f"toolkit regression. ({exc})"
            )
        raise


def test_predictions_match_recorded_reference(tracked):
    entry, _, emu, _ = tracked
    if not entry.reference_path.exists():
        pytest.fail(
            f"No reference for {entry.slug}; record it with "
            "tests/tracked_emulators/record_references.py under toolkit "
            f"{entry.written_by_toolkit}."
        )
    physical, expected, meta = read_reference(entry.reference_path)
    assert meta["revision"] == entry.revision

    predicted = emu.predict(to_model_inputs(emu, physical))

    _assert_trees_close(
        predicted,
        expected,
        rtol=entry.rtol,
        atol=entry.atol,
        what=f"{entry.slug} vs toolkit {meta['recorded_with_toolkit']}",
    )


def test_frozen_jax_apply_matches_predict(tracked):
    entry, _, emu, _ = tracked
    if not entry.reference_path.exists():
        pytest.skip(f"No reference for {entry.slug}")
    physical, _, _ = read_reference(entry.reference_path)
    x = to_model_inputs(emu, physical)

    frozen = emu.make_frozen_apply()(jax.tree.map(jnp.asarray, x))

    _assert_trees_close(
        frozen,
        emu.predict(x),
        rtol=entry.rtol,
        atol=entry.atol,
        what=f"{entry.slug} make_frozen_apply vs predict",
    )
