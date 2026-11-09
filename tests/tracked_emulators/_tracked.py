"""Registry, probe inputs and reference files for the tracked-emulator check.

Used by test_tracked_emulators.py (current toolkit) and by
record_references.py, which runs under the *old* toolkit that wrote a bundle.
So this module uses only public API that every tracked toolkit version has:
``Emulator.from_pretrained``, ``predict``, ``input_domain``,
``reference_scaling_inputs`` and ``normalize_tree``.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# The committed registry lists public emulators. A gitignored local registry
# (same format) can add private ones; its references live next to it.
REGISTRY_FILES = (
    HERE / "registry.toml",
    REPO_ROOT / "local" / "tracked_emulators" / "registry.toml",
)


@dataclass(frozen=True)
class TrackedEmulator:
    repo_id: str
    label: str
    revision: str
    owner: str
    written_by_toolkit: str
    references_dir: Path
    used_by: str = ""
    notes: str = ""
    rtol: float = 1e-5
    atol: float = 1e-6
    n_points: int = 8
    n_axis: int = 128
    allow_legacy_warning: bool = False

    @property
    def slug(self) -> str:
        return f"{self.repo_id.replace('/', '__')}@{self.label}"

    @property
    def reference_path(self) -> Path:
        return self.references_dir / f"{self.slug}.npz"


def load_registry() -> list[TrackedEmulator]:
    entries: list[TrackedEmulator] = []
    for path in REGISTRY_FILES:
        if not path.exists():
            continue
        for raw in tomllib.loads(path.read_text()).get("emulator", []):
            entries.append(
                TrackedEmulator(references_dir=path.parent / "references", **raw)
            )
    slugs = [entry.slug for entry in entries]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tracked emulators: {duplicates}.")
    return entries


def _map_tree(fn, *trees):
    if isinstance(trees[0], dict):
        return {key: _map_tree(fn, *(t[key] for t in trees)) for key in trees[0]}
    return fn(*trees)


def build_physical_probe(emu, *, n_points: int, n_axis: int, seed: int = 0):
    """Deterministic physical-space inputs inside the bundle's declared box.

    Vector leaves (e.g. stellar parameters) get the box midpoint followed by
    uniform draws in the inner 96% of the box. Scalar-range leaves are
    coordinate axes (e.g. log10 wavelength) and get the same evenly spaced
    grid of ``n_axis`` values for every point.
    """
    box = emu.input_domain or emu.reference_scaling_inputs
    if box is None:
        raise ValueError(
            "Bundle declares neither input_domain nor reference_scaling_inputs; "
            "cannot build probe inputs."
        )
    rng = np.random.default_rng(seed)

    def leaf(lo, hi):
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        if lo.ndim == 0:
            axis = np.linspace(lo, hi, n_axis)
            return np.broadcast_to(axis, (n_points, n_axis)).copy()
        u = rng.uniform(0.02, 0.98, size=(n_points, *lo.shape))
        u[0] = 0.5
        return lo + (hi - lo) * u

    return _map_tree(leaf, box["min_tree"], box["max_tree"])


def to_model_inputs(emu, physical):
    """Apply the bundle's explicit input scaling, as the README tells users."""
    from astro_emulators_toolkit import normalize_tree

    ref = emu.reference_scaling_inputs
    if ref is None:
        return physical
    return normalize_tree(physical, ref["min_tree"], ref["max_tree"])


def flatten_tree(tree, prefix: str) -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        flat: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            flat.update(flatten_tree(value, f"{prefix}/{key}"))
        return flat
    return {prefix: np.asarray(tree)}


def unflatten_tree(flat: dict[str, np.ndarray], prefix: str):
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        parts = path.split("/")
        if parts[0] != prefix:
            continue
        node = tree
        for part in parts[1:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def write_reference(path: Path, *, physical_inputs, outputs, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        **flatten_tree(physical_inputs, "inputs"),
        **flatten_tree(outputs, "outputs"),
    }
    np.savez(path, meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrays)


def read_reference(path: Path):
    with np.load(path, allow_pickle=False) as data:
        flat = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"]))
    return unflatten_tree(flat, "inputs"), unflatten_tree(flat, "outputs"), meta
