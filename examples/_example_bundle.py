"""Helpers shared by bundle-first example scripts."""

from __future__ import annotations

from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
REFERENCE_BUNDLE_DIR = EXAMPLES_DIR / "assets" / "reference_bundle_release"
REFERENCE_BUNDLE_WEIGHTS = REFERENCE_BUNDLE_DIR / "weights" / "weights.safetensors"
REFERENCE_BUNDLE_BUILD_SCRIPT = EXAMPLES_DIR / "assets" / "build_reference_bundle.py"


def require_reference_bundle() -> Path:
    if not REFERENCE_BUNDLE_WEIGHTS.exists():
        raise FileNotFoundError(
            "Released reference bundle weights are missing. Run: "
            f"python {REFERENCE_BUNDLE_BUILD_SCRIPT.as_posix()}"
        )
    return REFERENCE_BUNDLE_DIR
