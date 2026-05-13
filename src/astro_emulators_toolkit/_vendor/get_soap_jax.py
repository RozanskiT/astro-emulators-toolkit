"""Refresh the vendored SOAP_JAX snapshot from the pinned upstream commit.

This script documents how the vendored copy in
`astro_emulators_toolkit._vendor.soap_jax` was obtained. It clones the public
upstream repository, checks out the pinned commit, and copies the Python
sources plus license into this package.

The local `README.txt` is maintained by this repository and is intentionally
not overwritten by the refresh step.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


UPSTREAM_URL = "https://github.com/haydn-jones/SOAP_JAX"
UPSTREAM_COMMIT = "ddddc25724dfd629b0cd01584eca1e32ea8ac4de"
VENDOR_ROOT = Path(__file__).resolve().parent / "soap_jax"
UPSTREAM_PACKAGE_DIR = Path("src") / "soap_jax"
UPSTREAM_FILES = ("__init__.py", "soap.py")


def refresh_vendor_tree() -> None:
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="soap_jax_vendor_") as tmp_dir:
        repo_dir = Path(tmp_dir) / "SOAP_JAX"

        subprocess.run(["git", "clone", UPSTREAM_URL, str(repo_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", UPSTREAM_COMMIT], check=True
        )

        source_root = repo_dir / UPSTREAM_PACKAGE_DIR
        for filename in UPSTREAM_FILES:
            shutil.copy2(source_root / filename, VENDOR_ROOT / filename)
        shutil.copy2(repo_dir / "LICENSE", VENDOR_ROOT / "LICENSE")


def main() -> None:
    refresh_vendor_tree()
    print(f"Vendored SOAP_JAX from {UPSTREAM_URL} at {UPSTREAM_COMMIT}.")
    print(f"Updated files under: {VENDOR_ROOT}")


if __name__ == "__main__":
    main()
