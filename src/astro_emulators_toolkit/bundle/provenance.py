from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import flax
import jax
import numpy as np

from .. import __version__


def _safe_git_commit() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _optional_dependency_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def build_provenance() -> dict[str, Any]:
    return {
        "toolkit": "astro_emulators_toolkit",
        "toolkit_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": _optional_dependency_version("optax"),
        },
        "git_commit": _safe_git_commit(),
    }


__all__ = ["build_provenance"]
