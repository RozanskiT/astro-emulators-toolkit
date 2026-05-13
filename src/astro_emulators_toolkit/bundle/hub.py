# src/astro_emulators_toolkit/bundle/hub.py
from __future__ import annotations

import os
from pathlib import Path


def get_cache_dir() -> Path:
    # Preference order:
    # 1) ASTRO_EMU_CACHE_DIR
    # 2) XDG_CACHE_HOME/astro_emulators_toolkit
    # 3) ~/.cache/astro_emulators_toolkit
    env = os.environ.get("ASTRO_EMU_CACHE_DIR", None)
    if env:
        return Path(env).expanduser().resolve()

    xdg = os.environ.get("XDG_CACHE_HOME", None)
    if xdg:
        return (Path(xdg) / "astro_emulators_toolkit").expanduser().resolve()

    return (Path.home() / ".cache" / "astro_emulators_toolkit").resolve()


def snapshot_download(
    repo_id: str, *, revision: str | None = None, cache_dir: str | None = None
) -> Path:
    """
    Downloads a HF repo snapshot into a cache and returns the local directory.

    Tip: if you prefer a *project-local* cache (e.g. ./.emuspec_cache),
    pass cache_dir="./.emuspec_cache".
    """
    try:
        from huggingface_hub import snapshot_download as _snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Required Hugging Face download support is unavailable in this environment. "
            "Reinstall `astro-emulators-toolkit` or verify that `huggingface_hub` imports correctly."
        ) from e

    cache_dir = cache_dir or str(get_cache_dir() / "hub")
    local_dir = _snapshot_download(
        repo_id=repo_id, revision=revision, cache_dir=cache_dir
    )
    return Path(local_dir)
