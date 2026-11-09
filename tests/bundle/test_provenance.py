from __future__ import annotations

import subprocess

from astro_emulators_toolkit.bundle.provenance import build_provenance


def test_build_provenance_survives_missing_git_binary(monkeypatch):
    def _raise_missing_binary(*args, **kwargs):
        raise FileNotFoundError("No such file or directory: 'git'")

    monkeypatch.setattr(subprocess, "run", _raise_missing_binary)

    provenance = build_provenance()

    assert provenance["working_tree_git_commit"] is None
    assert provenance["toolkit_version"]


def test_build_provenance_survives_git_timeout(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["git", "rev-parse", "HEAD"], timeout=5.0)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    provenance = build_provenance()

    assert provenance["working_tree_git_commit"] is None
