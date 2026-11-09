from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_CONFIG_FILENAME = "run_config.json"
# JSON Lines log: one record appended per fit() call, so overrides applied in
# later segments (external optimizer, method override) never overwrite the
# provenance of earlier segments.
EFFECTIVE_RUNS_FILENAME = "effective_runs.jsonl"


def append_effective_run_record(
    workdir: str | Path,
    *,
    resolved_solver: str,
    requested_solver: str,
    optimizer_source: str | None,
    optim: dict[str, Any] | None,
    num_steps: int,
    resume: bool,
) -> Path:
    """Append a record of what a fit() call actually ran with.

    Saved configs alone cannot distinguish a config-built optimizer from an
    external override attached via ``configure_training``, nor an ``auto``
    solver from its resolution; this record captures the effective choices.
    """
    from .. import __version__
    from ..bundle.provenance import _safe_git_commit

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    record = {
        "resolved_solver": str(resolved_solver),
        "requested_solver": str(requested_solver),
        "optimizer_source": optimizer_source,
        "optim": optim,
        "num_steps": int(num_steps),
        "resume": bool(resume),
        "toolkit_version": __version__,
        "invoking_repo_git_commit": _safe_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = workdir / EFFECTIVE_RUNS_FILENAME
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


__all__ = [
    "RUN_CONFIG_FILENAME",
    "EFFECTIVE_RUNS_FILENAME",
    "append_effective_run_record",
]
