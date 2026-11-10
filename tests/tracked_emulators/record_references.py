"""Record reference predictions for the tracked emulators in registry.toml.

Run it under the toolkit version that wrote the bundles (each entry's
``written_by_toolkit``), so that the reference is what the bundle's owner
got. Export that revision and put it first on ``PYTHONPATH``:

    git archive v0.1.0 | tar -x -C /tmp/aet-0.1.0
    PYTHONPATH=/tmp/aet-0.1.0/src uv run python tests/tracked_emulators/record_references.py

    git archive v0.2.0.dev0 | tar -x -C /tmp/aet-0.2.0.dev0
    PYTHONPATH=/tmp/aet-0.2.0.dev0/src uv run python tests/tracked_emulators/record_references.py

By default it records every entry written by the running toolkit version
that has no reference yet. Options: ``--only SLUG`` (repeatable),
``--overwrite``, and ``--any-version`` to record under a different toolkit
version on purpose (the reference then says which version produced it).
"""

from __future__ import annotations

import os

# TransformerPayne bundles need x64, which must be set before JAX is imported.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import platform  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

import astro_emulators_toolkit as aet  # noqa: E402
from astro_emulators_toolkit import Emulator  # noqa: E402

from _tracked import (  # noqa: E402
    build_physical_probe,
    load_registry,
    to_model_inputs,
    write_reference,
)


def _check_published_fingerprint(entry, emu) -> dict:
    """Does the writing toolkit reproduce the published fingerprint here?

    Fingerprints are computed on the publisher's machine; float32 models can
    miss their tolerance on another platform even with the original toolkit.
    Recording the outcome lets the test tell that apart from a regression.
    """
    if not (emu.bundle_metadata or {}).get("fingerprint_evaluation"):
        return {"present": False}
    from astro_emulators_toolkit.bundle import verify_bundle_fingerprint_evaluation
    from astro_emulators_toolkit.bundle.hub import snapshot_download

    local_dir = snapshot_download(entry.repo_id, revision=entry.revision)
    try:
        summary = verify_bundle_fingerprint_evaluation(local_dir)
    except ValueError as exc:
        return {"present": True, "passed": False, "error": str(exc)}
    return {
        "present": True,
        "passed": True,
        "max_abs_error": float(summary["max_abs_error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", default=[], metavar="SLUG")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--any-version", action="store_true")
    args = parser.parse_args()

    print(f"astro-emulators-toolkit {aet.__version__} from {aet.__file__}")
    for entry in load_registry():
        if args.only and entry.slug not in args.only:
            continue
        if entry.written_by_toolkit != aet.__version__ and not args.any_version:
            print(f"skip   {entry.slug}: written by {entry.written_by_toolkit}")
            continue
        if entry.reference_path.exists() and not args.overwrite:
            print(f"skip   {entry.slug}: reference exists (use --overwrite)")
            continue

        emu = Emulator.from_pretrained(entry.repo_id, revision=entry.revision)
        provenance = (emu.bundle_metadata or {}).get("provenance") or {}
        if provenance.get("toolkit_version") != entry.written_by_toolkit:
            raise SystemExit(
                f"{entry.slug}: registry says written_by_toolkit="
                f"{entry.written_by_toolkit!r} but the bundle provenance says "
                f"{provenance.get('toolkit_version')!r}; fix registry.toml."
            )
        physical = build_physical_probe(
            emu, n_points=entry.n_points, n_axis=entry.n_axis
        )
        outputs = emu.predict(to_model_inputs(emu, physical))
        fingerprint = _check_published_fingerprint(entry, emu)
        write_reference(
            entry.reference_path,
            physical_inputs=physical,
            outputs=jax.tree.map(np.asarray, outputs),
            meta={
                "repo_id": entry.repo_id,
                "revision": entry.revision,
                "label": entry.label,
                "recorded_with_toolkit": aet.__version__,
                "jax": jax.__version__,
                "platform": platform.platform(),
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "published_on": provenance.get("platform"),
                "fingerprint": fingerprint,
                "probe": {
                    "n_points": entry.n_points,
                    "n_axis": entry.n_axis,
                    "seed": 0,
                },
            },
        )
        print(f"wrote  {entry.reference_path}")


if __name__ == "__main__":
    main()
