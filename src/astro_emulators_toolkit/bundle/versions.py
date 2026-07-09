"""Bundle compatibility contract: version fields, support windows, migrations.

This module is the single source of truth for how bundle compatibility works.

Version fields and what they mean
---------------------------------

- ``bundle_format_version`` (int): the on-disk shape of a bundle — which
  files exist, how metadata, weights, and sidecars are organized, and the
  metadata envelope itself. Bump it when the directory layout or the
  metadata envelope changes shape.
- ``config_schema_version`` (int): the ``RootConfig`` schema stored in the
  bundle's ``config.json``. Bump it when previously saved configs stop
  loading correctly under the current config dataclasses.
- ``spec.spec_version`` (int): the effective I/O spec structure carried in
  metadata (structure trees, scaling blocks, domain blocks). Bump it when
  the meaning or shape of spec blocks changes.
- ``weights_layout`` (str): the parameter-tree contract inside
  ``weights/weights.safetensors``. ``"params_plus_model_state_v1"`` means a
  ``{"params": ..., "model_state": ...}`` envelope whose ``params`` tree is
  the pure dict of the *canonical wrapped* model — i.e. rooted at the
  wrapper attribute ``core_model`` for stable model families. Renaming that
  wrapper attribute, or changing what is nested where, is a weights-layout
  change and requires a new layout string.

Compatibility policy
--------------------

1. Saving always stamps the CURRENT versions defined here.
2. Loading accepts exactly the versions listed in the ``SUPPORTED_*``
   collections below. Anything else fails with an error that names the
   supported versions.
3. When an older-but-supported ``bundle_format_version`` is loaded, the
   migration chain in ``BUNDLE_METADATA_MIGRATIONS`` upgrades the in-memory
   metadata one version step at a time until it reaches the current version.
   Loading never mutates the on-disk bundle.
4. Every version bump must: add the old version to the relevant
   ``SUPPORTED_*`` collection (or consciously decide to hard-break), register
   a migration for metadata-level changes, and document the change in the
   CHANGELOG.
5. Removing a version from a ``SUPPORTED_*`` collection is a breaking change
   for every bundle already in circulation and must be treated as such.
"""

from __future__ import annotations

from typing import Any, Callable

from ..config.schema import CONFIG_SCHEMA_VERSION
from ..spec import SPEC_VERSION


BUNDLE_FORMAT_VERSION = 1
SUPPORTED_BUNDLE_FORMAT_VERSIONS = frozenset({BUNDLE_FORMAT_VERSION})

SUPPORTED_CONFIG_SCHEMA_VERSIONS = frozenset({CONFIG_SCHEMA_VERSION})
SUPPORTED_SPEC_VERSIONS = frozenset({SPEC_VERSION})

WEIGHTS_LAYOUT = "params_plus_model_state_v1"
SUPPORTED_WEIGHTS_LAYOUTS = frozenset({WEIGHTS_LAYOUT})

# Maps bundle_format_version N to a function that upgrades a metadata dict
# from version N to version N+1 (and must set the new version itself).
# Registered migrations run in memory at load time; see migrate_bundle_metadata.
BUNDLE_METADATA_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate_bundle_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Upgrade loaded bundle metadata to the current format version in memory.

    Unsupported or malformed versions are returned unchanged so that
    ``validate_bundle_header`` reports its precise, user-facing error.
    """
    version = metadata.get("bundle_format_version")
    if not isinstance(version, int):
        return metadata
    if version not in SUPPORTED_BUNDLE_FORMAT_VERSIONS:
        return metadata
    while version < BUNDLE_FORMAT_VERSION:
        migrate = BUNDLE_METADATA_MIGRATIONS.get(version)
        if migrate is None:
            raise ValueError(
                f"bundle_format_version={version} is declared supported but no "
                f"migration to version {version + 1} is registered. This is a "
                "toolkit bug: every supported historical version needs a "
                "migration chain to the current version."
            )
        metadata = migrate(dict(metadata))
        new_version = metadata.get("bundle_format_version")
        if not isinstance(new_version, int) or new_version <= version:
            raise RuntimeError(
                f"Bundle metadata migration from version {version} did not "
                "advance bundle_format_version."
            )
        version = new_version
    return metadata


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_METADATA_MIGRATIONS",
    "CONFIG_SCHEMA_VERSION",
    "SPEC_VERSION",
    "SUPPORTED_BUNDLE_FORMAT_VERSIONS",
    "SUPPORTED_CONFIG_SCHEMA_VERSIONS",
    "SUPPORTED_SPEC_VERSIONS",
    "SUPPORTED_WEIGHTS_LAYOUTS",
    "WEIGHTS_LAYOUT",
    "migrate_bundle_metadata",
]
