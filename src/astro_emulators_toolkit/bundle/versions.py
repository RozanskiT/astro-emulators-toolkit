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
  loading correctly, or would change meaning, under the current config
  dataclasses. Version 1 was written by toolkit 0.1.x, version 2 by 0.2.0+.
  Older supported versions are upgraded in memory by the chain in
  ``config/migrations.py`` (``CONFIG_MIGRATIONS``), which also covers the
  ``resolved`` metadata block; settings whose value the upgrade changes are
  reported in one ``LegacyConfigWarning``. Current-version configs are
  validated strictly.
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
   a migration for metadata-level changes, record the first toolkit release
   that loads the new version in ``FIRST_LOADING_TOOLKIT`` (published model
   cards state it), and document the change in the CHANGELOG.
5. Removing a version from a ``SUPPORTED_*`` collection is a breaking change
   for every bundle already in circulation and must be treated as such.
"""

from __future__ import annotations

from typing import Any, Callable

from ..config.migrations import SUPPORTED_CONFIG_SCHEMA_VERSIONS
from ..config.schema import CONFIG_SCHEMA_VERSION
from ..spec import SPEC_VERSION


BUNDLE_FORMAT_VERSION = 1
SUPPORTED_BUNDLE_FORMAT_VERSIONS = frozenset({BUNDLE_FORMAT_VERSION})

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


# The first released toolkit version that can load each version stamp. Extend
# this table with every version bump, so that a bundle can state what its
# users need to install.
FIRST_LOADING_TOOLKIT: dict[str, dict[Any, str]] = {
    "bundle_format_version": {1: "0.1.0"},
    "config_schema_version": {1: "0.1.0", 2: "0.2.0"},
    "spec_version": {1: "0.1.0"},
    "weights_layout": {"params_plus_model_state_v1": "0.1.0"},
}


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def minimum_loading_toolkit_version(metadata: dict[str, Any]) -> str | None:
    """The oldest released toolkit version able to load a bundle, if known.

    This is what the bundle's users need to install, which can be older than
    ``provenance.toolkit_version`` (the version that *saved* it). Returns
    None when a stamp is missing or not in ``FIRST_LOADING_TOOLKIT``.
    """
    from ..config.migrations import wrote_v2_meaning

    needed: list[str] = []
    for field, table in FIRST_LOADING_TOOLKIT.items():
        stamp = (
            (metadata.get("spec") or {}).get("spec_version")
            if field == "spec_version"
            else metadata.get(field)
        )
        if stamp not in table:
            return None
        needed.append(table[stamp])
    provenance = metadata.get("provenance") or {}
    if wrote_v2_meaning(provenance.get("toolkit_version")):
        # 0.2.0 pre-releases stamped config schema v1 but wrote keys that
        # 0.1.x rejects; the first release that loads them is 0.2.0.
        needed.append("0.2.0")
    return max(needed, key=_version_key)


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_METADATA_MIGRATIONS",
    "CONFIG_SCHEMA_VERSION",
    "FIRST_LOADING_TOOLKIT",
    "SPEC_VERSION",
    "SUPPORTED_BUNDLE_FORMAT_VERSIONS",
    "SUPPORTED_CONFIG_SCHEMA_VERSIONS",
    "SUPPORTED_SPEC_VERSIONS",
    "SUPPORTED_WEIGHTS_LAYOUTS",
    "WEIGHTS_LAYOUT",
    "migrate_bundle_metadata",
    "minimum_loading_toolkit_version",
]
