"""Tests for the bundle compatibility contract in bundle/versions.py."""

from __future__ import annotations

import pytest

from astro_emulators_toolkit.bundle import versions
from astro_emulators_toolkit.bundle.metadata import validate_bundle_header
from astro_emulators_toolkit.bundle.versions import migrate_bundle_metadata


def _minimal_header() -> dict:
    return {
        "bundle_format_version": versions.BUNDLE_FORMAT_VERSION,
        "release": None,
        "config_schema_version": versions.CONFIG_SCHEMA_VERSION,
        "spec": {"spec_version": versions.SPEC_VERSION},
        "weights_layout": versions.WEIGHTS_LAYOUT,
        "model_family_id": "test.family",
    }


def test_validate_bundle_header_accepts_current_versions():
    validate_bundle_header(_minimal_header())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bundle_format_version", 999, "supported"),
        ("config_schema_version", 999, "supported"),
        ("weights_layout", "unknown_layout_v9", "supported"),
    ],
)
def test_validate_bundle_header_rejects_unsupported_versions_naming_supported(
    field, value, match
):
    header = _minimal_header()
    header[field] = value
    with pytest.raises(ValueError, match=match):
        validate_bundle_header(header)


def test_validate_bundle_header_rejects_unsupported_spec_version():
    header = _minimal_header()
    header["spec"] = {"spec_version": 999}
    with pytest.raises(ValueError, match="supported"):
        validate_bundle_header(header)


def test_migrate_is_identity_for_current_version():
    header = _minimal_header()
    assert migrate_bundle_metadata(dict(header)) == header


def test_migrate_leaves_unsupported_versions_for_header_validation():
    header = _minimal_header()
    header["bundle_format_version"] = 999
    # Unsupported versions pass through unchanged so validate_bundle_header
    # produces the user-facing error.
    assert migrate_bundle_metadata(dict(header)) == header


def test_migration_chain_upgrades_supported_older_versions(monkeypatch):
    # Simulate a future world: current format 2, with format 1 still
    # supported through a registered migration.
    def _v1_to_v2(metadata: dict) -> dict:
        metadata["bundle_format_version"] = 2
        metadata["migrated_marker"] = True
        return metadata

    monkeypatch.setattr(versions, "BUNDLE_FORMAT_VERSION", 2)
    monkeypatch.setattr(
        versions, "SUPPORTED_BUNDLE_FORMAT_VERSIONS", frozenset({1, 2})
    )
    monkeypatch.setattr(versions, "BUNDLE_METADATA_MIGRATIONS", {1: _v1_to_v2})

    header = _minimal_header()
    header["bundle_format_version"] = 1

    migrated = migrate_bundle_metadata(header)

    assert migrated["bundle_format_version"] == 2
    assert migrated["migrated_marker"] is True


def test_supported_version_without_migration_is_a_loud_toolkit_bug(monkeypatch):
    monkeypatch.setattr(versions, "BUNDLE_FORMAT_VERSION", 2)
    monkeypatch.setattr(
        versions, "SUPPORTED_BUNDLE_FORMAT_VERSIONS", frozenset({1, 2})
    )
    monkeypatch.setattr(versions, "BUNDLE_METADATA_MIGRATIONS", {})

    header = _minimal_header()
    header["bundle_format_version"] = 1

    with pytest.raises(ValueError, match="no\\s+migration"):
        migrate_bundle_metadata(header)


def test_migration_must_advance_version(monkeypatch):
    monkeypatch.setattr(versions, "BUNDLE_FORMAT_VERSION", 2)
    monkeypatch.setattr(
        versions, "SUPPORTED_BUNDLE_FORMAT_VERSIONS", frozenset({1, 2})
    )
    monkeypatch.setattr(
        versions, "BUNDLE_METADATA_MIGRATIONS", {1: lambda metadata: metadata}
    )

    header = _minimal_header()
    header["bundle_format_version"] = 1

    with pytest.raises(RuntimeError, match="did not\\s+advance"):
        migrate_bundle_metadata(header)
