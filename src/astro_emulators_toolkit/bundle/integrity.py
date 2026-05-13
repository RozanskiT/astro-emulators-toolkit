from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .bundle import (
    BUNDLE_INTEGRITY_FILENAME,
    CONFIG_FILENAME,
    METADATA_FILENAME,
    README_FILENAME,
    WEIGHTS_FILENAME,
    WEIGHTS_SUBDIR,
)
from .extras import normalize_extra_sidecar_descriptor


BUNDLE_INTEGRITY_FORMAT_VERSION = 1
BUNDLE_INTEGRITY_ALGORITHM = "sha256"


def validate_sha256_digest(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            f"{field_name} must contain a 64-character lowercase sha256 digest."
        )
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(
            f"{field_name} must contain a 64-character lowercase sha256 digest."
        )
    return value


def validate_bundle_id(bundle_id: Any, *, field_name: str = "Bundle bundle_id") -> None:
    if not isinstance(bundle_id, str) or not bundle_id.startswith("sha256:"):
        raise ValueError(f"{field_name} must be a 'sha256:...' string.")
    validate_sha256_digest(
        bundle_id.removeprefix("sha256:"),
        field_name=field_name,
    )


def validate_bundle_relpath(path: Any, *, field_name: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field_name} must be a non-empty relative POSIX path.")

    relpath = PurePosixPath(path)
    if relpath.is_absolute() or any(part in ("", ".", "..") for part in relpath.parts):
        raise ValueError(f"{field_name} must be a non-empty relative POSIX path.")

    normalized = relpath.as_posix()
    if normalized == ".":
        raise ValueError(f"{field_name} must be a non-empty relative POSIX path.")
    return normalized


def _bundle_payload_relpaths(metadata_without_bundle_id: dict[str, Any]) -> list[str]:
    relpaths = {
        CONFIG_FILENAME,
        METADATA_FILENAME,
        README_FILENAME,
        Path(WEIGHTS_SUBDIR, WEIGHTS_FILENAME).as_posix(),
    }
    bundle_format_version = int(metadata_without_bundle_id["bundle_format_version"])

    spec = metadata_without_bundle_id.get("spec")
    if isinstance(spec, dict):
        for block_name in (
            "reference_scaling_inputs",
            "reference_scaling_outputs",
            "input_domain",
        ):
            block = spec.get(block_name)
            if not isinstance(block, dict):
                continue
            storage = block.get("storage")
            if not isinstance(storage, dict):
                continue
            filename = storage.get("filename")
            if filename is None:
                continue
            relpaths.add(
                validate_bundle_relpath(
                    filename,
                    field_name=f"spec['{block_name}']['storage']['filename']",
                )
            )

    extras = metadata_without_bundle_id.get("extras")

    def _collect_extra_relpaths(value: Any, *, field_name: str) -> None:
        descriptor = normalize_extra_sidecar_descriptor(
            value,
            bundle_format_version=bundle_format_version,
            field_name=field_name,
        )
        if descriptor is not None:
            relpaths.add(
                validate_bundle_relpath(descriptor["path"], field_name=field_name)
            )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                _collect_extra_relpaths(child, field_name=f"{field_name}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                _collect_extra_relpaths(child, field_name=f"{field_name}[{idx}]")

    if isinstance(extras, dict):
        _collect_extra_relpaths(extras, field_name="extras")

    fingerprint = metadata_without_bundle_id.get("fingerprint_evaluation")
    if isinstance(fingerprint, dict):
        for block_name in ("inputs", "outputs"):
            block = fingerprint.get(block_name)
            if not isinstance(block, dict):
                continue
            filename = block.get("filename")
            if filename is None:
                continue
            relpaths.add(
                validate_bundle_relpath(
                    filename,
                    field_name=f"fingerprint_evaluation['{block_name}']['filename']",
                )
            )

    return sorted(relpaths)


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_bundle_integrity_tree(
    bundle_dir: Path,
    *,
    metadata_without_bundle_id: dict[str, Any],
) -> list[dict[str, str]]:
    metadata_bytes = json.dumps(
        metadata_without_bundle_id,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    tree: list[dict[str, str]] = []
    for relpath in _bundle_payload_relpaths(metadata_without_bundle_id):
        path = bundle_dir / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Bundle integrity payload file not found: {path}")
        payload = metadata_bytes if relpath == METADATA_FILENAME else path.read_bytes()
        tree.append({"path": relpath, "sha256": _sha256_hexdigest(payload)})
    return tree


def _canonical_integrity_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _build_bundle_integrity_manifest(
    bundle_dir: Path,
    *,
    metadata_without_bundle_id: dict[str, Any],
) -> dict[str, Any]:
    manifest_without_bundle_id = {
        "integrity_format_version": BUNDLE_INTEGRITY_FORMAT_VERSION,
        "algorithm": BUNDLE_INTEGRITY_ALGORITHM,
        "tree": _build_bundle_integrity_tree(
            bundle_dir,
            metadata_without_bundle_id=metadata_without_bundle_id,
        ),
    }
    return {
        **manifest_without_bundle_id,
        "bundle_id": (
            "sha256:"
            f"{_sha256_hexdigest(_canonical_integrity_payload_bytes(manifest_without_bundle_id))}"
        ),
    }


def write_bundle_integrity_manifest(
    bundle_dir: Path,
    *,
    metadata_without_bundle_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = bundle_dir / METADATA_FILENAME
    if metadata_without_bundle_id is None:
        metadata_without_bundle_id = json.loads(metadata_path.read_text())
    metadata_without_bundle_id = dict(metadata_without_bundle_id)
    metadata_without_bundle_id.pop("bundle_id", None)
    metadata_path.write_text(
        json.dumps(metadata_without_bundle_id, indent=2, sort_keys=True)
    )

    manifest = _build_bundle_integrity_manifest(
        bundle_dir,
        metadata_without_bundle_id=metadata_without_bundle_id,
    )
    (bundle_dir / BUNDLE_INTEGRITY_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def _load_bundle_integrity_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / BUNDLE_INTEGRITY_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Bundle is missing required integrity manifest: {path}"
        )
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode bundle integrity manifest: {path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Bundle integrity manifest must be a dictionary.")
    version = manifest.get("integrity_format_version")
    if version != BUNDLE_INTEGRITY_FORMAT_VERSION:
        raise ValueError(
            "Unsupported bundle integrity manifest version "
            f"{version!r} (expected {BUNDLE_INTEGRITY_FORMAT_VERSION})."
        )
    algorithm = manifest.get("algorithm")
    if algorithm != BUNDLE_INTEGRITY_ALGORITHM:
        raise ValueError(
            "Unsupported bundle integrity algorithm "
            f"{algorithm!r} (expected {BUNDLE_INTEGRITY_ALGORITHM!r})."
        )

    validate_bundle_id(
        manifest.get("bundle_id"),
        field_name="Bundle integrity manifest field 'bundle_id'",
    )

    tree = manifest.get("tree")
    if not isinstance(tree, list) or not tree:
        raise ValueError(
            "Bundle integrity manifest field 'tree' must be a non-empty list."
        )

    normalized_tree: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for idx, entry in enumerate(tree):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Bundle integrity manifest field 'tree[{idx}]' must be a dictionary."
            )
        relpath = validate_bundle_relpath(
            entry.get("path"),
            field_name=f"Bundle integrity manifest field 'tree[{idx}].path'",
        )
        if relpath == BUNDLE_INTEGRITY_FILENAME:
            raise ValueError(
                "Bundle integrity manifest tree must not include the integrity manifest itself."
            )
        if relpath in seen_paths:
            raise ValueError(
                f"Bundle integrity manifest has duplicate tree path: {relpath}"
            )
        seen_paths.add(relpath)
        normalized_tree.append(
            {
                "path": relpath,
                "sha256": validate_sha256_digest(
                    entry.get("sha256"),
                    field_name=f"Bundle integrity manifest field 'tree[{idx}].sha256'",
                ),
            }
        )

    return {
        "integrity_format_version": BUNDLE_INTEGRITY_FORMAT_VERSION,
        "algorithm": BUNDLE_INTEGRITY_ALGORITHM,
        "bundle_id": manifest["bundle_id"],
        "tree": normalized_tree,
    }


def compute_bundle_id(
    bundle_dir: Path,
    *,
    metadata_without_bundle_id: dict[str, Any],
) -> str:
    manifest = _build_bundle_integrity_manifest(
        bundle_dir,
        metadata_without_bundle_id=metadata_without_bundle_id,
    )
    return str(manifest["bundle_id"])


def verify_bundle_integrity(bundle_dir: Path) -> dict[str, Any]:
    manifest = _load_bundle_integrity_manifest(bundle_dir)
    metadata_path = bundle_dir / METADATA_FILENAME
    try:
        metadata_without_bundle_id = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to decode bundle metadata while verifying integrity: {metadata_path}"
        ) from exc
    if not isinstance(metadata_without_bundle_id, dict):
        raise ValueError(
            "Bundle metadata must be a dictionary while verifying integrity."
        )
    metadata_without_bundle_id = dict(metadata_without_bundle_id)
    metadata_without_bundle_id.pop("bundle_id", None)

    try:
        expected_manifest = _build_bundle_integrity_manifest(
            bundle_dir,
            metadata_without_bundle_id=metadata_without_bundle_id,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Bundle integrity verification failed: bundle-owned file referenced by metadata "
            f"is missing ({exc})."
        ) from exc

    manifest_tree = {entry["path"]: entry["sha256"] for entry in manifest["tree"]}
    expected_tree = {
        entry["path"]: entry["sha256"] for entry in expected_manifest["tree"]
    }

    missing_from_manifest = sorted(set(expected_tree).difference(manifest_tree))
    if missing_from_manifest:
        raise ValueError(
            "Bundle integrity verification failed: integrity manifest is missing bundle-owned "
            f"paths required by metadata ({missing_from_manifest})."
        )
    extra_in_manifest = sorted(set(manifest_tree).difference(expected_tree))
    if extra_in_manifest:
        raise ValueError(
            "Bundle integrity verification failed: integrity manifest includes paths that do not "
            f"belong to the current bundle contract ({extra_in_manifest})."
        )

    for relpath in sorted(expected_tree):
        actual_digest = expected_tree[relpath]
        recorded_digest = manifest_tree[relpath]
        if actual_digest != recorded_digest:
            raise ValueError(
                "Bundle integrity verification failed: bundle-owned file does not match the "
                f"integrity manifest ({relpath}; expected sha256:{recorded_digest}, "
                f"computed sha256:{actual_digest})."
            )
    actual_bundle_id = str(expected_manifest["bundle_id"])
    if actual_bundle_id != manifest["bundle_id"]:
        raise ValueError(
            "Bundle integrity verification failed: integrity manifest contents do not match the recorded "
            f"bundle_id (expected {manifest['bundle_id']}, computed {actual_bundle_id})."
        )
    return manifest


__all__ = [
    "BUNDLE_INTEGRITY_ALGORITHM",
    "BUNDLE_INTEGRITY_FORMAT_VERSION",
    "compute_bundle_id",
    "validate_bundle_id",
    "validate_bundle_relpath",
    "write_bundle_integrity_manifest",
    "verify_bundle_integrity",
]
