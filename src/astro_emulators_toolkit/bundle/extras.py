from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np

from ..spec import to_json_compatible
from .safetensors_io import load_arrays, save_arrays


EXTRAS_SIDECAR_DIRNAME = "extras"
EXTRAS_SIDECAR_SENTINEL = "__aet_sidecar__"
EXTRAS_SIDECAR_ARRAY_KEY = "value"
EXTRAS_SIDECAR_FORMAT = "safetensors_v1"
EXTRAS_SIDECAR_LAYOUT = "single_array_v1"
EXTRAS_SIDECAR_SUFFIX = ".safetensors"
EXTRAS_LONG_NUMERIC_ARRAY_MIN_SIZE = 32


def _sanitize_extras_path_segment(segment: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in str(segment)
    ).strip(".")
    return cleaned or "value"


def _extras_sidecar_filename(path_parts: tuple[str, ...]) -> str:
    segments = [_sanitize_extras_path_segment(part) for part in path_parts[:-1]]
    leaf = _sanitize_extras_path_segment(path_parts[-1]) + EXTRAS_SIDECAR_SUFFIX
    return Path(EXTRAS_SIDECAR_DIRNAME, *segments, leaf).as_posix()


def _coerce_long_numeric_extra_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        return None

    if arr.ndim == 0 or arr.size < EXTRAS_LONG_NUMERIC_ARRAY_MIN_SIZE:
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def _is_legacy_extra_sidecar_descriptor(value: Any) -> TypeGuard[dict[str, str]]:
    if not (
        isinstance(value, dict)
        and set(value) == {"path"}
        and isinstance(value.get("path"), str)
    ):
        return False
    relpath = Path(value["path"])
    return (
        not relpath.is_absolute()
        and bool(relpath.parts)
        and relpath.parts[0] == EXTRAS_SIDECAR_DIRNAME
        and relpath.suffix == EXTRAS_SIDECAR_SUFFIX
    )


def _is_extra_sidecar_descriptor(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and set(value) == {EXTRAS_SIDECAR_SENTINEL}


def normalize_extra_sidecar_descriptor(
    value: Any,
    *,
    bundle_format_version: int,
    field_name: str,
) -> dict[str, str] | None:
    if _is_extra_sidecar_descriptor(value):
        payload = value[EXTRAS_SIDECAR_SENTINEL]
        if not isinstance(payload, dict):
            raise ValueError(
                f"{field_name}.{EXTRAS_SIDECAR_SENTINEL} must be a dictionary."
            )
        required_keys = {"path", "format", "layout"}
        if set(payload) != required_keys:
            raise ValueError(
                f"{field_name}.{EXTRAS_SIDECAR_SENTINEL} must contain exactly "
                f"{sorted(required_keys)}."
            )
        path = payload.get("path")
        if not isinstance(path, str):
            raise ValueError(
                f"{field_name}.{EXTRAS_SIDECAR_SENTINEL}.path must be a string."
            )
        descriptor_format = payload.get("format")
        if descriptor_format != EXTRAS_SIDECAR_FORMAT:
            raise ValueError(
                f"{field_name}.{EXTRAS_SIDECAR_SENTINEL}.format must be "
                f"'{EXTRAS_SIDECAR_FORMAT}'."
            )
        layout = payload.get("layout")
        if layout != EXTRAS_SIDECAR_LAYOUT:
            raise ValueError(
                f"{field_name}.{EXTRAS_SIDECAR_SENTINEL}.layout must be "
                f"'{EXTRAS_SIDECAR_LAYOUT}'."
            )
        return {
            "path": path,
            "format": EXTRAS_SIDECAR_FORMAT,
            "layout": EXTRAS_SIDECAR_LAYOUT,
        }

    if _is_legacy_extra_sidecar_descriptor(value):
        return {
            "path": value["path"],
            "format": EXTRAS_SIDECAR_FORMAT,
            "layout": EXTRAS_SIDECAR_LAYOUT,
        }

    return None


def _build_extra_sidecar_descriptor(filename: str) -> dict[str, dict[str, str]]:
    return {
        EXTRAS_SIDECAR_SENTINEL: {
            "path": filename,
            "format": EXTRAS_SIDECAR_FORMAT,
            "layout": EXTRAS_SIDECAR_LAYOUT,
        }
    }


def _validate_extras_sidecar_filenames(extras: dict[str, Any]) -> None:
    seen_filenames: dict[str, tuple[str, ...]] = {}

    def _visit(value: Any, *, path_parts: tuple[str, ...]) -> None:
        arr = _coerce_long_numeric_extra_array(value)
        if arr is not None:
            filename = _extras_sidecar_filename(path_parts)
            previous = seen_filenames.get(filename)
            if previous is not None and previous != path_parts:
                previous_key = "/".join(previous)
                current_key = "/".join(path_parts)
                raise ValueError(
                    "extras sidecar filename collision after sanitization: "
                    f"{previous_key!r} and {current_key!r} both map to {filename!r}."
                )
            seen_filenames[filename] = path_parts
            return

        if isinstance(value, dict):
            for key, child in value.items():
                _visit(child, path_parts=path_parts + (str(key),))
            return

        if isinstance(value, tuple):
            value = list(value)

        if isinstance(value, list):
            for index, child in enumerate(value):
                _visit(child, path_parts=path_parts + (f"item_{index}",))

    for key, value in extras.items():
        _visit(value, path_parts=(str(key),))


def _externalize_extra_value(
    value: Any,
    *,
    dirpath: Path,
    path_parts: tuple[str, ...],
) -> Any:
    arr = _coerce_long_numeric_extra_array(value)
    if arr is not None:
        filename = _extras_sidecar_filename(path_parts)
        path = dirpath / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        save_arrays(path, {EXTRAS_SIDECAR_ARRAY_KEY: arr})
        return _build_extra_sidecar_descriptor(filename)

    if isinstance(value, dict):
        return {
            str(key): _externalize_extra_value(
                child,
                dirpath=dirpath,
                path_parts=path_parts + (str(key),),
            )
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return [
            _externalize_extra_value(
                child,
                dirpath=dirpath,
                path_parts=path_parts + (f"item_{index}",),
            )
            for index, child in enumerate(value)
        ]

    return to_json_compatible(value)


def canonicalize_bundle_extras(
    extras: dict[str, Any] | None,
    *,
    dirpath: Path,
) -> dict[str, Any] | None:
    extras_root = dirpath / EXTRAS_SIDECAR_DIRNAME
    if extras_root.exists():
        shutil.rmtree(extras_root)

    if extras is None:
        return None
    if not isinstance(extras, dict):
        raise ValueError("extras must be a dictionary when provided.")
    _validate_extras_sidecar_filenames(extras)

    normalized = {
        str(key): _externalize_extra_value(
            value,
            dirpath=dirpath,
            path_parts=(str(key),),
        )
        for key, value in dict(extras).items()
    }

    if extras_root.exists() and not any(extras_root.iterdir()):
        extras_root.rmdir()
    return normalized


def _hydrate_extra_value(
    value: Any,
    *,
    bundle_dir: Path,
    bundle_format_version: int,
    field_name: str,
) -> Any:
    descriptor = normalize_extra_sidecar_descriptor(
        value,
        bundle_format_version=bundle_format_version,
        field_name=field_name,
    )
    if descriptor is not None:
        from .integrity import validate_bundle_relpath

        filename = descriptor["path"]
        filename = validate_bundle_relpath(
            filename,
            field_name=f"{field_name}.path",
        )
        relpath = Path(filename)
        if not relpath.parts or relpath.parts[0] != EXTRAS_SIDECAR_DIRNAME:
            raise ValueError(
                f"Bundle metadata extras sidecar must live under '{EXTRAS_SIDECAR_DIRNAME}/'."
            )
        path = bundle_dir / relpath
        if not path.exists():
            raise FileNotFoundError(f"extras sidecar file not found: {path}")
        try:
            payload = load_arrays(path)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive for decoder/backend errors
            raise ValueError(f"Failed to decode extras sidecar: {path}") from exc
        if set(payload) != {EXTRAS_SIDECAR_ARRAY_KEY}:
            raise ValueError(
                "Extras sidecar must contain exactly one tensor named "
                f"'{EXTRAS_SIDECAR_ARRAY_KEY}'."
            )
        return to_json_compatible(payload[EXTRAS_SIDECAR_ARRAY_KEY])

    if isinstance(value, dict):
        return {
            str(key): _hydrate_extra_value(
                child,
                bundle_dir=bundle_dir,
                bundle_format_version=bundle_format_version,
                field_name=f"{field_name}.{key}",
            )
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [
            _hydrate_extra_value(
                child,
                bundle_dir=bundle_dir,
                bundle_format_version=bundle_format_version,
                field_name=f"{field_name}[{idx}]",
            )
            for idx, child in enumerate(value)
        ]

    return value


def hydrate_bundle_extras(
    extras: Any,
    *,
    bundle_dir: Path,
    bundle_format_version: int,
) -> dict[str, Any] | None:
    if extras is None:
        return None
    if not isinstance(extras, dict):
        raise ValueError(
            "Bundle metadata field 'extras' must be a dictionary when provided."
        )
    return {
        str(key): _hydrate_extra_value(
            value,
            bundle_dir=bundle_dir,
            bundle_format_version=bundle_format_version,
            field_name=f"extras.{key}",
        )
        for key, value in extras.items()
    }


__all__ = [
    "EXTRAS_SIDECAR_DIRNAME",
    "canonicalize_bundle_extras",
    "hydrate_bundle_extras",
    "normalize_extra_sidecar_descriptor",
]
