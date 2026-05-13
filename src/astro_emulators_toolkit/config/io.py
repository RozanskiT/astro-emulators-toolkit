# src/astro_emulators_toolkit/config/io.py
from __future__ import annotations

import json
import re
import types
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path, PurePath
from typing import Any, TypeGuard, Union, cast, get_args, get_origin, get_type_hints

from .parsing import parse_bool
from .schema import CONFIG_SCHEMA_VERSION, RootConfig, canonicalize_config_names


_YAML_SUFFIXES = {".yaml", ".yml"}
_SCIENTIFIC_NOTATION_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))[eE][+-]?\d+$"
)


def _is_dataclass_instance(value: Any) -> TypeGuard[object]:
    return is_dataclass(value) and not isinstance(value, type)


def _to_jsonable(x: Any) -> Any:
    if _is_dataclass_instance(x):
        return {k: _to_jsonable(v) for k, v in asdict(cast(Any, x)).items()}
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, PurePath):
        return str(x)
    return x


def _is_yaml_path(path: Path) -> bool:
    return path.suffix.lower() in _YAML_SUFFIXES


def _import_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "YAML config support requires the 'pyyaml' dependency."
        ) from e
    return yaml


def _normalize_loaded_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_loaded_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_loaded_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_loaded_tree(v) for v in value)
    if isinstance(value, str):
        stripped = value.strip()
        if _SCIENTIFIC_NOTATION_RE.fullmatch(stripped):
            return float(stripped)
    return value


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    return parse_bool(value, field_name=f"Config field '{field_name}'")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Config field '{field_name}' must be an integer.")
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"Config field '{field_name}' must be an integer.")


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be a float.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Config field '{field_name}' must be a float.") from exc
    raise ValueError(f"Config field '{field_name}' must be a float.")


def _coerce_value(value: Any, annotation: Any, *, field_name: str) -> Any:
    if annotation is Any:
        return value
    if annotation is None or annotation is type(None):
        if value is None:
            return None
        raise ValueError(f"Config field '{field_name}' must be null.")

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        if value is None and type(None) in args:
            return None
        non_none_args = [arg for arg in args if arg is not type(None)]
        for candidate in non_none_args:
            try:
                return _coerce_value(value, candidate, field_name=field_name)
            except (TypeError, ValueError):
                continue
        allowed = " | ".join(
            getattr(arg, "__name__", str(arg)) for arg in non_none_args
        )
        raise ValueError(f"Config field '{field_name}' must match one of: {allowed}.")

    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Config field '{field_name}' must be a sequence.")
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [
            _coerce_value(item, item_type, field_name=f"{field_name}[{idx}]")
            for idx, item in enumerate(value)
        ]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Config field '{field_name}' must be a sequence.")
        args = get_args(annotation)
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _coerce_value(item, args[0], field_name=f"{field_name}[{idx}]")
                for idx, item in enumerate(value)
            )
        if args and len(value) != len(args):
            raise ValueError(
                f"Config field '{field_name}' must have exactly {len(args)} entries."
            )
        return tuple(
            _coerce_value(item, args[idx], field_name=f"{field_name}[{idx}]")
            for idx, item in enumerate(value)
        )

    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"Config field '{field_name}' must be a mapping.")
        args = get_args(annotation)
        key_type = args[0] if len(args) >= 1 else Any
        value_type = args[1] if len(args) >= 2 else Any
        return {
            _coerce_value(
                key, key_type, field_name=f"{field_name}.<key>"
            ): _coerce_value(item, value_type, field_name=f"{field_name}.{key}")
            for key, item in value.items()
        }

    if annotation is bool:
        return _coerce_bool(value, field_name=field_name)
    if annotation is int:
        return _coerce_int(value, field_name=field_name)
    if annotation is float:
        return _coerce_float(value, field_name=field_name)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"Config field '{field_name}' must be a string.")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _build_dataclass(annotation, value, field_name=field_name)

    return value


def _build_dataclass(cls: type[Any], raw: Any, *, field_name: str) -> Any:
    if isinstance(raw, cls):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config field '{field_name}' must be a mapping.")

    raw_dict = {str(k): v for k, v in raw.items()}
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw_dict) - allowed)
    if unknown:
        raise ValueError(f"Unknown {field_name} config keys: {unknown}.")

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in raw_dict:
            continue
        kwargs[field.name] = _coerce_value(
            raw_dict[field.name],
            type_hints.get(field.name, Any),
            field_name=f"{field_name}.{field.name}",
        )
    return cls(**kwargs)


def save_config(cfg: RootConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(canonicalize_config_names(cfg))
    if _is_yaml_path(path):
        yaml = _import_yaml()
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_config(path: str | Path) -> RootConfig:
    path = Path(path)
    if _is_yaml_path(path):
        yaml = _import_yaml()
        raw = yaml.safe_load(path.read_text())
    else:
        raw = json.loads(path.read_text())
    raw = _normalize_loaded_tree(raw)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    allowed_top_level_keys = {
        "schema_version",
        "seed",
        "model",
        "task",
        "solver",
        "optim",
        "training",
        "bundle",
        "hub",
        "io",
    }
    unknown_keys = sorted(set(raw.keys()) - allowed_top_level_keys)
    if unknown_keys:
        raise ValueError(f"Unknown top-level config keys: {unknown_keys}.")

    defaults = RootConfig()

    schema_version = _coerce_value(
        raw.get("schema_version", defaults.schema_version),
        int,
        field_name="schema_version",
    )
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version} (expected {CONFIG_SCHEMA_VERSION})."
        )

    return canonicalize_config_names(
        RootConfig(
            schema_version=schema_version,
            seed=_coerce_value(raw.get("seed", defaults.seed), int, field_name="seed"),
            model=defaults.model
            if raw.get("model") is None
            else _build_dataclass(
                defaults.model.__class__, raw["model"], field_name="model"
            ),
            task=defaults.task
            if raw.get("task") is None
            else _build_dataclass(
                defaults.task.__class__, raw["task"], field_name="task"
            ),
            solver=defaults.solver
            if raw.get("solver") is None
            else _build_dataclass(
                defaults.solver.__class__, raw["solver"], field_name="solver"
            ),
            optim=defaults.optim
            if raw.get("optim") is None
            else _build_dataclass(
                defaults.optim.__class__, raw["optim"], field_name="optim"
            ),
            training=defaults.training
            if raw.get("training") is None
            else _build_dataclass(
                defaults.training.__class__, raw["training"], field_name="training"
            ),
            bundle=defaults.bundle
            if raw.get("bundle") is None
            else _build_dataclass(
                defaults.bundle.__class__, raw["bundle"], field_name="bundle"
            ),
            hub=defaults.hub
            if raw.get("hub") is None
            else _build_dataclass(defaults.hub.__class__, raw["hub"], field_name="hub"),
            io=defaults.io
            if raw.get("io") is None
            else _build_dataclass(defaults.io.__class__, raw["io"], field_name="io"),
        )
    )
