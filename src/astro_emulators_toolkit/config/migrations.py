"""Config schema migrations: read older ``RootConfig`` documents faithfully.

Config schema history
---------------------

- **v1** — written by astro-emulators-toolkit 0.1.x, and also stamped by the
  0.2.0 pre-releases (``0.2.0.devN``), which already wrote v2 *meaning* in
  the v1 shape. Bundles record the writing toolkit version, which tells the
  two apart (:func:`wrote_v2_meaning`).
- **v2** — written by 0.2.0 and later. Shape: ``training.val_fraction`` is
  gone (no toolkit version ever read it). Meaning, compared with 0.1.x:
  settings 0.1.x accepted but silently ignored are rejected; the default
  learning-rate schedule is warmup + cosine (0.1.x: constant, no warmup);
  ``lr_scaling="si"`` is a real mode (0.1.x: an alias for ``None``); and the
  MLP-style families store parameters in ``param_dtype`` (default:
  ``dtype``), where 0.1.x always stored them as float32.

Policy
------

A document that declares the current schema version is validated strictly
by the config dataclasses. A document that declares an older supported
version is first upgraded in memory, one version step at a time, so that it
means exactly what it meant when it was written: removed keys are dropped,
absent keys get the old defaults, and settings the old toolkit accepted but
ignored are replaced by the value that was actually in effect. Each
replacement that changes a value is reported in one
:class:`LegacyConfigWarning`. Files on disk are never modified.

The same rules apply to the ``resolved`` block of bundle metadata, which
bundle loading uses to rebuild the model and solver sections.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from .schema import CONFIG_SCHEMA_VERSION, _normalize_registry_name

SUPPORTED_CONFIG_SCHEMA_VERSIONS = frozenset({1, CONFIG_SCHEMA_VERSION})

_WRITTEN_BY = {1: "astro-emulators-toolkit 0.1.x"}

# Families whose layers store parameters in ``param_dtype`` since schema v2.
# Under 0.1.x their parameters were always float32, whatever ``dtype`` said.
_V1_FLOAT32_PARAM_FAMILIES = frozenset(
    {
        "mlp",
        "experimental/siren",
        "experimental/explicit_wavelength_mlp",
        "experimental/mlp_2d_regression",
    }
)


class LegacyConfigWarning(UserWarning):
    """A config written by an older toolkit version was upgraded in memory."""


def wrote_v2_meaning(toolkit_version: str | None) -> bool:
    """True for the 0.2.0 pre-releases, which stamped v1 but meant v2.

    Bundles record the toolkit version in ``metadata["provenance"]``;
    standalone config files have only the stamp, so ``None`` means 0.1.x.
    """
    return isinstance(toolkit_version, str) and toolkit_version.strip().startswith(
        "0.2.0.dev"
    )


def _same_dtype(a: Any, b: Any) -> bool:
    try:
        import jax.numpy as jnp

        return bool(jnp.dtype(str(a)) == jnp.dtype(str(b)))
    except TypeError:
        return str(a) == str(b)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _v1_to_v2_model(
    model: Mapping[str, Any], *, stored_param_dtype: str | None = None
) -> dict[str, Any]:
    model = dict(model)
    name = _normalize_registry_name(str(model.get("name", "mlp")))
    params = model.get("params")
    if (
        name in _V1_FLOAT32_PARAM_FAMILIES
        and isinstance(params, Mapping)
        and "param_dtype" not in params
    ):
        # 0.1.x stored these parameters as float32. When the saved weights are
        # at hand (bundles), their dtype is the ground truth: it also covers
        # pre-release builds that already stored them in ``dtype``. Either way
        # this reproduces how the model was built, so no note is needed.
        storage = stored_param_dtype or "float32"
        if not _same_dtype(storage, params.get("dtype", "float32")):
            model["params"] = {**params, "param_dtype": storage}
    return model


def _v1_to_v2_solver(solver: Mapping[str, Any], notes: list[str]) -> dict[str, Any]:
    solver = dict(solver)
    name = _normalize_registry_name(str(solver.get("name", "auto")))
    params = solver.get("params")
    if name == "gradient" and isinstance(params, Mapping) and params:
        notes.append(
            f"solver.params {sorted(str(k) for k in params)} were ignored by "
            "the 'gradient' solver; dropped."
        )
        solver["params"] = {}
    return solver


def _v1_to_v2_optim(optim: Mapping[str, Any], notes: list[str]) -> dict[str, Any]:
    optim = dict(optim)
    # Defaults that changed in v2: absent keys keep their 0.1.x meaning.
    optim.setdefault("schedule", "constant")
    optim.setdefault("warmup_steps", 0)

    lr_scaling = optim.get("lr_scaling")
    if isinstance(lr_scaling, str):
        normalized = _normalize_registry_name(lr_scaling)
        if normalized in {"", "none"}:
            lr_scaling = None
        elif normalized in {"si", "standard"}:
            notes.append(
                f"optim.lr_scaling={lr_scaling!r} was an alias for None (one "
                "shared learning rate); loaded as None."
            )
            lr_scaling = None
        optim["lr_scaling"] = lr_scaling

    scale = _as_float(optim.get("scale_embedding_lr", 1.0))
    if lr_scaling is None and scale is not None and scale != 1.0:
        notes.append(
            f"optim.scale_embedding_lr={optim['scale_embedding_lr']!r} had no "
            "effect with lr_scaling=None; loaded as 1.0."
        )
        optim["scale_embedding_lr"] = 1.0

    schedule = _normalize_registry_name(str(optim["schedule"]))
    decay_steps = _as_float(optim.get("decay_steps", 0))
    if schedule == "cosine" and decay_steps is not None and decay_steps > 0:
        notes.append(
            f"optim.decay_steps={optim['decay_steps']!r} was ignored by the "
            "'cosine' schedule; loaded as 0."
        )
        optim["decay_steps"] = 0
    return optim


def _migrate_v1_to_v2(
    raw: dict[str, Any], notes: list[str], *, toolkit_version: str | None
) -> dict[str, Any]:
    raw = dict(raw)
    # Shape, for every v1 stamp: drop keys v2 no longer has.
    training = raw.get("training")
    if isinstance(training, Mapping) and "val_fraction" in training:
        raw["training"] = {k: v for k, v in training.items() if k != "val_fraction"}
    # Meaning, only for documents written by 0.1.x.
    if not wrote_v2_meaning(toolkit_version):
        if isinstance(raw.get("model"), Mapping):
            raw["model"] = _v1_to_v2_model(raw["model"])
        if isinstance(raw.get("solver"), Mapping):
            raw["solver"] = _v1_to_v2_solver(raw["solver"], notes)
        optim = raw.get("optim")
        if optim is None or isinstance(optim, Mapping):
            raw["optim"] = _v1_to_v2_optim(optim or {}, notes)
    raw["schema_version"] = 2
    return raw


# Maps schema version N to a function upgrading a raw config dict from N to
# N + 1, given the toolkit version that wrote it (None if unknown). It must
# set the new ``schema_version`` and append a note for every setting whose
# value it changes.
CONFIG_MIGRATIONS: dict[int, Callable[..., dict[str, Any]]] = {1: _migrate_v1_to_v2}


def migrate_config_dict(
    raw: Mapping[str, Any],
    *,
    schema_version: int,
    toolkit_version: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Upgrade a raw config dict from ``schema_version`` to the current one.

    Returns the upgraded dict and the notes describing changed settings.
    """
    if schema_version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version={schema_version} "
            f"(supported: {sorted(SUPPORTED_CONFIG_SCHEMA_VERSIONS)})."
        )
    upgraded = dict(raw)
    notes: list[str] = []
    version = schema_version
    while version < CONFIG_SCHEMA_VERSION:
        upgraded = CONFIG_MIGRATIONS[version](
            upgraded, notes, toolkit_version=toolkit_version
        )
        version = int(upgraded["schema_version"])
    return upgraded, notes


def migrate_resolved_bundle_config(
    resolved: Mapping[str, Any],
    *,
    schema_version: int,
    toolkit_version: str | None = None,
    stored_param_dtype: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Upgrade a bundle's ``metadata["resolved"]`` block like ``config.json``.

    Only the ``model`` and ``solver`` sections are rebuilt from this block at
    load time, so only those are migrated. ``stored_param_dtype`` is the
    single floating dtype of the bundle's saved parameters, if uniform; it
    decides parameter storage for any v1 stamp.
    """
    upgraded = dict(resolved)
    notes: list[str] = []
    if schema_version < 2:
        if isinstance(upgraded.get("model"), Mapping):
            upgraded["model"] = _v1_to_v2_model(
                upgraded["model"], stored_param_dtype=stored_param_dtype
            )
        if not wrote_v2_meaning(toolkit_version) and isinstance(
            upgraded.get("solver"), Mapping
        ):
            upgraded["solver"] = _v1_to_v2_solver(upgraded["solver"], notes)
    return upgraded, notes


def warn_legacy_config(
    notes: list[str], *, source: str, schema_version: int, stacklevel: int = 2
) -> None:
    """Emit one :class:`LegacyConfigWarning` listing ``notes`` (deduplicated)."""
    unique = list(dict.fromkeys(notes))
    if not unique:
        return
    written_by = _WRITTEN_BY.get(schema_version, "an older toolkit version")
    lines = "\n".join(f"  - {note}" for note in unique)
    warnings.warn(
        f"{source} uses config schema v{schema_version} (written by "
        f"{written_by}) and was upgraded to v{CONFIG_SCHEMA_VERSION} in "
        "memory; files on disk are unchanged. Settings the old version "
        "accepted but ignored were replaced by the values actually in "
        f"effect:\n{lines}",
        LegacyConfigWarning,
        stacklevel=stacklevel + 1,
    )


__all__ = [
    "CONFIG_MIGRATIONS",
    "LegacyConfigWarning",
    "SUPPORTED_CONFIG_SCHEMA_VERSIONS",
    "migrate_config_dict",
    "migrate_resolved_bundle_config",
    "warn_legacy_config",
    "wrote_v2_meaning",
]
