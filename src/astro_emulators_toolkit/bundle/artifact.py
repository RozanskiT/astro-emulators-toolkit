from __future__ import annotations

from pathlib import Path
from typing import Any

from ..resolver import derive_model_family_id
from ..spec import materialize_effective_spec, to_json_compatible, validate_spec
from .bundle import Bundle
from .versions import (
    BUNDLE_FORMAT_VERSION,
    CONFIG_SCHEMA_VERSION,
    WEIGHTS_LAYOUT,
)
from .extras import canonicalize_bundle_extras, hydrate_bundle_extras
from .integrity import verify_bundle_integrity, write_bundle_integrity_manifest
from .metadata import (
    apply_resolved_bundle_metadata,
    build_representation_contract,
    build_resolved_bundle_config,
    get_bundle_model_init,
    validate_bundle_header,
    validate_fingerprint_evaluation_payload,
    validate_user_spec_for_bundle_save,
)
from .minmax_sidecars import (
    canonicalize_input_domain,
    canonicalize_reference_scaling_block,
    hydrate_input_domain_sidecar,
    hydrate_reference_scaling_block,
)
from .provenance import build_provenance
from .readme import render_bundle_readme

_REFERENCE_SCALING_BLOCKS = (
    {
        "block_name": "reference_scaling_inputs",
        "filename": "reference_scaling_inputs.safetensors",
        "field_name": "spec['reference_scaling_inputs']",
        "validator_name": "validate_reference_scaling_inputs",
    },
    {
        "block_name": "reference_scaling_outputs",
        "filename": "reference_scaling_outputs.safetensors",
        "field_name": "spec['reference_scaling_outputs']",
        "validator_name": "validate_reference_scaling_outputs",
    },
)


def save_bundle_artifact(
    *,
    cfg,
    graphdef,
    params,
    model_state,
    model_init: dict[str, Any],
    last_fit_method: str | None,
    last_fit_metadata: dict[str, Any] | None,
    dirpath: str | Path | None,
    spec: dict[str, Any] | None,
    extras: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if graphdef is None or params is None or model_state is None:
        raise RuntimeError(
            "Emulator is not initialized; call from_config() before saving a bundle."
        )
    workdir = Path(cfg.training.workdir)
    if dirpath is None:
        dirpath = workdir / cfg.bundle.bundle_subdir
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    validate_user_spec_for_bundle_save(spec)
    spec_payload = materialize_effective_spec(
        cfg,
        spec=None if spec is None else to_json_compatible(spec),
    )
    for block in _REFERENCE_SCALING_BLOCKS:
        spec_payload = canonicalize_reference_scaling_block(
            spec_payload,
            cfg,
            dirpath,
            block_name=block["block_name"],
            filename=block["filename"],
            field_name=block["field_name"],
            validator_name=block["validator_name"],
            model_init=model_init,
        )
    spec_payload = canonicalize_input_domain(
        spec_payload,
        cfg,
        dirpath,
        model_init=model_init,
    )
    validate_spec(spec_payload, cfg)
    extras_payload = canonicalize_bundle_extras(extras, dirpath=dirpath)

    family_id = derive_model_family_id(cfg.model.name)
    if family_id is None:
        raise RuntimeError(
            f"Bundle save requires a declared model_family_id for '{cfg.model.name}'."
        )
    metadata = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "release": None,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "spec": spec_payload,
        "resolved": build_resolved_bundle_config(cfg),
        "weights_layout": WEIGHTS_LAYOUT,
        "provenance": build_provenance(),
        **build_representation_contract(cfg, spec=spec_payload, model_init=model_init),
        "model_family_id": family_id,
    }
    if last_fit_method is not None:
        metadata["fit_method"] = last_fit_method
    if last_fit_metadata is not None:
        metadata["solver_metadata"] = to_json_compatible(dict(last_fit_metadata))
    if extras_payload is not None:
        metadata["extras"] = extras_payload

    bundle = Bundle(
        cfg=cfg,
        params_pure={"params": params, "model_state": model_state},
        metadata=metadata,
        readme_text=render_bundle_readme(cfg, metadata, last_fit_method),
    )
    bundle.save(dirpath)
    manifest = write_bundle_integrity_manifest(
        dirpath,
        metadata_without_bundle_id=dict(metadata),
    )
    metadata["bundle_id"] = manifest["bundle_id"]
    validate_bundle_header(metadata)
    return dirpath, metadata


def load_bundle_artifact(dirpath: str | Path):
    bundle_dir = Path(dirpath)
    loaded = Bundle.load(bundle_dir)
    validate_bundle_header(loaded.metadata)
    manifest = verify_bundle_integrity(bundle_dir)
    loaded.metadata["bundle_id"] = manifest["bundle_id"]
    validate_fingerprint_evaluation_payload(bundle_dir, loaded.metadata)
    loaded.cfg = apply_resolved_bundle_metadata(loaded.cfg, loaded.metadata)
    model_init = get_bundle_model_init(loaded.metadata, cfg=loaded.cfg)
    for block in _REFERENCE_SCALING_BLOCKS:
        loaded.metadata["spec"] = hydrate_reference_scaling_block(
            loaded.metadata["spec"],
            bundle_dir,
            loaded.cfg,
            block_name=block["block_name"],
            field_name=block["field_name"],
            validator_name=block["validator_name"],
            model_init=model_init,
        )
    loaded.metadata["spec"] = hydrate_input_domain_sidecar(
        loaded.metadata["spec"],
        bundle_dir,
        loaded.cfg,
        model_init=model_init,
    )
    loaded.metadata["extras"] = hydrate_bundle_extras(
        loaded.metadata.get("extras"),
        bundle_dir=bundle_dir,
        bundle_format_version=int(loaded.metadata["bundle_format_version"]),
    )
    loaded.metadata["spec"] = materialize_effective_spec(
        loaded.cfg,
        loaded.metadata["spec"],
    )
    validate_spec(loaded.metadata["spec"], loaded.cfg)
    return bundle_dir, loaded


__all__ = ["load_bundle_artifact", "save_bundle_artifact"]
