from __future__ import annotations

from typing import Any

from ..spec import format_spec_for_display


def _solver_metadata_value(solver_metadata: Any, key: str) -> Any:
    if not isinstance(solver_metadata, dict):
        return "not provided"
    return solver_metadata.get(key, "not provided")


def render_bundle_readme(cfg, metadata: dict[str, Any], fit_method: str | None) -> str:
    provenance = metadata.get("provenance", {})
    runtime_contract = metadata.get("runtime_contract") or {}
    extras = metadata.get("extras") or {}
    solver_metadata = metadata.get("solver_metadata") or {}
    release = metadata.get("release")
    release_label = "unreleased"
    if isinstance(release, dict):
        name = release.get("name")
        version = release.get("version")
        status = release.get("status")
        if isinstance(name, str) and isinstance(version, str):
            release_label = f"{name}@{version}"
            if isinstance(status, str) and status:
                release_label = f"{release_label} ({status})"
    lines = [
        "Astro Emulators Toolkit Bundle",
        "",
        "Summary:",
        f"  model: {cfg.model.name}",
        f"  release: {release_label}",
        f"  bundle_format_version: {metadata.get('bundle_format_version', 'unknown')}",
        f"  config_schema_version: {metadata.get('config_schema_version', 'unknown')}",
        f"  spec_version: {metadata['spec'].get('spec_version', 'unknown')}",
        f"  weights_layout: {metadata.get('weights_layout', 'unknown')}",
        f"  model_family_id: {metadata.get('model_family_id', 'unknown')}",
        f"  fingerprint_evaluation: {'present' if metadata.get('fingerprint_evaluation') is not None else 'absent'}",
        f"  task: {cfg.task.name}",
        f"  fit_method: {fit_method if fit_method is not None else 'unknown'}",
        f"  solver_params: {_solver_metadata_value(solver_metadata, 'params')}",
        f"  solver_diagnostics: {_solver_metadata_value(solver_metadata, 'diagnostics')}",
        f"  solver_design_matrix: {_solver_metadata_value(solver_metadata, 'design_matrix')}",
        f"  role_paths: {runtime_contract.get('role_paths', 'not provided')}",
        "",
        "Domain:",
        f"  input_domain: {metadata['spec'].get('input_domain', 'not provided')}",
        f"  reference_scaling_inputs: {metadata['spec'].get('reference_scaling_inputs', 'not provided')}",
        f"  reference_scaling_outputs: {metadata['spec'].get('reference_scaling_outputs', 'not provided')}",
        f"  extras: {sorted(extras) if isinstance(extras, dict) and extras else 'not provided'}",
        "",
        "Provenance:",
        f"  toolkit_version: {provenance.get('toolkit_version', 'unknown')}",
        f"  created_at: {provenance.get('created_at', 'unknown')}",
        f"  python_version: {provenance.get('python_version', 'unknown')}",
        f"  git_commit: {provenance.get('git_commit', 'unknown')}",
        "",
        "spec:",
        format_spec_for_display(metadata["spec"]),
        "",
        "Note: this bundle is the canonical emulator artifact. Physical-space composition is external.",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["render_bundle_readme"]
