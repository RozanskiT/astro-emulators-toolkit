from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_example(
    script_relpath: str, *args: str, smoke: bool = False
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / script_relpath
    env = os.environ.copy()
    if smoke:
        env["ASTRO_EMU_EXAMPLE_SMOKE"] = "1"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )


def _extract_saved_bundle_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Saved bundle:"):
            bundle_path = line.split(":", 1)[1].strip()
            if bundle_path:
                return Path(bundle_path)
    raise AssertionError("Example output missing 'Saved bundle:' line")


def _extract_saved_paths(stdout: str) -> list[Path]:
    saved_paths = []
    for line in stdout.splitlines():
        if line.startswith("Saved:"):
            saved_path = line.split(":", 1)[1].strip()
            if saved_path:
                saved_paths.append(Path(saved_path))
    if not saved_paths:
        raise AssertionError("Example output missing 'Saved:' lines")
    return saved_paths


def _extract_int_line(stdout: str, prefix: str) -> int:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return int(line.split(":", 1)[1].strip())
    raise AssertionError(f"Example output missing {prefix!r} line")


def _require_reference_bundle() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = repo_root / "examples/assets/reference_bundle_release"
    weights = bundle_dir / "weights" / "weights.safetensors"
    if not weights.exists():
        raise AssertionError(
            "Missing shipped reference bundle asset at "
            f"{weights}. Update it intentionally with "
            "`python examples/assets/build_reference_bundle.py`."
        )
    return bundle_dir


def test_basic_bundle_load_example_runs() -> None:
    _require_reference_bundle()
    completed = _run_example("examples/basic/02_load_bundle_predict.py")
    assert "Prediction shape:" in completed.stdout


def test_basic_dataset_viz_example_writes_repo_level_pngs_from_script_dir() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples/basic/00_visualize_datasets.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_dir = repo_root / "examples/runs/basic_dataset_viz"
    saved_paths = _extract_saved_paths(completed.stdout)
    assert len(saved_paths) == 3
    for saved_path in saved_paths:
        assert saved_path.parent == expected_dir
        assert saved_path.exists()


def test_development_spectral_resolution_postprocess_example_writes_plot() -> None:
    completed = _run_example(
        "examples/development/18_spectral_resolution_postprocess.py"
    )
    saved_paths = _extract_saved_paths(completed.stdout)
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()


def test_reference_flux_bundle_is_loadable() -> None:
    import numpy as np

    from astro_emulators_toolkit import Emulator, normalize_tree
    from astro_emulators_toolkit.bundle.bundle import BUNDLE_INTEGRITY_FILENAME
    import json

    bundle_dir = _require_reference_bundle()
    raw_bundle_dir = (
        Path(__file__).resolve().parents[2] / "examples/assets/reference_bundle_raw"
    )
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    assert (raw_bundle_dir / "weights" / "weights.safetensors").exists()
    raw_metadata = json.loads((raw_bundle_dir / "metadata.json").read_text())
    assert raw_metadata["release"] is None
    assert "fingerprint_evaluation" not in raw_metadata
    assert "bundle_id" not in raw_metadata
    assert "bundle_id" not in metadata
    assert raw_metadata["extras"]["wavelength_angstrom"] == {
        "__aet_sidecar__": {
            "path": "extras/wavelength_angstrom.safetensors",
            "format": "safetensors_v1",
            "layout": "single_array_v1",
        }
    }
    assert metadata["extras"]["wavelength_angstrom"] == {
        "__aet_sidecar__": {
            "path": "extras/wavelength_angstrom.safetensors",
            "format": "safetensors_v1",
            "layout": "single_array_v1",
        }
    }
    assert metadata["release"]["name"] == "payne-flux-reference-example"
    assert metadata["release"]["version"] == "0.1.0"
    assert metadata["release"]["status"] == "released"
    assert (raw_bundle_dir / BUNDLE_INTEGRITY_FILENAME).exists()
    assert (bundle_dir / BUNDLE_INTEGRITY_FILENAME).exists()
    assert (raw_bundle_dir / "extras" / "wavelength_angstrom.safetensors").exists()
    assert (bundle_dir / "extras" / "wavelength_angstrom.safetensors").exists()
    assert (bundle_dir / "fingerprint_evaluation" / "inputs.safetensors").exists()
    assert (bundle_dir / "fingerprint_evaluation" / "outputs.safetensors").exists()
    emu = Emulator.from_bundle(bundle_dir)
    ref = emu.reference_scaling_inputs
    x = {"parameters": np.asarray([[5600.0, 4.2, -0.1]], dtype=np.float32)}
    x_scaled = normalize_tree(x, ref["min_tree"], ref["max_tree"])
    pred = emu.predict(x_scaled)
    assert pred["flux"].shape == (1, 500)
    assert len(emu.bundle_extras["wavelength_angstrom"]) == 500


def test_bundle_metadata_example_uses_reference_bundle_asset() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/basic/03_inspect_bundle_metadata.py"
    )
    script_text = script.read_text()
    assert "reference_bundle" in script_text


def test_basic_bundle_metadata_example_runs() -> None:
    _require_reference_bundle()
    completed = _run_example("examples/basic/03_inspect_bundle_metadata.py")
    assert "Loaded emulator bundle" in completed.stdout
    assert "Output keys:" in completed.stdout
    assert "Domain summary:" in completed.stdout


def test_examples_metadata_names_match_actual_scripts() -> None:
    expected = {
        "test_basic_bundle_load_example_runs": "examples/basic/02_load_bundle_predict.py",
        "test_basic_stable_training_example_runs": "examples/basic/01_train_payne_flux_mlp.py",
        "test_bundle_metadata_portability_example_runs": "examples/advanced/04_bundle_metadata_and_portability.py",
    }
    for test_name, script_relpath in expected.items():
        assert Path(script_relpath).exists(), (
            f"{test_name} points to missing script: {script_relpath}"
        )


def test_basic_stable_training_example_runs() -> None:
    completed = _run_example("examples/basic/01_train_payne_flux_mlp.py", smoke=True)
    assert "Bundle:" in completed.stdout


def test_bundle_metadata_portability_example_runs() -> None:
    completed = _run_example("examples/advanced/04_bundle_metadata_and_portability.py")
    assert "Bundle path:" in completed.stdout
    assert "Companion recipe:" in completed.stdout
    assert "Input reference scaling kind: affine_minmax_v1" in completed.stdout
    assert "Output reference scaling kind: affine_minmax_v1" in completed.stdout


def test_basic_map_fit_example_runs() -> None:
    _require_reference_bundle()
    completed = _run_example("examples/basic/04_use_bundle_in_map_fit.py")
    assert "Applied spectral resolution: 30000" in completed.stdout
    assert "MAP parameters [teff, logg, feh]:" in completed.stdout
    assert "Final log-posterior:" in completed.stdout


def test_basic_cannon_example_writes_bundle_dir() -> None:
    completed = _run_example("examples/basic/05_train_cannon_flux.py", smoke=True)
    bundle_path = _extract_saved_bundle_path(completed.stdout)
    assert bundle_path.exists()
    assert bundle_path.is_dir()
    metadata = json.loads((bundle_path / "metadata.json").read_text())
    assert metadata["solver_metadata"]["name"] == "closed_form_linear"
    assert metadata["solver_metadata"]["params"]["regularize_intercept"] is False


def test_basic_isochrone_mlp_example_writes_bundle_dir() -> None:
    completed = _run_example("examples/basic/06_train_isochrone_mlp.py", smoke=True)
    bundle_path = _extract_saved_bundle_path(completed.stdout)
    assert bundle_path.exists()
    assert bundle_path.is_dir()


def test_basic_transformer_payne_flux_example_writes_bundle_dir() -> None:
    completed = _run_example(
        "examples/basic/07_train_transformer_payne_flux.py", smoke=True
    )
    bundle_path = _extract_saved_bundle_path(completed.stdout)
    assert bundle_path.exists()
    assert bundle_path.is_dir()


def test_experimental_isochrone_siren_example_writes_bundle_dir() -> None:
    completed = _run_example(
        "examples/experimental/07_train_isochrone_siren.py", smoke=True
    )
    bundle_path = _extract_saved_bundle_path(completed.stdout)
    assert bundle_path.exists()
    assert bundle_path.is_dir()


def test_advanced_transformer_payne_intensity_example_writes_bundle_dir() -> None:
    completed = _run_example(
        "examples/advanced/03_train_transformer_payne_intensity.py", smoke=True
    )
    assert "Periodic training log interval:" in completed.stdout
    assert "Explicit training log steps:" in completed.stdout
    assert "Recorded validation steps:" in completed.stdout
    assert "Explicit checkpoint steps:" in completed.stdout
    assert "Recorded checkpoint steps:" in completed.stdout
    bundle_path = _extract_saved_bundle_path(completed.stdout)
    assert bundle_path.exists()
    assert bundle_path.is_dir()


def test_advanced_resume_training_example_runs() -> None:
    _run_example("examples/basic/01_train_payne_flux_mlp.py", smoke=True)
    completed = _run_example("examples/advanced/02_resume_training.py", smoke=True)
    original_target = _extract_int_line(completed.stdout, "Original target step:")
    final_step = _extract_int_line(completed.stdout, "Resumed final step:")
    assert final_step > original_target
    assert "Bundle:" in completed.stdout


def test_advanced_training_internals_example_runs() -> None:
    completed = _run_example("examples/advanced/05_training_internals.py")
    assert "History keys:" in completed.stdout
    assert "Last fit method:" in completed.stdout


def test_advanced_config_driven_lr_scan_pair_runs(tmp_path) -> None:
    output_root = tmp_path / "lr_grid"
    completed = _run_example(
        "examples/advanced/07_grid_search_payne_flux_mlp_lr.py",
        "--output-root",
        str(output_root),
        "--prepare-only",
    )
    assert "Preparation complete." in completed.stdout

    config_path = output_root / "lr_1e-3" / "config.yaml"
    assert config_path.exists()

    completed = _run_example(
        "examples/advanced/06_train_payne_flux_mlp_from_config.py",
        str(config_path),
        "--label",
        "smoke-check",
    )
    assert "Run label: smoke-check" in completed.stdout
    assert "Summary:" in completed.stdout

    result_path = output_root / "lr_1e-3" / "tuning_result.json"
    result = json.loads(result_path.read_text())
    assert result["label"] == "smoke-check"
    assert result["learning_rate"] == 1e-3
    assert Path(result["bundle_dir"]).exists()


def test_advanced_blackjax_example_runs_when_extra_installed() -> None:
    pytest.importorskip("blackjax")
    _require_reference_bundle()
    completed = _run_example("examples/advanced/01_use_bundle_in_blackjax.py")
    assert "Sampled mu:" in completed.stdout
