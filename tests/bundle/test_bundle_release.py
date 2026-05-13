from __future__ import annotations

import json

import numpy as np
import pytest

from astro_emulators_toolkit.bundle import (
    load_bundle_fingerprint_evaluation,
    prepare_bundle_release,
    verify_bundle_fingerprint_evaluation,
)
from astro_emulators_toolkit.bundle.bundle import BUNDLE_INTEGRITY_FILENAME
from astro_emulators_toolkit.bundle.release import load_fingerprint_evaluation_artifacts
from astro_emulators_toolkit.bundle.safetensors_io import save_arrays
from astro_emulators_toolkit.config import (
    IOTreeSpec,
    IOSpec,
    ModelSpec,
    OptimConfig,
    RootConfig,
    TaskSpec,
    TrainConfig,
)
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.emulator import Emulator
from astro_emulators_toolkit.bundle.integrity import write_bundle_integrity_manifest


def _cfg(*, input_last_axis: int = 3, output_last_axis: int = 2) -> RootConfig:
    return RootConfig(
        io=IOSpec(
            inputs=IOTreeSpec(structure_tree={"parameters": None}),
            outputs=IOTreeSpec(structure_tree={"flux": None}),
        ),
        model=ModelSpec(
            name="mlp",
            params={},
            init_hints={
                "input_last_axis": input_last_axis,
                "output_last_axis": output_last_axis,
            },
        ),
    )


def _spec(
    *,
    input_min: tuple[float, ...] = (4500.0, 2.5, -0.3),
    input_max: tuple[float, ...] = (7000.0, 5.0, 0.3),
    output_min: tuple[float, ...] = (0.0, 0.0),
    output_max: tuple[float, ...] = (1.0, 1.0),
) -> dict[str, object]:
    return {
        "input_domain": {
            "kind": "box_v1",
            "value_space": "physical_input_dict_tree_v1",
            "min_tree": {"parameters": list(input_min)},
            "max_tree": {"parameters": list(input_max)},
        },
        "reference_scaling_inputs": {
            "kind": "affine_minmax_v1",
            "applies_to": "inputs",
            "source_space": "physical_input_dict_tree_v1",
            "target_space": "canonical_input_dict_tree_v1",
            "min_tree": {"parameters": list(input_min)},
            "max_tree": {"parameters": list(input_max)},
        },
        "reference_scaling_outputs": {
            "kind": "affine_minmax_v1",
            "applies_to": "outputs",
            "source_space": "canonical_output_dict_tree_v1",
            "target_space": "physical_output_dict_tree_v1",
            "min_tree": {"flux": list(output_min)},
            "max_tree": {"flux": list(output_max)},
        },
    }


def _refresh_bundle_integrity(bundle_dir):
    write_bundle_integrity_manifest(bundle_dir)


def test_prepare_bundle_release_writes_distinct_released_bundle_and_midpoint_fingerprint(
    tmp_path,
):
    raw_out = Emulator.from_config(_cfg()).save_bundle(
        tmp_path / "bundle_release_raw", spec=_spec()
    )
    released_out = tmp_path / "bundle_release_released"

    prepare_bundle_release(
        raw_out,
        path=released_out,
        release_name="payne-flux-reference-example",
        release_version="0.1.0",
    )

    raw_metadata = json.loads((raw_out / "metadata.json").read_text())
    assert raw_metadata["release"] is None
    assert "fingerprint_evaluation" not in raw_metadata
    assert not (raw_out / "fingerprint_evaluation").exists()

    metadata = json.loads((released_out / "metadata.json").read_text())
    manifest = json.loads((released_out / BUNDLE_INTEGRITY_FILENAME).read_text())
    assert metadata["release"] == {
        "name": "payne-flux-reference-example",
        "version": "0.1.0",
        "status": "released",
    }
    assert metadata["fingerprint_evaluation"]["kind"] == "canonical_inputs_outputs_v1"
    assert (
        metadata["fingerprint_evaluation"]["selection_strategy"]
        == "midpoint_from_input_domain_then_reference_scaling_inputs_v1"
    )
    assert (released_out / "fingerprint_evaluation" / "inputs.safetensors").exists()
    assert (released_out / "fingerprint_evaluation" / "outputs.safetensors").exists()
    assert {entry["path"] for entry in manifest["tree"]} >= {
        "README.txt",
        "config.json",
        "fingerprint_evaluation/inputs.safetensors",
        "fingerprint_evaluation/outputs.safetensors",
        "metadata.json",
        "reference_scaling_inputs.safetensors",
        "reference_scaling_outputs.safetensors",
        "weights/weights.safetensors",
    }

    fingerprint = load_bundle_fingerprint_evaluation(released_out)
    np.testing.assert_allclose(
        np.asarray(fingerprint["inputs"]["parameters"]),
        np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
    )
    assert np.asarray(fingerprint["outputs"]["flux"]).shape == (1, 2)

    summary = verify_bundle_fingerprint_evaluation(released_out)
    assert summary["checked_output_paths"] == ["flux"]
    assert summary["max_abs_error"] == pytest.approx(0.0)

    loaded = Emulator.from_bundle(released_out)
    assert "fingerprint_evaluation=present" in loaded.describe_bundle()
    assert (
        "release=payne-flux-reference-example@0.1.0 (released)"
        in loaded.describe_bundle()
    )


def test_prepare_bundle_release_accepts_explicit_canonical_inputs(tmp_path):
    out = Emulator.from_config(_cfg()).save_bundle(
        tmp_path / "bundle_release_manual", spec=_spec()
    )

    prepare_bundle_release(
        out,
        release_name="manual-fingerprint-bundle",
        release_version="0.1.0",
        fingerprint_inputs={"parameters": [0.25, 0.75, 0.5]},
    )

    metadata = json.loads((out / "metadata.json").read_text())
    assert (
        metadata["fingerprint_evaluation"]["selection_strategy"]
        == "provided_canonical_inputs_v1"
    )
    fingerprint = load_bundle_fingerprint_evaluation(out)
    np.testing.assert_allclose(
        np.asarray(fingerprint["inputs"]["parameters"]),
        np.asarray([[0.25, 0.75, 0.5]], dtype=np.float32),
    )


def test_prepare_bundle_release_normalizes_single_parameter_fingerprint_input_to_batch_one(
    tmp_path,
):
    out = Emulator.from_config(_cfg(input_last_axis=1, output_last_axis=1)).save_bundle(
        tmp_path / "bundle_release_single_parameter",
        spec=_spec(
            input_min=(0.0,),
            input_max=(1.0,),
            output_min=(0.0,),
            output_max=(1.0,),
        ),
    )

    prepare_bundle_release(
        out,
        release_name="single-parameter-fingerprint-bundle",
        release_version="0.1.0",
        fingerprint_inputs={"parameters": [0.0]},
    )

    fingerprint = load_bundle_fingerprint_evaluation(out)
    np.testing.assert_allclose(
        np.asarray(fingerprint["inputs"]["parameters"]),
        np.asarray([[0.0]], dtype=np.float32),
    )


def test_prepare_bundle_release_accepts_explicit_canonical_inputs_without_input_metadata(
    tmp_path,
):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    y = (x @ np.asarray([[0.5], [-0.2], [0.1]], dtype=np.float32)).astype(np.float32)
    ds = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (8,), "activation": "tanh"},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "workdir"),
            batch_size=8,
            num_steps=2,
            evaluation_interval_steps=1,
        ),
    )

    emu = Emulator.from_config(cfg)
    emu.fit(ds, validation_dataset=ds, callbacks=[])
    raw_out = emu.save_bundle(tmp_path / "bundle_release_identity")

    released_out = prepare_bundle_release(
        raw_out,
        path=tmp_path / "bundle_release_identity_released",
        release_name="identity-scaling-bundle",
        release_version="0.1.0",
        fingerprint_inputs={"parameters": [0.0, 0.0, 0.0]},
    )

    metadata = json.loads((released_out / "metadata.json").read_text())
    assert (
        metadata["fingerprint_evaluation"]["selection_strategy"]
        == "provided_canonical_inputs_v1"
    )
    fingerprint = load_bundle_fingerprint_evaluation(released_out)
    np.testing.assert_allclose(
        np.asarray(fingerprint["inputs"]["parameters"]),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    summary = verify_bundle_fingerprint_evaluation(released_out)
    assert summary["checked_output_paths"] == ["predictions"]


def test_prepare_bundle_release_rejects_multi_example_explicit_fingerprint_inputs(
    tmp_path,
):
    out = Emulator.from_config(_cfg()).save_bundle(
        tmp_path / "bundle_release_multi_example", spec=_spec()
    )

    with pytest.raises(ValueError, match="must describe exactly one example"):
        prepare_bundle_release(
            out,
            release_name="multi-example-fingerprint-bundle",
            release_version="0.1.0",
            fingerprint_inputs={"parameters": np.zeros((2, 3), dtype=np.float32)},
        )


def test_prepare_bundle_release_requires_explicit_input_metadata_when_inputs_not_provided(
    tmp_path,
):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    y = (x @ np.asarray([[0.5], [-0.2], [0.1]], dtype=np.float32)).astype(np.float32)
    ds = TreeArrayDataset(x={"parameters": x}, y={"predictions": y})
    cfg = RootConfig(
        io=IOSpec(),
        model=ModelSpec(
            name="mlp",
            params={"hidden_sizes": (8,), "activation": "tanh"},
        ),
        task=TaskSpec(name="regression", params={"loss": "mse"}),
        optim=OptimConfig(name="adam", lr=1e-3),
        training=TrainConfig(
            workdir=str(tmp_path / "workdir"),
            batch_size=8,
            num_steps=2,
            evaluation_interval_steps=1,
        ),
    )

    emu = Emulator.from_config(cfg)
    emu.fit(ds, validation_dataset=ds, callbacks=[])
    raw_out = emu.save_bundle(tmp_path / "bundle_release_identity")

    with pytest.raises(
        ValueError,
        match="Cannot synthesize fingerprint inputs without explicit input_domain or reference_scaling_inputs metadata",
    ):
        prepare_bundle_release(
            raw_out,
            path=tmp_path / "bundle_release_identity_released",
            release_name="identity-scaling-bundle",
            release_version="0.1.0",
        )


def test_verify_bundle_fingerprint_evaluation_rejects_output_drift(tmp_path):
    out = Emulator.from_config(_cfg()).save_bundle(
        tmp_path / "bundle_release_drift", spec=_spec()
    )
    prepare_bundle_release(
        out,
        release_name="drift-check-bundle",
        release_version="0.1.0",
    )

    payload = load_bundle_fingerprint_evaluation(out)
    expected = np.asarray(payload["outputs"]["flux"]).copy()
    expected.reshape(-1)[0] += np.asarray(1e-2, dtype=expected.dtype)
    save_arrays(
        out / "fingerprint_evaluation" / "outputs.safetensors",
        {"flux": expected},
    )
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="Bundle fingerprint evaluation failed"):
        verify_bundle_fingerprint_evaluation(out)


def test_from_bundle_rejects_malformed_fingerprint_evaluation_metadata(tmp_path):
    out = Emulator.from_config(_cfg()).save_bundle(
        tmp_path / "bundle_release_bad_meta", spec=_spec()
    )
    prepare_bundle_release(
        out,
        release_name="bad-metadata-bundle",
        release_version="0.1.0",
    )

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fingerprint_evaluation"]["inputs"]["filename"] = "inputs.safetensors"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (out / "inputs.safetensors").write_bytes(
        (out / "fingerprint_evaluation" / "inputs.safetensors").read_bytes()
    )
    _refresh_bundle_integrity(out)

    with pytest.raises(ValueError, match="must live under 'fingerprint_evaluation/'"):
        Emulator.from_bundle(out)


def test_direct_fingerprint_sidecar_loader_rejects_path_traversal(tmp_path):
    fingerprint = {
        "kind": "canonical_inputs_outputs_v1",
        "selection_strategy": "manual",
        "rtol": 1e-5,
        "atol": 1e-7,
        "inputs": {
            "format": "safetensors_v1",
            "filename": "fingerprint_evaluation/../inputs.safetensors",
            "layout": "numeric_dict_tree_v1",
            "space": "canonical_input_dict_trees_v1",
        },
        "outputs": {
            "format": "safetensors_v1",
            "filename": "fingerprint_evaluation/outputs.safetensors",
            "layout": "numeric_dict_tree_v1",
            "space": "canonical_output_dict_trees_v1",
        },
    }

    with pytest.raises(ValueError, match="relative POSIX path"):
        load_fingerprint_evaluation_artifacts(tmp_path, fingerprint)
