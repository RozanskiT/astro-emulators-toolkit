# Astro Emulators Toolkit

JAX + Flax NNX tools for training, validating, packaging, sharing, and using astrophysical emulators.

> **Status:** This package is under active development and currently being tested with collaborators. It is pre-1.0, so APIs and bundle metadata may still change. Please open an issue or contact me if you try it and something does not work.

The project is built around a narrow stable surface and explicit contracts. The goal is not only to fit emulators, but to make them usable by other astronomers in downstream inference code without requiring them to reverse-engineer training details. In practice that means:

- a small high-level facade (`Emulator`) for loading, training, inference, and bundle export;
- canonical dict-tree inputs and outputs instead of model-family-specific calling conventions;
- explicit preprocessing and domain metadata instead of hidden transforms;
- portable **bundles** as the artifact that another astronomer should receive, inspect, validate, and load.

This README is the main user documentation for the current pre-1.0 phase. The package API is intentionally small, but on-disk bundle details may still evolve before 1.0.

## What this toolkit is for

Use this toolkit when you need one or more of the following:

- load a trained emulator and call it inside your custom inference pipeline;
- train a new emulator on your data;
- publish a trained emulator with explicit metadata about inputs, outputs, scaling, valid domain, provenance, and release checks;
- develop new emulator architectures while keeping the user-facing inference surface simple.

Current stable examples focus on flux spectra, intensity spectra, and isochrone-like tabular regression.

## Design philosophy

### 1. One public facade

The central public object is `Emulator`. It provides the main supported workflows:

- `Emulator.from_bundle(...)` / `Emulator.from_pretrained(...)` for loading a shared artifact;
- `Emulator.from_config(...)` for building from a config;
- `fit(...)` for training;
- `predict(...)` for host-side inference returning NumPy arrays;
- `apply_jax(...)` and `make_frozen_apply(...)` for JAX-side inference;
- `save_bundle(...)` for exporting a portable artifact.

### 2. Explicit contracts

The toolkit does **not** hide physical-to-training-space preprocessing. If an emulator uses scaled inputs or outputs and the author explicitly records those transforms, the bundle can store `reference_scaling_inputs` and `reference_scaling_outputs`, but the caller still applies normalization or denormalization explicitly with `normalize_tree(...)` and `denormalize_tree(...)`. If either block is absent, the bundle is not claiming that the corresponding transform was documented. Likewise, `input_domain` records intended-use bounds, but the user remains responsible for checking whether proposed inputs are scientifically appropriate.

### 3. Composition over framework lock-in

Stable inference is one predictive mapping. The library does not try to own every downstream inversion or sampling workflow. Instead, it gives you a callable that can be inserted into your own JAX likelihood code.

### 4. Shared artifacts should be inspectable

A bundle is the artifact you share with collaborators: weights, config, effective I/O spec, metadata, provenance, integrity information, and optional release fingerprint checks. The point is portability and responsible reuse, not just serialization.

## Who should start where

If you received a trained emulator from somebody else, start with:

1. [Installation](#installation)
2. [Load a shared bundle and predict](#load-a-shared-bundle-and-predict)
3. [Inspecting bundle metadata at runtime](#inspecting-bundle-metadata-at-runtime)
4. [Using the same emulator inside JAX inference code](#using-the-same-emulator-inside-jax-inference-code)

If you are training your own emulator, start with:

1. [Installation](#installation)
2. [Main public components](#main-public-components)
3. [Training a new emulator](#training-a-new-emulator)
4. [Recording the scientific contract of the emulator](#recording-the-scientific-contract-of-the-emulator)
5. [Preparing a release bundle for collaborators](#preparing-a-release-bundle-for-collaborators)

If you are extending the library, also read:

- [Configuration model](#configuration-model)
- [Stable model families and tasks](#stable-model-families-and-tasks)
- [Data flow](#data-flow)
- [Bundle structure and semantics](#bundle-structure-and-semantics)
- [`AGENTS.md`](AGENTS.md)

For contributors and coding agents, repository guidance lives in [AGENTS.md](https://github.com/RozanskiT/astro-emulators-toolkit/blob/main/AGENTS.md).

## Installation

Python `>=3.11` is required.

Published-package usage with `pip`:

```bash
pip install astro-emulators-toolkit
```

Optional extras:

```bash
pip install "astro-emulators-toolkit[blackjax]"  # BlackJAX examples/integration
pip install "astro-emulators-toolkit[viz]"       # plotting snippets/examples
```

The same packages can be added with `uv`:

```bash
uv add astro-emulators-toolkit
uv add "astro-emulators-toolkit[blackjax]"
uv add "astro-emulators-toolkit[viz]"
```

Base installation already includes local bundle loading, Hugging Face bundle download, training, bundle export, safetensors serialization, JSON/YAML config I/O, and the stable model/task/runtime surface.

Hugging Face support in the stable public API is **download-only** through `Emulator.from_pretrained(...)`.

## Main public components

These are the components most users need to know.

### `astro_emulators_toolkit.Emulator`

High-level facade for:

- loading a bundle from disk with `from_bundle(...)`;
- downloading and loading a bundle from Hugging Face with `from_pretrained(...)`;
- building from config with `from_config(...)`;
- optional explicit initialization with `initialize(...)`;
- host-side inference with `predict(...)`;
- JAX-side inference with `apply_jax(...)` and `make_frozen_apply(...)`;
- training with `fit(...)`;
- exporting a portable artifact with `save_bundle(...)`.

Useful bundle metadata accessors on a loaded emulator:

- `spec`
- `input_spec`, `output_spec`
- `reference_scaling_inputs`, `reference_scaling_outputs`
- `input_domain`
- `bundle_extras`
- `input_channel_names_tree`, `output_channel_names_tree`
- `describe_bundle()` and `describe_domain()`

### `astro_emulators_toolkit.config`

Primitive-only configuration dataclasses plus config I/O:

- `RootConfig`
- `ModelSpec`, `TaskSpec`, `SolverConfig`, `OptimConfig`, `TrainConfig`
- `IOSpec`, `IOTreeSpec`, `MinMaxTreeSpec`
- `load_config(...)`, `save_config(...)`

Configs remain JSON/YAML-friendly by using string registry names for model families, tasks, solvers, and optimizers.

### `astro_emulators_toolkit.presets`

Stable starting points for the most common workflows:

- `payne_flux_mlp(...)`
- `isochrone_mlp(...)`
- `cannon_flux(...)`
- `transformer_payne_flux(...)`
- `transformer_payne_intensity(...)`

Use presets when you want a supported, top-down starting point instead of assembling a full `RootConfig` by hand.

### `astro_emulators_toolkit.data`

Core data-layer building blocks:

- `Batch`, `DatasetProtocol`, `DeviceBatchTransform`
- `TreeArrayDataset`
- `XYArrayDataset`
- `MappedDataset`
- `NpyTableDataset`
- `train_val_split(...)`
- `SubsetDataset`
- `DataLoader`

The intended split is:

- host side for storage access, semantic mapping, batching, and loader-added `valid_mask`;
- device side for explicit JAX-compatible numeric preprocessing through `device_batch_transform`.

Model families with runtime adapters can expose concrete device-side preprocessing objects through `emu.make_device_batch_transform(...)`.

### `astro_emulators_toolkit.training`

Training callbacks and helpers:

- `build_callbacks_from_config(...)`
- `CSVLogger`
- `ProgressBarLogger`
- `ModelCheckpoint`

`ModelCheckpoint` belongs to run management for ongoing training. It is not the collaborator-facing distribution artifact; that role belongs to bundles saved with `save_bundle(...)`.

### `astro_emulators_toolkit.bundle`

Release-oriented helpers for shared artifacts:

- `prepare_bundle_release(...)`
- `verify_bundle_fingerprint_evaluation(...)`
- `load_bundle_fingerprint_evaluation(...)`

## Load a shared bundle and predict

The first supported user workflow is: load a bundle, apply explicit preprocessing if required, and call `predict(...)`.

```python
import numpy as np

from astro_emulators_toolkit import Emulator, normalize_tree

# Load from Hugging Face. Local bundles use Emulator.from_bundle(path).
emu = Emulator.from_pretrained("RozanskiT/example_bundle")

x_physical = {
    "parameters": np.asarray([[5600.0, 4.2, -0.1]], dtype=np.float32)
}

# The library does not auto-apply preprocessing. When the bundle records an
# explicit reference_scaling_inputs block, reproduce the training-space input from it.
ref = emu.reference_scaling_inputs
if ref is None:
    raise ValueError("This bundle does not declare reference_scaling_inputs.")
x_scaled = normalize_tree(
    x_physical,
    ref["min_tree"],
    ref["max_tree"],
)

pred = emu.predict(x_scaled)
flux = pred["flux"]
wave = np.asarray(emu.bundle_extras.get("wavelength_angstrom", []), dtype=np.float32)

print(flux.shape)
print(wave.shape)
print(emu.describe_domain())
```

What to remember:

- `predict(...)` is the host-side path and returns NumPy-backed outputs.
- Stable public inference expects canonical dict-tree inputs with an explicit batch axis on every leaf. A single example still uses shape `(1, n)`, not `(n,)`.
- Scaling is explicit; `predict(...)` does not normalize for you.
- Fixed grids and convenience arrays often live in `bundle_extras`.

Runnable examples:

- [`examples/basic/02_load_bundle_predict.py`](examples/basic/02_load_bundle_predict.py)
- [`examples/basic/03_inspect_bundle_metadata.py`](examples/basic/03_inspect_bundle_metadata.py)

## Inspecting bundle metadata at runtime

A loaded emulator gives direct access to the scientific and portability metadata carried by the bundle.

```python
print(emu.describe_bundle())
print(emu.input_spec)
print(emu.output_spec)
print(emu.reference_scaling_inputs)
print(emu.reference_scaling_outputs)
print(emu.input_domain)
print(emu.bundle_extras.keys())
```

Typical uses:

- checking channel names for input labels or output channels;
- retrieving a stored wavelength grid or companion recipe;
- reconstructing physical-space preprocessing;
- auditing provenance or release identity;
- inspecting solver details recorded for a trained bundle, such as Cannon ridge settings and observed condition number;
- checking whether fingerprint evaluation metadata is present.

## Using the same emulator inside JAX inference code

For downstream inference, the main supported pattern is to freeze a pure callable once and compose it into your own JAX code.

```python
import jax
import jax.numpy as jnp
import numpy as np

from astro_emulators_toolkit import Emulator, normalize_tree

emu = Emulator.from_pretrained("RozanskiT/example_bundle")
apply_flux = emu.make_frozen_apply(jit=False)
ref = emu.reference_scaling_inputs

y_obs = jnp.asarray(observed_flux, dtype=jnp.float32)
y_err = jnp.asarray(observed_sigma, dtype=jnp.float32)

@jax.jit
def log_likelihood(theta):
    x_physical = {"parameters": theta[None, :]}
    x_scaled = normalize_tree(
        x_physical,
        ref["min_tree"],
        ref["max_tree"],
    )
    y_model = apply_flux(x_scaled)["flux"][0]
    resid = (y_obs - y_model) / y_err
    return -0.5 * jnp.sum(resid**2)

log_likelihood(jnp.asarray([5600.0, 4.2, -0.1], dtype=jnp.float32))
```

Important distinction:

- `apply_jax(...)` uses the current live emulator state.
- `make_frozen_apply(...)` captures the current parameter/state snapshot and returns a callable suitable for composition into jitted code.
- In repeated optimization or sampling loops, usually keep `make_frozen_apply(jit=False)` and jit the outer likelihood, optimizer step, or sampler transition. This gives XLA one compiled objective that includes your normalization, nuisance terms, masks, priors, and emulator call.
- `make_frozen_apply(jit=True)` is available when you specifically want the emulator callable pre-jitted, but it is not a replacement for jitting the full downstream objective or transition kernel.

If you retrain or otherwise mutate the emulator state, recreate the frozen callable afterwards.

The JAX-side spectral-resolution helper lives under `astro_emulators_toolkit.inference.compose`.
For example, if a bundle stores a wavelength grid and your likelihood needs a lower
instrumental resolution, wrap the frozen callable once:

```python
from astro_emulators_toolkit.inference.compose import downgrade_spectral_resolution

log_wavelength = jnp.asarray(bundle_log_wavelength_grid, dtype=jnp.float32)
apply_flux_lowres = downgrade_spectral_resolution(
    apply_flux,
    log_wavelength,
    resolution=30000.0,
    output_path="flux",
    axis=-1,
)
```

Here `bundle_log_wavelength_grid` is the grid matching the selected flux leaf.
The resolution helper assumes that leaf is already sampled on a uniform log10
wavelength grid. Use `axis=-1` for arrays shaped like `(batch, n_wavelength)`
and `axis=-2` for arrays shaped like `(batch, n_wavelength, channels)`. A bare
`output_path`, such as `"flux"`, smooths all leaves with that name; a
slash-delimited path, such as `"spectra/flux"`, selects one exact leaf.

For mixed output trees, pass an `axis_tree` with the same structure as the
output tree and leaves set to `None`, `-1`, or `-2`:

```python
apply_lowres = downgrade_spectral_resolution(
    apply_flux,
    log_wavelength,
    resolution=30000.0,
    axis_tree={"flux": -1, "intensity": -2, "diagnostics": None},
)
```

This changes only the selected output leaves; it does not hide input scaling,
output scaling, masking, or scientific validation.

### Numerical reproducibility in JAX likelihoods

Most released neural-network bundles use float32 weights. In float32, eager JAX evaluation and XLA-compiled evaluation can differ by tiny roundoff-level amounts because the compiler may fuse or reorder arithmetic. Flux differences at the `1e-6` level are usually negligible physically, but a small-noise likelihood can amplify them when it sums over many pixels.

For deterministic tests, synthetic observations, and tight reproducibility checks:

- generate synthetic observations with the same compiled model path used by the objective;
- compare likelihoods that are evaluated in the same execution mode, preferably inside the same outer `jax.jit`;
- do not expect `predict(...)` or an eager scalar likelihood to be bitwise identical to a compiled likelihood at very small tolerances;
- enable float64 only when the full model and weights are intended to run in float64. Float64 likelihood accumulation alone cannot make a float32 bundle evaluate as a float64 emulator.

Runnable examples:

- [`examples/basic/04_use_bundle_in_map_fit.py`](examples/basic/04_use_bundle_in_map_fit.py)
- [`examples/advanced/01_use_bundle_in_blackjax.py`](examples/advanced/01_use_bundle_in_blackjax.py)

## Training a new emulator

The typical supported training path is:

1. choose a stable preset or build a `RootConfig`;
2. use `TreeArrayDataset` for canonical trees, or start from `XYArrayDataset` / `NpyTableDataset` and wrap with `MappedDataset` when you need an explicit host-side semantic mapping;
3. call `fit(...)`;
4. save a portable bundle with `save_bundle(...)`.

```python
from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import TreeArrayDataset
from astro_emulators_toolkit.presets import payne_flux_mlp
from astro_emulators_toolkit.training import build_callbacks_from_config

# x_train, x_val: (n_samples, n_labels)
# y_train, y_val: (n_samples, n_wavelengths)

cfg = payne_flux_mlp(
    workdir="runs/payne_flux_mlp",
    profile="cpu_recommended",
)

emu = Emulator.from_config(cfg)
callbacks = build_callbacks_from_config(cfg)

history = emu.fit(
    TreeArrayDataset(x={"parameters": x_train}, y={"flux": y_train}),
    validation_dataset=TreeArrayDataset(x={"parameters": x_val}, y={"flux": y_val}),
    callbacks=callbacks,
)

pred = emu.predict({"parameters": x_val[:2]})["flux"]
bundle_dir = emu.save_bundle()

print(sorted(history.logs))
print(pred.shape)
print(bundle_dir)
```

### Training notes

- `Emulator.from_config(...)` can initialize immediately if enough shape information is available through `init_example=...` or `cfg.model.init_hints`.
- If not, `fit(...)` lazily initializes from the first training batch. When `device_batch_transform` is provided, initialization uses `device_batch_transform.for_init(...)`.
- `configure_training(optimizer=None, task=None)` is the advanced hook for attaching a task override and/or optimizer before `fit(...)`. It only configures training components; it does not call `jax.jit`, trigger XLA compilation, or initialize optimizer state.
- `training.num_steps` is the total target optimization step count. With `resume=True`, training continues until the restored state reaches `training.num_steps`; `max_steps` only caps the additional steps taken by the current `fit(...)` call and does not train past `training.num_steps`.
- For `closed_form_linear`, a scalar target leaf with shape `(N,)` is treated as one output channel; fitted coefficients and predictions keep an explicit trailing dimension of 1.
- Presets use `profile="smoke"` for short example runs and `profile="cpu_recommended"` for longer CPU-oriented defaults.
- Training and inference are tested for CPU/single-device JAX execution. Single-GPU execution should work with a correct JAX GPU installation. Multi-GPU training is not yet part of the stable contract; the current trainer does not shard batches or replicate/partition training state across devices.
- Use `build_callbacks_from_config(cfg)` when you want config-driven progress, evaluation, and run-state callbacks.

### Advanced training: learning-rate scaling

Most training recipes should leave `optim.lr_scaling=None`, which gives one shared
learning-rate schedule to the whole model. For scaling-law work and larger model
sweeps, `optim.lr_scaling` can instead request muP-style per-parameter learning
rate factors. This is an optimizer update rule: it does not change the public
I/O contract, hide preprocessing, or make bundle fingerprints a substitute for
scientific validation.

Supported settings are:

- `mlp` with `optim.lr_scaling="mup"`: requires `model.params.reference_width`.
  Dense kernels are scaled by the ratio between the reference width and the
  current hidden width; biases and otherwise unclassified parameters keep the
  base learning rate.
- `transformer_payne` with `optim.lr_scaling="mup"`: requires
  `model.params.reference_width` and applies the Transformer Payne
  component-wise scaling used by the optimizer implementation.
- `transformer_payne` with `optim.lr_scaling="mup_depth"`: additionally
  requires `model.params.reference_depth` and applies the depth factor to the
  attention and feed-forward blocks.

`optim.scale_embedding_lr` is available for Transformer Payne parameter
embedding layers. `optim.grad_clip=0.0`, the default, disables gradient clipping;
set a positive value to clip by global norm before the optimizer update.

Representative examples:

- [`examples/basic/01_train_payne_flux_mlp.py`](examples/basic/01_train_payne_flux_mlp.py)
- [`examples/basic/05_train_cannon_flux.py`](examples/basic/05_train_cannon_flux.py)
- [`examples/basic/06_train_isochrone_mlp.py`](examples/basic/06_train_isochrone_mlp.py)
- [`examples/basic/07_train_transformer_payne_flux.py`](examples/basic/07_train_transformer_payne_flux.py)
- [`examples/advanced/03_train_transformer_payne_intensity.py`](examples/advanced/03_train_transformer_payne_intensity.py)

## Recording the scientific contract of the emulator

For a shared artifact, the important question is not only “what weights did I save?” but “what does this emulator mean, what inputs does it expect, what outputs does it produce, and over what domain is it intended to be used?”

That contract lives in the effective bundle `spec` and `extras`.

There are two main ways to supply it:

1. put it directly into `cfg.io` so it becomes part of the resolved config-level spec;
2. pass `spec=...` and `extras=...` to `save_bundle(...)`.

A typical portability-oriented save looks like this:

```python
spec = {
    "input_domain": {
        "kind": "box_v1",
        "value_space": "physical_input_dict_tree_v1",
        "min_tree": {"parameters": label_min},
        "max_tree": {"parameters": label_max},
    },
    "reference_scaling_inputs": {
        "kind": "affine_minmax_v1",
        "applies_to": "inputs",
        "source_space": "physical_input_dict_tree_v1",
        "target_space": "canonical_input_dict_tree_v1",
        "min_tree": {"parameters": label_min},
        "max_tree": {"parameters": label_max},
    },
    "reference_scaling_outputs": {
        "kind": "affine_minmax_v1",
        "applies_to": "outputs",
        "source_space": "canonical_output_dict_tree_v1",
        "target_space": "physical_output_dict_tree_v1",
        "min_tree": {"flux": flux_min},
        "max_tree": {"flux": flux_max},
    },
}

bundle_dir = emu.save_bundle(
    "dist/my_flux_bundle",
    spec=spec,
    extras={
        "wavelength_angstrom": wavelength_grid,
        "companion_recipe": {
            "kind": "hf_repo_file_v1",
            "repo_id": "my-org/my-emulator-recipes",
            "revision": "v0.1.0",
            "path": "predict_physical.py",
            "role": "physical_inference_wrapper",
        },
    },
)
```

What belongs here:

- canonical input/output structure;
- channel names, units, and meanings when known;
- `reference_scaling_inputs` describing physical-to-training-space input transforms when known;
- `reference_scaling_outputs` describing canonical-to-physical output transforms when known;
- `input_domain` describing intended-use bounds;
- fixed coordinate grids or other conveniences in `extras`;
- optional companion-recipe pointers for wrappers that live outside the core bundle-owned payload.

## Preparing a release bundle for collaborators

For day-to-day development you may save a raw bundle. For a bundle you expect other astronomers to rely on, a release step is recommended.

```python
from astro_emulators_toolkit.bundle import (
    prepare_bundle_release,
    verify_bundle_fingerprint_evaluation,
)

released_dir = prepare_bundle_release(
    bundle_dir,
    path="dist/my_flux_bundle_released",
    release_name="my-flux-emulator",
    release_version="0.1.0",
)

summary = verify_bundle_fingerprint_evaluation(released_dir)
print(summary)
```

What this gives you:

- a human-readable release identity;
- a release-time fingerprint input/output pair stored inside the bundle;
- a reproducibility check that the released bundle still produces the expected output on that canonical fingerprint input;
- an updated integrity manifest for the release artifact.

If you omit `fingerprint_inputs=...`, automatic fingerprint synthesis requires the bundle to carry explicit `input_domain` or `reference_scaling_inputs` metadata.
When you do provide `fingerprint_inputs=...`, pass one canonical example. Public inference APIs refuse ambiguous 1D leaves; release preparation stores a single unbatched vector/list as a batch-one fingerprint.

### Scientific validation versus portability checks

These are not the same thing.

**Scientific validation** should include, as appropriate:

- held-out errors and calibration diagnostics;
- residual structure versus stellar labels, wavelength, age, metallicity, or other physical coordinates;
- behavior near the edges of the intended domain;
- comparisons against the expensive forward model or reference calculation;
- tests inside at least one realistic downstream likelihood/inference workflow.

**Portability checks** include:

- bundle integrity verification;
- load/save round-trips;
- fingerprint reproducibility checks.

A release fingerprint is a useful reproducibility smoke test. It is not a substitute for physical validation.

## API reference: the `Emulator` surface

### Construction

- `Emulator.from_config(cfg, init_example=None)`
- `Emulator.initialize(inputs=..., outputs=None, init_hints=None)`
- `Emulator.from_bundle(path, verbose=False)`
- `Emulator.from_pretrained(repo_id, revision=None, cache_dir=None, verbose=False)`

Use `from_config(...)` when building a new emulator. Use `from_bundle(...)` or `from_pretrained(...)` when the model already exists as a portable artifact.

### Inference

- `predict(x)` returns NumPy-backed outputs.
- `apply_jax(x, rng=None, postprocess=True, train=False)` returns JAX arrays.
- `make_frozen_apply(jit=False)` returns a callable over the current parameter/state snapshot.
- `make_device_batch_transform(...)` exposes model-family-specific device batch transforms when the runtime adapter provides them.

### Training and packaging

- `configure_training(optimizer=None, task=None)` for advanced training overrides. It only attaches a task object and/or optimizer for later `fit(...)`; it does not JIT/XLA-compile the model.
- `fit(train_dataset, validation_dataset=None, callbacks=None, resume=False, max_steps=None, device_batch_transform=None, method=None)`;
- `save_bundle(path=None, spec=None, extras=None)`.

### Metadata accessors

- `spec`
- `input_spec`, `output_spec`
- `reference_scaling_inputs`, `reference_scaling_outputs`, `input_domain`
- `bundle_extras`
- `input_channel_names_tree`, `output_channel_names_tree`
- `describe_bundle()`, `describe_domain()`

## Configuration model

`RootConfig` is the main configuration object. Its sections are:

- `model`: model-family name plus primitive parameters and optional init hints;
- `task`: training task name plus primitive parameters;
- `solver`: training algorithm selection;
- `optim`: optimizer and learning-rate schedule settings for gradient training;
- `training`: workdir, batch size, step schedule, logging/evaluation cadence, and other run controls;
- `io`: canonical input/output structure plus optional metadata, scaling, and domain;
- `bundle`: bundle-export options;
- `hub`: Hugging Face bundle source metadata.

A good default pattern is to start from a preset and then replace the pieces you need.

```python
from dataclasses import replace

from astro_emulators_toolkit.presets import payne_flux_mlp
from astro_emulators_toolkit.config import IOSpec, IOTreeSpec

cfg0 = payne_flux_mlp(workdir="runs/demo", profile="cpu_recommended")
cfg = replace(
    cfg0,
    io=IOSpec(
        inputs=IOTreeSpec(
            structure_tree={"parameters": None},
            channel_names_tree={"parameters": ["teff", "logg", "feh"]},
        ),
        outputs=cfg0.io.outputs,
        reference_scaling_inputs=cfg0.io.reference_scaling_inputs,
        reference_scaling_outputs=cfg0.io.reference_scaling_outputs,
        input_domain=cfg0.io.input_domain,
    ),
)
```

Config files can be saved and loaded as JSON or YAML based on the filename suffix:

```python
from astro_emulators_toolkit import load_config, save_config

save_config(cfg, "runs/demo/config.yaml")
cfg = load_config("runs/demo/config.yaml")
```

## Stable model families and tasks

### Stable model families

- `mlp`: general dense regression model. This is the basis for the stable Payne-style flux example and the stable isochrone example.
- `cannon`: closed-form quadratic baseline. Useful as a simple reference model and regression baseline, with an optional intercept term that is left unregularized by default in the closed-form solver.
- `transformer_payne`: wavelength-explicit spectral/intensity model family with runtime helpers for explicit wavelength handling.

### Stable task surface

- `regression`

### Experimental namespace

Prototype model families and tasks live under `experimental/...`. They are useful for research and development but are intentionally outside the stable public contract.

### Transformer-specific note

`transformer_payne` examples keep the wavelength interpolation/encoding path in float64 until the final embedding cast. For those workflows, enable:

```bash
export JAX_ENABLE_X64=1
```

See [`examples/basic/07_train_transformer_payne_flux.py`](examples/basic/07_train_transformer_payne_flux.py) and [`examples/advanced/03_train_transformer_payne_intensity.py`](examples/advanced/03_train_transformer_payne_intensity.py).

## Data flow

The intended pipeline is:

```text
raw storage
  -> dataset
  -> host-side semantic mapping
  -> loader / batching / valid_mask
  -> device-side preprocessing
  -> model
  -> task / loss / metrics
```

Validation follows the same split:

```text
validation raw storage
  -> validation dataset
  -> same host-side semantic mapping
  -> deterministic loader
  -> same device-side preprocessing, but train=False
  -> model
  -> task / metrics
```

The important boundary is:

- host side owns structure, naming, semantics, batching, and deterministic storage access;
- device side owns JAX-compatible numeric transforms that may be model-specific and optionally stochastic during training.

### 1. Batch contract

Every training or evaluation batch follows the same outer contract:

```python
{
    "x": ...,
    "y": ...,
    "sample_weight": ...,  # optional
    "valid_mask": ...,     # optional; loader adds this for padded eval batches
}
```

What each field means:

- `x`: model inputs in the semantic structure expected by the runtime/task;
- `y`: targets in the semantic structure expected by the task;
- `sample_weight`: optional per-sample weights;
- `valid_mask`: optional float mask for ignoring padded or invalid examples during evaluation.

Inside that batch, stable user-facing semantics are still canonical dict trees.

### 2. Raw datasets

Dataset responsibilities are intentionally narrow:

- deterministic indexed access;
- storage/decoding and sample fetching;
- assembling host batches with at least `x` and `y`;
- optional dataset-provided leaves such as `sample_weight` or `valid_mask`.

Loader responsibilities are separate:

- batching;
- shuffle and seed policy;
- deterministic evaluation ordering;
- adding `valid_mask` for padded eval batches.

Use the dataset type that matches your storage layer:

- `XYArrayDataset` for in-memory raw arrays returning `{"x": x_array, "y": y_array}`;
- `NpyTableDataset` for memmap-friendly loading from a single `.npy` table;
- `TreeArrayDataset` for in-memory data that is already organized as canonical semantic trees;
- a custom `DatasetProtocol` implementation when your data lives in many files or another external format.

Examples:

```python
from astro_emulators_toolkit.data import TreeArrayDataset, XYArrayDataset

raw_xy = XYArrayDataset(x=x_train, y=y_train)

tree_ds = TreeArrayDataset(
    x={"parameters": x_train},
    y={"flux": y_train},
)
```

```python
from astro_emulators_toolkit.config import NpyTableConfig
from astro_emulators_toolkit.data import NpyTableDataset

cfg = NpyTableConfig(
    path="data/isochrones.npy",
    inputs=(0, 1, 2),
    targets=(3, 4, 5),
    memmap=True,
)

raw_table = NpyTableDataset.from_config(cfg)
```

For many-file storage, implement the dataset contract directly:

```python
import numpy as np

class MyFileDataset:
    def __init__(self, paths: np.ndarray) -> None:
        self.paths = np.asarray(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def get_batch(self, idx: np.ndarray) -> dict[str, object]:
        x = load_label_rows(self.paths[idx])
        y = load_flux_rows(self.paths[idx])
        return {"x": x, "y": y}
```

`train_val_split(...)` and `SubsetDataset` handle deterministic splitting/subsetting on top of any dataset that follows the contract.

### 3. Host-side semantic mapping

`MappedDataset` is the explicit host-side boundary where raw storage batches become the semantic batch consumed by the model and task. This step is:

- host-side;
- deterministic;
- solver-agnostic;
- the right place to rename leaves, attach extra host metadata, or preserve `sample_weight`.

For the common case of raw `x` / `y` arrays that should become canonical trees:

```python
from astro_emulators_toolkit.data import MappedDataset, XYArrayDataset, pack_xy_as_tree

raw = XYArrayDataset(x=x_train, y=y_train)
train = MappedDataset(
    raw,
    map_batch=pack_xy_as_tree(x_leaf="parameters", y_leaf="flux"),
)
```

`map_batch` is intentionally general. It can also attach metadata-derived leaves or reshape several fields together when your canonical batch is more complex than one `x` leaf and one `y` leaf.

### 4. Device-side preprocessing

After a host batch has been fetched and batched, `device_batch_transform` is the explicit place for model-specific JAX-side preprocessing. This is not a storage concern and not a replacement for `MappedDataset`.

Typical uses include:

- injecting wavelength query grids for explicit-wavelength models;
- interpolating or resampling targets on device;
- applying train-time stochastic transforms that require an RNG.

Concrete device transforms expose:

- `for_init(batch)` for deterministic initialization-time shaping;
- `__call__(batch, train=..., rng=...)` for train/eval preprocessing.

For example, `transformer_payne` can provide a concrete device transform object through the runtime adapter:

```python
device_batch_transform = emu.make_device_batch_transform(
    wavelength_grid=wave,
    n_wavelength=wave.shape[0],
)
```

When `fit(...)` lazily initializes a model from data, it uses `device_batch_transform.for_init(...)` so initialization happens against the actual preprocessed batch shape.

### 5. Validation flow

Validation is not a different data model. It follows the same semantic pipeline as training:

- the validation dataset produces the same host-side batch contract;
- the same `MappedDataset` logic can be applied to both train and validation datasets;
- evaluation uses deterministic loader ordering rather than shuffled batch sampling;
- the same `device_batch_transform` runs with `train=False`;
- the loader pads the final eval batch when needed and records the real examples through `valid_mask`.

In other words, training and validation should differ in ordering and train/eval mode, not in the meaning of the batch itself.

### 6. Worked examples

Example A: in-memory raw arrays with one explicit semantic mapping step.

```python
from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import MappedDataset, XYArrayDataset, pack_xy_as_tree
from astro_emulators_toolkit.presets import payne_flux_mlp

emu = Emulator.from_config(
    payne_flux_mlp(workdir="runs/payne_flux_demo", profile="smoke")
)

train = MappedDataset(
    XYArrayDataset(x=x_train, y=y_train),
    map_batch=pack_xy_as_tree(x_leaf="parameters", y_leaf="flux"),
)

emu.fit(train)
```

Example B: a single `.npy` table with raw columns mapped into canonical trees.

```python
from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.config import NpyTableConfig
from astro_emulators_toolkit.data import MappedDataset, NpyTableDataset, pack_xy_as_tree
from astro_emulators_toolkit.presets import payne_flux_mlp

raw = NpyTableDataset.from_config(
    NpyTableConfig(
        path="data/train_table.npy",
        inputs=(0, 1, 2),
        targets=tuple(range(3, 503)),
        memmap=True,
    )
)
train = MappedDataset(
    raw,
    map_batch=pack_xy_as_tree(x_leaf="parameters", y_leaf="flux"),
)

emu = Emulator.from_config(payne_flux_mlp(workdir="runs/table_demo", profile="smoke"))
emu.fit(train)
```

Example C: a custom many-file dataset plus `transformer_payne` device-side preprocessing.

```python
import numpy as np

from astro_emulators_toolkit import Emulator
from astro_emulators_toolkit.data import MappedDataset
from astro_emulators_toolkit.presets import transformer_payne_flux


class MyFileDataset:
    def __init__(self, paths: np.ndarray) -> None:
        self.paths = np.asarray(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def get_batch(self, idx: np.ndarray) -> dict[str, object]:
        return {
            "x": load_parameter_rows(self.paths[idx]),
            "y": load_flux_rows(self.paths[idx]),
        }


def map_flux_batch(batch: dict[str, object]) -> dict[str, object]:
    return {
        "x": {"parameters": batch["x"]},
        "y": {"flux": batch["y"]},
    }


emu = Emulator.from_config(
    transformer_payne_flux(workdir="runs/transformer_demo", profile="smoke")
)
train = MappedDataset(MyFileDataset(train_paths), map_batch=map_flux_batch)
val = MappedDataset(MyFileDataset(val_paths), map_batch=map_flux_batch)
device_batch_transform = emu.make_device_batch_transform(
    wavelength_grid=wave,
    n_wavelength=wave.shape[0],
)

emu.fit(
    train,
    validation_dataset=val,
    device_batch_transform=device_batch_transform,
)
```

## Canonical dict-tree I/O and metadata

Stable user-facing inference and bundle metadata use nested dictionaries with numeric leaves.
This is intentional: the public I/O shape trades the flatness of one anonymous tensor for scientific clarity about what each leaf means. Examples should therefore keep trees shallow and use plain semantic keys, rather than exposing model-family-specific tensor layouts.

Examples:

```python
{"parameters": x}
{"flux": y}
{"parameters": x, "wavelengths": wave}
{"flux": y_flux}
```

The I/O spec can also carry metadata trees parallel to the structure tree:

- `channel_names_tree`
- `leaf_units_tree`
- `channel_units_tree`
- `leaf_meanings_tree`
- `channel_meanings_tree`

Important convention: channel metadata applies only to the **last axis** of a leaf.

## Scaling and domain metadata

Three metadata blocks matter especially for responsible use.

### `reference_scaling_inputs`

Describes the explicit affine transform between physical-space inputs and the canonical numeric representation used by the emulator. This block is optional and is only present when the author explicitly provided it in the config or bundle spec.

### `reference_scaling_outputs`

Describes the explicit affine transform between canonical emulator outputs and physical-space output values. This block is optional and is only present when the author explicitly provided it in the config or bundle spec.

Typical use:

```python
from astro_emulators_toolkit import normalize_tree, denormalize_tree

x_scaled = normalize_tree(x_physical, ref_inputs["min_tree"], ref_inputs["max_tree"])
y_physical = denormalize_tree(y_scaled, ref_outputs["min_tree"], ref_outputs["max_tree"])
```

### `input_domain`

Describes the intended-use box in physical input space using the same public input tree shape accepted by `predict(...)`. This is a statement about where the model is meant to be used, not a promise that all points inside are equally accurate.

## Bundle structure and semantics

A saved bundle is the portable artifact intended for sharing. Typical contents are:

- `README.txt`
- `config.json`
- `metadata.json`
- `weights/weights.safetensors`
- `bundle_integrity.json`
- optional sidecars such as `reference_scaling_inputs.safetensors`, `reference_scaling_outputs.safetensors`, `input_domain.safetensors`, `extras/*.safetensors`, and `fingerprint_evaluation/*.safetensors`

What the loader does:

- validates the bundle metadata structure;
- verifies the integrity manifest;
- materializes or hydrates sidecar-backed metadata;
- reconstructs the emulator from the saved config and runtime contract.

Common metadata fields exposed on load include:

- compatibility fields such as `bundle_format_version`, `config_schema_version`, `weights_layout`, and `model_family_id`;
- `spec` containing canonical I/O structure plus optional domain/scaling metadata;
- `release` identity if the bundle has been prepared as a release;
- `provenance` such as toolkit version, Python version, creation time, and git commit;
- optional `fingerprint_evaluation` descriptors.

## Bundles versus internal training state

These artifacts serve different purposes.

A **bundle** is the artifact you share with another astronomer. It is about portability, runtime meaning, provenance, and validation-oriented metadata.

Internal training state is for continuing a training run. It includes optimizer/run-state information and supports workflows such as `resume=True` during training. That is not the portability contract and should not be treated as the scientific release artifact.

In practical terms:

- collaborators should receive a bundle;
- `save_bundle(...)` is the public export step;
- run-continuation state is an internal training concern.

## Examples and recommended reading order

### Stable onboarding

Start here:

- [`examples/basic/00_visualize_datasets.py`](examples/basic/00_visualize_datasets.py): inspect the shipped example datasets;
- [`examples/basic/01_train_payne_flux_mlp.py`](examples/basic/01_train_payne_flux_mlp.py): first end-to-end stable training workflow;
- [`examples/basic/02_load_bundle_predict.py`](examples/basic/02_load_bundle_predict.py): first bundle-consumer workflow;
- [`examples/basic/03_inspect_bundle_metadata.py`](examples/basic/03_inspect_bundle_metadata.py): inspect what a bundle carries;
- [`examples/basic/04_use_bundle_in_map_fit.py`](examples/basic/04_use_bundle_in_map_fit.py): embed the emulator in a small JAX fitting loop.

Then continue with:

- [`examples/basic/05_train_cannon_flux.py`](examples/basic/05_train_cannon_flux.py)
- [`examples/basic/06_train_isochrone_mlp.py`](examples/basic/06_train_isochrone_mlp.py)
- [`examples/basic/07_train_transformer_payne_flux.py`](examples/basic/07_train_transformer_payne_flux.py)

### Advanced supported workflows

- [`examples/advanced/01_use_bundle_in_blackjax.py`](examples/advanced/01_use_bundle_in_blackjax.py)
- [`examples/advanced/02_resume_training.py`](examples/advanced/02_resume_training.py)
- [`examples/advanced/03_train_transformer_payne_intensity.py`](examples/advanced/03_train_transformer_payne_intensity.py)
- [`examples/advanced/04_bundle_metadata_and_portability.py`](examples/advanced/04_bundle_metadata_and_portability.py)
- [`examples/advanced/05_training_internals.py`](examples/advanced/05_training_internals.py)
- [`examples/advanced/06_train_payne_flux_mlp_from_config.py`](examples/advanced/06_train_payne_flux_mlp_from_config.py)
- [`examples/advanced/07_grid_search_payne_flux_mlp_lr.py`](examples/advanced/07_grid_search_payne_flux_mlp_lr.py)

### Maintainer and research surfaces

- `examples/development/`: maintainer benchmarks, longer recipes, and bundle-building utilities;
- `examples/experimental/`: prototype models/tasks and research workflows outside the stable contract.

## Development from source

For work from a source checkout, `uv` is the maintained path.

```bash
uv python install 3.12
uv sync
uv sync --extra blackjax
uv sync --extra viz
uv run pytest -q
```

Notes:

- the committed `uv.lock` is the reproducible source of truth for source checkouts;
- package support targets Python 3.11+;
- the package ships inline type hints and a `py.typed` marker;
- published users can use either `pip install ...` or `uv add ...`, while maintainers should prefer `uv sync`.

## Status

This is an alpha project. The stable public surface is intentionally small and the README is the primary documentation surface. The design emphasis is clear contracts, explicit preprocessing, portable sharing artifacts, and responsible downstream use.
