# AGENTS instructions (root scope)

## Purpose

This repository is for building, validating, sharing, and using astrophysical emulators in a way that is usable by other astronomers in downstream inference code.

The two documentation roles are:

- `README.md`: main user documentation for the current pre-1.0 phase;
- `AGENTS.md`: repo-level guidance for coding agents and assistants so changes preserve the intended public surface, artifact semantics, and validation posture.

## Documentation contract

`README.md` must remain a real manual, not just a landing page.

It should stay top-down and cover, at minimum:

1. project purpose and philosophy;
2. installation;
3. the main public components;
4. loading a shared bundle and predicting;
5. using the emulator inside downstream JAX inference;
6. training a new emulator;
7. recording I/O/scaling/domain metadata;
8. preparing a release bundle;
9. the important conceptual reference sections (config, data, model families, bundle semantics, examples, development notes).

Do not push essential semantics only into examples or tests.

## Artifact semantics

Be precise here.

- A **bundle** is the shareable artifact for another astronomer.
- Its role is portability, inspectability, provenance, runtime contract, and release/reproducibility checks.
- Do **not** describe a bundle as a generalized or “more complete” checkpoint.
- Internal run-continuation state is a different artifact with a different purpose.
- If docs mention `ModelCheckpoint` or `resume=True`, state clearly that this is internal training/run-management, not the public sharing contract.

The README should focus primarily on bundles and responsible use, not on internal run-state details.

## Public API and stable user workflows

The stable user-facing center of gravity is `astro_emulators_toolkit.Emulator`.

Supported stable workflows:

- `Emulator.from_bundle(...)` / `Emulator.from_pretrained(...)`
- `predict(...)` for host-side inference
- `apply_jax(...)` and `make_frozen_apply(...)` for JAX-side composition
- `Emulator.from_config(...)` and `fit(...)` for training
- `save_bundle(...)` for exporting a portable shared artifact

Stable supporting modules that README should document:

- `astro_emulators_toolkit.config`
- `astro_emulators_toolkit.presets`
- `astro_emulators_toolkit.data`
- `astro_emulators_toolkit.training`
- `astro_emulators_toolkit.bundle`

Avoid making users read deep internals for normal workflows.

## Core philosophy to preserve

### Keep the surface small

The public API should remain simple and Keras-like at the top level, even if internals stay modular for research work.

### Keep contracts explicit

- canonical nested dict-tree inputs/outputs;
- explicit preprocessing;
- explicit `reference_scaling_inputs`, `reference_scaling_outputs`, and `input_domain`;
- explicit metadata for channel names, units, and meanings where available.

Do not introduce hidden autoscaling or opaque preprocessing behind `predict(...)` or the frozen JAX callable.

### Support downstream composition

Sampling, inversion, and fitting should compose around the predictive mapping. Avoid adding broad, opinionated inference frameworks to the stable surface unless there is a strong reason and corresponding tests/docs.

### Distinguish scientific validation from portability checks

- scientific validation: held-out errors, residual structure, boundary behavior, comparison to the expensive model, realistic inference tests;
- portability checks: integrity verification, roundtrip load/save, release fingerprint checks.

Do not let docs imply that integrity or fingerprint verification is sufficient scientific validation.

## Stable versus experimental surface

Stable model families:

- `mlp`
- `cannon`
- `transformer_payne`

Stable task surface:

- `regression`

Experimental models/tasks must remain namespaced under `experimental/...` until intentionally promoted.

Promotion requires coordinated updates to:

- registries;
- runtime adapters and spec materialization/validation;
- bundle save/load path;
- tests;
- examples;
- README.

Do not quietly blur experimental and stable namespaces.

## Repo map

Key paths:

- `src/astro_emulators_toolkit/__init__.py`: top-level public package surface
- `src/astro_emulators_toolkit/emulator.py`: high-level facade
- `src/astro_emulators_toolkit/config/`: config dataclasses and JSON/YAML I/O
- `src/astro_emulators_toolkit/presets/`: stable config builders
- `src/astro_emulators_toolkit/models/`: stable model families and runtime adapters
- `src/astro_emulators_toolkit/tasks/`: stable tasks
- `src/astro_emulators_toolkit/training/`: trainer, callbacks, run-state helpers
- `src/astro_emulators_toolkit/data/`: datasets, loader, preprocessing helpers
- `src/astro_emulators_toolkit/bundle/`: portable artifact serialization, integrity, release helpers, Hub download
- `src/astro_emulators_toolkit/experimental/`: prototype models/tasks
- `examples/`: astronomer-facing and maintainer-facing scripts
- `tests/`: behavioral contracts

## Example hierarchy

Preserve the meaning of the example directories.

- `examples/basic/`: stable onboarding path for astronomers
- `examples/advanced/`: supported integration patterns, config-driven tuning patterns, and selected internals
- `examples/development/`: maintainer benchmarks, longer recipes, bundle builders
- `examples/experimental/`: prototype/research workflows outside the stable contract

When adding or revising examples, keep docstrings explicit about:

- data source;
- what the script creates;
- approximate runtime class;
- required optional extras or environment settings.

## Important runtime and metadata invariants

### Canonical I/O

- stable inference expects canonical nested dict trees with numeric leaves;
- this intentionally trades flat tensor calling conventions for scientific clarity, so examples should keep trees shallow and readable;
- runtime adapters own family-specific wrapping/packing/unpacking;
- channel metadata applies only to the last axis of a leaf;
- for stable families, docs/examples should show canonical keys, not internal tensor layouts.

### Metadata semantics

- `reference_scaling_inputs`, `reference_scaling_outputs`, and `input_domain` use explicit `min_tree` / `max_tree` semantics;
- `reference_scaling_inputs` is about reproducible transforms between physical inputs and model representation;
- `reference_scaling_outputs` is about reproducible transforms between model outputs and physical representation;
- `input_domain` is about intended-use bounds;
- fixed grids and similar conveniences belong in `bundle_extras`, not hidden runtime logic.

### Bundle behavior

- `save_bundle(...)` writes the portable artifact, including `README.txt`, `config.json`, `metadata.json`, weights, and integrity manifest;
- `from_bundle(...)` validates integrity and reconstructs runtime metadata;
- release helpers may add `release` metadata and `fingerprint_evaluation` sidecars;
- keep compatibility, runtime-contract, and metadata changes synchronized across save/load/tests/docs.

## Configuration guidance

Configs are deliberately primitive-only and JSON/YAML-friendly.

Preserve the pattern that:

- model, task, solver, and optimizer are selected by string names;
- presets are the supported starting point for common workflows;
- immutable config dataclasses can be customized by building a fresh `RootConfig` or using `dataclasses.replace(...)`.

Do not make normal user workflows depend on custom Python config classes or non-serializable callables.

## Data-layer guidance

Maintain the dataset/loader split.

Dataset responsibilities:

- deterministic indexed access;
- storage/decoding/column mapping;
- assembly of `x` and `y` payloads;
- optional metadata exposure.

Loader responsibilities:

- batching;
- ordering;
- shuffle/seed policy;
- `drop_last` behavior.

Keep preprocessing explicit. Do not move hidden physical-to-training-space transforms into datasets or loaders.

## Dependency boundaries

Current package facts:

- Python requirement: `>=3.11`
- base dependencies include JAX, Flax, Optax, Orbax checkpointing, safetensors, PyYAML, and Hugging Face download support
- optional extras are `blackjax` and `viz`
- Hugging Face support in the stable surface is download-only through `Emulator.from_pretrained(...)`

Avoid eager imports that make the base install require optional packages.

## Testing expectations

When behavior changes, update or add focused tests.

Common targeted commands:

```bash
uv run pytest tests/api -q
uv run pytest tests/bundle -q
uv run pytest tests/config -q
uv run pytest tests/data -q
uv run pytest tests/models -q
uv run pytest tests/tasks -q
uv run pytest tests/training -q
uv run pytest tests/examples -q
uv run mypy
```

Use the most relevant subset rather than always everything, but when changing bundle semantics or public docs/examples, run the corresponding focused tests.

## Common change patterns

### New stable model family

Requires coordinated work across:

- config/model implementation;
- runtime adapter;
- registry entry;
- bundle roundtrip tests;
- runtime-contract tests;
- stable examples;
- README.

### Bundle metadata or format change

Update together:

- save/load path;
- integrity logic;
- release helpers;
- metadata validation;
- relevant tests under `tests/bundle/`;
- user-facing docs/examples.

### Runtime I/O change

Update together:

- canonical pack/unpack logic;
- spec materialization/validation;
- channel metadata expectations;
- examples;
- tests.

## Review and summary expectations

When summarizing changes for a PR or user:

- call out user-facing API changes clearly;
- call out bundle compatibility changes clearly;
- distinguish scientific-validation guidance from portability/reproducibility checks;
- list the tests and example scripts actually run.
