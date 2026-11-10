# Changelog

All notable user-facing changes to this project will be documented here.

## [Unreleased]

### Added

- Publishing released bundles on the Hugging Face Hub: the `aemu` command
  (`aemu card`, `aemu check`, `aemu publish`) and the functions behind it in
  `astro_emulators_toolkit.bundle` (`write_model_card`,
  `check_bundle_for_publishing`, `plan_bundle_upload`, `publish_bundle`).
  Publishing uploads the bundle contract plus documentation files in one
  commit, tags it `v<release_version>`, and verifies the published copy. A
  default `README.md` model card has the author's sections plus a block
  generated from the bundle metadata (interface table, usage snippet,
  requirements, provenance). Tags are not moved, older releases and
  interface changes without a major version are refused, unless overridden.
- `Emulator.from_pretrained(...)` records the repository, requested revision
  and resolved commit in `bundle_metadata["hub_source"]`.
- `bundle.versions.minimum_loading_toolkit_version(metadata)`: the oldest
  toolkit release that loads a bundle.

### Deprecated

- `RootConfig.hub` (`HubConfig`): never read; setting it warns. It will be
  removed in 0.3.0 together with the next config schema version.

## [0.2.0] - 2026-11-09

### Added

- `optim.lr_scaling="si"`: legacy standard parameterization (embedding groups
  at `lr * scale_embedding_lr`, no width division).
- Best-model checkpoint tracking: `training.checkpoint_best_metric` /
  `checkpoint_best_mode`, the `BestModelCheckpoint` callback, and
  `Emulator.load_best_checkpoint()`; the best `TrainState` is kept under
  `checkpoints_best/`, separate from resume checkpoints.
- `eval_mode="passthrough"` on the transformer device batch transforms:
  validation batches are used exactly as the dataset provides them
  (per-example query wavelengths + targets).
- Run provenance: `run_config.json` is never overwritten (changed configs go
  to `run_config.resume-N.json`); every `fit(...)` appends its effective
  choices to `effective_runs.jsonl`; checkpoints store an optimizer/schedule
  fingerprint and resuming with a changed config warns.
- Bundle compatibility contract: supported-version windows and a load-time
  metadata migration registry in `bundle/versions.py`; loading never mutates
  on-disk bundles.
- `training.async_checkpointing=False` forces synchronous checkpoint writes
  (GPU-instability bisection).
- `RootConfig.with_updates(...)` accepts per-section dict updates.
- `model.params.param_dtype` for `mlp` (and the experimental `siren`,
  `explicit_wavelength_mlp`, `mlp_2d_regression`): parameter storage dtype,
  defaulting to `dtype`.
- Config schema migrations (`config/migrations.py`): bundles and configs
  written by 0.1.x (config schema v1) load with their v1 meaning; settings
  0.1.x accepted but ignored are replaced by the value in effect and reported
  in one `LegacyConfigWarning`. Bundles from the `0.2.0.devN` pre-releases
  (stamped v1, written with v2 meaning) load unchanged.
- `cuda12` optional extra (`pip install "astro-emulators-toolkit[cuda12]"`).
- Guard warnings: wavelength-coordinate span vs frequency-encoding periods,
  spectral-resolution kernels large relative to the spectrum length,
  `resume=True` with no checkpoint found, no-op `si` scaling.

### Changed

- `CONFIG_SCHEMA_VERSION` is 2. Bundles saved by 0.1.x still load (see
  Added); bundles saved by 0.2.0 need toolkit >= 0.2.0.
- Default learning-rate schedule is linear warmup + cosine decay
  (`warmup_steps=None` means automatic 10% of `num_steps`; previously the
  default was a constant LR). Every applied step now has a strictly positive
  LR; the cosine schedule rejects `decay_steps`.
- `Emulator.fit(callbacks=None)` builds config-declared callbacks by default
  (`training.checkpoint_*` now takes effect without extra wiring); pass
  `callbacks=[]` to disable.
- `optim.lr_scaling="standard"` is rejected as ambiguous; setting
  `scale_embedding_lr != 1` with `lr_scaling=None` is a config error.
- `CSVLogger` appends to existing files on resume instead of truncating.
- The gradient solver rejects `solver.params` (previously silently ignored).
- Bundle provenance `git_commit` renamed to `working_tree_git_commit`;
  bundle metadata `resolved` now records `optim` and `optimizer_source`.
- Shuffle streams are keyed by `SeedSequence((seed, cycle))`: the same
  `shuffle_seed` produces a different (better-isolated) batch order than
  previous releases.
- Model parameters are stored in the configured dtype (`param_dtype` now
  follows `dtype`); SIREN hidden layers default to the paper's `omega0=30`.

### Fixed

- `optim.lr_scaling` ("mup"/"mup_depth") was a silent no-op for models built
  through `Emulator.from_config(...)`: parameter labeling now handles the
  canonical wrapper nesting and `make_tx` fails closed when a scaled group
  matches no parameters.
- Removed a per-step `jnp.sort` of sampled query wavelengths from the
  transformer training step (unnecessary, and implicated in long-run GPU
  command-buffer crashes).
- `NpyTableDataset` with memmap no longer materializes entire rows per batch;
  only the touched columns are read.
- `save_bundle()` no longer crashes on machines without a `git` binary.
- Scientific-notation string coercion applies only to YAML configs; JSON
  strings are preserved as written.
- Experimental explicit-wavelength adapters no longer crash on custom output
  structures with `channels > 1`.
- The final trained state always receives validation metrics when evaluation
  is enabled; the checkpoint manager is released on exceptions;
  `on_checkpoint_end` is dispatched to every registered callback.

### Removed

- `training.val_fraction`: no toolkit version ever read it (`fit()` never
  splits). Build the validation set with `data.train_val_split(...)` and
  pass it as `validation_dataset`. Older configs and bundles that contain it
  still load; the key is dropped in memory.
- The explicit `jaxlib` dependency: `jax` pins its own compatible `jaxlib`,
  preventing jax/jaxlib/CUDA-plugin version skew on GPU clusters. Use the
  `cuda12` extra for GPU installs.

## [0.1.0] - 2026-05-13

### Added

- Initial public alpha release.
- Bundle-first emulator workflow.
- Stable model families: `mlp`, `cannon`, `transformer_payne`.
- Base install includes training and Hugging Face download support.
- Optional BlackJAX integration.

### Changed

- Public documentation is README-first.
