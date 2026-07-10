# Changelog

All notable user-facing changes to this project will be documented here.

## [Unreleased]

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
- `cuda12` optional extra (`pip install "astro-emulators-toolkit[cuda12]"`).
- Guard warnings: wavelength-coordinate span vs frequency-encoding periods,
  spectral-resolution kernels large relative to the spectrum length,
  `resume=True` with no checkpoint found, no-op `si` scaling.

### Changed

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
