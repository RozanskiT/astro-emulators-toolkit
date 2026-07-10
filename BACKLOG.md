# Backlog

Deferred, intentionally-not-yet-done work. One entry per item: what, why it
is deferred, and what doing it properly requires. Remove entries when done
(and add a `CHANGELOG.md` entry if user-facing). Point-in-time context for
many of these lives in `ARCHITECTURE_REVIEW_2026-07.md`.

## Config / schema

- **Remove `training.val_fraction`.** It is advisory-only today (`fit()`
  never auto-splits — the split is a scientific decision; use
  `data.train_val_split`). Removal is deliberately deferred to the first
  `CONFIG_SCHEMA_VERSION` bump because `load_config` rejects unknown keys:
  doing it requires a v1→v2 config migration (the machinery exists in
  `bundle/versions.py`) and regenerating the shipped reference bundle assets
  (`examples/assets/build_reference_bundle.py`).
- **Optional `min_lr` floor for the cosine schedule.** The original
  transfer-learning code supported `min_learning_rate`; aemu cosine always
  decays toward zero (reaching it one count past the final applied step).
  Add an `optim.min_lr` field if a nonzero floor is needed for ported
  recipes.

## Presets / bundle compatibility

- **Rename the intensity preset's output leaf from `flux`.** An intensity
  emulator whose canonical output leaf is named `flux` contradicts the
  "plain semantic keys" doctrine. Deferred to its own PR (if at all): it is
  bundle-compatibility breaking and needs a spec/weights compat decision
  plus regenerated example assets.

## Training

- **Pin explicit `checkpoint_steps` milestones against retention GC.**
  `max_saved_checkpoints` (orbax `max_to_keep`) can garbage-collect
  explicitly requested milestone checkpoints once enough newer ones exist.
  Investigate orbax `should_keep_fn` semantics and pin explicit steps.
- **Upstream the vendored SOAP weight-decay/LR one-step schedule skew.**
  In `_vendor/soap_jax/soap.py` the wd term resolves the schedule at
  `count+1` while the gradient update uses `count`. A local patch would be
  silently lost on a vendor refresh — fix upstream in soap_jax, then
  re-vendor via `_vendor/get_soap_jax.py`.

## Data path

- **Lower-I/O explicit-wavelength training path.** The TransformerPayne
  device batch transform still interpolates targets against the full source
  spectrum per step (the storage layer now reads only touched columns, but
  the device step carries the whole grid as a jit constant and gathers from
  it). For very large source grids, add a dataset-side/host-side sampled
  wavelength path or precomputed query-index + interpolation-weight tables.
