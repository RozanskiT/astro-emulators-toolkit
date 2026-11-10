# Backlog

Deferred, intentionally-not-yet-done work. One entry per item: what, why it
is deferred, and what doing it properly requires. Remove entries when done
(and add a `CHANGELOG.md` entry if user-facing).

## Config / schema

- **Remove `RootConfig.hub` (`HubConfig`) in 0.3.0.** Deprecated in 0.2.1:
  never read, and a bundle cannot meaningfully record where it will be
  published. Removing the field changes the saved config, so 0.2.0 could not
  read the new bundles; it waits for the next config schema version (v3),
  whose migration drops the key from older documents (as v2 did for
  `training.val_fraction`).
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
  plus regenerated example assets. Published intensity bundles already
  depend on the `flux` leaf (the TransformerPayne entries in
  `tests/tracked_emulators/registry.toml`, used by SPICE).

## Bundles / release

- **Fingerprint tolerance across platforms.** `prepare_bundle_release`
  stamps `rtol=1e-5, atol=1e-7`. For float32 TransformerPayne bundles
  published from Gadi (Linux x86), a macOS arm64 CPU misses that by up to
  ~6x even with the toolkit that wrote them (two tracked bundles), so
  downstream users can see fingerprint failures that are only platform
  numerics. Consider a dtype-aware default tolerance, or documenting the
  fingerprint as a same-platform check.

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

## Examples / downstream composition

- **BlackJAX example fails when `JAX_ENABLE_X64=1` is set.** With x64 on,
  `examples/advanced/01_use_bundle_in_blackjax.py` stops with a `lax.cond`
  dtype mismatch: the emulator's log-density stays float32 while BlackJAX's
  sampler state is float64. TransformerPayne users need x64 globally, so
  their samplers can hit the same error. Decide whether the example (or a
  documented pattern) casts the log-density to the sampler's dtype, or
  whether frozen-apply outputs should follow the input dtype, and test the
  example under both settings.
