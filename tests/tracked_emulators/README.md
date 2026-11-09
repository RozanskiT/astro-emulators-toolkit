# Tracked emulators

Pretrained bundles published by collaborators (and by us) that toolkit
changes must not break. For each entry in [`registry.toml`](registry.toml),
the check confirms that the current code:

1. loads it with `Emulator.from_pretrained(...)` at the pinned commit,
   without an unexpected `LegacyConfigWarning`;
2. reproduces the bundle's published fingerprint;
3. predicts, on recorded probe inputs, what the toolkit version that
   wrote the bundle predicted (`references/<slug>.npz`);
4. gives the same result through `make_frozen_apply(...)` as through
   `predict(...)`.

This is a portability and regression check. It does not validate the
emulators scientifically.

## Run it

It downloads every bundle from the Hugging Face Hub (cached after the first
run), so it is opt-in:

```bash
AEMU_TRACKED_EMULATORS=1 uv run pytest tests/tracked_emulators -q -rx
```

Run it before every release and after any change to bundle loading, config
parsing, model code or runtime adapters.

A published fingerprint was computed on the publisher's machine. If even the
toolkit that wrote the bundle misses it on the machine the reference was
recorded on, the reference says so and the fingerprint test reports an
expected failure (`xfail`) with the reason. It still fails if the original
toolkit passed it. Check 3 compares on the same machine either way.

## Add an emulator

1. Add an `[[emulator]]` entry to `registry.toml`. Pin `revision` to a commit
   (resolve a tag with
   `HfApi().model_info(repo_id, revision="v1.2.0").sha`), and set
   `written_by_toolkit` to the bundle's `metadata.json` →
   `provenance.toolkit_version`.
2. Record its reference **with that toolkit version**, by exporting the
   matching git revision and putting it first on `PYTHONPATH`:

   ```bash
   git archive v0.1.0 | tar -x -C /tmp/aet-0.1.0     # or the commit for that version
   PYTHONPATH=/tmp/aet-0.1.0/src uv run python tests/tracked_emulators/record_references.py
   ```

   It records only entries written by the running toolkit version that have
   no reference yet (`--only SLUG`, `--overwrite`, `--any-version` change
   that). Toolkit `0.2.0.dev0` is commit `65feb0b`.
3. Run the check, then commit the entry and its reference.

## Private emulators

Entries that should not appear in this public repository go in
`local/tracked_emulators/registry.toml`, which is gitignored and uses the
same format; their references go in `local/tracked_emulators/references/`.
Both commands pick that registry up automatically. Private Hub repositories
use your Hugging Face login (`hf auth login`) or `HF_TOKEN`.
