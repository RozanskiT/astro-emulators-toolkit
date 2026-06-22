# Contributing

## Scope

- The stable public surface is bundle-first inference and training; keep stable behavior simple and Keras-like.
- `README.md` is the public source of truth for pre-1.0 releases and should stay aligned with the implementation.
- New research ideas should start under `src/astro_emulators_toolkit/experimental/` and `examples/experimental/` unless they are being intentionally promoted.

## Setup

```bash
uv sync
uv sync --extra blackjax
```

## Development workflow

This project is currently maintained by one primary developer and this documentation supports structured development.

- Keep `main` installable, tested, and close to releasable.
- Protect `main` on GitHub when possible and require CI to pass before merging.
- Do normal work on short-lived branches from current `main`, such as
  `feature/...`, `fix/...`, `docs/...`, or `experimental/...`.
- Do not keep a permanent `develop` branch unless the project grows into
  multiple concurrent release trains. For larger work, prefer a temporary
  integration branch such as `next/...` or `release/...`, then delete it after
  merge or release.
- Open draft PRs for non-trivial changes.
- Merge only when the branch is coherent: tests are appropriate for the change,
  docs are aligned, and any public-surface or bundle-compatibility impact is
  named explicitly.

## Feature branch checklist

Use this for ordinary fixes, docs changes, and new features.

1. Start from an up-to-date local `main`.

```bash
git switch main
git pull --ff-only origin main
```

2. Create a short-lived branch.

```bash
git switch -c feature/short-description
```

Use a prefix that matches the work, such as `feature/...`, `fix/...`,
`docs/...`, `experimental/...`, or `release/...`.

3. Make the change, then run the focused checks that match the change.

```bash
uv run pytest tests/api -q
```

4. Review and commit the branch.

```bash
git status
git diff
git add path/to/changed_file.py
git commit -m "Short imperative summary"
```

5. Push the branch to GitHub.

```bash
git push -u origin feature/short-description
```

6. Open a GitHub PR from the feature branch into `main`.

Keep it as a draft while still working. In the PR summary, name any user-facing
API, bundle compatibility, metadata, README, example, or dependency impact.

7. Wait for CI, review the diff, then merge the PR on GitHub.

8. Update local `main` and remove the finished branch.

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/short-description
git fetch --prune
```

If GitHub did not delete the remote branch automatically, delete it manually:

```bash
git push origin --delete feature/short-description
```

Stop here for docs-only changes, internal cleanup, experiments, or small changes
that do not need a new installable package release.

## Before opening a PR

- Run the most targeted tests you reasonably can, usually with `uv run pytest ...`.
- For changes that may affect the published package, also run the relevant
  lint/type checks:

```bash
uv run --frozen ruff format --check
uv run --frozen ruff check .
uv run --frozen mypy
```

- Keep changes small and focused.
- Mention any user-facing behavior, metadata, README, or example changes explicitly in the PR summary.
- Add a `CHANGELOG.md` entry for user-facing changes.

## Change review checklist

Before merging, check the parts of the public contract touched by the change.

- Public API: update tests and README when stable `Emulator` workflows or stable
  supporting modules change.
- Bundle compatibility: update save/load code, validation, release helpers, and
  bundle tests together when metadata, format, runtime contract, or integrity
  behavior changes.
- Runtime I/O: keep canonical nested dict-tree inputs/outputs explicit and add
  focused tests for packing, unpacking, specs, and channel metadata.
- Scientific validation: document held-out errors, residual structure, boundary
  behavior, or realistic inference checks separately from portability checks.
- Dependencies: keep optional functionality behind optional extras and avoid
  eager imports that make the base install heavier.

Useful focused test groups:

```bash
uv run pytest tests/api -q
uv run pytest tests/bundle -q
uv run pytest tests/config -q
uv run pytest tests/data -q
uv run pytest tests/models -q
uv run pytest tests/tasks -q
uv run pytest tests/training -q
uv run pytest tests/examples -q
```

## Release checklist

Do not release every merged PR. Release when `main` has a user-facing fix,
feature, packaging change, or documentation update that package users should get
from `pip`.

Use patch versions for small fixes, for example `0.1.1`. Use the next minor
version for new supported features or meaningful public-contract changes, for
example `0.2.0`. PyPI will not accept re-uploading the same version, so choose
the version deliberately.

1. Start a release branch from an up-to-date `main`.

```bash
git switch main
git pull --ff-only origin main
git switch -c release/0.1.1
```

2. Update the release files.

- Move completed `CHANGELOG.md` entries out of `Unreleased`.
- Bump `src/astro_emulators_toolkit/_version.py`.
- Update `CITATION.cff` when the public release citation should change.

3. Run the checks appropriate for a package release.

```bash
uv run --frozen ruff format --check
uv run --frozen ruff check .
uv run --frozen mypy
uv run pytest tests/api -q
uv run pytest tests/bundle -q
uv run pytest tests/examples -q
```

Run additional focused suites when the release touches config, data, models,
tasks, training, or examples.

4. Commit, push, open a release PR, and merge only after CI passes, including
   the release-smoke job.

```bash
git add CHANGELOG.md CITATION.cff src/astro_emulators_toolkit/_version.py
git commit -m "Prepare 0.1.1 release"
git push -u origin release/0.1.1
```

5. After the release PR is merged, update local `main`.

```bash
git switch main
git pull --ff-only origin main
git branch -d release/0.1.1
git fetch --prune
```

6. Build the source distribution and wheel from the merged release commit.

Make sure `dist/` contains only artifacts for the version being published.

```bash
rm -rf dist
uv build
uv run --frozen twine check dist/*
```

`uv build` creates both the source distribution and wheel under `dist/`.

7. Tag the exact commit being published.

```bash
git tag -a v0.1.1 -m "astro-emulators-toolkit 0.1.1"
git push origin v0.1.1
```

8. Upload the checked artifacts to PyPI.

```bash
uv run --frozen twine upload dist/*
```

9. Optionally verify the published package in a fresh environment.

```bash
python -m pip install --upgrade astro-emulators-toolkit==0.1.1
python -c "import astro_emulators_toolkit as aet; print(aet.__version__)"
```

## Adding a new model or task

- For stable additions, update the relevant config, registry, runtime adapter or contract, tests, and at least one example together.
- Do not expand the stable surface casually; keep new research work experimental unless promotion is intentional.

### Stable family checklist

- Implement the family config and model under `src/astro_emulators_toolkit/models/`.
- Implement the family runtime adapter so it owns init-context derivation, wrapped predictive I/O, spec defaults and validation, bundle runtime metadata, and any batch transform helpers.
- Use the built-in stable families as the template: `MLPRuntimeAdapter`, `CannonRuntimeAdapter`, and `TransformerPayneRuntimeAdapter`.
- Register one `ModelRegistryEntry` in `src/astro_emulators_toolkit/models/__init__.py`.
- Add family tests and at least one example, plus a preset only if the stable onboarding path needs it.
- If a new stable family needs edits outside registration, family-local code, tests, examples, or presets, pause and tighten the shared abstraction before adding more global branching.

## Docs and examples

- Update `README.md` when public behavior changes.
- Keep examples aligned with the current code and tests.
