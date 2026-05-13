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

## Before opening a PR

- Run the most targeted tests you reasonably can, usually with `uv run pytest ...`.
- Keep changes small and focused.
- Mention any user-facing behavior, metadata, README, or example changes explicitly in the PR summary.

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
