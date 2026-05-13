# Development examples

Audience: maintainers and smoke-test workflows.

These are not the primary stable onboarding path.
The longer CPU benchmark recipes in `11` through `15` are the reference point for the stable preset `profile="cpu_recommended"` defaults.
The earlier RFF and maintenance scripts stay here as maintainer-only workflows and are intentionally out of the main README path.
Script `16_benchmark_transformer_payne_flux_training_components.py` is the measurement harness for loader, host-to-device, train-step, and validation-path timing snapshots.
Script `17_benchmark_cannon_design_matrix.py` compares the current Cannon quadratic feature path against a vectorized candidate before any package-level optimization lands.
Script `18_spectral_resolution_postprocess.py` visualizes the JAX-side spectral-resolution wrapper on irregular flux and intensity example datasets.

Typical setup from a clone:

```bash
uv sync
```

Add extras when needed:

```bash
uv sync --extra blackjax
```

The base install already includes training, checkpointing, and Hugging Face bundle download support. Plotting support is part of the default `dev` group.

Scripts `11` to `15` are relatively long CPU runs and are best treated as benchmark/reference recipes rather than first-use examples.
