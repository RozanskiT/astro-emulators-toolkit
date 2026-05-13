# Basic examples (stable)

Audience: first-time users (astronomers and collaborators).
`00_visualize_datasets.py` is an optional orientation step that shows the shipped toy/example datasets before the core bundle, fitting, and training workflows.
This folder includes one small MAP-fit example early because fitting is a core bundle workflow, not a niche addon.

All training/data-facing scripts in this folder use canonical nested dict I/O backed by arrays from `examples/examples_datasets/`.
The runnable training scripts use preset `profile="smoke"` so they finish quickly on CPU.
For genuinely useful CPU training runs, use the same presets with `profile="cpu_recommended"`.
The transformer example enables `JAX_ENABLE_X64=1` because its wavelength path stays float64 through interpolation and encoding.

If `examples/assets/reference_bundle_release` does not yet contain safetensors files,
generate them with:
`python examples/assets/build_reference_bundle.py`.
That script creates both `reference_bundle_raw` and `reference_bundle_release`.

Suggested order:
0. `00_visualize_datasets.py` (optional dataset tour for understanding the shipped toy/example data)
1. `01_train_payne_flux_mlp.py` (predicts flux at all available training wavelengths)
2. `02_load_bundle_predict.py` (loads the shipped `examples/assets/reference_bundle_release` and predicts with host-side `predict()`)
3. `03_inspect_bundle_metadata.py` (prints a bundle summary, channel metadata, extras, and input-domain information)
4. `04_use_bundle_in_map_fit.py` (uses `make_frozen_apply()` inside a small jitted MAP fit loop)
5. `05_train_cannon_flux.py`
6. `06_train_isochrone_mlp.py`
7. `07_train_transformer_payne_flux.py`
