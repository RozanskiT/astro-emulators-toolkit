# Advanced examples

Audience: users integrating bundles into inference loops and custom training workflows.
These scripts are where `make_frozen_apply()` is front-and-center for downstream JAX integration; the smaller MAP fit has moved to `examples/basic/04_use_bundle_in_map_fit.py`.

All dataset-backed scripts use canonical nested dict I/O with in-repo arrays from `examples/examples_datasets/` (or bundles derived from them).
The training examples here also use preset `profile="smoke"` for short CPU runtime; switch to `profile="cpu_recommended"` when you want the longer preset defaults tuned from the development benchmarks.
The transformer_payne intensity example enables `JAX_ENABLE_X64=1` so wavelength interpolation and encoding stay in float64 until the final embedding cast.

Includes:
- BlackJAX usage via `make_frozen_apply()`,
- resume training,
- transformer_payne intensity training,
- metadata portability checks,
- training internals,
- config-driven single-run training and a small learning-rate scan.

The config-driven tuning pair is:
- `06_train_payne_flux_mlp_from_config.py`: run one JSON/YAML `RootConfig` and write `tuning_result.json`;
- `07_grid_search_payne_flux_mlp_lr.py`: generate one config per learning rate, launch the single-run worker, and collate `grid_results.json`.
