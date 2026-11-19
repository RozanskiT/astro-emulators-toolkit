# Experimental examples

Audience: research/prototyping users.

These examples use namespaced experimental models/tasks (`experimental/...`) and randomized datasets.

- `07_train_isochrone_siren.py` trains the experimental `experimental/siren` model on the shipped example isochrone table and writes a portable bundle with explicit min/max scaling metadata.
- `08_train_wavelength_explicit_flux.py` trains, exports, and reloads any of four models on the same flux data. Pass `--model transformer_payne`, `mhe_mlp`, `mhe_transformer_payne`, or `spectral_basis_mlp`. The default smoke profile runs on CPU in seconds with the base dependencies. Use `--profile cpu_recommended` for a longer fit.
