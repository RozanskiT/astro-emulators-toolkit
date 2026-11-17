# Experimental examples

Audience: research/prototyping users.

These examples use namespaced experimental models/tasks (`experimental/...`) and randomized datasets.

- `07_train_isochrone_siren.py` trains the experimental `experimental/siren` model on the shipped example isochrone table and writes a portable bundle with explicit min/max scaling metadata.
- `08_train_mhe_flux.py --model mhe_mlp` (or `--model mhe_transformer_payne`) trains on the shipped flux spectra with explicit input normalization and exports a portable bundle. The default `smoke` profile takes seconds; `--profile cpu_recommended` runs a longer fit. No optional extras are needed; the script enables JAX float64 wavelengths.
