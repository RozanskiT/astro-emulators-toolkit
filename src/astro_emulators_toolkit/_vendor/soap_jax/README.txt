Vendored SOAP_JAX
=================

This directory contains a vendored copy of the public `SOAP_JAX` project by
Haydn Jones. Many thanks to the upstream author for publishing and maintaining
the implementation.

Upstream repository:
https://github.com/haydn-jones/SOAP_JAX

Pinned upstream commit:
ddddc25724dfd629b0cd01584eca1e32ea8ac4de

What is included here:
- `__init__.py`
- `soap.py`
- `LICENSE`

Why this is vendored:
- SOAP is an internal optimizer implementation detail for this toolkit.
- `uv sync` already installs this project editable, so keeping SOAP in `src/`
  avoids an extra separately-installed package.
- This keeps the integration surface small while preserving upstream
  provenance.
- It should make a future migration simpler if comparable functionality lands
  directly in Optax and we want to swap implementations behind
  `astro_emulators_toolkit.optimizers`.

Local adjustments:
- The vendored file keeps the upstream algorithm and structure.
- Type-only dependencies on `chex` and `jaxtyping` were removed to keep the
  toolkit dependency surface smaller.

Refresh workflow:
- Run `python -m astro_emulators_toolkit._vendor.get_soap_jax` to re-fetch the
  pinned upstream snapshot.
- Review the diff afterward before committing any update.
