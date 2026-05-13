"""Use a resolution-degraded bundle apply function inside a small MAP fit loop.

Data: shipped reference flux bundle + synthetic observations.
Creates: nothing.
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax

from astro_emulators_toolkit import Emulator, normalize_tree
from astro_emulators_toolkit.inference.compose import downgrade_spectral_resolution

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_bundle import require_reference_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a flux and residual plot for the initial and final MAP model.",
    )
    return parser.parse_args()


def _logit(p: jax.Array) -> jax.Array:
    return jnp.log(p) - jnp.log1p(-p)


def _show_fit_plot(
    *,
    wavelength: np.ndarray,
    true_flux: np.ndarray,
    initial_flux: np.ndarray,
    final_flux: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_flux, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    ax_flux.plot(wavelength, true_flux, color="0.2", lw=1.4, label="true degraded flux")
    ax_flux.plot(
        wavelength,
        initial_flux,
        color="tab:orange",
        lw=1.0,
        alpha=0.9,
        label="initial guess",
    )
    ax_flux.plot(
        wavelength,
        final_flux,
        color="tab:blue",
        lw=1.1,
        alpha=0.9,
        label="final MAP fit",
    )
    ax_flux.set_ylabel("Normalized flux")
    ax_flux.set_title("MAP fit with R=30,000 spectral-resolution postprocessing")
    ax_flux.grid(alpha=0.2)
    ax_flux.legend(fontsize=8)

    ax_resid.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax_resid.plot(
        wavelength,
        initial_flux - true_flux,
        color="tab:orange",
        lw=1.0,
        label="initial - true",
    )
    ax_resid.plot(
        wavelength,
        final_flux - true_flux,
        color="tab:blue",
        lw=1.0,
        label="final - true",
    )
    ax_resid.set_xlabel("Wavelength [Angstrom]")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(alpha=0.2)
    ax_resid.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = _parse_args()
    bundle = require_reference_bundle()
    emu = Emulator.from_bundle(bundle)
    wave = np.asarray(emu.bundle_extras["wavelength_angstrom"], dtype=np.float32)
    log_wavelength = np.linspace(
        np.log10(float(wave[0])),
        np.log10(float(wave[-1])),
        num=wave.shape[0],
        dtype=np.float64,
    )
    # Freeze once, add the spectral-resolution postprocess, then jit the
    # downstream objective around the composed callable.
    apply_flux = downgrade_spectral_resolution(
        emu.make_frozen_apply(jit=False),
        log_wavelength,
        resolution=30_000.0,
        output_path="flux",
        axis=-1,
        jit=False,
    )

    ref = emu.reference_scaling_inputs or {}
    theta_min = jnp.asarray(ref["min_tree"]["parameters"], dtype=jnp.float32)
    theta_max = jnp.asarray(ref["max_tree"]["parameters"], dtype=jnp.float32)
    theta_true = jnp.asarray([5600.0, 4.2, -0.1], dtype=jnp.float32)
    theta_unit = (theta_true - theta_min) / (theta_max - theta_min)
    z_true = _logit(jnp.clip(theta_unit, 1e-5, 1.0 - 1e-5))

    def theta_from_latent(z: jax.Array) -> jax.Array:
        return theta_min + jax.nn.sigmoid(z) * (theta_max - theta_min)

    def model_flux(z: jax.Array) -> jax.Array:
        theta = theta_from_latent(z)
        x_scaled = normalize_tree(
            {"parameters": theta[None, :]},
            ref["min_tree"],
            ref["max_tree"],
        )
        return apply_flux(x_scaled)["flux"][0]

    compiled_model_flux = jax.jit(model_flux)
    y_obs = compiled_model_flux(z_true)
    y_err = jnp.asarray(np.full_like(np.asarray(y_obs), 0.02, dtype=np.float32))

    def logpost(z: jax.Array) -> jax.Array:
        y_model = model_flux(z)
        resid = (y_obs - y_model) / y_err
        return -0.5 * jnp.sum(resid**2)

    def objective(z: jax.Array) -> jax.Array:
        return -logpost(z)

    z = z_true + jnp.asarray([0.5, -0.5, 0.20], dtype=jnp.float32)
    z_initial = z
    initial_logpost = logpost(z)
    tx = optax.adam(0.03)
    state = tx.init(z)

    @jax.jit
    def step(z, state):
        value, grad = jax.value_and_grad(objective)(z)
        updates, state = tx.update(grad, state)
        return optax.apply_updates(z, updates), state, value

    for _ in range(150):
        z, state, _ = step(z, state)
    theta_map = theta_from_latent(z)
    print("Applied spectral resolution:", 30_000)
    print("True log-posterior:", float(logpost(z_true)))
    print("Initial offset log-posterior:", float(initial_logpost))
    print("Final log-posterior:", float(logpost(z)))
    print("MAP parameters [teff, logg, feh]:", np.asarray(theta_map))
    if args.show:
        _show_fit_plot(
            wavelength=wave,
            true_flux=np.asarray(y_obs),
            initial_flux=np.asarray(compiled_model_flux(z_initial)),
            final_flux=np.asarray(compiled_model_flux(z)),
        )


if __name__ == "__main__":
    main()
