"""Visualize spectral-resolution postprocessing on example flux and intensity data.

Data: examples_datasets/irregular_flux and examples_datasets/irregular_intensity.
Creates: examples/runs/development_spectral_resolution_postprocess/resolution_postprocess.png.
Runtime: a few seconds on CPU.

Use --show to display the figure interactively after saving it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from astro_emulators_toolkit.inference.compose import downgrade_spectral_resolution

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import load_randomized_flux_arrays, load_randomized_intensity_arrays


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO_ROOT / "examples" / "runs" / "development_spectral_resolution_postprocess"
)
PLOT_PATH = RUN_DIR / "resolution_postprocess.png"
RESOLUTIONS = (100_000.0, 30_000.0, 10_000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset row to visualize.",
    )
    parser.add_argument(
        "--n-wave",
        type=int,
        default=500,
        help="Number of points in the common log-wavelength grid.",
    )
    return parser.parse_args()


def _load_pyplot(*, show: bool) -> Any:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _common_log_grid(
    wavelength: np.ndarray, *, n_wave: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_wave < 2:
        raise ValueError("n_wave must be at least 2.")
    lo = float(np.min(wavelength))
    hi = float(np.max(wavelength))
    log_wavelength = np.linspace(np.log10(lo), np.log10(hi), n_wave, dtype=np.float64)
    return log_wavelength, np.power(10.0, log_wavelength)


def _interp_to_grid(
    source_wavelength: np.ndarray,
    values: np.ndarray,
    target_wavelength: np.ndarray,
) -> np.ndarray:
    return np.interp(target_wavelength, source_wavelength, values).astype(np.float32)


def _downgrade_payload(
    *,
    log_wavelength: np.ndarray,
    flux: np.ndarray,
    intensity: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    def apply_fn() -> dict[str, jnp.ndarray]:
        return {
            "flux": jnp.asarray(flux[None, :], dtype=jnp.float32),
            "intensity": jnp.asarray(intensity[None, :, :], dtype=jnp.float32),
        }

    apply_lowres = downgrade_spectral_resolution(
        apply_fn,
        jnp.asarray(log_wavelength, dtype=jnp.float32),
        resolution=resolution,
        axis_tree={"flux": -1, "intensity": -2},
    )
    result = apply_lowres()
    return np.asarray(result["flux"][0]), np.asarray(result["intensity"][0])


def main() -> None:
    args = _parse_args()
    plt = _load_pyplot(show=args.show)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    _x_flux, flux_all, wave_flux = load_randomized_flux_arrays()
    _x_intensity, intensity_all, wave_intensity = load_randomized_intensity_arrays()

    flux_index = args.sample_index % flux_all.shape[0]
    intensity_index = args.sample_index % intensity_all["lines"].shape[0]
    log_wavelength, wavelength = _common_log_grid(wave_flux, n_wave=args.n_wave)

    flux = _interp_to_grid(wave_flux, flux_all[flux_index], wavelength)
    intensity_lines = _interp_to_grid(
        wave_intensity["lines"], intensity_all["lines"][intensity_index], wavelength
    )
    intensity_continuum = _interp_to_grid(
        wave_intensity["continuum"],
        intensity_all["continuum"][intensity_index],
        wavelength,
    )
    intensity = np.stack([intensity_lines, intensity_continuum], axis=-1)
    if np.any(~np.isfinite(intensity)) or np.any(intensity <= 0.0):
        raise ValueError("Intensity sample must contain only positive finite values.")

    fig, axes = plt.subplots(
        len(RESOLUTIONS),
        2,
        figsize=(12, 9),
        sharex="col",
        constrained_layout=True,
    )

    for row, resolution in enumerate(RESOLUTIONS):
        flux_lowres, intensity_lowres = _downgrade_payload(
            log_wavelength=log_wavelength,
            flux=flux,
            intensity=intensity,
            resolution=resolution,
        )

        ax_flux = axes[row, 0]
        ax_intensity = axes[row, 1]

        ax_flux.plot(wavelength, flux, color="0.72", lw=1.0, label="original")
        ax_flux.plot(
            wavelength,
            flux_lowres,
            color="tab:blue",
            lw=1.3,
            label=f"R={resolution:,.0f}",
        )
        ax_flux.set_ylabel(f"R={resolution:,.0f}\nnormalized flux")
        ax_flux.grid(alpha=0.2)

        ax_intensity.plot(
            wavelength,
            intensity[:, 0],
            color="0.72",
            lw=1.0,
            label="original lines",
        )
        ax_intensity.plot(
            wavelength,
            intensity[:, 1],
            color="0.72",
            lw=1.0,
            ls="--",
            label="original continuum",
        )
        ax_intensity.plot(
            wavelength,
            intensity_lowres[:, 0],
            color="tab:purple",
            lw=1.3,
            label="lines",
        )
        ax_intensity.plot(
            wavelength,
            intensity_lowres[:, 1],
            color="tab:purple",
            lw=1.3,
            ls="--",
            label="continuum",
        )
        ax_intensity.set_yscale("log")
        ax_intensity.set_ylabel("intensity")
        ax_intensity.grid(alpha=0.2)

        if row == 0:
            ax_flux.set_title("Flux leaf, axis -1")
            ax_intensity.set_title("Intensity leaf, axis -2")
            ax_flux.legend(fontsize=8)
            ax_intensity.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Wavelength [Angstrom]")
    axes[-1, 1].set_xlabel("Wavelength [Angstrom]")
    fig.suptitle("Spectral-resolution postprocessing with shared batch/channel kernels")
    fig.savefig(PLOT_PATH, dpi=180)
    print("Saved:", PLOT_PATH)
    print("Flux sample index:", flux_index)
    print("Intensity sample index:", intensity_index)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
