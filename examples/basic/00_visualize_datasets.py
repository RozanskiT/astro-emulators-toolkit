"""Visualize bundled example datasets and save figures.

Data: irregular_flux, irregular_intensity, isochrones.
Creates: examples/runs/basic_dataset_viz/*.png
Runtime: a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _example_data import (
    ISOCHRONE_X_COLUMNS,
    load_isochrones_table,
    load_randomized_flux,
    load_randomized_intensity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    out_dir = REPO_ROOT / "examples" / "runs" / "basic_dataset_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_flux, y_flux, wave_flux = load_randomized_flux()
    _x_int, y_int, wave = load_randomized_intensity()
    flux = y_flux["flux"]
    wave_lines = wave["lines"]
    wave_cont = wave["continuum"]
    iso_table, iso_columns = load_isochrones_table()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(6):
        ax.plot(wave_flux, flux[i], lw=1.0, alpha=0.8, label=f"spectrum {i + 1}")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("Normalized flux [dimensionless]")
    ax.set_title("Randomized flux dataset: sample spectra")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    flux_fig = out_dir / "flux_samples.png"
    fig.tight_layout()
    fig.savefig(flux_fig, dpi=160)
    plt.close(fig)

    i_log_teff = iso_columns.index("log_Teff")
    i_log_g = iso_columns.index("log_g")
    i_feh = iso_columns.index("feh")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sc = ax.scatter(
        iso_table[:, i_log_teff],
        iso_table[:, i_log_g],
        c=iso_table[:, i_feh],
        s=5,
        cmap="viridis",
        alpha=0.7,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("[Fe/H] [dex]")
    ax.set_xlabel("log10(Teff / K)")
    ax.set_ylabel("log10(g / cm s$^{-2}$)")
    ax.set_title("Isochrones grid coverage")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    iso_fig = out_dir / "isochrones_logg_logteff.png"
    fig.tight_layout()
    fig.savefig(iso_fig, dpi=160)
    plt.close(fig)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)
    for i in range(4):
        ax_top.plot(wave_lines, y_int["lines"][i], lw=1.0, alpha=0.85)
    ax_top.set_title("Randomized intensity dataset: line intensities")
    ax_top.set_ylabel("Intensity [erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$ sr$^{-1}$]")
    ax_top.set_xlabel("Wavelength [Angstrom]")
    ax_top.grid(alpha=0.2)

    for i in range(4):
        ax_bottom.plot(wave_cont, y_int["continuum"][i], lw=1.0, alpha=0.85)
    ax_bottom.set_title("Randomized intensity dataset: continuum intensities")
    ax_bottom.set_xlabel("Wavelength [Angstrom]")
    ax_bottom.set_ylabel("Intensity [erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$ sr$^{-1}$]")
    ax_bottom.grid(alpha=0.2)
    int_fig = out_dir / "intensity_lines_and_continuum.png"
    fig.tight_layout()
    fig.savefig(int_fig, dpi=160)
    plt.close(fig)

    print("Saved:", flux_fig)
    print("Saved:", iso_fig)
    print("Saved:", int_fig)
    print("Isochrone X columns reference:", ISOCHRONE_X_COLUMNS)
    print("Flux input shape:", x_flux["parameters"].shape)
    print(
        "Flux shape:",
        flux.shape,
        "Intensity lines shape:",
        y_int["lines"].shape,
        "continuum shape:",
        y_int["continuum"].shape,
    )


if __name__ == "__main__":
    main()
