"""Physical-intensity inference for wavelength-explicit ExoJAX PayneTraining bundles."""

import jax
import jax.numpy as jnp


def _canonical_intensity_apply(emu):
    ref = emu.reference_scaling_outputs
    if ref is None or "log10_intensity" not in ref["min_tree"]:
        raise ValueError("Expected output scaling for the ExoJAX log10_intensity leaf")
    low = ref["min_tree"]["log10_intensity"]
    span = ref["max_tree"]["log10_intensity"] - low
    frozen = emu.make_frozen_apply(jit=False)

    def apply(labels, wavelengths):
        w = jnp.broadcast_to(wavelengths, (len(labels), len(wavelengths)))
        y = frozen({"parameters": labels, "wavelengths": w})["log10_intensity"]
        return jnp.power(jnp.float64(10), y.astype(jnp.float64) * span + low)

    return apply


def make_exojax_predictor(emu):
    """Map physical labels (8,) or (B,8) and shared vacuum wavelengths (Q,) in Å
    to float64 specific intensity (B,Q) in the original per-wavenumber units.

    Older ExoJAX TransformerPayne bundles use the recorded affine wavelength
    scale. The released MHE and 128-mode bundles declare their log10 conversion.
    Recorded detector gaps and out-of-band wavelengths return NaN.
    """
    family = emu.cfg.model.name
    if (
        family
        not in (
            "transformer_payne",
            "experimental/mhe_mlp",
            "experimental/mhe_transformer_payne",
            "experimental/interpolated_payne",
        )
        or emu.cfg.model.params.get("channels", 1) != 1
    ):
        raise ValueError("Expected a single-channel wavelength-explicit ExoJAX bundle")
    ref = emu.reference_scaling_inputs
    if ref is None or "parameters" not in ref["min_tree"]:
        raise ValueError("Expected input scaling for the eight ExoJAX parameters")
    low = jnp.asarray(ref["min_tree"]["parameters"], dtype=jnp.float64)
    span = jnp.asarray(ref["max_tree"]["parameters"], dtype=jnp.float64) - low
    if low.shape != (8,) or span.shape != (8,):
        raise ValueError("Expected input scaling for the eight ExoJAX parameters")

    extras = emu.bundle_extras
    coordinate = extras.get("wavelength_coordinate")
    if coordinate == "log10_angstrom_legacy_roundtrip":
        w0, w1 = extras["preprocessing"]["legacy_wavelength_bounds"]
    elif coordinate == "affine_angstrom" or (
        coordinate is None and family == "transformer_payne"
    ):
        w0, w1 = ref["min_tree"]["wavelengths"], ref["max_tree"]["wavelengths"]
    else:
        raise ValueError(f"Unsupported ExoJAX wavelength_coordinate: {coordinate!r}")
    intervals = extras.get("chip_intervals_angstrom")
    if intervals is not None:
        intervals = jnp.asarray(intervals, dtype=jnp.float64)
        if intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) == 0:
            raise ValueError("Expected chip_intervals_angstrom with shape (chips, 2)")

    if family == "experimental/interpolated_payne":
        from .models.interpolated_payne import _make_direct_intensity_apply

        apply = _make_direct_intensity_apply(emu)
    else:
        apply = _canonical_intensity_apply(emu)

    @jax.jit
    def predict(parameters, wavelength_angstrom):
        p = jnp.atleast_2d(jnp.asarray(parameters, dtype=jnp.float64))
        w = jnp.asarray(wavelength_angstrom, dtype=jnp.float64)
        if p.ndim != 2 or p.shape[-1] != 8 or w.ndim != 1:
            raise ValueError(
                "Expected physical labels (8,) or (B,8) and wavelengths (Q,)"
            )
        labels = ((p - low) / span).astype(jnp.float32)
        if intervals is not None:
            valid = jnp.any(
                (w[:, None] >= intervals[:, 0]) & (w[:, None] <= intervals[:, 1]),
                axis=1,
            )
            # Keep invalid coordinates out of the model and its reverse-mode pass.
            w = jnp.where(valid, w, intervals[0, 0])
        q = (w - w0) / (w1 - w0)
        if coordinate == "log10_angstrom_legacy_roundtrip":
            q = jnp.log10(w0 + q * (w1 - w0))
        intensity = apply(labels, q)
        if intervals is not None:
            intensity = jnp.where(valid, intensity, jnp.nan)
        return intensity

    return predict
