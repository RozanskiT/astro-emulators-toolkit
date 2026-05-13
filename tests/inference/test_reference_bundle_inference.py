from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from astro_emulators_toolkit import Emulator, denormalize_tree, normalize_tree
from astro_emulators_toolkit.bundle import load_bundle_fingerprint_evaluation


REFERENCE_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "assets"
    / "reference_bundle_release"
)


@dataclass(frozen=True)
class ReferenceBundle:
    emulator: Emulator
    reference_scaling_inputs: dict[str, Any]
    input_domain: dict[str, Any]
    apply_eager: Callable[[dict[str, Any]], dict[str, Any]]
    apply_jitted: Callable[[dict[str, Any]], dict[str, Any]]
    fingerprint_canonical_parameters: jax.Array
    fingerprint_physical_parameters: jax.Array
    fingerprint_flux: jax.Array
    wavelength_angstrom: jax.Array
    parameter_min: jax.Array
    parameter_max: jax.Array


@pytest.fixture(scope="module")
def reference_bundle() -> ReferenceBundle:
    emu = Emulator.from_bundle(REFERENCE_BUNDLE)
    reference_scaling_inputs = emu.reference_scaling_inputs
    input_domain = emu.input_domain
    assert reference_scaling_inputs is not None
    assert input_domain is not None

    fingerprint = load_bundle_fingerprint_evaluation(REFERENCE_BUNDLE)
    fingerprint_inputs = {
        "parameters": jnp.asarray(
            fingerprint["inputs"]["parameters"], dtype=jnp.float32
        )
    }
    fingerprint_physical = denormalize_tree(
        fingerprint_inputs,
        reference_scaling_inputs["min_tree"],
        reference_scaling_inputs["max_tree"],
    )
    parameter_min = jnp.asarray(
        reference_scaling_inputs["min_tree"]["parameters"], dtype=jnp.float32
    )
    parameter_max = jnp.asarray(
        reference_scaling_inputs["max_tree"]["parameters"], dtype=jnp.float32
    )

    return ReferenceBundle(
        emulator=emu,
        reference_scaling_inputs=reference_scaling_inputs,
        input_domain=input_domain,
        apply_eager=emu.make_frozen_apply(jit=False),
        apply_jitted=emu.make_frozen_apply(jit=True),
        fingerprint_canonical_parameters=fingerprint_inputs["parameters"][0],
        fingerprint_physical_parameters=fingerprint_physical["parameters"][0],
        fingerprint_flux=jnp.asarray(
            fingerprint["outputs"]["flux"][0], dtype=jnp.float32
        ),
        wavelength_angstrom=jnp.asarray(
            emu.bundle_extras["wavelength_angstrom"], dtype=jnp.float32
        ),
        parameter_min=parameter_min,
        parameter_max=parameter_max,
    )


def _canonical_input_from_physical(
    theta_phys: jax.Array,
    reference_scaling_inputs: dict[str, Any],
) -> dict[str, jax.Array]:
    theta = jnp.asarray(theta_phys, dtype=jnp.float32)
    if theta.ndim == 1:
        theta = theta[None, :]
    return normalize_tree(
        {"parameters": theta},
        reference_scaling_inputs["min_tree"],
        reference_scaling_inputs["max_tree"],
    )


def _predict_flux_physical(
    apply_fn: Callable[[dict[str, Any]], dict[str, Any]],
    reference_scaling_inputs: dict[str, Any],
    theta_phys: jax.Array,
) -> jax.Array:
    y = apply_fn(_canonical_input_from_physical(theta_phys, reference_scaling_inputs))
    return jnp.asarray(y["flux"][0], dtype=jnp.float32)


def _physical_apply(
    apply_fn: Callable[[dict[str, Any]], dict[str, Any]],
    reference_scaling_inputs: dict[str, Any],
) -> Callable[[jax.Array], dict[str, Any]]:
    def wrapped(theta_phys: jax.Array) -> dict[str, Any]:
        return apply_fn(
            _canonical_input_from_physical(theta_phys, reference_scaling_inputs)
        )

    return wrapped


def _diagonal_gaussian_loglik(
    model_flux: jax.Array,
    observed_flux: jax.Array,
    sigma: float | jax.Array,
) -> jax.Array:
    residual = (model_flux - observed_flux) / sigma
    return -0.5 * jnp.sum(jnp.square(residual))


def _loglik_physical(
    apply_fn: Callable[[dict[str, Any]], dict[str, Any]],
    reference_scaling_inputs: dict[str, Any],
    theta_phys: jax.Array,
    observed_flux: jax.Array,
    sigma: float | jax.Array,
) -> jax.Array:
    model_flux = _predict_flux_physical(apply_fn, reference_scaling_inputs, theta_phys)
    return _diagonal_gaussian_loglik(model_flux, observed_flux, sigma)


def _logit(p: jax.Array) -> jax.Array:
    return jnp.log(p) - jnp.log1p(-p)


def _physical_from_latent(z: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    return lo + jax.nn.sigmoid(z) * (hi - lo)


def test_reference_bundle_fingerprint_point_is_best_among_small_perturbations(
    reference_bundle: ReferenceBundle,
):
    sigma = 0.02
    theta_true = reference_bundle.fingerprint_physical_parameters
    true_loglik = _loglik_physical(
        reference_bundle.apply_eager,
        reference_bundle.reference_scaling_inputs,
        theta_true,
        reference_bundle.fingerprint_flux,
        sigma,
    )

    perturbations = jnp.asarray(
        [
            [50.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.05],
        ],
        dtype=jnp.float32,
    )
    perturbed_logliks = [
        _loglik_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_true + perturbation,
            reference_bundle.fingerprint_flux,
            sigma,
        )
        for perturbation in perturbations
    ]

    assert all(float(true_loglik) > float(value) for value in perturbed_logliks)


def test_physical_space_forward_wrapper_matches_manual_canonical_path(
    reference_bundle: ReferenceBundle,
):
    theta_phys = jnp.asarray([5600.0, 4.2, -0.1], dtype=jnp.float32)

    canonical_inputs = _canonical_input_from_physical(
        theta_phys, reference_bundle.reference_scaling_inputs
    )
    manual = reference_bundle.apply_eager(canonical_inputs)
    physical_apply = _physical_apply(
        reference_bundle.apply_eager, reference_bundle.reference_scaling_inputs
    )
    wrapped = physical_apply(theta_phys)

    np.testing.assert_allclose(
        np.asarray(wrapped["flux"]),
        np.asarray(manual["flux"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_outer_jitted_loglik_matches_eager_and_prejitted_frozen_apply(
    reference_bundle: ReferenceBundle,
):
    theta_phys = jnp.asarray([5650.0, 3.9, -0.04], dtype=jnp.float32)

    def loglik_with(apply_fn: Callable[[dict[str, Any]], dict[str, Any]]):
        return lambda theta: _loglik_physical(
            apply_fn,
            reference_bundle.reference_scaling_inputs,
            theta,
            reference_bundle.fingerprint_flux,
            0.02,
        )

    eager_apply_loglik = jax.jit(loglik_with(reference_bundle.apply_eager))(theta_phys)
    prejitted_apply_loglik = jax.jit(loglik_with(reference_bundle.apply_jitted))(
        theta_phys
    )

    np.testing.assert_allclose(
        np.asarray(prejitted_apply_loglik),
        np.asarray(eager_apply_loglik),
        rtol=1e-6,
        atol=1e-6,
    )


def test_loglik_gradient_wrt_physical_labels_is_finite(
    reference_bundle: ReferenceBundle,
):
    z0 = _logit(jnp.asarray([0.57, 0.43, 0.62], dtype=jnp.float32))

    def loglik_from_latent(z: jax.Array) -> jax.Array:
        theta_phys = _physical_from_latent(
            z, reference_bundle.parameter_min, reference_bundle.parameter_max
        )
        return _loglik_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_phys,
            reference_bundle.fingerprint_flux,
            0.02,
        )

    value, grad = jax.value_and_grad(loglik_from_latent)(z0)

    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 1e-3


def test_vmap_loglik_matches_scalar_loop(reference_bundle: ReferenceBundle):
    theta0 = reference_bundle.fingerprint_physical_parameters
    candidates = jnp.stack(
        [
            theta0,
            theta0 + jnp.asarray([40.0, 0.0, 0.0], dtype=jnp.float32),
            theta0 + jnp.asarray([0.0, -0.08, 0.0], dtype=jnp.float32),
            theta0 + jnp.asarray([0.0, 0.0, 0.04], dtype=jnp.float32),
        ]
    )

    def scalar_loglik(theta_phys: jax.Array) -> jax.Array:
        return _loglik_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_phys,
            reference_bundle.fingerprint_flux,
            0.02,
        )

    batched = jax.vmap(scalar_loglik)(candidates)
    looped = jnp.asarray([scalar_loglik(theta) for theta in candidates])

    np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), rtol=1e-6)


def test_masked_likelihood_ignores_corrupted_region(
    reference_bundle: ReferenceBundle,
):
    model_flux = _predict_flux_physical(
        reference_bundle.apply_eager,
        reference_bundle.reference_scaling_inputs,
        reference_bundle.fingerprint_physical_parameters,
    )
    clean_observation = reference_bundle.fingerprint_flux
    bad_region = (reference_bundle.wavelength_angstrom >= 5003.0) & (
        reference_bundle.wavelength_angstrom <= 5004.0
    )
    good_region = ~bad_region
    assert int(jnp.sum(bad_region)) > 0

    corrupted_observation = clean_observation + jnp.where(bad_region, 5.0, 0.0)

    def masked_loglik(observed_flux: jax.Array) -> jax.Array:
        residual = (model_flux - observed_flux) / 0.02
        return -0.5 * jnp.sum(jnp.where(good_region, jnp.square(residual), 0.0))

    masked_clean = masked_loglik(clean_observation)
    masked_corrupted = masked_loglik(corrupted_observation)
    unmasked_clean = _diagonal_gaussian_loglik(model_flux, clean_observation, 0.02)
    unmasked_corrupted = _diagonal_gaussian_loglik(
        model_flux, corrupted_observation, 0.02
    )

    np.testing.assert_allclose(
        np.asarray(masked_corrupted), np.asarray(masked_clean), atol=1e-6
    )
    assert float(unmasked_corrupted) < float(unmasked_clean) - 100_000.0


def test_domain_penalty_distinguishes_in_domain_and_out_of_domain_points(
    reference_bundle: ReferenceBundle,
):
    domain_min = jnp.asarray(
        reference_bundle.input_domain["min_tree"]["parameters"], dtype=jnp.float32
    )
    domain_max = jnp.asarray(
        reference_bundle.input_domain["max_tree"]["parameters"], dtype=jnp.float32
    )

    def loglik_with_domain(theta_phys: jax.Array) -> jax.Array:
        in_domain = jnp.all((theta_phys >= domain_min) & (theta_phys <= domain_max))
        loglik = _loglik_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_phys,
            reference_bundle.fingerprint_flux,
            0.02,
        )
        return jnp.where(in_domain, loglik, -jnp.inf)

    in_domain = loglik_with_domain(reference_bundle.fingerprint_physical_parameters)
    too_hot = loglik_with_domain(jnp.asarray([7001.0, 3.75, 0.0], dtype=jnp.float32))
    too_metal_poor = loglik_with_domain(
        jnp.asarray([5750.0, 3.75, -0.4], dtype=jnp.float32)
    )

    assert bool(jnp.isfinite(in_domain))
    assert bool(jnp.isneginf(too_hot))
    assert bool(jnp.isneginf(too_metal_poor))


def test_heteroscedastic_diagonal_noise_model_behaves_correctly(
    reference_bundle: ReferenceBundle,
):
    theta_phys = reference_bundle.fingerprint_physical_parameters + jnp.asarray(
        [50.0, 0.0, 0.0], dtype=jnp.float32
    )
    model_flux = _predict_flux_physical(
        reference_bundle.apply_eager,
        reference_bundle.reference_scaling_inputs,
        theta_phys,
    )
    wave01 = (
        reference_bundle.wavelength_angstrom
        - jnp.min(reference_bundle.wavelength_angstrom)
    ) / (
        jnp.max(reference_bundle.wavelength_angstrom)
        - jnp.min(reference_bundle.wavelength_angstrom)
    )
    sigma = 0.01 + 0.01 * wave01

    helper_loglik = _diagonal_gaussian_loglik(
        model_flux, reference_bundle.fingerprint_flux, sigma
    )
    manual_residual = (model_flux - reference_bundle.fingerprint_flux) / sigma
    manual_loglik = -0.5 * jnp.sum(jnp.square(manual_residual))

    np.testing.assert_allclose(
        np.asarray(helper_loglik), np.asarray(manual_loglik), rtol=1e-7, atol=1e-7
    )


def test_map_recovers_fingerprint_labels_from_flux(
    reference_bundle: ReferenceBundle,
):
    z = _logit(jnp.asarray([0.6, 0.4, 0.65], dtype=jnp.float32))
    tx = optax.adam(0.05)
    state = tx.init(z)

    def objective(z_current: jax.Array) -> jax.Array:
        theta_phys = _physical_from_latent(
            z_current, reference_bundle.parameter_min, reference_bundle.parameter_max
        )
        return -_loglik_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_phys,
            reference_bundle.fingerprint_flux,
            0.02,
        )

    @jax.jit
    def step(
        z_current: jax.Array, opt_state: optax.OptState
    ) -> tuple[jax.Array, optax.OptState, jax.Array]:
        value, grad = jax.value_and_grad(objective)(z_current)
        updates, opt_state = tx.update(grad, opt_state)
        return optax.apply_updates(z_current, updates), opt_state, value

    initial_objective = objective(z)
    for _ in range(120):
        z, state, _ = step(z, state)
    final_objective = objective(z)

    recovered_unit = jax.nn.sigmoid(z)
    recovered_theta = _physical_from_latent(
        z, reference_bundle.parameter_min, reference_bundle.parameter_max
    )
    normalized_label_error = jnp.max(
        jnp.abs(recovered_unit - reference_bundle.fingerprint_canonical_parameters)
    )

    assert float(final_objective) < float(initial_objective) * 0.01
    assert float(normalized_label_error) < 0.01
    assert bool(
        jnp.all(
            jnp.abs(recovered_theta - reference_bundle.fingerprint_physical_parameters)
            < jnp.asarray([25.0, 0.03, 0.01], dtype=jnp.float32)
        )
    )


def test_map_jointly_recovers_labels_and_scalar_additive_offset(
    reference_bundle: ReferenceBundle,
):
    beta_true = jnp.asarray(0.015, dtype=jnp.float32)
    observed_flux = reference_bundle.fingerprint_flux + beta_true
    u = jnp.concatenate(
        [
            _logit(jnp.asarray([0.57, 0.43, 0.6], dtype=jnp.float32)),
            jnp.asarray([0.0], dtype=jnp.float32),
        ]
    )
    tx = optax.adam(0.05)
    state = tx.init(u)

    def objective(u_current: jax.Array) -> jax.Array:
        theta_phys = _physical_from_latent(
            u_current[:3],
            reference_bundle.parameter_min,
            reference_bundle.parameter_max,
        )
        model_flux = _predict_flux_physical(
            reference_bundle.apply_eager,
            reference_bundle.reference_scaling_inputs,
            theta_phys,
        )
        model_with_offset = model_flux + u_current[3]
        return -_diagonal_gaussian_loglik(model_with_offset, observed_flux, 0.02)

    @jax.jit
    def step(
        u_current: jax.Array, opt_state: optax.OptState
    ) -> tuple[jax.Array, optax.OptState, jax.Array]:
        value, grad = jax.value_and_grad(objective)(u_current)
        updates, opt_state = tx.update(grad, opt_state)
        return optax.apply_updates(u_current, updates), opt_state, value

    initial_objective = objective(u)
    for _ in range(150):
        u, state, _ = step(u, state)
    final_objective = objective(u)

    recovered_unit = jax.nn.sigmoid(u[:3])
    normalized_label_error = jnp.max(
        jnp.abs(recovered_unit - reference_bundle.fingerprint_canonical_parameters)
    )
    recovered_beta = u[3]

    assert float(final_objective) < float(initial_objective) * 0.01
    assert float(normalized_label_error) < 0.02
    assert float(jnp.abs(recovered_beta - beta_true)) < 5e-4
