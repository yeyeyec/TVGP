#!/usr/bin/env python3
"""Post-process a public CEM JumpGP model without changing its source code."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular


PUBLIC_ABSOLUTE_JITTER = 1e-8
DEFAULT_RELATIVE_JITTERS = (
    0.0,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
)
DEFAULT_MAX_CONDITION = 1e12


class PredictionAdapterError(RuntimeError):
    """Raised when the public model cannot be post-processed reliably."""


@dataclass(frozen=True)
class AdapterSettings:
    public_absolute_jitter: float = PUBLIC_ABSOLUTE_JITTER
    relative_jitters: tuple = DEFAULT_RELATIVE_JITTERS
    max_condition: float = DEFAULT_MAX_CONDITION


def _scalar(value, name):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise PredictionAdapterError("Invalid scalar {}.".format(name))
    return float(array[0])


def _finite_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise PredictionAdapterError("Non-finite {}.".format(name))
    return array


def _prediction_tolerance(*values):
    scale = max(1.0, *(abs(float(value)) for value in values))
    return 2e-6 * scale


def _attempt_prediction(
    covariance,
    cross_covariance,
    signal_test_variance,
    response_test_variance,
    y_selected,
    sample_mean,
    noise_variance,
    total_jitter,
    max_condition,
):
    regularized = covariance + total_jitter * np.eye(covariance.shape[0])
    factor = cholesky(regularized, lower=True, check_finite=False)
    condition_number = float(np.linalg.cond(regularized))
    if not np.isfinite(condition_number) or condition_number > max_condition:
        raise PredictionAdapterError(
            "Regularized covariance condition number {:.6g} exceeds {:.6g}.".format(
                condition_number, max_condition
            )
        )

    transformed_cross = solve_triangular(
        factor, cross_covariance, lower=True, check_finite=False
    )
    transformed_sample_residual = solve_triangular(
        factor, y_selected - sample_mean, lower=True, check_finite=False
    )
    ones = np.ones_like(y_selected)
    transformed_ones = solve_triangular(
        factor, ones, lower=True, check_finite=False
    )
    transformed_y = solve_triangular(
        factor, y_selected, lower=True, check_finite=False
    )
    arrays = (
        transformed_cross,
        transformed_sample_residual,
        transformed_ones,
        transformed_y,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise PredictionAdapterError("A triangular solve returned non-finite values.")

    plugin_mean = _scalar(
        transformed_cross.T @ transformed_sample_residual + sample_mean,
        "reconstructed public mean",
    )
    posterior_reduction = float(np.sum(transformed_cross * transformed_cross))
    plugin_latent_variance = signal_test_variance - posterior_reduction
    plugin_response_variance = response_test_variance - posterior_reduction

    mean_information = float(np.sum(transformed_ones * transformed_ones))
    if not np.isfinite(mean_information) or mean_information <= 0.0:
        raise PredictionAdapterError("Invalid GLS mean information.")
    gls_numerator = float(np.sum(transformed_ones * transformed_y))
    gls_mean = gls_numerator / mean_information
    if not np.isfinite(gls_mean):
        raise PredictionAdapterError("The GLS local mean is non-finite.")

    transformed_gls_residual = solve_triangular(
        factor, y_selected - gls_mean, lower=True, check_finite=False
    )
    if not np.all(np.isfinite(transformed_gls_residual)):
        raise PredictionAdapterError("The GLS residual solve is non-finite.")
    uk_mean = _scalar(
        transformed_cross.T @ transformed_gls_residual + gls_mean,
        "UK mean",
    )
    one_k_inv_k = float(np.sum(transformed_ones * transformed_cross))
    mean_uncertainty = (1.0 - one_k_inv_k) ** 2 / mean_information
    uk_latent_variance = plugin_latent_variance + mean_uncertainty
    uk_response_variance = uk_latent_variance + noise_variance

    quantities = (
        posterior_reduction,
        plugin_latent_variance,
        plugin_response_variance,
        mean_information,
        one_k_inv_k,
        mean_uncertainty,
        uk_latent_variance,
        uk_response_variance,
    )
    if not all(np.isfinite(value) for value in quantities):
        raise PredictionAdapterError("A predictive quantity is non-finite.")
    variance_tolerance = 1e-9 * max(
        1.0,
        abs(signal_test_variance),
        abs(response_test_variance),
        abs(noise_variance),
    )
    for name, value in (
        ("plug-in latent variance", plugin_latent_variance),
        ("plug-in response variance", plugin_response_variance),
        ("UK latent variance", uk_latent_variance),
        ("UK response variance", uk_response_variance),
        ("UK mean-estimation increment", mean_uncertainty),
    ):
        if value < -variance_tolerance:
            raise PredictionAdapterError(
                "Materially negative {}: {:.12g}.".format(name, value)
            )

    return {
        "plugin_mean_external": plugin_mean,
        "plugin_latent_variance_external": max(plugin_latent_variance, 0.0),
        "plugin_response_variance_external": max(plugin_response_variance, 0.0),
        "uk_mean": uk_mean,
        "uk_latent_variance": max(uk_latent_variance, 0.0),
        "uk_response_variance": max(uk_response_variance, 0.0),
        "noise_variance": noise_variance,
        "mean_uncertainty": max(mean_uncertainty, 0.0),
        "gls_mean": gls_mean,
        "mean_information": mean_information,
        "log_mean_information": float(np.log(mean_information)),
        "one_k_inv_k": one_k_inv_k,
        "condition_number": condition_number,
    }


def compute_external_predictions(model, settings=None):
    """Return public native and external UK predictions from one CEM model."""
    settings = settings or AdapterSettings()
    selected = np.asarray(model["r"], dtype=bool).reshape(-1)
    x = _finite_array(model["x"], "local inputs")
    y = _finite_array(model["y"], "local responses").reshape(-1, 1)
    xt = _finite_array(model["xt"], "test input")
    logtheta = _finite_array(model["logtheta"], "covariance parameters").reshape(-1)
    if selected.size != x.shape[0] or y.shape[0] != x.shape[0]:
        raise PredictionAdapterError("CEM membership and local data dimensions differ.")
    if not np.any(selected):
        raise PredictionAdapterError("CEM returned an empty local region.")
    if xt.shape[0] != 1:
        raise PredictionAdapterError("The adapter expects one test input per local fit.")

    x_selected = x[selected, :]
    y_selected = y[selected, :]
    sample_mean = _scalar(model["ms"], "public sample mean")
    public_mean = _scalar(model["mu_t"], "public predictive mean")
    public_response_variance = _scalar(
        model["sig2_t"], "public response variance"
    )
    public_variance_tolerance = 1e-9 * max(1.0, abs(public_response_variance))
    if public_response_variance < -public_variance_tolerance:
        raise PredictionAdapterError(
            "Public code returned a materially negative response variance."
        )

    covariance_function, covariance_parts = model["cv"]
    covariance = _finite_array(
        covariance_function(covariance_parts, logtheta, x_selected),
        "local covariance matrix",
    )
    response_test, cross_covariance = covariance_function(
        covariance_parts, logtheta, x_selected, xt
    )
    response_test_variance = _scalar(response_test, "response test variance")
    cross_covariance = _finite_array(cross_covariance, "cross covariance")
    noise_variance = float(np.exp(2.0 * logtheta[-1]))
    signal_test_variance = response_test_variance - noise_variance
    if not np.isfinite(noise_variance) or noise_variance < 0.0:
        raise PredictionAdapterError("Invalid fitted nugget variance.")
    if covariance.shape != (len(y_selected), len(y_selected)):
        raise PredictionAdapterError("Unexpected local covariance dimensions.")
    if cross_covariance.shape != (len(y_selected), 1):
        raise PredictionAdapterError("Unexpected cross-covariance dimensions.")

    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    covariance_scale = max(
        float(np.mean(np.abs(diagonal))),
        abs(response_test_variance),
        1e-12,
    )
    errors = []
    for attempt, relative_jitter in enumerate(settings.relative_jitters, start=1):
        extra_jitter = float(relative_jitter) * covariance_scale
        total_jitter = settings.public_absolute_jitter + extra_jitter
        try:
            output = _attempt_prediction(
                covariance=covariance,
                cross_covariance=cross_covariance,
                signal_test_variance=signal_test_variance,
                response_test_variance=response_test_variance,
                y_selected=y_selected,
                sample_mean=sample_mean,
                noise_variance=noise_variance,
                total_jitter=total_jitter,
                max_condition=settings.max_condition,
            )
        except (PredictionAdapterError, ValueError, np.linalg.LinAlgError) as error:
            errors.append(
                "relative_jitter={:.3g}: {}".format(relative_jitter, error)
            )
            continue

        mean_difference = abs(output["plugin_mean_external"] - public_mean)
        variance_difference = abs(
            output["plugin_response_variance_external"]
            - public_response_variance
        )
        reconstruction_checked = int(relative_jitter == 0.0)
        if reconstruction_checked:
            tolerance = _prediction_tolerance(public_mean, public_response_variance)
            if max(mean_difference, variance_difference) > tolerance:
                raise PredictionAdapterError(
                    "The external reconstruction does not match the public prediction: "
                    "mean difference {:.6g}, variance difference {:.6g}, tolerance {:.6g}."
                    .format(mean_difference, variance_difference, tolerance)
                )
        output.update({
            "public_mean": public_mean,
            "public_response_variance": max(public_response_variance, 0.0),
            "sample_mean": sample_mean,
            "selected_neighbors": int(np.sum(selected)),
            "public_absolute_jitter": settings.public_absolute_jitter,
            "extra_jitter": extra_jitter,
            "relative_jitter": float(relative_jitter),
            "total_jitter": total_jitter,
            "covariance_scale": covariance_scale,
            "stabilization_attempts": attempt,
            "used_extra_jitter": int(relative_jitter > 0.0),
            "public_reconstruction_checked": reconstruction_checked,
            "public_mean_abs_difference": mean_difference,
            "public_variance_abs_difference": variance_difference,
        })
        return output

    raise PredictionAdapterError(
        "External UK stabilization failed after {} attempts: {}".format(
            len(settings.relative_jitters), " | ".join(errors)
        )
    )
