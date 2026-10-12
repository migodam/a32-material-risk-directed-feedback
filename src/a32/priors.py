"""Proper material priors from the 64 independent TRAIN parents only.

``T`` maps the fixed real coefficient vector to unweighted stacked real and
imaginary material pixels.  ``train_mass`` contains the *actual-amplitude*
TRAIN material in mass coordinates, sqrt(cell_volume) [Re, Im].  The caller
owns TRAIN provenance: this module accepts arrays and never reads a split,
target, development score, file, or physical model.  A prior precision Lambda
is returned without any numerical damping.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.linalg import cho_solve

from .frozen_representations import PIXELS, haar24_transform


TRAIN_PARENTS = 64
_LEVELS = (("coarse_dct", 3, 0, 9), ("detail", 6, 9, 36),
           ("detail", 12, 36, 144), ("detail", 24, 144, 576))
_OAS_SOURCE = "https://arxiv.org/pdf/0907.4698"


def _real_array(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must use fixed real coordinates")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _symmetric(value: np.ndarray) -> np.ndarray:
    return (value + value.T) / 2


def _spd_inverse(matrix: np.ndarray, name: str) -> np.ndarray:
    """Invert an SPD matrix by Cholesky; never add a floor, jitter or pinv."""
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} is not numerically SPD; no jitter/pinv allowed") from error
    return _symmetric(cho_solve((factor, True), np.eye(len(matrix)),
                              check_finite=False))


@lru_cache(maxsize=1)
def _default_haar() -> np.ndarray:
    return haar24_transform()[0]


def _oas_covariance(centered: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Original finite-dimension OAS, Chen et al. equation (23).

    S = Xc.T Xc / n, F = tr(S)/d I, Sigma = (1-rho) S + rho F.
    The 2/d terms are retained; they are omitted by some library variants.
    Centering uses the same 64-parent TRAIN mean; n is explicitly 64.
    """
    n, dimension = centered.shape
    empirical = _symmetric(centered.T @ centered / n)
    trace = float(np.trace(empirical))
    target_variance = trace / dimension
    if not np.isfinite(target_variance) or target_variance <= 0:
        raise ValueError("TRAIN coefficient variance is zero; exact OAS cannot yield an SPD prior")
    target = target_variance * np.eye(dimension)
    trace_square = float(np.sum(empirical * empirical))
    numerator = (1 - 2 / dimension) * trace_square + trace * trace
    # This form avoids subtracting nearly equal traces for an isotropic S.
    departure_square = float(np.sum((empirical - target) ** 2))
    denominator = (n + 1 - 2 / dimension) * departure_square
    shrinkage = 1.0 if dimension == 1 or denominator == 0 else min(1.0, numerator / denominator)
    covariance = _symmetric((1 - shrinkage) * empirical + shrinkage * target)
    return covariance, {
        "estimator": "ORIGINAL_OAS_EQ23_FINITE_DIMENSION",
        "formula_source": _OAS_SOURCE,
        "empirical_covariance_denominator": n,
        "oas_n": n, "oas_dimension": dimension,
        "oas_2_over_dimension_retained": True,
        "shrinkage": float(shrinkage),
        "isotropic_target_variance": target_variance,
        "empirical_covariance_trace": trace,
        "empirical_covariance_frobenius_squared": trace_square,
        "oas_numerator": float(numerator), "oas_denominator": float(denominator),
    }


def _entry(mean: np.ndarray, covariance: np.ndarray, precision: np.ndarray,
           metadata: dict[str, Any]) -> dict[str, Any]:
    return {"mean": mean.copy(), "covariance": covariance, "precision": precision,
            "metadata": {"data_scope": "TRAIN_ONLY_64_INDEPENDENT_PARENTS",
                         "independent_train_parents": TRAIN_PARENTS,
                         "mean_scope": "TRAIN_COEFFICIENT_SAMPLE_MEAN",
                         "real_dimension": len(mean),
                         "proper": True, "numerical_damping_added": False,
                         "jitter_added": False, "pseudoinverse_used": False,
                         "covariance_trace": float(np.trace(covariance)),
                         "precision_trace": float(np.trace(precision)),
                         **metadata}}


def build_priors(train_coeff: Any, T: Any, train_mass: Any,
                 haar_analysis: Any = None,
                 cell_volume: float = 1 / 576) -> dict[str, dict[str, Any]]:
    """Build gaussian/ridge/multiscale priors in the same fixed real chart.

    Inputs have shapes (64,d), (1152,d), (64,1152), respectively.  The optional
    Haar analysis is the frozen orthogonal (576,576) matrix (or its
    ``(matrix, metadata)`` return tuple).  T already includes the physical
    amplitude.  Ridge matches *covariance* trace: Sigma_r = tr(Sigma_g)/d I;
    precision traces are also exposed because they generally do not match.

    Multiscale variance is the centered MLE variance pooled across coordinates
    within each real/imaginary channel and frozen Haar level.  Its precision is
    pulled back through sqrt(cell_volume) T, not projected as a covariance.
    A zero-variance level can only be omitted when T has exactly zero response
    there.  Other degenerate TRAIN priors fail explicitly without a variance
    floor.  No pixel clipping, target fitting, or tuning is performed.
    """
    coefficients = _real_array(train_coeff, "train_coeff")
    basis = _real_array(T, "T")
    mass = _real_array(train_mass, "train_mass")
    if coefficients.ndim != 2 or coefficients.shape[0] != TRAIN_PARENTS or coefficients.shape[1] < 1:
        raise ValueError("train_coeff must have shape (64,d) with d>=1")
    dimension = coefficients.shape[1]
    if basis.shape != (2 * PIXELS, dimension):
        raise ValueError("T must be the fixed real physical basis with shape (1152,d)")
    if mass.shape != (TRAIN_PARENTS, 2 * PIXELS):
        raise ValueError("train_mass must have shape (64,1152)")
    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("cell_volume must be finite and positive")
    if haar_analysis is None:
        haar = _default_haar()
    else:
        raw_haar = haar_analysis[0] if isinstance(haar_analysis, tuple) else haar_analysis
        haar = _real_array(raw_haar, "haar_analysis")
    if haar.shape != (PIXELS, PIXELS):
        raise ValueError("haar_analysis must be the frozen (576,576) analysis matrix")

    mean = coefficients.mean(axis=0)
    covariance, gaussian_metadata = _oas_covariance(coefficients - mean)
    precision = _spd_inverse(covariance, "OAS covariance")
    gaussian = _entry(mean, covariance, precision, gaussian_metadata)
    ridge_variance = float(np.trace(covariance) / dimension)
    ridge = _entry(mean, ridge_variance * np.eye(dimension),
                   np.eye(dimension) / ridge_variance,
                   {"estimator": "ISOTROPIC_COVARIANCE_TRACE_MATCHED",
                    "matched_prior": "gaussian", "matched_quantity": "covariance_trace",
                    "isotropic_variance": ridge_variance,
                    "precision_trace_matching_claimed": False})

    mass_haar = np.hstack((mass[:, :PIXELS] @ haar.T,
                           mass[:, PIXELS:] @ haar.T))
    centered_haar = mass_haar - mass_haar.mean(axis=0)
    basis_haar = np.vstack((haar @ (np.sqrt(volume) * basis[:PIXELS]),
                            haar @ (np.sqrt(volume) * basis[PIXELS:])))
    whitened_basis = np.zeros_like(basis_haar)
    level_records = []
    for channel, channel_name in enumerate(("real", "imaginary")):
        for kind, size, start, stop in _LEVELS:
            selection = slice(channel * PIXELS + start, channel * PIXELS + stop)
            variance = float(np.mean(centered_haar[:, selection] ** 2))
            response = basis_haar[selection]
            zero_response = not np.any(response != 0)
            if variance == 0 and not zero_response:
                raise ValueError(f"zero TRAIN Haar variance for {channel_name}/{size} with nonzero basis response")
            if not np.isfinite(variance):
                raise ValueError("TRAIN Haar variance is not finite")
            if variance > 0:
                whitened_basis[selection] = response / np.sqrt(variance)
            level_records.append({"channel": channel_name, "kind": kind, "size": size,
                                  "coordinate_count": stop - start,
                                  "centered_mle_variance": variance,
                                  "zero_response": zero_response,
                                  "omitted_zero_variance_zero_response": variance == 0 and zero_response})
    multiscale_precision = _symmetric(whitened_basis.T @ whitened_basis)
    multiscale_covariance = _spd_inverse(multiscale_precision, "multiscale precision pullback")
    multiscale = _entry(mean, multiscale_covariance, multiscale_precision,
                        {"estimator": "TRAIN_CHANNEL_HAAR_LEVEL_VARIANCE_PRECISION_PULLBACK",
                         "cell_volume": volume,
                         "train_mass_convention": "ACTUAL_AMPLITUDE_SQRT_VOLUME_STACKED_REAL_IMAG",
                         "basis_convention": "UNWEIGHTED_PHYSICAL_T_INCLUDES_AMPLITUDE",
                         "variance_denominator": TRAIN_PARENTS,
                         "pullback": "Lambda=T_mass_haar.T diag(1/level_variance) T_mass_haar",
                         "variance_floor_added": False, "levels": level_records})
    return {"gaussian": gaussian, "ridge": ridge, "multiscale": multiscale}
