"""Lloyd-Max optimal codebook computation for Beta/Gaussian distributions.

Precomputes the optimal scalar quantizer centroids for the coordinate
distribution induced by random rotation on the unit sphere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import gamma as gamma_fn, gammaln


# ---------------------------------------------------------------------------
# Beta PDF for coordinates on the unit hypersphere (Lemma 1 in the paper)
# ---------------------------------------------------------------------------

def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a uniformly random point on S^{d-1}.

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    For d >= 64, this converges to N(0, 1/d) (paper Section 3.1).
    We use the Gaussian approximation for numerical stability.
    """
    if d >= 64:
        # Gaussian approximation: N(0, 1/d)
        variance = 1.0 / d
        return np.exp(-x**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    if d <= 2:
        return 1.0 / (np.pi * np.sqrt(np.maximum(1 - x**2, 1e-30)))
    # Use log-gamma for numerical stability at moderate d
    log_coeff = (gammaln(d / 2) - gammaln((d - 1) / 2)
                 - 0.5 * np.log(np.pi))
    exponent = (d - 3) / 2
    log_pdf = log_coeff + exponent * np.log(np.maximum(1 - x**2, 1e-30))
    return np.exp(log_pdf)



# ---------------------------------------------------------------------------
# Lloyd-Max algorithm for optimal scalar quantization
# ---------------------------------------------------------------------------

def _lloyd_max(d: int, n_levels: int, n_iter: int = 200,
               n_grid: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Find optimal scalar quantizer centroids via Lloyd-Max iteration.

    Args:
        d: Dimensionality (determines the Beta PDF shape).
        n_levels: Number of quantization levels (2^b).
        n_iter: Max iterations for convergence.
        n_grid: Number of grid points for numerical integration.

    Returns:
        (centroids, boundaries) — sorted arrays.
        centroids has shape (n_levels,), boundaries has shape (n_levels + 1,).
    """
    # Determine the effective support of the distribution
    # For d >= 64 (Gaussian N(0,1/d)), support is ~[-4/sqrt(d), 4/sqrt(d)]
    # For small d, the Beta distribution has support [-1, 1]
    if d >= 64:
        half_range = 4.0 / np.sqrt(d)
    else:
        half_range = 1.0 - 1e-10

    # Set up a fine grid over the support
    x = np.linspace(-half_range, half_range, n_grid)
    pdf = _beta_pdf(x, d)
    dx = x[1] - x[0]

    # Normalize the PDF
    pdf = pdf / (np.sum(pdf) * dx)

    # Initialize centroids evenly spaced within the support
    centroids = np.linspace(-half_range, half_range, n_levels + 2)[1:-1]

    for _ in range(n_iter):
        old_centroids = centroids.copy()

        # Boundaries = midpoints between adjacent centroids
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = x[0]
        boundaries[-1] = x[-1]
        for i in range(n_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        # Update centroids as conditional expectations within each bin
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (x >= lo) & (x < hi)
            if i == n_levels - 1:
                mask = (x >= lo) & (x <= hi)
            weighted_sum = np.sum(x[mask] * pdf[mask]) * dx
            total_weight = np.sum(pdf[mask]) * dx
            if total_weight > 1e-15:
                centroids[i] = weighted_sum / total_weight

        # Check convergence
        if np.max(np.abs(centroids - old_centroids)) < 1e-12:
            break

    # Final boundaries
    boundaries = np.empty(n_levels + 1)
    boundaries[0] = -half_range
    boundaries[-1] = half_range
    for i in range(n_levels - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

    return centroids.astype(np.float32), boundaries.astype(np.float32)


def _compute_mse(centroids: np.ndarray, boundaries: np.ndarray,
                 d: int, n_grid: int = 100_000) -> float:
    """Compute the MSE of the scalar quantizer (per coordinate)."""
    x = np.linspace(-1 + 1e-10, 1 - 1e-10, n_grid)
    pdf = _beta_pdf(x, d)
    dx = x[1] - x[0]
    pdf = pdf / (np.sum(pdf) * dx)

    mse = 0.0
    n_levels = len(centroids)
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (x >= lo) & (x <= hi)
        mse += np.sum((x[mask] - centroids[i])**2 * pdf[mask]) * dx
    return float(mse)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Cache directory for precomputed codebooks
_CACHE_DIR = Path(__file__).parent / "_codebook_cache"


def compute_codebook(d: int, b: int, use_cache: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Compute or load the optimal Lloyd-Max codebook for given (d, b).

    Args:
        d: Vector dimensionality.
        b: Bit-width (each coordinate quantized to b bits → 2^b levels).
        use_cache: Whether to cache results to disk.

    Returns:
        (centroids, boundaries):
            centroids — shape (2^b,) float32, sorted ascending.
            boundaries — shape (2^b + 1,) float32, sorted ascending.
    """
    n_levels = 2 ** b
    cache_file = _CACHE_DIR / f"d{d}_b{b}.json"

    # Try loading from cache
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return (
            np.array(data["centroids"], dtype=np.float32),
            np.array(data["boundaries"], dtype=np.float32),
        )

    # Compute
    centroids, boundaries = _lloyd_max(d, n_levels)

    # Cache
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "d": d, "b": b,
                "centroids": centroids.tolist(),
                "boundaries": boundaries.tolist(),
                "mse_per_coord": _compute_mse(centroids, boundaries, d),
            }, f, indent=2)

    return centroids, boundaries
