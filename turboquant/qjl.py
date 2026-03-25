"""QJL (Quantized Johnson-Lindenstrauss) — 1-bit inner product quantizer.

Implements the sign-based random projection that provides unbiased
inner product estimation. Used as the second stage in TurboQuantProd.
"""

from __future__ import annotations

import numpy as np


def generate_qjl_matrix(d: int, seed: int) -> np.ndarray:
    """Generate the random projection matrix S for QJL.

    Args:
        d: Dimensionality.
        seed: Random seed (should differ from the rotation seed).

    Returns:
        (d, d) float32 matrix with i.i.d. N(0,1) entries.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d)).astype(np.float32)


def qjl_quantize(
    residuals: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize residual vectors using QJL (1-bit sign projection).

    Algorithm 2, line 7: qjl = sign(S · r)

    Args:
        residuals: (n, d) float32 residual vectors.
        S: (d, d) float32 random projection matrix.

    Returns:
        (signs, norms):
            signs — (n, d) int8 array of +1/-1
            norms — (n,) float32 array of ||r||_2 values (gamma)
    """
    norms = np.linalg.norm(residuals, axis=1).astype(np.float32)  # (n,)

    # Normalize residuals (avoid division by zero for zero residuals)
    safe_norms = np.maximum(norms, 1e-10)
    normalized = residuals / safe_norms[:, np.newaxis]  # (n, d)

    # Project and take signs: sign(S @ r_normalized^T) for each vector
    # projected shape: (n, d) — each row is S @ r_i
    projected = normalized @ S.T  # (n, d)

    # Sign: +1 or -1 (treat 0 as +1)
    signs = np.where(projected >= 0, np.int8(1), np.int8(-1))

    return signs, norms


def qjl_dequantize(
    signs: np.ndarray,
    norms: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Dequantize QJL-compressed vectors.

    Algorithm 2, line 11: x_qjl = sqrt(pi/2) / d * gamma * S^T @ qjl

    Args:
        signs: (n, d) int8 array of +1/-1.
        norms: (n,) float32 array of ||r||_2 (gamma).
        S: (d, d) float32 random projection matrix.

    Returns:
        (n, d) float32 approximate residual vectors.
    """
    d = S.shape[0]
    scale = np.sqrt(np.pi / 2) / d  # scalar constant

    # S^T @ qjl for each vector: signs @ S (since (S^T @ qjl_i)^T = qjl_i^T @ S)
    reconstructed = signs.astype(np.float32) @ S  # (n, d)

    # Scale by gamma * sqrt(pi/2) / d
    reconstructed = reconstructed * (scale * norms[:, np.newaxis])

    return reconstructed.astype(np.float32)
