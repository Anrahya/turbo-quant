"""Random rotation matrix generation for TurboQuant.

Provides a reproducible random orthogonal matrix via QR decomposition
of a Gaussian random matrix. This is the 'scramble' step that makes
every coordinate follow the same Beta distribution.
"""

from __future__ import annotations

import numpy as np


def generate_rotation(d: int, seed: int) -> np.ndarray:
    """Generate a uniformly random d×d orthogonal matrix.

    Uses QR decomposition of a random Gaussian matrix with sign
    correction to ensure uniform Haar measure.

    Args:
        d: Dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        (d, d) float32 orthogonal matrix Π such that Π @ Π.T = I.
    """
    rng = np.random.default_rng(seed)
    # Random Gaussian matrix
    G = rng.standard_normal((d, d)).astype(np.float32)
    # QR decomposition
    Q, R = np.linalg.qr(G)
    # Sign correction: ensure uniqueness by making diagonal of R positive
    # This guarantees Q is drawn from the Haar measure on O(d)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


def rotate(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply rotation: y = vectors @ rotation.T (each row is rotated).

    Args:
        vectors: (n, d) float32 input vectors.
        rotation: (d, d) float32 orthogonal matrix.

    Returns:
        (n, d) float32 rotated vectors.
    """
    return vectors @ rotation.T


def rotate_back(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply inverse rotation: x = vectors @ rotation.

    Since rotation is orthogonal, rotation^{-1} = rotation^T,
    so x = vectors @ rotation (which is vectors @ (rotation.T).T).

    Args:
        vectors: (n, d) float32 rotated vectors.
        rotation: (d, d) float32 orthogonal matrix.

    Returns:
        (n, d) float32 vectors in original space.
    """
    return vectors @ rotation
