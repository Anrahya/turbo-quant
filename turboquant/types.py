"""Data classes and abstract interfaces for TurboQuant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantizerConfig:
    """Configuration for a TurboQuant quantizer.

    Attributes:
        bit_width: Number of bits per coordinate (1, 2, 3, 4, ...).
        dimension: Dimensionality of input vectors.
        seed: Random seed for reproducibility of rotation / QJL matrices.
    """
    bit_width: int
    dimension: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.bit_width < 1:
            raise ValueError(f"bit_width must be >= 1, got {self.bit_width}")
        if self.dimension < 2:
            raise ValueError(f"dimension must be >= 2, got {self.dimension}")


# ---------------------------------------------------------------------------
# Quantized data containers
# ---------------------------------------------------------------------------

@dataclass
class QuantizedMSE:
    """Result of MSE-optimal TurboQuant quantization.

    Attributes:
        indices: (n, d) array of uint8 centroid indices (b bits each).
        norms: (n,) array of float32 original vector L2 norms.
        config: The QuantizerConfig used during quantization.
    """
    indices: np.ndarray          # (n, d) uint8
    norms: np.ndarray            # (n,) float32
    config: QuantizerConfig

    @property
    def n_vectors(self) -> int:
        return self.indices.shape[0]

    def size_bytes(self) -> int:
        """Actual compressed size in bytes (indices bit-packed + norms)."""
        n, d = self.indices.shape
        b = self.config.bit_width
        # Each coordinate takes b bits, packed into bytes
        bits_total = n * d * b
        idx_bytes = (bits_total + 7) // 8
        norm_bytes = n * 4  # float32 norms
        return idx_bytes + norm_bytes


@dataclass
class QuantizedProd:
    """Result of inner-product-optimal TurboQuant quantization.

    Attributes:
        indices: (n, d) array of uint8 centroid indices ((b-1) bits each).
        qjl_signs: (n, d) array of int8 sign bits (+1/-1).
        residual_norms: (n,) array of float32 residual L2 norms (gamma).
        norms: (n,) array of float32 original vector L2 norms.
        config: The QuantizerConfig used during quantization.
    """
    indices: np.ndarray          # (n, d) uint8
    qjl_signs: np.ndarray       # (n, d) int8 (+1 or -1)
    residual_norms: np.ndarray   # (n,) float32
    norms: np.ndarray            # (n,) float32
    config: QuantizerConfig

    @property
    def n_vectors(self) -> int:
        return self.indices.shape[0]

    def size_bytes(self) -> int:
        """Actual compressed size in bytes."""
        n, d = self.indices.shape
        b = self.config.bit_width
        # MSE part: (b-1) bits per coordinate
        mse_bits = n * d * (b - 1)
        mse_bytes = (mse_bits + 7) // 8
        # QJL part: 1 bit per coordinate
        qjl_bits = n * d
        qjl_bytes = (qjl_bits + 7) // 8
        # Scalars: norm + residual_norm per vector
        scalar_bytes = n * 4 * 2
        return mse_bytes + qjl_bytes + scalar_bytes


# ---------------------------------------------------------------------------
# Abstract base class for future extensibility
# ---------------------------------------------------------------------------

class BaseQuantizer(ABC):
    """Abstract interface for vector quantizers.

    Subclass this to implement GPU backends, alternative algorithms, etc.
    """

    @abstractmethod
    def quantize(self, vectors: np.ndarray) -> Any:
        """Quantize a batch of vectors.

        Args:
            vectors: (n, d) float32 array of input vectors.

        Returns:
            A quantized data container (QuantizedMSE or QuantizedProd).
        """
        ...

    @abstractmethod
    def dequantize(self, data: Any) -> np.ndarray:
        """Dequantize back to approximate float vectors.

        Args:
            data: A quantized data container.

        Returns:
            (n, d) float32 array of reconstructed vectors.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        data: Any,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-k nearest vectors by inner product.

        Args:
            query: (d,) or (nq, d) float32 query vector(s).
            data: A quantized data container.
            k: Number of top results to return.

        Returns:
            (indices, scores) — both arrays of shape (nq, k).
        """
        ...
