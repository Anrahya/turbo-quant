"""Core TurboQuant quantizers — MSE-optimal and inner-product-optimal.

Implements Algorithm 1 (TurboQuantMSE) and Algorithm 2 (TurboQuantProd)
from the paper.
"""

from __future__ import annotations

import numpy as np

CHUNK_SIZE = 100_000  # Process in chunks to avoid OOM on large datasets

from turboquant.types import (
    BaseQuantizer,
    QuantizerConfig,
    QuantizedMSE,
    QuantizedProd,
)
from turboquant.codebook import compute_codebook
from turboquant.rotation import generate_rotation, rotate, rotate_back
from turboquant.qjl import generate_qjl_matrix, qjl_quantize, qjl_dequantize


# ---------------------------------------------------------------------------
# TurboQuantMSE — Algorithm 1
# ---------------------------------------------------------------------------

class TurboQuantMSE(BaseQuantizer):
    """MSE-optimal TurboQuant quantizer.

    Pipeline (Algorithm 1):
        Quant:   y = Π · x → idx_j = nearest centroid for each y_j
        DeQuant: ỹ_j = codebook[idx_j] → x̃ = Πᵀ · ỹ
    """

    def __init__(self, config: QuantizerConfig) -> None:
        self.config = config
        self.d = config.dimension
        self.b = config.bit_width

        # Generate rotation matrix (one-time)
        self.rotation = generate_rotation(self.d, config.seed)

        # Compute codebook (one-time, cached to disk)
        self.centroids, self.boundaries = compute_codebook(self.d, self.b)

    def quantize(self, vectors: np.ndarray) -> QuantizedMSE:
        """Quantize vectors using MSE-optimal TurboQuant.

        Args:
            vectors: (n, d) float32 input vectors.

        Returns:
            QuantizedMSE with centroid indices and norms.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        n, d = vectors.shape
        assert d == self.d, f"Expected dimension {self.d}, got {d}"

        # Process in chunks to avoid OOM on large datasets
        all_indices = np.empty((n, d), dtype=np.uint8)
        norms = np.empty(n, dtype=np.float32)

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            chunk = vectors[start:end]

            # Store norms and normalize
            chunk_norms = np.linalg.norm(chunk, axis=1).astype(np.float32)
            safe_norms = np.maximum(chunk_norms, 1e-10)
            x_norm = chunk / safe_norms[:, np.newaxis]

            # Step 1: Rotate
            y = rotate(x_norm, self.rotation)

            # Step 2: Nearest centroid (searchsorted returns int64, but per-chunk it's small)
            all_indices[start:end] = self._find_nearest_centroids(y)
            norms[start:end] = chunk_norms

        return QuantizedMSE(
            indices=all_indices,
            norms=norms,
            config=self.config,
        )

    def dequantize(self, data: QuantizedMSE) -> np.ndarray:
        """Dequantize MSE-quantized data back to approximate vectors.

        Args:
            data: QuantizedMSE from quantize().

        Returns:
            (n, d) float32 reconstructed vectors.
        """
        # Step 1: Look up centroids → ỹ
        y_approx = self.centroids[data.indices]  # (n, d)

        # Step 2: Rotate back → x̃ = Πᵀ · ỹ
        x_approx = rotate_back(y_approx, self.rotation)

        # Step 3: Rescale by original norms
        x_approx = x_approx * data.norms[:, np.newaxis]

        return x_approx.astype(np.float32)

    def dequantize_normalized(self, data: QuantizedMSE) -> np.ndarray:
        """Dequantize without rescaling (unit sphere). Used internally."""
        y_approx = self.centroids[data.indices]
        return rotate_back(y_approx, self.rotation).astype(np.float32)

    def search(
        self,
        query: np.ndarray,
        data: QuantizedMSE,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-k nearest vectors by inner product (asymmetric search).

        Query stays at full precision, DB is dequantized in chunks.
        """
        query = np.atleast_2d(np.asarray(query, dtype=np.float32))
        n = data.n_vectors
        nq = query.shape[0]

        # For small datasets, do it all at once
        if n <= CHUNK_SIZE:
            return _topk_inner_product(query, self.dequantize(data), k)

        # Chunked search: dequantize a chunk, compute scores, keep top-k
        top_indices = np.zeros((nq, k), dtype=np.int64)
        top_scores = np.full((nq, k), -np.inf, dtype=np.float32)

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            chunk_data = QuantizedMSE(
                indices=data.indices[start:end],
                norms=data.norms[start:end],
                config=data.config,
            )
            chunk_recon = self.dequantize(chunk_data)
            chunk_scores = query @ chunk_recon.T  # (nq, chunk_size)

            # Merge with running top-k
            combined_scores = np.concatenate([top_scores, chunk_scores], axis=1)
            combined_indices = np.concatenate(
                [top_indices, np.arange(start, end)[np.newaxis, :].repeat(nq, axis=0)],
                axis=1,
            )
            # Pick top-k from combined
            best = np.argpartition(-combined_scores, k, axis=1)[:, :k]
            top_scores = np.take_along_axis(combined_scores, best, axis=1)
            top_indices = np.take_along_axis(combined_indices, best, axis=1)

            del chunk_recon, chunk_scores  # free memory

        # Final sort
        sort_order = np.argsort(-top_scores, axis=1)
        return (
            np.take_along_axis(top_indices, sort_order, axis=1),
            np.take_along_axis(top_scores, sort_order, axis=1),
        )

    def _find_nearest_centroids(self, y: np.ndarray) -> np.ndarray:
        """Find the index of the nearest centroid for each coordinate.

        Uses vectorized binary search on the boundaries for efficiency.
        """
        # boundaries: (n_levels + 1,) — sorted thresholds
        # For each value, find which bin it falls into
        # np.searchsorted gives the insertion point in boundaries
        # bin index = insertion_point - 1, clamped to [0, n_levels - 1]
        n_levels = len(self.centroids)
        bin_indices = np.searchsorted(self.boundaries, y, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, n_levels - 1)
        return bin_indices.astype(np.uint8)


# ---------------------------------------------------------------------------
# TurboQuantProd — Algorithm 2
# ---------------------------------------------------------------------------

class TurboQuantProd(BaseQuantizer):
    """Inner-product-optimal TurboQuant quantizer (two-stage).

    Pipeline (Algorithm 2):
        Quant:   (b-1)-bit MSE quant → residual → QJL sign bits
        DeQuant: x̃ = x̃_mse + γ · √(π/2)/d · Sᵀ · qjl
    """

    def __init__(self, config: QuantizerConfig) -> None:
        if config.bit_width < 2:
            raise ValueError(
                "TurboQuantProd requires bit_width >= 2 "
                "(uses b-1 bits for MSE + 1 bit for QJL)"
            )
        self.config = config
        self.d = config.dimension
        self.b = config.bit_width

        # Stage 1: MSE quantizer at (b-1) bits
        mse_config = QuantizerConfig(
            bit_width=self.b - 1,
            dimension=self.d,
            seed=config.seed,
        )
        self.mse_quantizer = TurboQuantMSE(mse_config)

        # Stage 2: QJL random projection matrix
        # Use a different seed offset so S ≠ Π
        self.S = generate_qjl_matrix(self.d, seed=config.seed + 1_000_000)

    def quantize(self, vectors: np.ndarray) -> QuantizedProd:
        """Quantize vectors using inner-product-optimal TurboQuant.

        Args:
            vectors: (n, d) float32 input vectors.

        Returns:
            QuantizedProd with MSE indices, QJL signs, and norms.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        # Store original norms and normalize
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        safe_norms = np.maximum(norms, 1e-10)
        x_normalized = vectors / safe_norms[:, np.newaxis]

        # Step 1: MSE quantize at (b-1) bits
        mse_data = self.mse_quantizer.quantize(x_normalized)

        # Step 2: Compute residual r = x - DeQuant_mse(idx)
        x_mse_approx = self.mse_quantizer.dequantize_normalized(mse_data)
        residuals = x_normalized - x_mse_approx

        # Step 3: QJL on residual
        qjl_signs, residual_norms = qjl_quantize(residuals, self.S)

        return QuantizedProd(
            indices=mse_data.indices,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            norms=norms,
            config=self.config,
        )

    def dequantize(self, data: QuantizedProd) -> np.ndarray:
        """Dequantize inner-product-quantized data.

        Args:
            data: QuantizedProd from quantize().

        Returns:
            (n, d) float32 reconstructed vectors.
        """
        # Reconstruct MSE part (on unit sphere, norms=1 in mse_data)
        mse_data = QuantizedMSE(
            indices=data.indices,
            norms=np.ones(data.n_vectors, dtype=np.float32),
            config=QuantizerConfig(
                bit_width=self.b - 1,
                dimension=self.d,
                seed=self.config.seed,
            ),
        )
        x_mse = self.mse_quantizer.dequantize(mse_data)  # (n, d)

        # Reconstruct QJL part
        x_qjl = qjl_dequantize(data.qjl_signs, data.residual_norms, self.S)

        # Combine and rescale
        x_approx = (x_mse + x_qjl) * data.norms[:, np.newaxis]

        return x_approx.astype(np.float32)

    def search(
        self,
        query: np.ndarray,
        data: QuantizedProd,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-k nearest vectors by inner product."""
        query = np.atleast_2d(np.asarray(query, dtype=np.float32))
        reconstructed = self.dequantize(data)
        return _topk_inner_product(query, reconstructed, k)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _topk_inner_product(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force top-k inner product search.

    Args:
        queries: (nq, d) float32 query vectors.
        database: (n, d) float32 database vectors.
        k: Number of top results.

    Returns:
        (indices, scores) each of shape (nq, k), sorted by descending score.
    """
    # Compute all inner products: (nq, n)
    scores = queries @ database.T

    # Get top-k indices per query
    if k >= scores.shape[1]:
        # Return all, sorted
        indices = np.argsort(-scores, axis=1)
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return indices, sorted_scores

    # Use argpartition for efficiency (O(n) instead of O(n log n))
    top_k_unsorted = np.argpartition(-scores, k, axis=1)[:, :k]
    top_k_scores = np.take_along_axis(scores, top_k_unsorted, axis=1)

    # Sort the top-k by score (descending)
    sort_order = np.argsort(-top_k_scores, axis=1)
    top_k_sorted = np.take_along_axis(top_k_unsorted, sort_order, axis=1)
    top_k_scores_sorted = np.take_along_axis(top_k_scores, sort_order, axis=1)

    return top_k_sorted, top_k_scores_sorted
