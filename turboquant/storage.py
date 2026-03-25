"""Bit-packed storage for quantized vector databases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from turboquant.types import QuantizerConfig, QuantizedMSE, QuantizedProd


def pack_indices(indices: np.ndarray, bit_width: int) -> np.ndarray:
    """Pack b-bit indices into a compact byte array.

    Args:
        indices: (n, d) uint8 array where each value is in [0, 2^b - 1].
        bit_width: Number of bits per index.

    Returns:
        1D uint8 array with packed bits.
    """
    n, d = indices.shape
    flat = indices.ravel().astype(np.uint64)
    total_bits = n * d * bit_width

    # Build a bit stream
    packed = np.zeros((total_bits + 7) // 8, dtype=np.uint8)

    for bit_pos in range(bit_width):
        # Extract bit `bit_pos` from each index
        bit_values = ((flat >> bit_pos) & 1).astype(np.uint8)
        # Place into packed array
        for i, bv in enumerate(bit_values):
            global_bit = i * bit_width + bit_pos
            byte_idx = global_bit // 8
            bit_idx = global_bit % 8
            packed[byte_idx] |= bv << bit_idx

    return packed


def unpack_indices(packed: np.ndarray, bit_width: int,
                   n: int, d: int) -> np.ndarray:
    """Unpack a byte array back to b-bit indices.

    Args:
        packed: 1D uint8 packed array.
        bit_width: Number of bits per index.
        n: Number of vectors.
        d: Dimensionality.

    Returns:
        (n, d) uint8 array of indices.
    """
    total_elements = n * d
    flat = np.zeros(total_elements, dtype=np.uint8)

    for i in range(total_elements):
        value = 0
        for bit_pos in range(bit_width):
            global_bit = i * bit_width + bit_pos
            byte_idx = global_bit // 8
            bit_idx = global_bit % 8
            bit_val = (packed[byte_idx] >> bit_idx) & 1
            value |= bit_val << bit_pos
        flat[i] = value

    return flat.reshape(n, d)


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """Pack +1/-1 sign array into bits (1 = positive, 0 = negative).

    Args:
        signs: (n, d) int8 array of +1/-1.

    Returns:
        1D uint8 packed array.
    """
    # Convert +1/-1 to 1/0
    bits = ((signs.ravel() + 1) // 2).astype(np.uint8)
    return np.packbits(bits)


def unpack_signs(packed: np.ndarray, n: int, d: int) -> np.ndarray:
    """Unpack bit-packed signs back to +1/-1 array.

    Args:
        packed: 1D uint8 packed array.
        n: Number of vectors.
        d: Dimensionality.

    Returns:
        (n, d) int8 array of +1/-1.
    """
    bits = np.unpackbits(packed)[:n * d]
    signs = (bits.astype(np.int8) * 2 - 1)
    return signs.reshape(n, d)


def save_mse(data: QuantizedMSE, path: str | Path) -> int:
    """Save MSE-quantized database to disk.

    Args:
        data: QuantizedMSE to save.
        path: File path (recommended: .tqdb extension).

    Returns:
        File size in bytes.
    """
    path = Path(path)
    packed_idx = pack_indices(data.indices, data.config.bit_width)

    np.savez_compressed(
        path,
        packed_indices=packed_idx,
        norms=data.norms,
        bit_width=np.array([data.config.bit_width]),
        dimension=np.array([data.config.dimension]),
        seed=np.array([data.config.seed]),
        n_vectors=np.array([data.n_vectors]),
    )
    return path.stat().st_size


def load_mse(path: str | Path) -> QuantizedMSE:
    """Load MSE-quantized database from disk.

    Args:
        path: Path to .tqdb file.

    Returns:
        QuantizedMSE.
    """
    path = Path(path)
    loaded = np.load(path)

    b = int(loaded["bit_width"][0])
    d = int(loaded["dimension"][0])
    n = int(loaded["n_vectors"][0])
    seed = int(loaded["seed"][0])

    config = QuantizerConfig(bit_width=b, dimension=d, seed=seed)
    indices = unpack_indices(loaded["packed_indices"], b, n, d)
    norms = loaded["norms"]

    return QuantizedMSE(indices=indices, norms=norms, config=config)


def save_prod(data: QuantizedProd, path: str | Path) -> int:
    """Save inner-product-quantized database to disk."""
    path = Path(path)
    b = data.config.bit_width
    packed_idx = pack_indices(data.indices, b - 1)
    packed_qjl = pack_signs(data.qjl_signs)

    np.savez_compressed(
        path,
        packed_indices=packed_idx,
        packed_qjl=packed_qjl,
        residual_norms=data.residual_norms,
        norms=data.norms,
        bit_width=np.array([b]),
        dimension=np.array([data.config.dimension]),
        seed=np.array([data.config.seed]),
        n_vectors=np.array([data.n_vectors]),
    )
    return path.stat().st_size


def load_prod(path: str | Path) -> QuantizedProd:
    """Load inner-product-quantized database from disk."""
    path = Path(path)
    loaded = np.load(path)

    b = int(loaded["bit_width"][0])
    d = int(loaded["dimension"][0])
    n = int(loaded["n_vectors"][0])
    seed = int(loaded["seed"][0])

    config = QuantizerConfig(bit_width=b, dimension=d, seed=seed)
    indices = unpack_indices(loaded["packed_indices"], b - 1, n, d)
    qjl_signs = unpack_signs(loaded["packed_qjl"], n, d)

    return QuantizedProd(
        indices=indices,
        qjl_signs=qjl_signs,
        residual_norms=loaded["residual_norms"],
        norms=loaded["norms"],
        config=config,
    )
