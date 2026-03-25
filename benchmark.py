"""TurboQuant Benchmark — Compare compressed vs. raw vector search.

Usage:
    python benchmark.py
    python benchmark.py --n 50000 --d 768 --bits 2 3 4
    python benchmark.py --dataset path/to/vectors.npy --bits 2 3 4
"""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np

from turboquant.types import QuantizerConfig
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.search import brute_force_search, recall_at_k


def generate_synthetic(n: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic random vectors for benchmarking.

    Returns:
        (database, queries) — database is (n, d), queries is (n_queries, d).
    """
    rng = np.random.default_rng(seed)
    database = rng.standard_normal((n, d)).astype(np.float32)
    # Normalize to unit vectors (cosine similarity = inner product)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    n_queries = min(200, n // 10)
    queries = rng.standard_normal((n_queries, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    return database, queries


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors from a .npy file. Last 200 vectors become queries."""
    data = np.load(path).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    data /= norms

    n_queries = min(200, len(data) // 10)
    queries = data[-n_queries:]
    database = data[:-n_queries]
    return database, queries


def run_benchmark(
    database: np.ndarray,
    queries: np.ndarray,
    bit_widths: list[int],
    k_values: list[int] = [1, 10, 100],
    seed: int = 42,
) -> None:
    """Run the full benchmark and print results."""
    n, d = database.shape
    nq = queries.shape[0]
    raw_bytes = n * d * 4  # float32

    print(f"\n{'='*80}")
    print(f"  TurboQuant Benchmark")
    print(f"  Database: {n:,} vectors × {d} dimensions")
    print(f"  Queries:  {nq}")
    print(f"  Raw size: {raw_bytes / 1e6:.1f} MB ({raw_bytes / 1e9:.2f} GB)")
    print(f"{'='*80}\n")

    # Ground truth
    print("Computing ground truth (brute-force on raw vectors)...")
    max_k = max(k_values)
    t0 = time.perf_counter()
    gt_indices, gt_scores = brute_force_search(queries, database, max_k)
    gt_time = time.perf_counter() - t0
    print(f"  Done in {gt_time:.2f}s\n")

    # Header
    k_headers = "".join(f"  R@{k:<4}" for k in k_values)
    print(f"{'Method':<22} {'Bits':>4} {'Comp':>6} {'Size':>10} "
          f"{'MSE':>10} {'Bias':>8} "
          f"{k_headers} "
          f"{'Q-time':>8} {'S-time':>8}")
    print("-" * (100 + 8 * len(k_values)))

    # Raw baseline
    raw_recalls = []
    for k in k_values:
        raw_recalls.append(f"  {1.0:<5.3f}")
    raw_recall_str = "".join(raw_recalls)
    print(f"{'Raw (float32)':<22} {'32':>4} {'1.0×':>6} "
          f"{raw_bytes/1e6:>8.1f}MB "
          f"{'0':>10} {'0':>8} "
          f"{raw_recall_str} "
          f"{'—':>8} {gt_time:>7.2f}s")

    # Test each bit-width with both MSE and Prod quantizers
    for b in bit_widths:
        for method_name, QuantizerClass in [("TurboQuant-MSE", TurboQuantMSE),
                                             ("TurboQuant-Prod", TurboQuantProd)]:
            if method_name == "TurboQuant-Prod" and b < 2:
                continue  # Prod needs b >= 2

            config = QuantizerConfig(bit_width=b, dimension=d, seed=seed)

            # Quantize
            t0 = time.perf_counter()
            quantizer = QuantizerClass(config)
            data = quantizer.quantize(database)
            quant_time = time.perf_counter() - t0

            # Compressed size
            comp_bytes = data.size_bytes()
            comp_ratio = raw_bytes / comp_bytes

            # Dequantize in chunks and compute MSE + bias
            chunk_sz = 50_000
            mse_sum = 0.0
            ips_true_sum = 0.0
            ips_approx_sum = 0.0
            n_bias_pairs = min(500, n)

            from turboquant.types import QuantizedMSE as _QMSE
            for start in range(0, n, chunk_sz):
                end = min(start + chunk_sz, n)
                # Build a chunk view of the quantized data
                if hasattr(data, 'qjl_signs'):
                    # Prod type — skip chunked MSE for now
                    chunk_recon = None
                else:
                    chunk_data = _QMSE(
                        indices=data.indices[start:end],
                        norms=data.norms[start:end],
                        config=data.config,
                    )
                    chunk_recon = quantizer.dequantize(chunk_data)

                if chunk_recon is not None:
                    chunk_db = database[start:end]
                    mse_sum += float(np.sum(np.sum((chunk_db - chunk_recon)**2, axis=1)))

                    # Bias: only use first n_bias_pairs vectors
                    if start < n_bias_pairs:
                        bias_end = min(end, n_bias_pairs)
                        bias_slice = slice(start, bias_end)
                        r_start = start - start  # relative start in chunk
                        r_end = bias_end - start
                        ips_true_sum += float(np.sum(queries @ chunk_db[r_start:r_end].T))
                        ips_approx_sum += float(np.sum(queries @ chunk_recon[r_start:r_end].T))

                    del chunk_recon

            mse = mse_sum / n
            bias = ips_approx_sum / ips_true_sum if abs(ips_true_sum) > 1e-6 else 1.0


            # Search
            t0 = time.perf_counter()
            pred_indices, pred_scores = quantizer.search(queries, data, max_k)
            search_time = time.perf_counter() - t0

            # Recall
            recall_strs = []
            for k in k_values:
                r = recall_at_k(gt_indices, pred_indices, k)
                recall_strs.append(f"  {r:<5.3f}")
            recall_str = "".join(recall_strs)

            label = f"{method_name} ({b}b)"
            print(f"{label:<22} {b:>4} {comp_ratio:>5.1f}× "
                  f"{comp_bytes/1e6:>8.1f}MB "
                  f"{mse:>10.5f} {bias:>8.4f} "
                  f"{recall_str} "
                  f"{quant_time:>7.2f}s {search_time:>7.2f}s")

    print()


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .npy file with vectors (optional)")
    parser.add_argument("--n", type=int, default=50_000,
                        help="Number of synthetic vectors (default: 50000)")
    parser.add_argument("--d", type=int, default=256,
                        help="Dimensionality for synthetic (default: 256)")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Bit-widths to test (default: 1 2 3 4)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100],
                        help="K values for recall@k (default: 1 10 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        database, queries = load_dataset(args.dataset)
    else:
        print(f"Generating synthetic dataset: {args.n:,} × {args.d}d...")
        database, queries = generate_synthetic(args.n, args.d, seed=args.seed)

    run_benchmark(database, queries, args.bits, args.k, args.seed)


if __name__ == "__main__":
    main()
