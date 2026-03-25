# How TurboQuant Works — Technical Deep Dive

A step-by-step explanation of the algorithm, written for developers who know basic linear algebra.

## The Problem

You have millions of embedding vectors (from OpenAI, Cohere, etc.) — each one is 1536 floating-point numbers. Storing them as float32 costs **6 GB per million vectors** and searching requires computing millions of dot products in full precision.

**Goal**: Store each number as 4 bits instead of 32 bits. That's a 8× reduction — but how do you pick which 16 values (2⁴ = 16) to bucket into, and how much accuracy do you lose?

## Step 1: Random Rotation (The Key Trick)

Every embedding vector has a different distribution of coordinate values. Some coordinates are large, some are clustered, some are spread out. Designing a universal quantizer for arbitrary distributions seems impossible.

**TurboQuant's insight**: If you multiply every vector by a random orthogonal matrix `Π`, every coordinate of the rotated vector follows the **exact same distribution** — a bell curve `N(0, 1/d)`.

```python
# Generate Π once (random orthogonal matrix via QR decomposition)
G = np.random.randn(d, d)      # random Gaussian matrix
Π, _ = np.linalg.qr(G)         # Q is uniformly random orthogonal

# Rotate
y = x @ Π.T                    # y has same length as x, but now each
                                # coordinate follows N(0, 1/d)
```

**Why this works**: A random rotation is like choosing a random coordinate system. The vector hasn't changed — you just described it in new axes. But in these random axes, no single coordinate is "special", so they all have the same statistics.

**Physics analogy**: It's like measuring the displacement of a ball in a randomly chosen direction. No matter where the ball is, the measurement follows the same distribution.

## Step 2: Lloyd-Max Quantizer (Optimal Bins)

Now that every coordinate is `N(0, 1/d)`, we solve a classic signal processing problem: what are the best 16 bin centers for quantizing this Gaussian?

This is the **Lloyd-Max algorithm** (1960) — it finds centroids that minimize mean squared error:

```
For N(0, 1/d) with 16 bins (4-bit):

Centroids (scaled by √d):
  [-2.73, -2.07, -1.62, -1.26, -0.94, -0.66, -0.39, -0.13,
    0.13,  0.39,  0.66,  0.94,  1.26,  1.62,  2.07,  2.73]

These are symmetric, precomputed once, and work for any dataset.
```

To quantize: find the nearest centroid for each coordinate, store the 4-bit index.

## Step 3: Dequantize — Reverse the Rotation

```python
# Dequantize
ỹ = codebook[indices]           # look up centroid values
x̃ = ỹ @ Π                      # rotate back to original space
```

The reconstructed vector `x̃` is close to `x` — the MSE is ~0.009 per unit vector at 4-bit.

## Why Does This Preserve Search Quality?

For vector search, you care about **ranking**, not exact values. Consider two database vectors with true similarities to a query:

```
sim(query, doc_A) = 0.85   (best match)
sim(query, doc_B) = 0.60
```

After quantization, both get some noise:

```
sim(query, doc_A_quantized) ≈ 0.83
sim(query, doc_B_quantized) ≈ 0.59
```

The ranking is preserved — doc_A still wins. Quantization noise only causes errors when two vectors are so close in similarity that noise flips their order. At 4-bit with d=1536, this happens only ~1% of the time.

**Key mathematical fact**: In 1536 dimensions, each coordinate contributes 1/1536 of the total information. Small per-coordinate errors average out across all dimensions. The higher the dimension, the more robust the ranking.

## The Two Variants

### TurboQuantMSE (Algorithm 1) — For Search

Minimizes reconstruction error. Best for nearest neighbor search.

```
Quantize:   y = Π · x  →  idx = nearest_centroid(y)
Dequantize: ỹ = codebook[idx]  →  x̃ = Πᵀ · ỹ
```

### TurboQuantProd (Algorithm 2) — For LLM KV Cache

MSE quantization introduces a small **multiplicative bias** in dot products (inner products are systematically shrunk). For LLM attention, this bias matters.

Fix: spend 1 of the b bits on a **QJL correction** — compute `sign(S · residual)` for an unbiased estimate:

```
Quantize:   MSE at (b-1) bits → residual → sign(S · residual)
Dequantize: x̃_mse + γ · √(π/2)/d · Sᵀ · signs
```

This gives unbiased inner products at the cost of slightly higher MSE.

## Theoretical Guarantees

TurboQuant's MSE distortion is within **2.7×** of the Shannon lower bound — the absolute information-theoretic minimum. No algorithm, however complex, can do much better than 2.7× lower MSE. This is a near-optimal result.

| Bits | MSE (per unit vector) | Shannon lower bound | Ratio |
|---|---|---|---|
| 1 | 0.363 | 0.250 | 1.45× |
| 2 | 0.117 | 0.063 | 1.87× |
| 3 | 0.030 | 0.016 | 1.94× |
| 4 | 0.009 | 0.004 | 2.36× |

## References

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- Lloyd, S. (1982). Least squares quantization in PCM. IEEE Trans. Info. Theory.
- Johnson, W. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space.
