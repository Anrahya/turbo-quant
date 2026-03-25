<p align="center">
  <h1 align="center">⚡ TurboQuant</h1>
  <p align="center">
    <strong>8× vector database compression with 99% search accuracy. Zero training required.</strong>
  </p>
  <p align="center">
    Python implementation of <a href="https://arxiv.org/abs/2504.19874">TurboQuant</a> (ICLR 2026) — near-optimal vector quantization for vector database compression.
  </p>
</p>

---

## What is this?

TurboQuant compresses high-dimensional embedding vectors (like OpenAI's `text-embedding-3-large`) from 32-bit to 4-bit per coordinate, **cutting storage by 8×** while maintaining **99% search accuracy**.

Unlike traditional methods (Product Quantization, etc.), TurboQuant:
- **Requires zero training** — no k-means, no codebook fitting
- **Works on any data** — same quantizer for any dataset, any embedding model
- **Near-optimal** — within 2.7× of the information-theoretic lower bound
- **Online** — quantize each vector independently, no batch processing needed

## Benchmark Results (1M Real OpenAI Embeddings)

Tested on 1M vectors from [DBpedia](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) (OpenAI `text-embedding-3-large`, d=1536):

| Metric | Raw float32 | TurboQuant 4-bit |
|---|---|---|
| **Storage** | 6,143 MB | **772 MB** (8×) |
| **Recall@1** | 100% | **99.0%** |
| **Recall@10** | 100% | **95.5%** |
| **MSE** | 0 | 0.009 |
| **Quantize time** | — | 47s (one-time) |

### The Dimension Tradeoff is Dead

Many teams truncate embeddings from 1536-d → 768-d to save storage. TurboQuant makes this obsolete:

| Config | Storage per 1M | Quality |
|---|---|---|
| ❌ float32 at 768-d | 3.1 GB | Mediocre |
| ✅ **TurboQuant 4-bit at 1536-d** | **0.77 GB** | **Excellent** |

**4× less storage AND better quality.** Keep your full dimensions.

## Quick Start

```bash
pip install numpy scipy
git clone https://github.com/Anrahya/turbo-quant.git
cd turbo-quant
```

### Compress and search

```python
import numpy as np
from turboquant import TurboQuantMSE, QuantizerConfig

# Your embedding vectors (n, 1536)
vectors = np.load("my_embeddings.npy")

# Quantize — zero training, just works
config = QuantizerConfig(bit_width=4, dimension=1536, seed=42)
quantizer = TurboQuantMSE(config)
compressed = quantizer.quantize(vectors)

print(f"Compressed: {compressed.size_bytes() / 1e6:.0f} MB")
# Compressed: 772 MB (was 6,143 MB)

# Search
query = np.random.randn(1, 1536).astype(np.float32)
indices, scores = quantizer.search(query, compressed, k=10)
```

### Run the benchmark

```bash
# Quick test with synthetic data
python benchmark.py --n 50000 --d 1536 --bits 4

# With your own vectors
python benchmark.py --dataset path/to/vectors.npy --bits 4

# Full sweep
python benchmark.py --dataset vectors.npy --bits 1 2 3 4 --k 1 10 100
```

## How It Works

TurboQuant is a 3-step process:

```
1. ROTATE      — Multiply by a random orthogonal matrix
                 → Every coordinate now follows the same bell curve

2. QUANTIZE    — Snap each coordinate to the nearest of 16 precomputed
                 optimal bin centers (Lloyd-Max quantizer)
                 → Store as 4-bit index

3. DEQUANTIZE  — Look up bin centers, reverse the rotation
                 → Approximate original vector (MSE ≈ 0.009)
```

**Why does random rotation help?** It converts any input distribution into a known one (Gaussian), letting us use the same optimal quantizer for all data — no training needed.

For details, see the [technical deep-dive](docs/HOW_IT_WORKS.md) and the [paper](https://arxiv.org/abs/2504.19874).

## Project Structure

```
turbo-quant/
├── turboquant/
│   ├── types.py       — Data classes and interfaces (BaseQuantizer ABC)
│   ├── codebook.py    — Lloyd-Max optimal scalar quantizer
│   ├── rotation.py    — Random orthogonal rotation (QR of Gaussian)
│   ├── qjl.py         — QJL 1-bit sign projection (for inner-product variant)
│   ├── quantizer.py   — TurboQuantMSE + TurboQuantProd (Algorithms 1 & 2)
│   ├── storage.py     — Bit-packed save/load (.tqdb format)
│   └── search.py      — Brute-force search + recall@k metrics
├── docs/
│   ├── HOW_IT_WORKS.md   — Technical deep-dive (the math explained simply)
│   ├── USE_CASES.md      — Real-world applications with code examples
│   └── REPRODUCING.md    — Steps to reproduce our benchmark results
├── benchmark.py       — End-to-end benchmark script
├── FINDINGS.md        — Infrastructure cost analysis & results
└── pyproject.toml
```

## Two Quantizer Variants

| Variant | Use case | Bit budget |
|---|---|---|
| `TurboQuantMSE` | **Vector search** (nearest neighbor) | All b bits for MSE |
| `TurboQuantProd` | KV cache compression (unbiased attention) | (b-1) bits MSE + 1 bit QJL |

For vector databases, use **`TurboQuantMSE`** — it achieves lower reconstruction error, which means better search ranking.

## Infrastructure Impact

| Scale | Raw float32 | TurboQuant 4-bit | Cloud savings |
|---|---|---|---|
| 1M vectors | 6.1 GB | 0.77 GB | 70% |
| 10M vectors | 61 GB | 7.7 GB | **87%** |
| 50M vectors | 307 GB | 38 GB | **88%** |
| 100M vectors | 614 GB | 77 GB | **90%** |

At 10M+ vectors, databases that required distributed clusters now fit on a single server.

## Citation

```bibtex
@inproceedings{shen2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author={Shen, Hao and Silwal, Sandeep and Zakynthinou, Lydia and Silvestri, Francesco and Fanti, Giulia and Indyk, Piotr},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This is an independent implementation of the TurboQuant algorithm. The original paper is by researchers at Google Research and Carnegie Mellon University.
