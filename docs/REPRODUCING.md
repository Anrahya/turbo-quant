# Reproducing Our Results

Step-by-step instructions to reproduce the benchmark numbers in our README.

## Prerequisites

```bash
# Python 3.10+
pip install numpy scipy pandas pyarrow
```

## 1. Quick Validation (2 minutes)

Synthetic data — no downloads needed:

```bash
python benchmark.py --n 10000 --d 1536 --bits 4
```

Expected output:
- MSE ≈ 0.02 (higher than real data due to uniform random distribution)
- Recall@1 ≈ 0.77 (lower because synthetic vectors lack natural clusters)

## 2. Full Reproduction (30 minutes)

### Download the dataset

We use [DBpedia entities embedded with OpenAI text-embedding-3-large](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) (1M vectors, d=1536).

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M',
    repo_type='dataset',
    local_dir='dbpedia_dataset',
    allow_patterns='*.parquet'
)
"
```

### Convert to NumPy

```python
import pandas as pd
import numpy as np
from pathlib import Path

files = sorted(Path('dbpedia_dataset/data').glob('*.parquet'))
vectors = np.vstack([
    np.array(pd.read_parquet(f, columns=['text-embedding-3-large-1536-embedding'])
        ['text-embedding-3-large-1536-embedding'].tolist(), dtype=np.float32)
    for f in files
])
np.save('dbpedia_full_1m.npy', vectors)
print(f'Saved {vectors.shape[0]:,} vectors, {vectors.nbytes/1e9:.1f} GB')
```

### Run the benchmark

```bash
# 4-bit only (fastest, matches README results)
python benchmark.py --dataset dbpedia_full_1m.npy --bits 4

# Full sweep
python benchmark.py --dataset dbpedia_full_1m.npy --bits 1 2 3 4 --k 1 10 100
```

### Expected results (4-bit, 1M vectors)

| Metric | Expected |
|---|---|
| MSE | 0.009 ± 0.001 |
| Recall@1 | 0.99 ± 0.01 |
| Recall@10 | 0.96 ± 0.02 |
| Compression ratio | 8.0× |
| Quantize time | 30-60s (CPU dependent) |

## 3. Using Your Own Data

Save your embeddings as a NumPy array and run:

```bash
# vectors.npy should be shape (n_vectors, dimensions), dtype float32
python benchmark.py --dataset vectors.npy --bits 4
```

The benchmark auto-normalizes vectors and splits the last 200 as queries.

## Hardware Used

Our results were obtained on:
- CPU: Intel i5-12400F (6 cores)
- RAM: 32 GB DDR4 3200
- OS: Windows 11
- Python 3.13, NumPy 2.x, SciPy 1.x
