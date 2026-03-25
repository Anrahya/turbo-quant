# Real-World Use Cases

Practical applications of TurboQuant for production vector systems.

---

## 1. RAG (Retrieval-Augmented Generation)

**Problem**: Your RAG app searches through 5M document chunks. At d=1536 float32, that's 30 GB — needs a costly high-memory server.

**With TurboQuant**: 3.8 GB at 4-bit. Runs on a $50/month VPS.

```python
from turboquant import TurboQuantMSE, QuantizerConfig

# Quantize your document embeddings once
config = QuantizerConfig(bit_width=4, dimension=1536, seed=42)
quantizer = TurboQuantMSE(config)
compressed_db = quantizer.quantize(document_embeddings)

# Save to disk (3.8 GB instead of 30 GB)
from turboquant.storage import save_mse
save_mse(compressed_db, "rag_database.tqdb")

# At query time
query_embedding = embed("What is the refund policy?")
indices, scores = quantizer.search(query_embedding, compressed_db, k=5)
context_chunks = [documents[i] for i in indices[0]]
```

**Impact**: 99% of queries return the same top-5 chunks as uncompressed. Your LLM sees virtually identical context.

---

## 2. Semantic Search at Scale

**Problem**: An e-commerce site with 10M product embeddings. Full-precision search requires a 64 GB instance.

**With TurboQuant**: 7.7 GB at 4-bit. Fits in a standard 8 GB instance.

```python
# Quantize product catalog
compressed_catalog = quantizer.quantize(product_embeddings)

# Search
query = embed("wireless noise cancelling headphones under $200")
top_indices, scores = quantizer.search(query, compressed_catalog, k=20)
```

---

## 3. Multi-Tenant SaaS

**Problem**: Each customer has their own vector collection. 100 customers × 100K vectors each = 10M total. Need to keep all in RAM for low latency.

**With TurboQuant**: Each customer's 100K vectors takes just 77 MB instead of 614 MB. All 100 customers fit in 7.7 GB total.

**Bonus**: No per-customer training. Same quantizer works for all customers — add new customer instantly.

---

## 4. Edge Deployment

**Problem**: Run vector search on a Raspberry Pi, mobile device, or IoT gateway with 2-4 GB RAM.

**With TurboQuant at 2-bit**: 1M vectors × 1536d = 390 MB. Fits on a Raspberry Pi with room to spare. ~93% recall.

---

## 5. Replace Dimension Reduction

**Problem**: Teams using OpenAI's `dimensions` parameter to truncate from 1536 → 768 to save cost.

**Instead**: Keep full 1536 dimensions, apply TurboQuant 4-bit.

| Approach | Storage per 1M | MTEB Quality |
|---|---|---|
| float32, 768-d | 3.1 GB | ~60% of full quality |
| TurboQuant 4-bit, 1536-d | 0.77 GB | ~99% of full quality |

You get **4× less storage** and **much better quality**. No tradeoff.

---

## Integration Patterns

### Offline batch processing

```python
# 1. Quantize entire database (one-time)
compressed = quantizer.quantize(all_vectors)
save_mse(compressed, "database.tqdb")

# 2. At query time, load and search
compressed = load_mse("database.tqdb")
results = quantizer.search(query, compressed, k=10)
```

### Online / streaming

```python
# New vector arrives — quantize individually
new_compressed = quantizer.quantize(new_vector)

# Append to existing database
combined_indices = np.vstack([db.indices, new_compressed.indices])
combined_norms = np.concatenate([db.norms, new_compressed.norms])
# No retraining needed — same codebook works for all vectors
```

### Hybrid with HNSW / IVF

TurboQuant handles compression. For sub-linear search, combine with an index:

```
1. Build HNSW/IVF index on full-precision vectors (or compressed)
2. Store compressed vectors for the re-ranking step
3. At query time: HNSW finds top-1000 candidates → re-rank with compressed dot products → return top-10
```

This gives you both fast search (HNSW) and low memory (TurboQuant).
