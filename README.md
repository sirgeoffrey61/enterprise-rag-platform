# Enterprise RAG Platform

Production-oriented retrieval-augmented generation (RAG) stack built on English Wikipedia: ingestion, hybrid search, cross-encoder reranking, citation-grounded LLM answers, Redis caching, Prometheus metrics, and continuous evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENTERPRISE RAG PLATFORM                              │
└─────────────────────────────────────────────────────────────────────────────┘

  [1] INGESTION                    [2] CHUNKING
  download.py ──► wikipedia_raw    chunker.py ──► chunks.jsonl
       │                                │
       └──────────────┬─────────────────┘
                      ▼
  [3] EMBEDDINGS                   [4] VECTOR DB (Qdrant)
  embed.py (BGE-small) ──►         index_qdrant.py ──► enterprise_rag
  embeddings.npy                   collection (384-dim, cosine, on_disk)

                      ▼
  [5] HYBRID RETRIEVAL             [6] RERANKING
  hybrid_retriever.py              reranker.py
  • BM25 (sparse)                  • cross-encoder/ms-marco-MiniLM-L-6-v2
  • Qdrant dense (BGE query)       • RRF fusion → top-k chunks
  • RRF fusion (k=60)

                      ▼
  [7] GROUNDED GENERATION          [8] REDIS CACHE
  generator.py (Groq)              cache.py
  • llama-3.1-8b-instant           • MD5 query keys, 1h TTL
  • Inline citations [1][2]…         • ~83,000× speedup on repeat queries

                      ▼
  [9] OBSERVABILITY                [10] EVALUATION
  metrics.py (Prometheus)          benchmark.py
  tracer.py (QueryTracer)          • 20-query benchmark suite
  • Latency histograms             • Recall / precision / grounding
  • Cache hit counters             • results/benchmark_results.json

  Optional API layer: FastAPI-ready `api/` module (generator + cache)
```

## Benchmark Results

| Metric | Result |
|--------|--------|
| **Recall (answerable)** | **10/10** |
| **Precision (unanswerable)** | **10/10** |
| **Avg grounding ratio** (answerable) | **0.34** |
| **Avg latency** (no cache) | **~26 s** |
| **Insufficient evidence rate** | **50%** (by design: 10/20 queries unanswerable) |
| **Cache speedup** | **~83,000×** (29.5 s → 0.4 ms on repeat query) |

Run the benchmark: `python evaluation/benchmark.py`

## Stack

| Layer | Technology |
|-------|------------|
| Vector database | **Qdrant** (cosine, on-disk vectors) |
| Embeddings | **BAAI/bge-small-en-v1.5** (384-dim) |
| Sparse retrieval | **BM25** (rank-bm25) |
| Reranking | **cross-encoder/ms-marco-MiniLM-L-6-v2** |
| Cache | **Redis** |
| Metrics | **Prometheus** (prometheus-client) |
| LLM | **Groq** (`llama-3.1-8b-instant`) |
| API | **FastAPI**-ready Python modules in `api/` |

## Project Structure

```
enterprise-rag-platform/
├── ingestion/          # Wikipedia download, chunking
├── embeddings/         # BGE embedding pipeline
├── retrieval/          # Hybrid search + Qdrant indexing
├── reranking/          # Cross-encoder reranker
├── api/                # Grounded generation + Redis cache
├── observability/      # Prometheus metrics + tracing
├── evaluation/         # Benchmark runner + query set
├── infra/              # Qdrant/Redis helper scripts
├── data/               # Raw/processed data (gitignored)
└── docs/
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (CUDA; PyTorch cu128 nightly for RTX 50-series / sm_120)
- Redis on `localhost:6379`
- Qdrant on `localhost:6333`
- [Groq API key](https://console.groq.com/)

### Install

```powershell
cd enterprise-rag-platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment

Copy `.env` and set:

```env
GROQ_API_KEY=your_key_here
```

### Services

```powershell
# Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:alpine

# Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
  -v "./data/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Or use `infra/start_redis.ps1` and `infra/start_qdrant.ps1` for local binaries on Windows.

### Pipeline (in order)

```powershell
.\venv\Scripts\python.exe ingestion\download.py
.\venv\Scripts\python.exe ingestion\chunker.py
.\venv\Scripts\python.exe embeddings\embed.py
.\venv\Scripts\python.exe retrieval\index_qdrant.py
.\venv\Scripts\python.exe retrieval\hybrid_retriever.py
.\venv\Scripts\python.exe reranking\reranker.py
.\venv\Scripts\python.exe api\generator.py
.\venv\Scripts\python.exe evaluation\benchmark.py
```

## Designed to Scale to 10M+ Documents

This architecture is built for horizontal and vertical scaling:

- **Qdrant `on_disk=True`** — vectors served from disk to keep RAM bounded as collections grow into millions of points.
- **Batch ingestion** — embedding (`batch_size=128`) and Qdrant upserts (`batch_size=256`) amortize I/O and GPU cost.
- **Hybrid retrieval** — BM25 + dense search improves recall on large corpora where pure vector search can miss lexical matches.
- **Two-stage ranking** — cheap hybrid recall (`candidate_k=25`) then cross-encoder rerank on a small set controls latency at scale.
- **Redis query cache** — identical queries skip the full pipeline; critical for production API traffic patterns.
- **Prometheus metrics** — per-stage latency histograms and cache counters for SLO monitoring and autoscaling signals.
- **Grounded generation** — citation validation and insufficient-evidence responses reduce hallucination risk on enterprise knowledge bases.
- **Continuous evaluation** — `evaluation/benchmark.py` supports regression testing as the corpus and models evolve.

For 10M+ documents, expect to shard Qdrant collections, run ingestion/embed jobs offline (or distributed), and front the stack with a FastAPI service using the existing `api/` modules.

## License

MIT (add your license file as needed).
