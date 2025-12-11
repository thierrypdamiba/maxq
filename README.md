# MaxQ

MaxQ is a small, opinionated layer on top of **Qdrant Cloud** that helps you **ingest data** and **evaluate retrieval** from either:
- a **terminal-first CLI**, or
- a **local Studio UI** (web app + API)

MaxQ uses **Qdrant Cloud Inference** for embeddings (no local model downloads).

## What you can do

- **Ingest (ETL):** embed + upsert + index data from URLs, local files, snapshots, or datasets
- **Hybrid search:** Dense (MiniLM / mxbai / BGE) + Sparse (BM25 / SPLADE) with fusion (e.g., RRF)
- **Auto-config templates:** task-specific presets for QA, semantic search, etc.
- **Evaluate & tune:** playground + evaluation suite + project-based workflows
- **Cloud-native:** embeddings via Qdrant Cloud Inference

---

## Quickstart

### 1) Create a Qdrant Cloud cluster
Sign up at https://cloud.qdrant.io (free tier available), create a cluster, and grab:
- `QDRANT_URL`
- `QDRANT_API_KEY` (read/write)

### 2) Install
```bash
pip install maxq
```

### 3) Configure

```bash
export QDRANT_URL="https://your-cluster.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key"
```

### 4) Start Studio (recommended)

```bash
maxq studio
```

Open:
* Studio UI: http://localhost:3000
* API docs: http://localhost:8000/docs

### 5) Or use the CLI

```bash
maxq --help
maxq doctor
maxq demo
maxq search "space adventure with robots"
```

---

## Common commands

```bash
maxq studio               # Start Studio UI + API
maxq doctor               # System health check
maxq demo                 # Load sample dataset + run a search
maxq search "query"       # Search indexed data
maxq import               # Import / ingest a dataset
maxq run                  # Run index + eval pipeline
maxq setup                # Re-run setup wizard
```

---

## Configuration

| Variable            | Required | Description                  | Default                                  |
| ------------------- | -------- | ---------------------------- | ---------------------------------------- |
| `QDRANT_URL`        | Yes      | Qdrant Cloud cluster URL     | -                                        |
| `QDRANT_API_KEY`    | Yes      | Qdrant Cloud API key         | -                                        |
| `MAXQ_DENSE_MODEL`  | No       | Dense embedding model        | `sentence-transformers/all-MiniLM-L6-v2` |
| `MAXQ_SPARSE_MODEL` | No       | Sparse embedding model       | `Qdrant/bm25`                            |
| `OPENAI_API_KEY`    | No       | Optional (HyDE/RAG features) | -                                        |

---

## Supported Models (Qdrant Cloud Inference)

**Dense**
* `sentence-transformers/all-MiniLM-L6-v2` (384 dims, default)
* `mixedbread-ai/mxbai-embed-large-v1` (1024 dims)
* `BAAI/bge-base-en-v1.5` (768 dims)

**Sparse**
* `Qdrant/bm25` (default)
* `prithivida/Splade_PP_en_v1`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Studio UI (Next.js)                     │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend (:8000)                   │
│   Projects · Indexing · Search · Evaluation · Export         │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼──────────────────────────────┐
│                        Qdrant Cloud                          │
│   ┌─────────────────┐    ┌─────────────────┐                │
│   │  Cloud Inference │    │  Vector Storage │                │
│   │  (embeddings)    │    │  (search)       │                │
│   └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## License

MIT
