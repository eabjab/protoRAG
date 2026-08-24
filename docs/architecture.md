# Architecture

protoRAG is a small layered system. The `ProtoRAG` facade owns three
retrieval subsystems behind strict `typing.Protocol` interfaces, so every
backend is swappable and every layer testable in isolation:

```
                         ┌─────────────────────────────────────────────┐
                         │                 ProtoRAG                    │
                         │  (core/engine.py — ingestion + retrieval)   │
                         └──┬──────────────┬───────────────┬───────────┘
        ingestion           │              │               │
   ┌──────────────┐   ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼──────────┐
   │  chunkers    │   │ embeddings│  │  lexical  │  │    storage      │
   │ BaseChunker  │   │ BaseEmbedder       │  BM25Engine│  BaseVectorStore │
   │  Recursive…  │   │  fastembed │  (Okapi,     │   numpy / usearch /
   │  Simple…     │   │  torch     │   inverted   │    chromadb
   │  (custom)    │   │  sentence- │   index)     │                       │
   └──────────────┘   │  transform.│  └─────┬─────┘  └───────────────────┘
                      └────────────┘        │
                                 ┌──────────▼──────────┐
                                 │       hybrid        │
                                 │  normalizers + fuse │
                                 │  (RRF / linear)     │
                                 └─────────────────────┘
        cross-cutting:  serialization/ (manifest, save/load) · tools/ (ProtoRAGTool)
```

## Ingestion pipeline

1. **Chunk** — `add_texts` wraps each text in a `Document` (uuid id), then
   `add_documents` splits each document with the active `BaseChunker`
   (default `RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)`).
   Chunks are addressed as `f"{doc_id}_chunk_{i}"` and carry the document
   metadata plus `chunk_index` / `total_chunks`.
   Re-adding a document id replaces its previous chunks (upsert).
2. **Embed** — one batched `embed_documents` call produces float32 vectors;
   embedders L2-normalize internally, which is what makes cosine search over
   the numpy backend exact.
3. **Index** — vectors go to the `BaseVectorStore`, raw text to the
   in-process `BM25Engine` inverted index. Both stores are keyed by chunk id.

## Retrieval pipeline

`search(query, top_k, mode, alpha, fusion_strategy, filter_metadata)`:

- **BM25 mode** — lexical ranking only.
- **VECTOR mode** — `embed_query` (query-aware; see below) + vector-store
  ranking.
- **HYBRID mode** (default) — both rankings fused:
  - `FusionStrategy.RRF` (default): Reciprocal Rank Fusion,
    `score(c) = Σ_r w_r / (k + rank_r(c))` with `k = 60` and equal weights.
    Ranks only — raw scores are ignored, so no normalization is needed.
  - `FusionStrategy.LINEAR`: each side is min-max normalized, then combined
    as `alpha * vector + (1 - alpha) * bm25`.
- **Metadata filter** — when `filter_metadata` is given, an exact-match
  candidate set is computed (chunk metadata, falling back to document
  metadata) and retrievers run over `max(4 * top_k, 40)` candidates before
  the filter trims results.
- Every hit is a `QueryResult` carrying `score`, `rank`, plus the per-side
  `vector_score` / `lexical_score` (when the corresponding side ran).

### Vector score convention

All backends convert native distances into a **higher-is-better** similarity
score (`storage/base.py`):

| Metric | Score |
| --- | --- |
| `cosine` | cosine similarity in `[-1, 1]` (usearch/chroma: `1 - distance`; numpy: normalized dot product) |
| `inner_product` | raw dot product, unbounded (usearch/chroma: `-distance`) |
| `l2` | `1 / (1 + squared_distance)` in `(0, 1]` |

The default metric is `cosine`. `usearch` add is upserting: re-adding an id
removes the previous vector first.

### Lexical engine

`BM25Engine` implements Okapi BM25 (defaults `k1 = 1.5`, `b = 0.75`,
lowercasing on, no stopword removal by default):

```
Score(D, Q) = Σ_i IDF(qi) · f(qi, D) · (k1 + 1) / (f(qi, D) + k1 · (1 − b + b · |D| / avgdl))
IDF(qi)     = ln( (N − n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
```

with a dictionary-backed inverted index and a cached IDF table — pure
Python/NumPy, no external service.

### Embedding backends

`BaseEmbedder` requires `backend`, `model_name`, `dimension`,
`embed_documents(texts)`, `embed_query(text)`, and an `init_kwargs` dict
(recorded into the manifest for faithful reconstruction).

- **`fastembed`** (core default; ONNX Runtime, CPU): default model
  `BAAI/bge-small-en-v1.5` (384-dim). BGE models are *asymmetric*:
  `embed_query` transparently prepends the canonical BAAI instruction
  (`"Represent this sentence for searching relevant passages: "`) when the
  model name contains `bge`. Documents are embedded as-is.
- **`torch`** (`protorag[full]`): Hugging Face `AutoModel` + mean pooling.
- **`sentence-transformers`** (`protorag[sentence-transformers]`).

Custom embedders are accepted via `ProtoRAG(embedder_instance=...)`; their
`backend` string is recorded in the manifest, so a saved index can only be
reloaded where that backend is registered (or with an explicit
`override_embedder`).

## Persistence

`save(path)` writes a self-describing directory:

```
<path>/
├── vector_store/     # backend-native artifacts
├── bm25_index.json   # postings, doc lengths, IDF table
├── chunks.jsonl      # document + chunk records (content, metadata)
└── manifest.json     # written LAST — the validity marker
```

The manifest records:

- `schema_version` (major-1 forward compatible), `protorag_version`,
  `created_at_utc`;
- `embedding_config`: backend, model name, dimension, constructor kwargs;
- `vector_store_config`: backend, metric, dimension, constructor kwargs;
- `lexical_config`: BM25 `k1` / `b` / `lowercase`;
- `stats`: total documents and chunks.

`load(path)` rehydrates without touching the class `__init__` (it uses
`cls.__new__`), then cross-checks consistency:

| Condition | Error |
| --- | --- |
| `manifest.json` missing | `FileNotFoundError` |
| unsupported schema major / corrupt file / chunk-count mismatch between manifest, `chunks.jsonl`, vector store, and BM25 index | `SerializationError` |
| recorded embedding or vector backend unavailable, or embedder dimension mismatch | `IncompatibleBackendError` (message names expected vs actual dimensions and an install hint) |

All writes are atomic (temp file + rename). Score round-trip across
save/load is within `1e-5` for `numpy`, `usearch`, and `chromadb`.

## Registries

`EmbedderRegistry` and `VectorStoreRegistry` map backend keys to factories.
Unknown keys raise `ProtoRAGException` with the list of available backends;
missing optional dependencies surface as `ImportError` (converted to
`IncompatibleBackendError` on the load path).

## Tool layer

`ProtoRAGTool` wraps a search call as:

- a plain callable: `tool(query="…")` → retrieved context text;
- a Hugging Face transformers tool schema via `to_json_schema()`
  (`type: "function"`, parameters `{query: string}`, `required: ["query"]`);
- optionally a smolagents `Tool` subclass via `to_hf_tool()` when
  `smolagents` is importable; otherwise the plain callable is returned.

## Performance targets

Benchmark corpus: 1,000 chunks of ~500 characters. Single-query latency
budgets: **vector < 5 ms, BM25 < 2 ms, hybrid < 8 ms**. Measured on a
modern laptop CPU (usearch, 384-dim BGE-small, 500-char chunks): vector
≈ 0.1 ms, BM25 ≈ 0.6 ms, hybrid ≈ 0.7 ms.

## Error taxonomy

`ProtoRAGException` is the base class; the subsystems raise
`ChunkingError`, `EmbeddingError`, `VectorStoreError`, `LexicalError`,
`SerializationError`, and `IncompatibleBackendError` so callers can catch at
the level they need.
