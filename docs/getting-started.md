# Getting started

## Installation

Requires Python 3.9 – 3.13.

```bash
# Core (CPU): numpy + pydantic + fastembed + usearch
pip install protorag

# Optional extras
pip install "protorag[full]"              # + PyTorch, transformers, sentence-transformers, chromadb
pip install "protorag[chroma]"            # + chromadb only
pip install "protorag[sentence-transformers]"  # + sentence-transformers only
```

From a source checkout:

```bash
pip install ".[dev]"      # core + test/lint/type-check tooling
pip install ".[full,dev]" # core + PyTorch backends + chromadb + tooling
```

## Quickstart

```python
from protorag import ProtoRAG, SearchMode

# CPU-friendly: in-process usearch index + ONNX fastembed embeddings.
rag = ProtoRAG(vector_backend="numpy", embedding_backend="fastembed")

rag.add_texts(
    [
        "The Apollo 11 mission landed humans on the Moon in July 1969.",
        "Python is a high-level, general-purpose programming language.",
        "Transformers and self-attention mechanisms revolutionized "
        "natural language processing.",
    ],
    metadatas=[
        {"source": "history"},
        {"source": "programming"},
        {"source": "nlp"},
    ],
)

# Pure lexical (BM25) search.
results = rag.search("Apollo 11 Moon 1969", top_k=1, mode=SearchMode.BM25)

# Vector search.
results = rag.search("coding in modern scripting languages", top_k=1, mode=SearchMode.VECTOR)

# Hybrid: RRF fusion of vector + BM25 rankings (the default mode).
results = rag.search("Apollo space mission NLP", top_k=2, mode=SearchMode.HYBRID)
for hit in results:
    print(f"{hit.score:.4f}  {hit.content[:60]!r}  {hit.metadata}")
```

`add_texts` / `add_documents` return the generated chunk IDs. `len(rag)`
reports the number of stored chunks; `rag.clear()` empties the index.

Documents can be added without chunking when you already have passages:

```python
from protorag import Document

rag.add_documents([Document(id="doc1", content="...", metadata={"topic": "AI"})])
```

### Choosing backends

| `vector_backend` | Package | Notes |
| --- | --- | --- |
| `"numpy"` | core | exact brute-force; great for prototyping and tests |
| `"usearch"` (default) | core | HNSW ANN, the recommended default |
| `"chromadb"` | `protorag[chroma]` | embedded Chroma client |

| `embedding_backend` | Package | Notes |
| --- | --- | --- |
| `"fastembed"` (default) | core | ONNX/CPU; default model `BAAI/bge-small-en-v1.5` |
| `"torch"` | `protorag[full]` | Hugging Face `AutoModel` via PyTorch |
| `"sentence-transformers"` | `protorag[sentence-transformers]` | `SentenceTransformer` models |

BGE-family models are asymmetric: `embed_query` automatically prepends the
canonical BGE instruction prefix (`"Represent this sentence for searching
relevant passages: "`) so query and passage embeddings line up. You can also
pass your own embedder with `embedder_instance=` (see
[architecture.md](architecture.md) for the `BaseEmbedder` contract).

## Hybrid search and metadata filters

```python
from protorag import FusionStrategy, SearchMode

results = rag.search(
    "learning from data",
    top_k=5,
    mode=SearchMode.HYBRID,              # or SearchMode.BM25 / SearchMode.VECTOR
    alpha=0.5,                           # linear-fusion weight on the vector scores
    fusion_strategy=FusionStrategy.RRF,  # or FusionStrategy.LINEAR
    filter_metadata={"source": "nlp"},   # exact-match pre-filter, any keys
)
```

- `SearchMode.HYBRID` runs both retrievers and fuses their rankings.
  `FusionStrategy.RRF` (default) is rank-based (scores ignored);
  `FusionStrategy.LINEAR` min-max-normalizes each side then combines
  `alpha * vector + (1 - alpha) * bm25`.
- `filter_metadata` is an exact-match predicate: each key must equal the
  given value in the chunk's metadata (falling back to the parent document's
  metadata). When a filter is active the retrievers run over a wider
  candidate set before the filter is applied, so `top_k` still fills up when
  possible.
- BM25 tuning is exposed on the engine constructor:
  `ProtoRAG(bm25_k1=1.5, bm25_b=0.75, bm25_lowercase=True, bm25_stopwords=None)`.

## Persistence

```python
rag.save("./my_index")          # writes manifest.json, chunks.jsonl, bm25_index.json, vector_store/
rag2 = ProtoRAG.load("./my_index")
```

`save` is atomic: the manifest (written last) records embedding model,
dimensions, backend names, and index statistics. `load` enforces a
compatibility matrix — a dimension mismatch raises
`IncompatibleBackendError` with both expected and actual values, a missing
directory raises `FileNotFoundError`, and a corrupt/missing file raises
`SerializationError`. If the recorded embedding backend is unavailable in
the loading environment, pass an explicit substitute:

```python
rag2 = ProtoRAG.load("./my_index", override_embedder=my_embedder)
```

Rankings and scores round-trip within `1e-5` for all three vector backends.

## Tool / agent integration

```python
tool = rag.to_tool(
    name="colorado_kb",
    description="Knowledge base about Colorado facts.",
    top_k=5,
    mode=SearchMode.HYBRID,
)

schema = tool.to_json_schema()   # valid HF transformers tool schema
# schema["type"] == "function"; parameters: {query: string}, required: ["query"]

answer_context = tool(query="What is the elevation of Denver?")
```

`tool` is directly callable with a `query` keyword. When `smolagents` is
installed, `tool.to_hf_tool()` additionally returns a smolagents `Tool`
subclass; without it, the plain callable is returned unchanged.

## Next steps

- [architecture.md](architecture.md) — internals: score conventions, fusion
  math, serialization format, and the compatibility matrix.
