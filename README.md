# protoRAG

Zero-infrastructure, in-memory RAG prototyping framework for Python.
Vector, BM25, and hybrid search with pluggable backends, durable
persistence, and Hugging Face tool integration — all in one process, no
external service.

## Features

- **Hybrid retrieval** — vector + Okapi BM25 with RRF or linear score
  fusion, exact-match metadata filters
- **Pluggable backends** — vector stores: `numpy`, `usearch` (default),
  `chromadb`; embedders: `fastembed` (default, CPU/ONNX), `torch`,
  `sentence-transformers`
- **CPU-first** — the base install needs no GPU and no PyTorch
- **Durable indexes** — atomic `save()` / `load()` with a self-describing
  manifest and explicit compatibility errors; score round-trip < 1e-5
- **Agent-ready** — `to_tool()` emits a valid HF transformers tool schema
  and a plain callable (optional smolagents wrapper)
- **Strictly typed** — `mypy --strict` clean, `py.typed` marker included

## Installation

Python 3.9 – 3.13.

```bash
pip install protorag                 # core: numpy + fastembed + usearch
pip install "protorag[full]"         # + PyTorch / transformers / sentence-transformers / chromadb
pip install "protorag[chroma]"       # + chromadb only
```

## Quickstart

```python
from protorag import ProtoRAG, SearchMode

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

for hit in rag.search("Apollo 11 Moon 1969", top_k=1, mode=SearchMode.BM25):
    print(hit.score, hit.content, hit.metadata)

# Hybrid is the default mode:
rag.search("Apollo space mission NLP", top_k=2)

# Persistence + agent tooling:
rag.save("./my_index")
rag2 = ProtoRAG.load("./my_index")
tool = rag2.to_tool(name="my_kb", description="A knowledge base.")
tool(query="What happened on the Moon in 1969?")
tool.to_json_schema()
```

## Documentation

- [Getting started](docs/getting-started.md) — installation, backends,
  hybrid search, persistence, tool integration
- [Architecture](docs/architecture.md) — component map, score conventions,
  fusion math, serialization format, performance targets

## Development

```bash
pip install ".[dev]"
ruff check src/protorag     # lint
mypy --strict src/protorag  # type check
pytest -m "not network" -q  # offline test suite
pytest -q                   # full suite (downloads the default BGE model once)
```

CI (`.github/workflows/`): `test-cpu.yml` runs the py3.9–3.13 matrix with
lint + mypy + offline tests; `test-full.yml` runs the full suite with
PyTorch backends and network tests; `release.yml` builds and publishes on
`v*` tags.

## License

[Apache License 2.0](LICENSE)
