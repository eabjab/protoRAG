# protoRAG documentation

protoRAG is a zero-infrastructure, in-memory Retrieval-Augmented Generation
(RAG) prototyping framework for Python. It provides vector, lexical (BM25),
and hybrid search with pluggable storage and embedding backends, persistence
with strict compatibility checks, and a one-call adapter for Hugging Face /
agent tooling — all in-process, with no external service.

## Documentation

| Document | Contents |
| --- | --- |
| [Getting started](getting-started.md) | Installation, quickstart, hybrid search, persistence, tool integration |
| [Architecture](architecture.md) | Component map, score conventions, fusion formulas, serialization layout, compatibility rules |

## Design goals

- **Zero infrastructure** — everything runs in a single Python process; no
  database server, broker, or compiled extension is required.
- **Prototyping speed** — construct a working hybrid RAG index in a few lines
  of Python.
- **CPU-first** — the default installation is CPU-only (NumPy + ONNX-based
  `fastembed` + `usearch`); PyTorch/CUDA backends are optional extras.
- **Swappable backends** — vector stores (`numpy`, `usearch`, `chromadb`) and
  embedders (`fastembed`, `torch`, `sentence-transformers`) are registered
  interfaces, not hard dependencies.
- **Durable indexes** — `save()` / `load()` round-trip all three vector
  backends byte-for-byte in ranking, with an explicit compatibility matrix.
- **Agent-ready** — `ProtoRAG.to_tool()` emits a valid Hugging Face
  transformers tool schema (`to_json_schema()`) and a plain callable.

## Package layout

```
src/protorag/
├── core/          # ProtoRAG engine, entities, chunkers, exceptions
├── embeddings/    # BaseEmbedder, fastembed / torch / sentence-transformers backends, registry
├── storage/       # BaseVectorStore, numpy / usearch / chromadb backends, registry
├── lexical/       # tokenizer + Okapi BM25 engine
├── hybrid/        # score normalizers + RRF / linear fusion
├── serialization/ # manifest, serializer, save/load + compatibility checks
└── tools/         # ProtoRAGTool (HF tool schema + optional smolagents wrapper)
```
