"""The ``ProtoRAG`` facade: high-level orchestrator over chunking, embedding,
lexical (BM25), vector, and hybrid-search subsystems.

Typical usage::

    from protorag import ProtoRAG, Document

    rag = ProtoRAG(vector_backend="usearch", embedding_backend="fastembed")
    rag.add_documents([Document(id="doc1", content="...")])
    results = rag.search("query", top_k=5, mode="hybrid")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

import numpy as np

from protorag.core.chunker import BaseChunker, RecursiveCharacterChunker
from protorag.core.entities import (
    Chunk,
    DistanceMetric,
    Document,
    FusionStrategy,
    QueryResult,
    SearchMode,
)
from protorag.core.exceptions import (
    IncompatibleBackendError,
    ProtoRAGException,
    SerializationError,
)
from protorag.embeddings.base import BaseEmbedder
from protorag.embeddings.registry import EmbedderRegistry
from protorag.hybrid.fusion import fuse
from protorag.lexical.bm25 import BM25Engine
from protorag.serialization.manifest import (
    EmbeddingConfig,
    IndexStats,
    LexicalConfig,
    Manifest,
    VectorStoreConfig,
    verify_schema_version,
)
from protorag.serialization.serializer import (
    read_json,
    read_jsonl,
    utc_now_iso,
    write_json_atomic,
    write_jsonl_atomic,
)
from protorag.storage.base import BaseVectorStore
from protorag.storage.registry import VectorStoreRegistry
from protorag.tools.hf_tool import ProtoRAGTool

_MANIFEST_FILE = "manifest.json"
_CHUNKS_FILE = "chunks.jsonl"
_BM25_FILE = "bm25_index.json"
_VECTOR_STORE_DIR = "vector_store"

_STR_INSTALL_HINTS = {
    "fastembed": "pip install fastembed",
    "sentence-transformers": "pip install 'protorag[sentence-transformers]'",
    "torch": "pip install torch",
}

_E = TypeVar("_E", bound=Enum)


def _coerce_enum(enum_cls: Type[_E], value: Union[str, _E]) -> _E:
    """Accepts either an enum member or its string value."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower().strip())
    except ValueError as err:
        valid = [member.value for member in enum_cls]
        raise ProtoRAGException(
            f"Invalid {enum_cls.__name__} value {value!r}. Valid values: {valid}."
        ) from err


class ProtoRAG:
    """Unified in-memory RAG orchestrator.

    Wires together an embedder, a vector store, and an in-process BM25
    lexical index behind a small, stable facade. State lives entirely in
    process memory and is (de)serialized through a self-describing
    directory layout (see :meth:`save` / :meth:`load`).
    """

    def __init__(
        self,
        vector_backend: str = "usearch",
        embedding_backend: str = "fastembed",
        embedding_model: Optional[str] = None,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
        embedder_instance: Optional[BaseEmbedder] = None,
        custom_chunker: Optional[BaseChunker] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_lowercase: bool = True,
        bm25_stopwords: Optional[Sequence[str]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        vector_store_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes storage backends, lexical index, and embedding models."""
        metric = _coerce_enum(DistanceMetric, distance_metric)
        self._embedder: BaseEmbedder = (
            embedder_instance
            if embedder_instance is not None
            else EmbedderRegistry.create(
                embedding_backend, model_name=embedding_model, **(embedding_kwargs or {})
            )
        )
        self._vector_store: BaseVectorStore = VectorStoreRegistry.create(
            vector_backend,
            dimension=self._embedder.dimension,
            metric=metric,
            **(vector_store_kwargs or {}),
        )
        self._bm25 = BM25Engine(
            k1=bm25_k1, b=bm25_b, lowercase=bm25_lowercase, stopwords=bm25_stopwords
        )
        self._custom_chunker = custom_chunker
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._distance_metric = metric
        self._vector_store_kwargs: Dict[str, Any] = dict(vector_store_kwargs or {})
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Chunks, embeds, and indexes plain text documents.

        Returns the generated document ids (chunks are addressable as
        ``f"{doc_id}_chunk_{i}"``).
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ProtoRAGException(
                f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})."
            )
        documents = [
            Document(id=f"doc_{uuid4().hex}", content=text, metadata=dict(metadata))
            for text, metadata in zip(texts, metadatas)
        ]
        return self.add_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Chunks, embeds, and indexes :class:`Document` objects.

        Re-adding a document with an existing id replaces its previous
        chunks. Returns the ids of all documents that produced chunks.
        """
        documents = list(documents)
        if not documents:
            return []
        chunker: BaseChunker = (
            self._custom_chunker
            if self._custom_chunker is not None
            else RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        planned: List[Tuple[Document, str, int, int]] = []
        for doc in documents:
            parts = chunker.split_text(doc.content)
            if not parts:
                continue
            self._remove_document(doc.id)
            for index, part in enumerate(parts):
                planned.append((doc, part, index, len(parts)))
        if not planned:
            return []

        embeddings = self._embedder.embed_documents([part for _, part, _, _ in planned])
        chunk_ids = [f"{doc.id}_chunk_{index}" for doc, _, index, _ in planned]
        self._vector_store.add(chunk_ids, embeddings)
        self._bm25.add_documents(chunk_ids, [part for _, part, _, _ in planned])
        indexed_doc_ids: List[str] = []
        for (doc, part, index, total), vector in zip(planned, embeddings):
            self._chunks[f"{doc.id}_chunk_{index}"] = Chunk(
                id=f"{doc.id}_chunk_{index}",
                document_id=doc.id,
                content=part,
                metadata={**doc.metadata, "chunk_index": index, "total_chunks": total},
                embedding=np.asarray(vector, dtype=np.float32),
            )
            if doc.id not in indexed_doc_ids:
                indexed_doc_ids.append(doc.id)
                self._documents[doc.id] = doc
        return indexed_doc_ids

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Union[str, SearchMode] = SearchMode.HYBRID,
        alpha: float = 0.5,
        fusion_strategy: Union[str, FusionStrategy] = FusionStrategy.RRF,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Executes Vector, BM25, or Hybrid search across the indexed corpus.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.
            mode: ``vector``, ``bm25``, or ``hybrid``.
            alpha: Vector weight for linear fusion (1.0 = vector, 0.0 = BM25).
            fusion_strategy: ``rrf`` or ``linear`` for hybrid mode.
            filter_metadata: Optional exact-match metadata predicate applied
                to chunk (and, by inheritance, document) metadata.
        """
        mode = _coerce_enum(SearchMode, mode)
        fusion_strategy = _coerce_enum(FusionStrategy, fusion_strategy)
        if top_k <= 0 or not self._chunks or not query or not query.strip():
            return []

        candidate_set: Optional[Set[str]] = None
        candidate_k = top_k
        if filter_metadata:
            candidate_set = {
                chunk_id
                for chunk_id in self._chunks
                if self._matches_filter(self._chunks[chunk_id], filter_metadata)
            }
            if not candidate_set:
                return []
            candidate_k = max(4 * top_k, 40)

        if mode is SearchMode.BM25:
            ranking = self._bm25.search(query, top_k=candidate_k)
            if candidate_set is not None:
                ranking = [(cid, score) for cid, score in ranking if cid in candidate_set]
            return [
                self._to_result(cid, score, rank, lexical_score=score)
                for rank, (cid, score) in enumerate(ranking[:top_k], start=1)
            ]

        query_vector = self._embedder.embed_query(query)
        vector_ranking = self._vector_store.search(query_vector, top_k=candidate_k)
        if mode is SearchMode.VECTOR:
            if candidate_set is not None:
                vector_ranking = [
                    (cid, score) for cid, score in vector_ranking if cid in candidate_set
                ]
            return [
                self._to_result(cid, score, rank, vector_score=score)
                for rank, (cid, score) in enumerate(vector_ranking[:top_k], start=1)
            ]

        # Hybrid mode: fuse vector + lexical rankings.
        lexical_ranking = self._bm25.search(query, top_k=candidate_k)
        fused = fuse(
            [vector_ranking, lexical_ranking], strategy=fusion_strategy, alpha=alpha
        )
        vector_scores: Dict[str, float] = dict(vector_ranking)
        lexical_scores: Dict[str, float] = dict(lexical_ranking)
        results: List[QueryResult] = []
        for cid, score in fused:
            if candidate_set is not None and cid not in candidate_set:
                continue
            if len(results) >= top_k:
                break
            results.append(
                self._to_result(
                    cid,
                    score,
                    len(results) + 1,
                    vector_score=vector_scores.get(cid),
                    lexical_score=lexical_scores.get(cid),
                )
            )
        return results

    def _matches_filter(self, chunk: Chunk, filter_metadata: Dict[str, Any]) -> bool:
        for key, value in filter_metadata.items():
            if key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
            else:
                doc = self._documents.get(chunk.document_id)
                if doc is None or key not in doc.metadata or doc.metadata[key] != value:
                    return False
        return True

    def _to_result(
        self,
        chunk_id: str,
        score: float,
        rank: int,
        vector_score: Optional[float] = None,
        lexical_score: Optional[float] = None,
    ) -> QueryResult:
        chunk = self._chunks[chunk_id]
        return QueryResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            score=float(score),
            metadata=dict(chunk.metadata),
            vector_score=vector_score,
            lexical_score=lexical_score,
            rank=rank,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serializes the current index state, chunks, and metadata.

        Layout: ``manifest.json``, ``chunks.jsonl``, ``bm25_index.json``,
        ``vector_store/<backend artifacts>``. The manifest is written last so
        a partial directory is never treated as a valid index.
        """
        os.makedirs(path, exist_ok=True)
        self._vector_store.save(os.path.join(path, _VECTOR_STORE_DIR))
        self._bm25.save(os.path.join(path, _BM25_FILE))
        records: List[Dict[str, Any]] = []
        for doc in self._documents.values():
            records.append(
                {
                    "record": "document",
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": dict(doc.metadata),
                }
            )
        for chunk in self._chunks.values():
            records.append(
                {
                    "record": "chunk",
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": dict(chunk.metadata),
                }
            )
        write_jsonl_atomic(os.path.join(path, _CHUNKS_FILE), records)

        init_kwargs = getattr(self._embedder, "init_kwargs", None)
        manifest = Manifest(
            created_at_utc=utc_now_iso(),
            embedding_config=EmbeddingConfig(
                backend=self._embedder.backend,
                model_name=self._embedder.model_name,
                dimension=self._embedder.dimension,
                kwargs=dict(init_kwargs or {}),
            ),
            vector_store_config=VectorStoreConfig(
                backend=self._vector_store.backend,
                metric=self._distance_metric,
                dimension=self._embedder.dimension,
                kwargs=dict(self._vector_store_kwargs),
            ),
            lexical_config=LexicalConfig(
                k1=self._bm25.k1, b=self._bm25.b, lowercase=self._bm25.lowercase
            ),
            stats=IndexStats(
                total_documents=len(self._documents), total_chunks=len(self._chunks)
            ),
        )
        write_json_atomic(os.path.join(path, _MANIFEST_FILE), manifest.to_dict())

    @classmethod
    def load(
        cls,
        path: str,
        override_embedder: Optional[BaseEmbedder] = None,
    ) -> ProtoRAG:
        """Loads and rehydrates a :class:`ProtoRAG` instance from disk.

        Raises:
            FileNotFoundError: when ``manifest.json`` is missing.
            SerializationError: on unsupported schema or inconsistent state.
            IncompatibleBackendError: when the persisted embedding backend is
                unavailable and no ``override_embedder`` was supplied.
        """
        manifest_path = os.path.join(path, _MANIFEST_FILE)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No {_MANIFEST_FILE!r} found in {path!r}; not a protoRAG index directory."
            )
        manifest = Manifest.model_validate(read_json(manifest_path))
        verify_schema_version(manifest.schema_version)

        if override_embedder is not None:
            embedder: BaseEmbedder = override_embedder
        else:
            try:
                embedder = EmbedderRegistry.create(
                    manifest.embedding_config.backend,
                    model_name=manifest.embedding_config.model_name,
                    **manifest.embedding_config.kwargs,
                )
            except ImportError as err:
                raise IncompatibleBackendError(
                    _incompatible_embedder_message(manifest, err)
                ) from err
        if embedder.dimension != manifest.embedding_config.dimension:
            raise IncompatibleBackendError(
                f"Embedder '{embedder.model_name}' produces "
                f"{embedder.dimension}-dimensional embeddings, but the index was built with "
                f"{manifest.embedding_config.dimension}-dimensional embeddings "
                f"({manifest.embedding_config.backend}/{manifest.embedding_config.model_name}). "
                f"Pass a dimension-matching override_embedder or re-index the corpus."
            )

        store_config = manifest.vector_store_config
        try:
            vector_store = VectorStoreRegistry.create(
                store_config.backend,
                dimension=store_config.dimension,
                metric=store_config.metric,
                **store_config.kwargs,
            )
            vector_store.load(os.path.join(path, _VECTOR_STORE_DIR))
        except ImportError as err:
            raise IncompatibleBackendError(
                f"Vector store backend '{store_config.backend}' is not available in this "
                f"environment ({err}). Install it or re-save the index with a different "
                f"vector backend."
            ) from err

        lexical_config = manifest.lexical_config
        bm25 = BM25Engine(
            k1=lexical_config.k1, b=lexical_config.b, lowercase=lexical_config.lowercase
        )
        bm25.load(os.path.join(path, _BM25_FILE))

        documents, chunks = _read_corpus(path)
        expected = manifest.stats.total_chunks
        actual = len(chunks)
        if actual != expected:
            raise SerializationError(
                f"Manifest reports {expected} chunks but {_CHUNKS_FILE} contains {actual}."
            )
        if len(vector_store) != actual:
            raise SerializationError(
                f"Vector store holds {len(vector_store)} vectors but {actual} chunks were loaded."
            )
        if len(bm25) != actual:
            raise SerializationError(
                f"BM25 index holds {len(bm25)} documents but {actual} chunks were loaded."
            )

        instance: ProtoRAG = cls.__new__(cls)
        instance._embedder = embedder
        instance._vector_store = vector_store
        instance._bm25 = bm25
        instance._custom_chunker = None
        instance._documents = documents
        instance._chunks = chunks
        instance._distance_metric = store_config.metric
        instance._vector_store_kwargs = dict(store_config.kwargs)
        instance._embedding_model = manifest.embedding_config.model_name
        return instance

    # ------------------------------------------------------------------ #
    # Tooling & lifecycle
    # ------------------------------------------------------------------ #

    def to_tool(
        self,
        name: str = "knowledge_retriever",
        description: Optional[str] = None,
        top_k: int = 5,
        mode: Union[str, SearchMode] = SearchMode.HYBRID,
    ) -> ProtoRAGTool:
        """Returns a callable tool for HF transformers / agents frameworks."""
        return ProtoRAGTool(
            self,
            name=name,
            description=description,
            top_k=top_k,
            search_mode=_coerce_enum(SearchMode, mode),
        )

    def clear(self) -> None:
        """Flushes all chunks, vectors, and inverted index structures."""
        if self._chunks:
            chunk_ids = list(self._chunks.keys())
            self._vector_store.delete(chunk_ids)
        self._bm25.clear()
        self._chunks.clear()
        self._documents.clear()

    def __len__(self) -> int:
        """Returns the number of indexed chunks."""
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _remove_document(self, doc_id: str) -> None:
        stale = [
            chunk_id for chunk_id in self._chunks if self._chunks[chunk_id].document_id == doc_id
        ]
        if stale:
            self._vector_store.delete(stale)
            self._bm25.delete(stale)
            for chunk_id in stale:
                del self._chunks[chunk_id]
        self._documents.pop(doc_id, None)


def _read_corpus(path: str) -> Tuple[Dict[str, Document], Dict[str, Chunk]]:
    chunks_path = os.path.join(path, _CHUNKS_FILE)
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"{_CHUNKS_FILE!r} not found in {path!r}.")
    documents: Dict[str, Document] = {}
    chunks: Dict[str, Chunk] = {}
    for record in read_jsonl(chunks_path):
        kind = record.get("record")
        if kind == "document":
            documents[str(record["id"])] = Document(
                id=str(record["id"]),
                content=str(record["content"]),
                metadata=dict(record.get("metadata") or {}),
            )
        elif kind == "chunk":
            chunks[str(record["id"])] = Chunk(
                id=str(record["id"]),
                document_id=str(record["document_id"]),
                content=str(record["content"]),
                metadata=dict(record.get("metadata") or {}),
            )
        else:
            raise SerializationError(f"Unknown record type {kind!r} in {_CHUNKS_FILE!r}.")
    return documents, chunks


def _incompatible_embedder_message(manifest: Manifest, original: ImportError) -> str:
    backend = manifest.embedding_config.backend
    hint = _STR_INSTALL_HINTS.get(backend, f"pip install the package providing '{backend}'")
    return (
        f"Embedding backend '{backend}' is not available in this environment ({original}). "
        f"Install it with: {hint}. "
        f"Alternatively, load the index with a compatible substitute embedder, e.g. "
        f"ProtoRAG.load(path, override_embedder=FastEmbedEmbedder()) — the substitute must "
        f"produce {manifest.embedding_config.dimension}-dimensional embeddings "
        f"(index model: {manifest.embedding_config.model_name!r})."
    )
