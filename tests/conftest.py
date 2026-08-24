"""Shared fixtures, sample corpora, and the offline deterministic embedder.

The offline suite (default) never touches the network: it uses
:class:`NGramHashingEmbedder`, a deterministic md5 n-gram hashing embedder
whose vectors are stable across processes (required for save/load identity
checks). Tests that exercise the real fastembed model are marked
``network`` and skip gracefully when the model cannot be downloaded.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np
import pytest

from protorag.core.entities import Document

SAMPLE_DOCUMENTS = [
    Document(id="doc1", content="The Apollo 11 mission landed humans on the Moon in July 1969."),
    Document(id="doc2", content="Python is a high-level, general-purpose programming language."),
    Document(
        id="doc3",
        content="Transformers and self-attention mechanisms revolutionized natural language processing.",
    ),
]


class NGramHashingEmbedder:
    """Deterministic character n-gram hashing embedder (offline test double).

    Maps each 2/3-character n-gram of the case-folded text to a signed bucket
    via md5, then L2-normalizes the resulting vector. Fully deterministic
    across processes, so saved corpora reload to bit-identical vectors.
    """

    backend = "fake"
    model_name = "fake-ngram-hash"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = int(dimension)

    def _vector(self, text: str) -> np.ndarray:
        out = np.zeros(self.dimension, dtype=np.float32)
        padded = f" {text.casefold()} "
        for n in (2, 3):
            for i in range(len(padded) - n + 1):
                gram = padded[i : i + n]
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                code = int.from_bytes(digest[:4], "big")
                bucket = code % self.dimension
                sign = 1.0 if (digest[4] % 2) == 0 else -1.0
                out[bucket] += sign
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-12 else out

    def embed_documents(self, texts, batch_size: int = 32) -> np.ndarray:
        vectors = [self._vector(t) for t in texts]
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture(scope="session")
def hash_embedder() -> NGramHashingEmbedder:
    return NGramHashingEmbedder()


@pytest.fixture(scope="session")
def sample_documents() -> List[Document]:
    return list(SAMPLE_DOCUMENTS)


@pytest.fixture(scope="session")
def real_embedder():
    """Real fastembed model; skips the dependent test when unavailable."""
    from protorag.core.exceptions import EmbeddingError
    from protorag.embeddings.fastembed_backend import FastEmbedEmbedder

    try:
        return FastEmbedEmbedder()
    except EmbeddingError as err:  # pragma: no cover - network/model availability
        pytest.skip(f"fastembed model unavailable (network/model download required): {err}")


@pytest.fixture
def make_rag():
    """Factory building a ProtoRAG on a deterministic embedder (no network)."""
    from protorag.core.engine import ProtoRAG

    def _make(
        vector_backend: str = "numpy",
        dimension: int = 64,
        **kwargs: Any,
    ) -> Any:
        params: Dict[str, Any] = {
            "vector_backend": vector_backend,
            "embedding_backend": "fastembed",
            "embedder_instance": NGramHashingEmbedder(dimension=dimension),
        }
        params.update(kwargs)
        return ProtoRAG(**params)

    return _make
