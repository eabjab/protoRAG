"""Unit tests for the embedding layer."""

from __future__ import annotations

import numpy as np
import pytest

from protorag import BaseEmbedder, EmbedderRegistry, EmbeddingError
from protorag.embeddings.base import l2_normalize
from protorag.embeddings.fastembed_backend import FastEmbedEmbedder
from tests.conftest import NGramHashingEmbedder

# --------------------------------------------------------------------------- #
# l2_normalize
# --------------------------------------------------------------------------- #


def test_l2_normalize_unit_rows() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    assert normalized.shape == (2, 2)
    assert normalized.dtype == np.float32
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_l2_normalize_zero_row_untouched() -> None:
    vectors = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    np.testing.assert_array_equal(normalized[0], [0.0, 0.0])
    assert abs(float(np.linalg.norm(normalized[1])) - 1.0) < 1e-6


def test_l2_normalize_1d_input() -> None:
    normalized = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert normalized.shape == (2,)
    assert abs(float(np.linalg.norm(normalized)) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Offline hashing embedder (contract compliance)
# --------------------------------------------------------------------------- #


def test_hash_embedder_contract() -> None:
    embedder = NGramHashingEmbedder(dimension=32)
    assert embedder.dimension == 32
    assert isinstance(embedder, BaseEmbedder)

    matrix = embedder.embed_documents(["hello world", "the moon", "hello world"])
    assert matrix.dtype == np.float32
    assert matrix.shape == (3, 32)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    # Deterministic: identical inputs -> identical vectors.
    np.testing.assert_array_equal(matrix[0], matrix[2])
    # Deterministic across instances.
    other = NGramHashingEmbedder(dimension=32).embed_documents(["hello world"])[0]
    np.testing.assert_array_equal(matrix[0], other)

    query = embedder.embed_query("hello world")
    assert query.shape == (32,)
    np.testing.assert_array_equal(query, matrix[0])


def test_hash_embedder_empty_batch() -> None:
    embedder = NGramHashingEmbedder(dimension=8)
    matrix = embedder.embed_documents([])
    assert matrix.shape == (0, 8)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        EmbedderRegistry.create("no-such-backend")
    with pytest.raises(ValueError):
        EmbedderRegistry.is_available("no-such-backend")


def test_registry_is_available_fastembed() -> None:
    assert EmbedderRegistry.is_available("fastembed") is True
    # torch / sentence-transformers are not core dependencies.
    assert EmbedderRegistry.is_available("torch") in (True, False)


def test_registry_sentence_transformers_uninstalled_raises_actionable_error() -> None:
    if EmbedderRegistry.is_available("sentence-transformers"):
        pytest.skip("sentence-transformers is installed; ImportError path not applicable")
    with pytest.raises(ImportError, match="sentence-transformers is not installed"):
        EmbedderRegistry.create("sentence-transformers")


def test_registry_fastembed_rejects_bad_model(capfd: pytest.CaptureFixture[str]) -> None:
    with pytest.raises((EmbeddingError, Exception)):
        FastEmbedEmbedder(model_name="protorag/nonexistent-model-xyz-404")


# --------------------------------------------------------------------------- #
# Real fastembed backend (network-gated)
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_fastembed_real_model_contract(real_embedder: FastEmbedEmbedder) -> None:
    assert real_embedder.backend == "fastembed"
    assert real_embedder.dimension > 0

    matrix = real_embedder.embed_documents(
        ["The quick brown fox jumps.", "Python is a programming language."], batch_size=2
    )
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, real_embedder.dimension)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    query = real_embedder.embed_query("The quick brown fox jumps.")
    assert query.shape == (real_embedder.dimension,)
    # BGE models are asymmetric: query embeddings are not expected to equal
    # document embeddings of the same text (query instruction prefix).
    assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-5

    # Semantic ordering sanity: a related query should be closer to doc 0.
    related = real_embedder.embed_query("A fast fox leapt across the yard.")
    unrelated = real_embedder.embed_query("Quarterly revenue grew by nine percent.")
    sim_related = float(np.dot(related, matrix[0]))
    sim_unrelated = float(np.dot(unrelated, matrix[0]))
    assert sim_related > sim_unrelated
