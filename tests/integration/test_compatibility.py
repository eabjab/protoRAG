"""Load-time compatibility and error-handling tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from protorag import (
    IncompatibleBackendError,
    ProtoRAG,
    SerializationError,
)
from protorag.embeddings.registry import EmbedderRegistry
from protorag.serialization.serializer import read_json, write_json_atomic
from tests.conftest import NGramHashingEmbedder

TEXTS = [
    "Machine learning enables systems to learn from data.",
    "Neural networks mimic human brain structures.",
]


def _saved_index(tmp_path: Path) -> Path:
    rag = ProtoRAG(
        vector_backend="numpy",
        embedding_backend="fastembed",
        embedder_instance=NGramHashingEmbedder(dimension=64),
    )
    rag.add_texts(TEXTS)
    path = tmp_path / "index"
    rag.save(str(path))
    return path


def _rewrite_manifest(path: Path, mutate: Callable[[Dict[str, Any]], None]) -> None:
    manifest_path = path / "manifest.json"
    payload = read_json(str(manifest_path))
    mutate(payload)
    write_json_atomic(str(manifest_path), payload)


def test_load_missing_manifest_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        ProtoRAG.load(str(tmp_path / "empty"))


def test_unavailable_embedding_backend_raises_actionable_error(tmp_path: Path) -> None:
    if EmbedderRegistry.is_available("sentence-transformers"):
        pytest.skip("sentence-transformers is installed; ImportError path not applicable")
    path = _saved_index(tmp_path)

    def _swap_backend(payload: Dict[str, Any]) -> None:
        payload["embedding_config"]["backend"] = "sentence-transformers"

    _rewrite_manifest(path, _swap_backend)
    with pytest.raises(IncompatibleBackendError) as excinfo:
        ProtoRAG.load(str(path))
    message = str(excinfo.value)
    assert "sentence-transformers" in message
    assert "pip install" in message
    assert "override_embedder" in message


def test_override_embedder_bypasses_unavailable_backend(tmp_path: Path) -> None:
    if EmbedderRegistry.is_available("sentence-transformers"):
        pytest.skip("sentence-transformers is installed; ImportError path not applicable")
    path = _saved_index(tmp_path)

    def _swap_backend(payload: Dict[str, Any]) -> None:
        payload["embedding_config"]["backend"] = "sentence-transformers"

    _rewrite_manifest(path, _swap_backend)
    loaded = ProtoRAG.load(str(path), override_embedder=NGramHashingEmbedder(dimension=64))
    assert len(loaded) == 2


def test_unsupported_schema_version_raises_serialization_error(tmp_path: Path) -> None:
    path = _saved_index(tmp_path)
    _rewrite_manifest(path, lambda payload: payload.__setitem__("schema_version", "2.0.0"))
    with pytest.raises(SerializationError):
        ProtoRAG.load(str(path))


def test_embedder_dimension_mismatch_raises(tmp_path: Path) -> None:
    path = _saved_index(tmp_path)
    with pytest.raises(IncompatibleBackendError, match="dimension"):
        ProtoRAG.load(str(path), override_embedder=NGramHashingEmbedder(dimension=32))


def test_chunk_count_mismatch_raises_serialization_error(tmp_path: Path) -> None:
    path = _saved_index(tmp_path)

    def _lie_about_stats(payload: Dict[str, Any]) -> None:
        payload["stats"]["total_chunks"] = 999

    _rewrite_manifest(path, _lie_about_stats)
    # The manifest records the offline embedder's backend ("fake"), which is
    # not registry-creatable, so the load needs an override to reach the
    # chunk-count validation.
    with pytest.raises(SerializationError, match="chunks"):
        ProtoRAG.load(str(path), override_embedder=NGramHashingEmbedder(dimension=64))
