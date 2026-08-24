"""Unit tests for the chunking strategies."""

from __future__ import annotations

import pytest

from protorag import (
    BaseChunker,
    ChunkingError,
    RecursiveCharacterChunker,
    SimpleCharacterChunker,
)

PARAGRAPH_TEXT = (
    "First paragraph about the Moon.\n\n"
    "Second paragraph about rockets.\n"
    "Third paragraph about astronauts and training."
)


def test_simple_chunker_empty_input() -> None:
    assert SimpleCharacterChunker().split_text("") == []
    assert SimpleCharacterChunker().split_text("   \n\t  ") == []


def test_recursive_chunker_empty_input() -> None:
    assert RecursiveCharacterChunker().split_text("") == []
    assert RecursiveCharacterChunker().split_text(" \n ") == []


@pytest.mark.parametrize(
    "chunker",
    [
        SimpleCharacterChunker(chunk_size=50, chunk_overlap=10),
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10),
    ],
    ids=["simple", "recursive"],
)
def test_chunker_contract(chunker: BaseChunker) -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 10
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip(), "chunks must be non-empty"
        assert len(chunk) <= 50, f"chunk exceeds chunk_size: {len(chunk)}"


@pytest.mark.parametrize(
    "chunker",
    [
        SimpleCharacterChunker(chunk_size=40, chunk_overlap=10),
        RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10),
    ],
    ids=["simple", "recursive"],
)
def test_chunker_preserves_content_order(chunker: BaseChunker) -> None:
    text = "alpha beta gamma delta epsilon"
    chunks = chunker.split_text(text)
    # Every original token appears in order across the concatenated output.
    joined = " ".join(chunks)
    positions = [joined.find(token) for token in text.split()]
    assert all(pos >= 0 for pos in positions), joined
    assert positions == sorted(positions)


def test_simple_chunker_window_and_overlap() -> None:
    chunker = SimpleCharacterChunker(chunk_size=5, chunk_overlap=2)
    chunks = chunker.split_text("abcdefgh")
    assert chunks[0] == "abcde"
    assert chunks[1] == "defgh"  # step = 3, window slides
    assert all(len(c) <= 5 for c in chunks)


def test_recursive_chunker_prefers_separators() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
    chunks = chunker.split_text(PARAGRAPH_TEXT)
    # Each paragraph exceeds nothing at this budget, but the combined text
    # exceeds it, so the chunker must split on paragraph/line boundaries
    # rather than mid-sentence. First and Third must never be merged.
    assert any(chunk.startswith("First paragraph") for chunk in chunks)
    assert any(chunk.startswith("Third paragraph") for chunk in chunks)
    assert not any("First paragraph" in c and "Third paragraph" in c for c in chunks)


def test_recursive_chunker_hard_splits_long_words() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.split_text("x" * 25)
    assert [len(c) for c in chunks] == [10, 10, 5]


def test_chunker_validation_errors() -> None:
    with pytest.raises(ChunkingError):
        SimpleCharacterChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ChunkingError):
        SimpleCharacterChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ChunkingError):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=11)
    with pytest.raises(ChunkingError):
        RecursiveCharacterChunker(chunk_size=-3, chunk_overlap=0)


def test_chunkers_satisfy_protocol() -> None:
    assert isinstance(SimpleCharacterChunker(), BaseChunker)
    assert isinstance(RecursiveCharacterChunker(), BaseChunker)
