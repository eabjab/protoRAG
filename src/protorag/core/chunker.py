"""Text chunking strategies.

Both chunkers honor the same contract:

* ``split_text("")`` (or whitespace-only input) returns ``[]``.
* Every returned chunk is non-empty and at most ``chunk_size`` characters long.
* Non-whitespace content is preserved in original order (overlap chunkers may
  repeat characters at chunk boundaries, which is the intended behavior).
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from protorag.core.exceptions import ChunkingError

DEFAULT_SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@runtime_checkable
class BaseChunker(Protocol):
    """Structural interface for text chunkers."""

    def split_text(self, text: str) -> List[str]:
        """Splits ``text`` into ordered, overlapping chunks."""
        ...


def _validate_chunker_params(chunk_size: int, chunk_overlap: int) -> None:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ChunkingError(f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )


class SimpleCharacterChunker:
    """Fixed-size sliding window chunker.

    Splits text into windows of ``chunk_size`` characters where consecutive
    windows overlap by exactly ``chunk_overlap`` characters.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        _validate_chunker_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            piece = text[start:end]
            if piece.strip():
                chunks.append(piece)
            if end >= length:
                break
            start += self._step
        return chunks


class RecursiveCharacterChunker:
    """Langchain-style recursive chunker.

    Splits text on a priority list of separators (paragraph -> line ->
    sentence -> word), recursing into oversized fragments, then greedily
    re-merges fragments into chunks of at most ``chunk_size`` characters with
    up to ``chunk_overlap`` characters of carry-over between chunks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        _validate_chunker_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: Tuple[str, ...] = tuple(separators) if separators else DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split_recursive(text, 0)
        return self._merge_pieces(pieces)

    def _split_recursive(self, text: str, sep_idx: int) -> List[str]:
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if sep_idx >= len(self.separators) or not self.separators[sep_idx]:
            return self._hard_split(text)
        parts = self._split_keep_separator(text, self.separators[sep_idx])
        pieces: List[str] = []
        for part in parts:
            pieces.extend(self._split_recursive(part, sep_idx + 1))
        return pieces

    def _split_keep_separator(self, text: str, separator: str) -> List[str]:
        raw = re.split(f"({re.escape(separator)})", text)
        parts: List[str] = []
        for i, piece in enumerate(raw):
            if i % 2 == 1:  # captured separator: attach to previous part
                if parts:
                    parts[-1] += piece
                else:
                    parts.append(piece)
            elif piece:
                parts.append(piece)
        return [part for part in parts if part.strip()]

    def _hard_split(self, text: str) -> List[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _merge_pieces(self, pieces: List[str]) -> List[str]:
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if len(piece) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._hard_split(piece))
                continue
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current)
                room = self.chunk_size - len(piece)
                suffix_len = min(self.chunk_overlap, room)
                current = (current[-suffix_len:] + piece) if suffix_len > 0 else piece
            else:
                current += piece
        if current.strip():
            chunks.append(current)
        return [chunk for chunk in chunks if chunk.strip()]
