"""Fast Unicode text tokenization and normalization for the lexical engine."""

from __future__ import annotations

import re
from typing import FrozenSet, List, Optional

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: A pragmatic English stopword list. Tokenization is stopword-free by
#: default; pass ``stopwords=DEFAULT_STOPWORDS`` to the BM25 engine to enable.
DEFAULT_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
        "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
        "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
        "isn't", "it", "its", "itself", "just", "me", "more", "most", "mustn't",
        "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
        "same", "shan't", "she", "should", "shouldn't", "so", "some", "such",
        "t", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
        "weren't", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "won't", "would", "wouldn't", "you", "your",
        "yours", "yourself", "yourselves",
    }
)


def tokenize(
    text: str,
    lowercase: bool = True,
    stopwords: Optional[FrozenSet[str]] = None,
) -> List[str]:
    """Splits ``text`` into word tokens using a Unicode ``\\w+`` regex.

    Args:
        text: Input text.
        lowercase: Case-fold every token (``str.casefold``).
        stopwords: Optional set of tokens to drop (compared case-folded).

    Returns:
        List of tokens in order of appearance.
    """
    tokens = _WORD_RE.findall(text or "")
    if lowercase:
        tokens = [token.casefold() for token in tokens]
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    return tokens
