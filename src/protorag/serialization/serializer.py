"""Low-level atomic file (de)serialization helpers.

All JSON/JSONL writes go through a temp file + ``os.replace`` so a crash
mid-write never leaves a truncated artifact behind (deterministic, atomic
persistence per the project invariants).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def write_json_atomic(path: str, obj: Any) -> None:
    """Serializes ``obj`` as pretty JSON to ``path`` atomically."""
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path: str) -> Any:
    """Loads JSON from ``path``; raises ``FileNotFoundError`` when missing."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_atomic(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Writes one JSON object per line to ``path`` atomically."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    _atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads JSONL from ``path``; raises ``FileNotFoundError`` when missing."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
