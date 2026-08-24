"""Hugging Face / agent tool integration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from protorag import ProtoRAG
from tests.conftest import NGramHashingEmbedder

DENVER = "Denver is the capital of Colorado and has an elevation of 5,280 feet."


def _assert_schema(tool: Any) -> None:
    schema = tool.to_json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "colorado_kb"
    assert schema["function"]["description"] == "Knowledge base about Colorado facts."
    assert "query" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["query"]
    # The schema must round-trip through JSON unchanged.
    assert json.loads(json.dumps(schema)) == schema


@pytest.mark.network
def test_hf_tool_verbatim_spec(real_embedder: Any) -> None:
    """Agent tool schema and query with the real fastembed model."""
    rag = ProtoRAG(vector_backend="numpy", embedding_backend="fastembed")
    rag.add_texts([DENVER])
    tool = rag.to_tool(name="colorado_kb", description="Knowledge base about Colorado facts.")

    _assert_schema(tool)
    output = tool(query="What is the elevation of Denver?")
    assert "5,280 feet" in output
    assert "Denver is the capital" in output


def test_hf_tool_offline_twin() -> None:
    rag = ProtoRAG(
        vector_backend="numpy",
        embedding_backend="fastembed",
        embedder_instance=NGramHashingEmbedder(dimension=64),
    )
    rag.add_texts([DENVER])
    tool = rag.to_tool(name="colorado_kb", description="Knowledge base about Colorado facts.")

    _assert_schema(tool)
    output = tool(query="Denver capital Colorado elevation feet")
    assert "5,280 feet" in output
    assert "Denver is the capital" in output
