"""Unit tests for the agent tool wrapper (SPEC-001 §2.6, §4.1)."""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

from protorag import ProtoRAG, ProtoRAGTool

DENVER = "Denver is the capital of Colorado and has an elevation of 5,280 feet."
BOULDER = "Boulder is known for outdoor activities and hiking trails."


def _rag_with_docs(make_rag: Any) -> ProtoRAG:
    rag = make_rag(vector_backend="numpy")
    rag.add_texts([DENVER, BOULDER])
    return rag


def test_to_json_schema_shape(make_rag: Any) -> None:
    tool = _rag_with_docs(make_rag).to_tool(
        name="colorado_kb", description="Knowledge base about Colorado facts."
    )
    schema = tool.to_json_schema()
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "colorado_kb"
    assert function["description"] == "Knowledge base about Colorado facts."
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert "query" in parameters["properties"]
    assert parameters["properties"]["query"]["type"] == "string"
    assert parameters["required"] == ["query"]
    # Schema must be plain JSON-serializable.
    assert json.loads(json.dumps(schema)) == schema


def test_call_formats_results(make_rag: Any) -> None:
    tool = _rag_with_docs(make_rag).to_tool(top_k=2, mode="bm25")
    output = tool(query="elevation of Denver in feet")
    assert output.startswith("[1] (Score:")
    assert "Doc ID:" in output
    assert "5,280 feet" in output


def test_call_empty_kb_returns_fixed_message(make_rag: Any) -> None:
    tool = make_rag(vector_backend="numpy").to_tool()
    assert tool(query="anything at all") == "No relevant context found in knowledge base."


def test_top_k_respected(make_rag: Any) -> None:
    tool = _rag_with_docs(make_rag).to_tool(top_k=1, mode="vector")
    output = tool(query="Denver Colorado capital")
    assert "[1]" in output
    assert "[2]" not in output


def test_default_name_and_description(make_rag: Any) -> None:
    tool = ProtoRAGTool(make_rag(vector_backend="numpy"))
    assert tool.name == "knowledge_base_retriever"
    assert tool.description


def test_to_hf_tool_falls_back_to_self_without_smolagents(make_rag: Any) -> None:
    if importlib.util.find_spec("smolagents") is not None:
        pytest.skip("smolagents is installed; fallback path not applicable")
    tool = _rag_with_docs(make_rag).to_tool()
    assert tool.to_hf_tool() is tool
