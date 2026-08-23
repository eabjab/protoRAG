"""Agent-tool wrapper exposing a :class:`~protorag.core.engine.ProtoRAG`
instance to Hugging Face ``smolagents`` / ``transformers`` and any
OpenAI-compatible function-calling stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from protorag.core.entities import SearchMode


class ProtoRAGTool:
    """Standardized tool wrapper for ProtoRAG retrieval."""

    def __init__(
        self,
        rag_instance: Any,
        name: str = "knowledge_base_retriever",
        description: Optional[str] = None,
        top_k: int = 5,
        search_mode: SearchMode = SearchMode.HYBRID,
    ) -> None:
        self.rag = rag_instance
        self.name = name
        self.description = description or (
            "Searches the in-memory knowledge base for relevant documents "
            "and context chunks. Pass a natural language query to retrieve "
            "matching text snippets with relevance scores."
        )
        self.top_k = top_k
        self.search_mode = search_mode

    def __call__(self, query: str) -> str:
        """Executes the search and formats results as a context string."""
        results = self.rag.search(query=query, top_k=self.top_k, mode=self.search_mode)
        if not results:
            return "No relevant context found in knowledge base."
        formatted_chunks = []
        for i, res in enumerate(results, start=1):
            formatted_chunks.append(
                f"[{i}] (Score: {res.score:.4f}, Doc ID: {res.document_id})\n{res.content}"
            )
        return "\n\n".join(formatted_chunks)

    def to_json_schema(self) -> Dict[str, Any]:
        """OpenAI / Hugging Face compatible JSON Schema for tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query or keyword phrase to "
                                "retrieve relevant context for."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def to_hf_tool(self) -> Any:
        """Export to a Hugging Face smolagents / transformers Tool object.

        Falls back to returning ``self`` (a plain callable) when
        ``smolagents`` is not installed, since agents can invoke any
        callable and inspect ``to_json_schema()`` directly.
        """
        try:
            from smolagents import Tool
        except ImportError:
            return self

        name = self.name
        description = self.description

        class HfProtoRAGTool(Tool):  # type: ignore[misc]
            """smolagents ``Tool`` wrapper around a ProtoRAG retrieval call."""

            name = name
            description = description
            # Field default for the smolagents (pydantic) tool spec.
            inputs = {  # noqa: RUF012
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve context.",
                }
            }
            output_type = "string"

            def __init__(self, tool_parent: ProtoRAGTool) -> None:
                super().__init__()
                self.name = tool_parent.name
                self.description = tool_parent.description
                self.tool_parent = tool_parent

            def forward(self, query: str) -> str:
                return self.tool_parent(query)

        return HfProtoRAGTool(self)
