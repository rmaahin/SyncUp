"""Thin wrapper around Google Docs MCP tools.

Tool names are defined as constants — update them when the actual MCP server
is connected and tool names are known.
"""

from __future__ import annotations

from typing import Any

from mcp_layer.client import SyncUpMCPClient

# MCP tool name constants
TOOL_READ_DOCUMENT = "google_docs_read_document"
TOOL_CREATE_DOCUMENT = "google_docs_create_document"
TOOL_SEARCH_DOCUMENTS = "google_docs_search_documents"


class GoogleDocsMCP:
    """Typed interface to Google Docs MCP tools.

    All methods delegate to the underlying MCP tool via ``ainvoke``.
    """

    def __init__(self, mcp_client: SyncUpMCPClient) -> None:
        self._mcp = mcp_client

    async def read_document(self, document_id: str) -> dict[str, Any]:
        """Read the raw text content of a Google Doc.

        Args:
            document_id: The Google Docs document ID.

        Returns:
            Document content and metadata.
        """
        tool = self._mcp.require_tool(TOOL_READ_DOCUMENT)
        result = await tool.ainvoke({"document_id": document_id})
        return result if isinstance(result, dict) else {"content": result}

    async def create_document(
        self, title: str, content: str = ""
    ) -> dict[str, Any]:
        """Create a new Google Doc.

        Args:
            title: Document title.
            content: Initial document body text.

        Returns:
            Created document data including document ID.
        """
        tool = self._mcp.require_tool(TOOL_CREATE_DOCUMENT)
        result = await tool.ainvoke({"title": title, "content": content})
        return result if isinstance(result, dict) else {"document_id": result}

    async def search_documents(self, query: str) -> list[dict[str, Any]]:
        """Search for documents matching a query.

        Args:
            query: Search query string.

        Returns:
            List of matching document metadata.
        """
        tool = self._mcp.require_tool(TOOL_SEARCH_DOCUMENTS)
        result = await tool.ainvoke({"query": query})
        return result if isinstance(result, list) else [result]
