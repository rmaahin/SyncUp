"""MCP client manager for SyncUp.

Wraps ``langchain-mcp-adapters`` ``MultiServerMCPClient`` to manage connections
to Google Calendar, Google Docs, and GitHub MCP servers with graceful degradation.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _get_multi_server_client_class() -> type:
    """Lazy-import MultiServerMCPClient to avoid circular import.

    Our ``mcp`` package name shadows the pip ``mcp`` package.  By deferring
    the import we let ``langchain_mcp_adapters`` resolve correctly.
    """
    mod = importlib.import_module("langchain_mcp_adapters.client")
    return mod.MultiServerMCPClient  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class MCPToolNotFoundError(Exception):
    """Raised when a requested MCP tool is not available."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"MCP tool not found: {tool_name}")


# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------

_SERVER_CONFIGS = {
    "google_calendar": "GOOGLE_CALENDAR_MCP_URL",
    "google_docs": "GOOGLE_DOCS_MCP_URL",
    "github": "GITHUB_MCP_URL",
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SyncUpMCPClient:
    """Async context manager for MCP server connections.

    Connects to configured MCP servers on entry. If a server is unreachable,
    logs a warning and continues with the servers that are available.

    Usage::

        async with SyncUpMCPClient() as mcp:
            tools = mcp.get_tools()
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._connected_servers: set[str] = set()
        self._mcp_client: Any = None

    async def connect(self) -> None:
        """Connect to all configured MCP servers.

        Servers with missing or empty URL env vars are skipped.
        Servers that fail to connect are logged and skipped.

        Each server is connected independently so that one failing server
        does not prevent the others from being used.
        """
        MultiServerMCPClient = _get_multi_server_client_class()

        for server_name, env_var in _SERVER_CONFIGS.items():
            url = os.environ.get(env_var, "")
            if not url:
                logger.info("MCP server '%s' skipped — %s not set", server_name, env_var)
                continue

            try:
                client = MultiServerMCPClient(
                    {server_name: {"url": url, "transport": "sse"}}
                )
                raw_tools = await client.get_tools()
                for tool in raw_tools:
                    self._tools[tool.name] = tool
                self._connected_servers.add(server_name)
                logger.info(
                    "MCP server '%s' connected — %d tool(s)",
                    server_name,
                    len(raw_tools),
                )
            except Exception:
                logger.exception("MCP server '%s' failed to connect", server_name)

        if self._connected_servers:
            logger.info(
                "MCP connected to %d server(s): %s — %d total tool(s)",
                len(self._connected_servers),
                ", ".join(sorted(self._connected_servers)),
                len(self._tools),
            )
        else:
            logger.warning("No MCP servers connected")

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        self._mcp_client = None
        self._tools = {}
        self._connected_servers = set()

    async def __aenter__(self) -> SyncUpMCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    # -- Tool access --------------------------------------------------------

    def get_tools(self) -> list[BaseTool]:
        """Return all connected MCP tools as LangGraph-compatible tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> BaseTool | None:
        """Look up a single tool by name. Returns ``None`` if not found."""
        return self._tools.get(name)

    def require_tool(self, name: str) -> BaseTool:
        """Look up a tool by name, raising if not found.

        Raises:
            MCPToolNotFoundError: If the tool is not available.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise MCPToolNotFoundError(name)
        return tool

    def is_server_connected(self, server_name: str) -> bool:
        """Check whether a specific MCP server connected successfully."""
        return server_name in self._connected_servers
