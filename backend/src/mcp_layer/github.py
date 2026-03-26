"""Thin wrapper around GitHub MCP tools.

Tool names are defined as constants — update them when the actual MCP server
is connected and tool names are known.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mcp_layer.client import SyncUpMCPClient

# MCP tool name constants
TOOL_GET_COMMITS = "github_get_commits"
TOOL_GET_FILE_DIFF = "github_get_file_diff"
TOOL_GET_PULL_REQUESTS = "github_get_pull_requests"


class GitHubMCP:
    """Typed interface to GitHub MCP tools.

    All methods delegate to the underlying MCP tool via ``ainvoke``.
    """

    def __init__(self, mcp_client: SyncUpMCPClient) -> None:
        self._mcp = mcp_client

    async def get_commits(
        self,
        owner: str,
        repo: str,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get commits for a repository.

        Args:
            owner: Repository owner (user or org).
            repo: Repository name.
            since: Only return commits after this time.

        Returns:
            List of commit data.
        """
        tool = self._mcp.require_tool(TOOL_GET_COMMITS)
        params: dict[str, Any] = {"owner": owner, "repo": repo}
        if since is not None:
            params["since"] = since.isoformat()
        result = await tool.ainvoke(params)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
        return result if isinstance(result, list) else [result]

    async def get_file_diff(
        self, owner: str, repo: str, sha: str
    ) -> dict[str, Any]:
        """Get the diff for a specific commit.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: Commit SHA.

        Returns:
            Diff content and metadata.
        """
        tool = self._mcp.require_tool(TOOL_GET_FILE_DIFF)
        result = await tool.ainvoke({"owner": owner, "repo": repo, "sha": sha})
        if isinstance(result, str):
            try:
                return json.loads(result)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, TypeError):
                return {"diff": result}
        return result if isinstance(result, dict) else {"diff": result}

    async def get_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
    ) -> list[dict[str, Any]]:
        """List pull requests for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: Filter by PR state (``open``, ``closed``, ``all``).

        Returns:
            List of pull request data.
        """
        tool = self._mcp.require_tool(TOOL_GET_PULL_REQUESTS)
        result = await tool.ainvoke({"owner": owner, "repo": repo, "state": state})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
        return result if isinstance(result, list) else [result]
