"""GitHub MCP Server for SyncUp.

Exposes three tools over SSE transport:
  - github_get_commits
  - github_get_file_diff
  - github_get_pull_requests

Uses a GitHub Personal Access Token (PAT) for authentication.
Set GITHUB_TOKEN env var before running.

Usage:
    # With env var:
    GITHUB_TOKEN=ghp_xxx python server.py

    # With custom port:
    MCP_PORT=3003 python server.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import requests

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("github-mcp")

# ---------------------------------------------------------------------------
# GitHub API setup
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _headers() -> dict[str, str]:
    """Build request headers with auth token."""
    h: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(url: str, params: dict | None = None) -> requests.Response:
    """Make an authenticated GET request to the GitHub API."""
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

_port = int(os.environ.get("MCP_PORT", "3003"))

mcp = FastMCP(
    "GitHub MCP",
    instructions="MCP server for GitHub — get commits, file diffs, and pull requests",
    host="0.0.0.0",
    port=_port,
)


@mcp.tool()
def github_get_commits(
    owner: str,
    repo: str,
    since: str = "",
    author: str = "",
    per_page: int = 30,
) -> list[dict]:
    """Get commits for a repository.

    Args:
        owner: Repository owner (user or org).
        repo: Repository name.
        since: Only return commits after this ISO 8601 timestamp (optional).
        author: Filter by commit author GitHub username (optional).
        per_page: Number of commits to return (default 30, max 100).
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    params: dict[str, str | int] = {"per_page": min(per_page, 100)}
    if since:
        params["since"] = since
    if author:
        params["author"] = author

    resp = _get(url, params)
    commits = resp.json()

    logger.info("Found %d commits in %s/%s", len(commits), owner, repo)

    return [
        {
            "sha": c["sha"],
            "message": c["commit"]["message"],
            "author": c["commit"]["author"]["name"],
            "author_login": c.get("author", {}).get("login", "") if c.get("author") else "",
            "date": c["commit"]["author"]["date"],
            "url": c["html_url"],
        }
        for c in commits
    ]


@mcp.tool()
def github_get_file_diff(
    owner: str,
    repo: str,
    sha: str,
) -> dict:
    """Get the diff and file changes for a specific commit.

    Args:
        owner: Repository owner.
        repo: Repository name.
        sha: Commit SHA.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    resp = _get(url)
    data = resp.json()

    files = data.get("files", [])
    logger.info("Commit %s has %d changed files", sha[:8], len(files))

    return {
        "sha": data["sha"],
        "message": data["commit"]["message"],
        "author": data["commit"]["author"]["name"],
        "date": data["commit"]["author"]["date"],
        "stats": data.get("stats", {}),
        "files": [
            {
                "filename": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changes": f["changes"],
                "patch": f.get("patch", ""),
            }
            for f in files
        ],
    }


@mcp.tool()
def github_get_pull_requests(
    owner: str,
    repo: str,
    state: str = "open",
    per_page: int = 30,
) -> list[dict]:
    """List pull requests for a repository.

    Args:
        owner: Repository owner.
        repo: Repository name.
        state: Filter by PR state — 'open', 'closed', or 'all'.
        per_page: Number of PRs to return (default 30, max 100).
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
    params: dict[str, str | int] = {
        "state": state,
        "per_page": min(per_page, 100),
    }

    resp = _get(url, params)
    prs = resp.json()

    logger.info("Found %d %s PRs in %s/%s", len(prs), state, owner, repo)

    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "author": pr["user"]["login"],
            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "merged_at": pr.get("merged_at"),
            "url": pr["html_url"],
            "head_branch": pr["head"]["ref"],
            "base_branch": pr["base"]["ref"],
        }
        for pr in prs
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub MCP Server")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 3003, or MCP_PORT env var)")
    args = parser.parse_args()

    if args.port is not None:
        mcp._tcp_port = args.port
        _port = args.port

    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — API calls will be rate-limited (60 req/hr)")

    # Quick auth check
    logger.info("Verifying GitHub authentication...")
    try:
        resp = _get(f"{GITHUB_API_BASE}/user")
        user = resp.json()
        logger.info("Authenticated as: %s", user.get("login", "unknown"))
    except requests.exceptions.HTTPError:
        if GITHUB_TOKEN:
            logger.error("GitHub token is invalid or expired")
            sys.exit(1)
        logger.warning("Running unauthenticated — rate limits apply")

    logger.info("Starting GitHub MCP server on port %d...", _port)
    logger.info("SSE endpoint: http://localhost:%d/sse", _port)

    mcp.run(transport="sse")
