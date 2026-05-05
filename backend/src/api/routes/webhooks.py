"""Webhook ingestion routes for GitHub and Trello.

Receives external webhook HTTP requests, validates signatures, parses payloads,
pre-fetches additional context (e.g. diffs via MCP), and invokes the
Progress Tracking agent node.

All external data is treated as UNTRUSTED per CLAUDE.md security rules.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from agents.progress_tracking import progress_tracking
from integrations.webhooks import (
    WebhookParseError,
    parse_github_pr,
    parse_github_push,
    parse_trello_card_update,
    validate_github_signature,
)
from state.store import InMemoryStateStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pending event builders
# ---------------------------------------------------------------------------


def _build_push_pending_event(
    parsed: Any,
    diffs: dict[str, str],
) -> dict[str, Any]:
    """Build a pending_event dict from a parsed GitHub push event.

    Args:
        parsed: A ``GitHubPushEvent`` instance.
        diffs: Mapping of commit SHA → diff summary string.

    Returns:
        A pending_event dict for the progress tracking agent.
    """
    commits = []
    for commit in parsed.commits:
        commits.append({
            "sha": commit.id,
            "message": commit.message,
            "added": commit.added,
            "removed": commit.removed,
            "modified": commit.modified,
            "diff_summary": diffs.get(commit.id, ""),
        })

    return {
        "event_type": "github_push",
        "github_username": parsed.pusher.name,
        "repository_full_name": parsed.repository.full_name,
        "commits": commits,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_pr_pending_event(parsed: Any) -> dict[str, Any]:
    """Build a pending_event dict from a parsed GitHub PR event.

    Args:
        parsed: A ``GitHubPREvent`` instance.

    Returns:
        A pending_event dict for the progress tracking agent.
    """
    pr = parsed.pull_request
    return {
        "event_type": "github_pr",
        "github_username": pr.user.get("login", "") if isinstance(pr.user, dict) else "",
        "repository_full_name": "",
        "pr_title": pr.title,
        "pr_action": parsed.action,
        "pr_number": parsed.number,
        "pr_files_changed": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_trello_pending_event(parsed: Any) -> dict[str, Any]:
    """Build a pending_event dict from a parsed Trello card update event.

    Args:
        parsed: A ``TrelloCardUpdateEvent`` instance.

    Returns:
        A pending_event dict for the progress tracking agent.
    """
    return {
        "event_type": "trello_card_update",
        "trello_card_id": parsed.card_id,
        "trello_card_name": parsed.card_name,
        "trello_list_before": parsed.list_before_id,
        "trello_list_after": parsed.list_after_id,
        "member_creator_username": parsed.member_creator_username,
        "member_creator_full_name": parsed.member_creator_full_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _fetch_commit_diffs(
    mcp_client: Any,
    parsed: Any,
) -> dict[str, str]:
    """Pre-fetch diffs for each commit via GitHub MCP.

    Args:
        mcp_client: The ``SyncUpMCPClient`` instance (may be ``None``).
        parsed: A ``GitHubPushEvent`` instance.

    Returns:
        A dict mapping commit SHA → diff summary string.
        Returns empty diffs if MCP is unavailable.
    """
    if mcp_client is None:
        logger.warning("MCP client unavailable — skipping diff pre-fetch")
        return {}

    diffs: dict[str, str] = {}
    repo_full_name = parsed.repository.full_name
    parts = repo_full_name.split("/", 1)
    if len(parts) != 2:
        logger.warning("Cannot parse repo full_name: %s", repo_full_name)
        return {}

    owner, repo = parts

    try:
        from mcp_layer.github import GitHubMCP

        github_mcp = GitHubMCP(mcp_client)
        for commit in parsed.commits:
            try:
                result = await github_mcp.get_file_diff(owner, repo, commit.id)
                diffs[commit.id] = str(result.get("diff", ""))
            except Exception as exc:
                logger.warning("Failed to fetch diff for %s: %s", commit.id, exc)
                diffs[commit.id] = ""
    except Exception as exc:
        logger.warning("GitHub MCP initialization failed: %s", exc)

    return diffs


# ---------------------------------------------------------------------------
# Agent invocation helper
# ---------------------------------------------------------------------------


async def _process_webhook_event(
    store: InMemoryStateStore,
    project_id: str,
    pending_event: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the progress tracking agent and persist the updated state.

    Retrieves the current state, sets ``pending_event``, calls the agent
    synchronously in a thread pool, merges the result (manually appending
    to the contribution_ledger since ``operator.add`` reducers only work
    inside the LangGraph runtime), and saves the updated state.

    Args:
        store: The in-memory state store.
        project_id: The project identifier.
        pending_event: The event dict for the agent to process.

    Returns:
        A status dict with processing results.
    """
    state = await store.get(project_id)
    if state is None:
        return {"status": "error", "detail": "project not found"}

    # Set the pending event on state
    state_with_event = state.model_copy(update={"pending_event": pending_event})

    # Run the sync agent in a thread pool to avoid blocking the event loop
    result = await asyncio.to_thread(progress_tracking, state_with_event)

    # Manually append contribution_ledger (operator.add reducer doesn't apply
    # outside LangGraph — model_copy replaces the value instead of appending)
    new_records = result.get("contribution_ledger", [])
    new_ledger = state.contribution_ledger + new_records
    updated = state.model_copy(update={
        "contribution_ledger": new_ledger,
        "pending_event": None,
        "student_progress": result.get("student_progress", state.student_progress),
    })
    await store.save(project_id, updated)

    logger.info(
        "Processed webhook for project %s: %d new record(s), progress=%s",
        project_id,
        len(new_records),
        result.get("student_progress", {}),
    )

    try:
        from api.websockets import manager as ws_manager

        await ws_manager.broadcast(
            project_id,
            {
                "event": "contribution_logged",
                "data": {
                    "records_added": len(new_records),
                    "student_progress": result.get("student_progress", {}),
                },
            },
        )
    except Exception:
        logger.exception("WebSocket broadcast failed for webhook event")

    return {
        "status": "processed",
        "project_id": project_id,
        "records_added": len(new_records),
        "student_progress": result.get("student_progress", {}),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/github")
async def github_webhook(request: Request) -> dict[str, Any]:
    """Receive and process a GitHub webhook event.

    Validates the HMAC-SHA256 signature, parses the event based on
    ``X-GitHub-Event`` header, pre-fetches diffs for push events,
    invokes the progress tracking agent, and persists the updated state.

    Returns:
        A status dict with processing results.

    Raises:
        HTTPException: 401 for invalid signature, 400 for malformed payload.
    """
    # 1. Read raw body for signature validation
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

    if not validate_github_signature(body, signature, secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 2. Parse JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_header = request.headers.get("X-GitHub-Event", "")

    # 3. Parse and build pending event based on event type
    try:
        if event_header == "push":
            parsed = parse_github_push(payload)
            mcp_client = getattr(request.app.state, "mcp_client", None)
            diffs = await _fetch_commit_diffs(mcp_client, parsed)
            pending = _build_push_pending_event(parsed, diffs)
            repo_name = parsed.repository.full_name
        elif event_header == "pull_request":
            parsed = parse_github_pr(payload)
            pending = _build_pr_pending_event(parsed)
            repo_name = pending.get("repository_full_name", "")
        else:
            return {"status": "ignored", "event": event_header}
    except WebhookParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 4. Resolve project_id from repo → project mapping
    repo_map: dict[str, str] = getattr(request.app.state, "repo_project_map", {})
    project_id = repo_map.get(repo_name)
    if project_id is None:
        logger.warning("No project found for repo %s", repo_name)
        return {"status": "accepted", "detail": "no matching project"}

    # 5. Invoke agent and persist state
    store: InMemoryStateStore = request.app.state.state_store
    return await _process_webhook_event(store, project_id, pending)


@router.post("/trello")
async def trello_webhook(request: Request) -> dict[str, Any]:
    """Receive and process a Trello webhook event.

    Parses the card update payload, invokes the progress tracking agent,
    and persists the updated state.

    Returns:
        A status dict with processing results.

    Raises:
        HTTPException: 400 for malformed payload.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        parsed = parse_trello_card_update(payload)
    except WebhookParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Only process card moves (updateCard with list change)
    if parsed.action_type != "updateCard" or not parsed.list_after_id:
        return {"status": "ignored"}

    pending = _build_trello_pending_event(parsed)

    # Resolve project_id from board → project mapping
    board_id = payload.get("model", {}).get("id", "")
    board_map: dict[str, str] = getattr(request.app.state, "board_project_map", {})
    project_id = board_map.get(board_id)
    if project_id is None:
        logger.warning("No project found for board %s", board_id)
        return {"status": "accepted", "detail": "no matching project"}

    # Invoke agent and persist state
    store: InMemoryStateStore = request.app.state.state_store
    return await _process_webhook_event(store, project_id, pending)


@router.head("/trello")
async def trello_webhook_verification() -> Response:
    """Trello sends a HEAD request to verify the webhook URL exists.

    Returns:
        HTTP 200 response.
    """
    return Response(status_code=200)
