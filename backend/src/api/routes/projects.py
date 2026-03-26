"""Project management API routes.

Provides endpoints for bootstrapping project state into the in-memory store,
retrieving state for debugging, and registering external webhooks (Trello).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from integrations.trello import TrelloClient
from state.schema import SyncUpState
from state.store import InMemoryStateStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{project_id}/state")
async def bootstrap_state(project_id: str, request: Request) -> dict[str, Any]:
    """Load a SyncUpState into the in-memory store.

    Also populates the board-to-project and repo-to-project lookup maps so
    that incoming webhooks can be routed to the correct project.

    Args:
        project_id: The project identifier (path parameter).

    Returns:
        A summary of the bootstrapped state.
    """
    body = await request.json()
    state = SyncUpState.model_validate(body)

    store: InMemoryStateStore = request.app.state.state_store
    await store.save(project_id, state)

    # Populate Trello board → project lookup
    if state.trello_board_id:
        board_map: dict[str, str] = getattr(request.app.state, "board_project_map", {})
        board_map[state.trello_board_id] = project_id
        request.app.state.board_project_map = board_map
        logger.info("Mapped Trello board %s → project %s", state.trello_board_id, project_id)

    # Populate GitHub repo → project lookup
    if state.github_repo:
        repo_map: dict[str, str] = getattr(request.app.state, "repo_project_map", {})
        repo_map[state.github_repo] = project_id
        request.app.state.repo_project_map = repo_map
        logger.info("Mapped GitHub repo %s → project %s", state.github_repo, project_id)

    return {
        "status": "ok",
        "project_id": project_id,
        "tasks": len(state.task_array),
        "students": len(state.student_profiles),
        "trello_board_id": state.trello_board_id,
        "github_repo": state.github_repo,
    }


@router.get("/{project_id}/state")
async def get_state(project_id: str, request: Request) -> dict[str, Any]:
    """Retrieve the current state for a project.

    Args:
        project_id: The project identifier.

    Returns:
        The full SyncUpState as JSON.

    Raises:
        HTTPException: 404 if the project is not found.
    """
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return state.model_dump(mode="json")


@router.get("/")
async def list_projects(request: Request) -> dict[str, Any]:
    """List all project IDs currently stored.

    Returns:
        A dict with a ``projects`` key containing a list of project ID strings.
    """
    store: InMemoryStateStore = request.app.state.state_store
    projects = await store.list_projects()
    return {"projects": projects}


@router.post("/{project_id}/webhooks/register")
async def register_webhooks(
    project_id: str,
    request: Request,
    ngrok_url: str | None = None,
) -> dict[str, Any]:
    """Register a Trello webhook for a project.

    Trello will POST card update events to the ``/api/webhooks/trello``
    endpoint via the provided public URL.

    Args:
        project_id: The project identifier.
        ngrok_url: The public ngrok URL (query param).  Falls back to the
            ``NGROK_PUBLIC_URL`` environment variable.

    Returns:
        A dict with the registered webhook details.

    Raises:
        HTTPException: 404 if project not found, 400 if no public URL or
            no Trello board configured.
    """
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    base_url = ngrok_url or os.environ.get("NGROK_PUBLIC_URL", "")
    if not base_url:
        raise HTTPException(
            status_code=400,
            detail="No public URL provided. Pass ngrok_url query param or set NGROK_PUBLIC_URL.",
        )

    # Strip trailing slash
    base_url = base_url.rstrip("/")

    results: dict[str, Any] = {}

    # Register Trello webhook
    if state.trello_board_id:
        callback = f"{base_url}/api/webhooks/trello"
        try:
            async with TrelloClient() as client:
                webhook = await client.create_webhook(
                    callback_url=callback,
                    id_model=state.trello_board_id,
                    description=f"SyncUp webhook for {project_id}",
                )
                results["trello"] = {
                    "webhook_id": webhook.id,
                    "callback_url": callback,
                    "board_id": state.trello_board_id,
                }
                logger.info(
                    "Trello webhook registered: %s → %s", webhook.id, callback
                )
        except Exception as exc:
            logger.error("Failed to register Trello webhook: %s", exc)
            results["trello"] = {"error": str(exc)}
    else:
        results["trello"] = {"error": "No Trello board configured for this project"}

    # Mark webhooks as configured
    updated = state.model_copy(update={"webhook_configured": True})
    await store.save(project_id, updated)

    return {"status": "ok", "project_id": project_id, "webhooks": results}
