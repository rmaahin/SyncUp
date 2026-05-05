"""Project management API routes.

Provides endpoints for bootstrapping project state into the in-memory store,
retrieving state for debugging, and registering external webhooks (Trello).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.routes.onboarding import StudentStore, get_student_store
from api.websockets import manager as ws_manager
from guardrails.sanitizer import sanitize_document
from integrations.trello import TrelloClient
from state.schema import SyncUpState, TaskStatus
from state.store import InMemoryStateStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    """Request body for creating a new project."""

    name: str
    description: str = ""
    final_deadline: datetime
    meeting_interval_days: int = Field(default=7, ge=1)


class CreateProjectResponse(BaseModel):
    """Response for project creation."""

    project_id: str
    name: str
    status: str


class ProjectOverview(BaseModel):
    """Project-level overview for the GET /{project_id} endpoint."""

    project_id: str
    name: str
    description: str
    final_deadline: Optional[datetime]
    student_count: int
    task_count: int
    completion_pct: float
    publishing_status: Optional[dict[str, Any]] = None


class UploadBriefRequest(BaseModel):
    """Request body for setting the project brief."""

    brief: str


class PipelineStatusResponse(BaseModel):
    """Pipeline status summary."""

    project_id: str
    phase: str
    task_count: int
    delegated_count: int
    published: bool
    has_errors: bool


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


# ---------------------------------------------------------------------------
# Project lifecycle CRUD (Phase 10)
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=CreateProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    body: CreateProjectRequest, request: Request
) -> CreateProjectResponse:
    """Create a new project and initialize its SyncUpState."""
    project_id = uuid.uuid4().hex
    state = SyncUpState(
        project_id=project_id,
        project_name=body.name,
        project_brief=body.description,
        final_deadline=body.final_deadline,
        meeting_interval_days=body.meeting_interval_days,
    )
    store: InMemoryStateStore = request.app.state.state_store
    await store.save(project_id, state)
    return CreateProjectResponse(
        project_id=project_id, name=body.name, status="created"
    )


@router.get("/{project_id}", response_model=ProjectOverview)
async def project_overview(project_id: str, request: Request) -> ProjectOverview:
    """Return a project overview: name, deadline, student count, task count, completion %."""
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    total = len(state.task_array)
    done = sum(1 for t in state.task_array if t.status == TaskStatus.DONE)
    completion_pct = (done / total * 100.0) if total else 0.0

    return ProjectOverview(
        project_id=project_id,
        name=state.project_name,
        description=state.project_brief,
        final_deadline=state.final_deadline,
        student_count=len(state.student_profiles),
        task_count=total,
        completion_pct=round(completion_pct, 1),
        publishing_status=(
            state.publishing_status.model_dump(mode="json")
            if state.publishing_status
            else None
        ),
    )


@router.post("/{project_id}/brief")
async def upload_brief(
    project_id: str, body: UploadBriefRequest, request: Request
) -> dict[str, Any]:
    """Set/upload the project brief.  Sanitizes through the guardrails layer."""
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    sanitized = sanitize_document(body.brief, source="project_brief")
    updated = state.model_copy(update={"project_brief": sanitized})
    await store.save(project_id, updated)

    return {
        "status": "ok",
        "project_id": project_id,
        "brief_length": len(sanitized),
    }


@router.post(
    "/{project_id}/students/{student_id}",
    status_code=status.HTTP_201_CREATED,
)
async def add_student_to_project(
    project_id: str,
    student_id: str,
    request: Request,
    student_store: StudentStore = Depends(get_student_store),
) -> dict[str, Any]:
    """Link a student profile (from StudentStore) into a project's state."""
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    profile = student_store.get(student_id)
    if profile is None:
        raise HTTPException(
            status_code=404, detail=f"Student '{student_id}' not found"
        )

    if any(sp.student_id == student_id for sp in state.student_profiles):
        raise HTTPException(
            status_code=409,
            detail=f"Student '{student_id}' already in project '{project_id}'",
        )

    updated = state.model_copy(
        update={"student_profiles": state.student_profiles + [profile]}
    )
    await store.save(project_id, updated)

    return {
        "status": "ok",
        "project_id": project_id,
        "student_id": student_id,
        "student_count": len(updated.student_profiles),
    }


@router.post("/{project_id}/start")
async def start_pipeline(project_id: str, request: Request) -> dict[str, Any]:
    """Trigger the full LangGraph pipeline for a project.

    Validates that brief and students are present, invokes the graph
    synchronously in a thread pool, persists the result, and broadcasts
    a ``pipeline_complete`` event over WebSocket.
    """
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not state.project_brief:
        raise HTTPException(
            status_code=400, detail="Project brief is required before starting"
        )
    if not state.student_profiles:
        raise HTTPException(
            status_code=400,
            detail="At least one student must be added before starting",
        )

    # Lazy import to avoid module-load circular dependencies
    from graph.main import graph

    result = await asyncio.to_thread(graph.invoke, state.model_dump())
    updated = SyncUpState.model_validate(result)
    await store.save(project_id, updated)

    summary = {
        "status": "ok",
        "project_id": project_id,
        "task_count": len(updated.task_array),
        "delegated_count": len(updated.delegation_matrix),
        "published": updated.publishing_status is not None,
    }

    await ws_manager.broadcast(
        project_id, {"event": "pipeline_complete", "data": summary}
    )

    return summary


@router.get("/{project_id}/status", response_model=PipelineStatusResponse)
async def pipeline_status(
    project_id: str, request: Request
) -> PipelineStatusResponse:
    """Return the current pipeline phase derived from state."""
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not state.task_array:
        phase = "awaiting_decomposition"
    elif not state.delegation_matrix:
        phase = "awaiting_delegation"
    elif state.publishing_status is None:
        phase = "awaiting_publishing"
    else:
        phase = "complete"

    has_errors = bool(
        state.publishing_status and state.publishing_status.errors
    )

    return PipelineStatusResponse(
        project_id=project_id,
        phase=phase,
        task_count=len(state.task_array),
        delegated_count=len(state.delegation_matrix),
        published=state.publishing_status is not None,
        has_errors=has_errors,
    )
