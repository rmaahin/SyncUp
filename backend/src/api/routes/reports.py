"""Final-report API routes — generate and read TeamReports."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from services.peer_review_analysis import detect_bias
from services.report_generator import generate_team_report
from state.schema import SyncUpState
from state.store import InMemoryStateStore

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_state(request: Request, project_id: str) -> SyncUpState:
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return state


@router.post("/{project_id}/generate")
async def generate_report(project_id: str, request: Request) -> dict[str, Any]:
    """Synthesize the final TeamReport. Slow — N+1 LLM calls."""
    state = await _get_state(request, project_id)
    report = generate_team_report(state)
    new_state = state.model_copy(update={"team_report": report})
    store: InMemoryStateStore = request.app.state.state_store
    await store.save(project_id, new_state)
    return report.model_dump(mode="json")


@router.get("/{project_id}/team")
async def get_team_report(project_id: str, request: Request) -> dict[str, Any]:
    state = await _get_state(request, project_id)
    if state.team_report is None:
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return state.team_report.model_dump(mode="json")


@router.get("/{project_id}/student/{student_id}")
async def get_student_report(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    state = await _get_state(request, project_id)
    if state.team_report is None:
        raise HTTPException(status_code=404, detail="Report not generated yet")
    for sr in state.team_report.student_reports:
        if sr.student_id == student_id:
            return sr.model_dump(mode="json")
    raise HTTPException(status_code=404, detail="Student report not found")


@router.get("/{project_id}/bias-flags")
async def get_bias_flags(project_id: str, request: Request) -> dict[str, Any]:
    state = await _get_state(request, project_id)
    if state.team_report is not None:
        return {
            "bias_flags": [f.model_dump(mode="json") for f in state.team_report.bias_flags]
        }
    flags = detect_bias(
        state.peer_review_data,
        [s.student_id for s in state.student_profiles],
    )
    return {"bias_flags": [f.model_dump(mode="json") for f in flags]}
