"""Dashboard data routes for student and professor views.

All endpoints are read-only — they extract data from the project's
``SyncUpState`` and return JSON.  No LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from state.schema import SyncUpState, TaskStatus
from state.store import InMemoryStateStore

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_state(request: Request, project_id: str) -> SyncUpState:
    """Fetch project state or raise 404."""
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return state


def _validate_student_in_project(state: SyncUpState, student_id: str) -> None:
    """Raise 404 if the student is not part of the project."""
    if not any(sp.student_id == student_id for sp in state.student_profiles):
        raise HTTPException(status_code=404, detail="Student not found in project")


# ---------------------------------------------------------------------------
# Student endpoints
# ---------------------------------------------------------------------------


@router.get("/student/{project_id}/{student_id}/tasks")
async def student_tasks(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    """Return tasks assigned to a specific student."""
    state = await _get_state(request, project_id)
    _validate_student_in_project(state, student_id)

    assigned_task_ids = {
        tid for tid, sid in state.delegation_matrix.items() if sid == student_id
    }
    tasks = [
        t.model_dump(mode="json")
        for t in state.task_array
        if t.id in assigned_task_ids
    ]
    return {"tasks": tasks}


@router.get("/student/{project_id}/{student_id}/progress")
async def student_progress(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    """Return progress summary for a student."""
    state = await _get_state(request, project_id)
    _validate_student_in_project(state, student_id)

    assigned_task_ids = {
        tid for tid, sid in state.delegation_matrix.items() if sid == student_id
    }
    assigned_tasks = [t for t in state.task_array if t.id in assigned_task_ids]
    total = len(assigned_tasks)
    completed = sum(1 for t in assigned_tasks if t.status == TaskStatus.DONE)
    completion_pct = (completed / total * 100.0) if total else 0.0

    # Deadline adherence: tasks done by deadline or still before deadline
    now = datetime.now(timezone.utc)
    adherent = 0
    tasks_with_deadlines = [t for t in assigned_tasks if t.deadline is not None]
    for t in tasks_with_deadlines:
        assert t.deadline is not None  # for type narrowing
        if t.status == TaskStatus.DONE or t.deadline >= now:
            adherent += 1
    deadline_adherence_pct = (
        (adherent / len(tasks_with_deadlines) * 100.0) if tasks_with_deadlines else 100.0
    )

    progress_status = state.student_progress.get(student_id, "unknown")

    burn_down_data = [
        bd.model_dump(mode="json") for bd in state.project_timeline.burn_down_targets
    ]

    return {
        "student_id": student_id,
        "total_tasks": total,
        "completed_tasks": completed,
        "completion_pct": round(completion_pct, 1),
        "deadline_adherence_pct": round(deadline_adherence_pct, 1),
        "progress_status": progress_status,
        "burn_down_data": burn_down_data,
    }


@router.get("/student/{project_id}/{student_id}/meetings")
async def student_meetings(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    """Return meetings involving a student."""
    state = await _get_state(request, project_id)
    _validate_student_in_project(state, student_id)

    past_meetings = [
        m.model_dump(mode="json")
        for m in state.meeting_log
        if student_id in m.attendees
    ]

    return {
        "upcoming": (
            state.next_meeting_scheduled.isoformat()
            if state.next_meeting_scheduled
            else None
        ),
        "past_meetings": past_meetings,
    }


@router.get("/student/{project_id}/{student_id}/notifications")
async def student_notifications(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    """Return interventions/nudges targeted at a student, newest first."""
    state = await _get_state(request, project_id)
    _validate_student_in_project(state, student_id)

    interventions = [
        i.model_dump(mode="json")
        for i in state.intervention_history
        if i.target_student_id == student_id
    ]
    interventions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"notifications": interventions}


# ---------------------------------------------------------------------------
# Professor endpoints
# ---------------------------------------------------------------------------


@router.get("/professor/{project_id}/overview")
async def professor_overview(
    project_id: str, request: Request
) -> dict[str, Any]:
    """Return project-level aggregate stats for the professor dashboard."""
    state = await _get_state(request, project_id)

    total_tasks = len(state.task_array)
    done = sum(1 for t in state.task_array if t.status == TaskStatus.DONE)
    completion_pct = (done / total_tasks * 100.0) if total_tasks else 0.0

    on_track = sum(1 for s in state.student_progress.values() if s == "on_track")
    at_risk = sum(1 for s in state.student_progress.values() if s == "at_risk")
    behind = sum(1 for s in state.student_progress.values() if s == "behind")

    now = datetime.now(timezone.utc)
    days_remaining = (
        (state.final_deadline - now).days if state.final_deadline else None
    )

    return {
        "project_id": project_id,
        "project_name": state.project_name,
        "total_tasks": total_tasks,
        "completed_tasks": done,
        "completion_pct": round(completion_pct, 1),
        "student_count": len(state.student_profiles),
        "on_track_count": on_track,
        "at_risk_count": at_risk,
        "behind_count": behind,
        "days_remaining": days_remaining,
        "publishing_status": (
            state.publishing_status.model_dump(mode="json")
            if state.publishing_status
            else None
        ),
    }


@router.get("/professor/{project_id}/students")
async def professor_students(
    project_id: str, request: Request
) -> dict[str, Any]:
    """Return per-student summary for the professor dashboard."""
    state = await _get_state(request, project_id)

    # Build inverse delegation: student_id → [task_ids]
    student_tasks: dict[str, list[str]] = {}
    for tid, sid in state.delegation_matrix.items():
        student_tasks.setdefault(sid, []).append(tid)

    # Build contribution counts and quality
    student_contributions: dict[str, list[float]] = {}
    for cr in state.contribution_ledger:
        student_contributions.setdefault(cr.student_id, []).append(
            cr.semantic_quality_score
        )

    # Build intervention counts
    student_interventions: dict[str, int] = {}
    for iv in state.intervention_history:
        student_interventions[iv.target_student_id] = (
            student_interventions.get(iv.target_student_id, 0) + 1
        )

    task_map = {t.id: t for t in state.task_array}

    summaries = []
    for sp in state.student_profiles:
        assigned_ids = student_tasks.get(sp.student_id, [])
        assigned = [task_map[tid] for tid in assigned_ids if tid in task_map]
        completed_count = sum(1 for t in assigned if t.status == TaskStatus.DONE)

        quality_scores = student_contributions.get(sp.student_id, [])
        avg_quality = (
            round(sum(quality_scores) / len(quality_scores), 2)
            if quality_scores
            else 0.0
        )

        # Deadline adherence
        now = datetime.now(timezone.utc)
        tasks_with_dl = [t for t in assigned if t.deadline is not None]
        adherent = sum(
            1
            for t in tasks_with_dl
            if t.status == TaskStatus.DONE or (t.deadline is not None and t.deadline >= now)
        )
        adherence_pct = (
            (adherent / len(tasks_with_dl) * 100.0) if tasks_with_dl else 100.0
        )

        summaries.append({
            "student_id": sp.student_id,
            "name": sp.name,
            "tasks_assigned": len(assigned_ids),
            "tasks_completed": completed_count,
            "contribution_count": len(quality_scores),
            "avg_quality_score": avg_quality,
            "progress_status": state.student_progress.get(sp.student_id, "unknown"),
            "intervention_count": student_interventions.get(sp.student_id, 0),
            "deadline_adherence_pct": round(adherence_pct, 1),
        })

    return {"students": summaries}


@router.get("/professor/{project_id}/student/{student_id}/contributions")
async def professor_student_contributions(
    project_id: str,
    student_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return paginated contribution ledger for one student, newest first."""
    state = await _get_state(request, project_id)
    _validate_student_in_project(state, student_id)

    records = [
        cr for cr in state.contribution_ledger if cr.student_id == student_id
    ]
    records.sort(key=lambda cr: cr.timestamp, reverse=True)
    total = len(records)
    page = records[offset : offset + limit]

    return {
        "items": [r.model_dump(mode="json") for r in page],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/professor/{project_id}/interventions")
async def professor_interventions(
    project_id: str, request: Request
) -> dict[str, Any]:
    """Return all interventions across all students, newest first."""
    state = await _get_state(request, project_id)

    interventions = [
        i.model_dump(mode="json") for i in state.intervention_history
    ]
    interventions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"interventions": interventions}


@router.get("/professor/{project_id}/timeline")
async def professor_timeline(
    project_id: str, request: Request
) -> dict[str, Any]:
    """Return project timeline data for burn-down charting."""
    state = await _get_state(request, project_id)
    return state.project_timeline.model_dump(mode="json")
