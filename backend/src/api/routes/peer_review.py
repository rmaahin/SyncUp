"""Peer-review API routes — generate forms, fetch per-student forms, submit reviews."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agents.peer_review import DIMENSION_KEYS, generate_peer_review_form
from guardrails.sanitizer import sanitize_text
from state.schema import PeerReview, SyncUpState
from state.store import InMemoryStateStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_state(request: Request, project_id: str) -> SyncUpState:
    store: InMemoryStateStore = request.app.state.state_store
    state = await store.get(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return state


def _student_ids(state: SyncUpState) -> set[str]:
    return {s.student_id for s in state.student_profiles}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class _ReviewItem(BaseModel):
    reviewee_id: str
    ratings: dict[str, int]
    comments: dict[str, str] = Field(default_factory=dict)


class SubmitPayload(BaseModel):
    reviewer_id: str
    reviews: list[_ReviewItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/{project_id}/generate")
async def generate_forms(project_id: str, request: Request) -> dict[str, Any]:
    """Generate peer-review form template for a project."""
    state = await _get_state(request, project_id)
    update = generate_peer_review_form(state)
    new_state = state.model_copy(update=update)
    store: InMemoryStateStore = request.app.state.state_store
    await store.save(project_id, new_state)
    return update["peer_review_form_template"]


@router.get("/{project_id}/form/{student_id}")
async def get_form(
    project_id: str, student_id: str, request: Request
) -> dict[str, Any]:
    """Return the peer-review form for one student (excluding self)."""
    state = await _get_state(request, project_id)
    if not state.peer_review_forms_generated:
        raise HTTPException(status_code=404, detail="Forms not generated yet")
    if student_id not in _student_ids(state):
        raise HTTPException(status_code=404, detail="Student not found in project")

    if any(r.reviewer_id == student_id for r in state.peer_review_data):
        raise HTTPException(status_code=409, detail="Reviewer has already submitted")

    template = state.peer_review_form_template
    forms_by_student = template.get("forms_by_student", {})
    if student_id not in forms_by_student:
        raise HTTPException(status_code=404, detail="No form for this student")
    return {
        "dimensions": template.get("dimensions", []),
        "teammates": forms_by_student[student_id].get("teammates", []),
    }


@router.post("/{project_id}/submit", status_code=201)
async def submit_review(
    project_id: str, payload: SubmitPayload, request: Request
) -> dict[str, Any]:
    """Submit peer reviews for all teammates in one call."""
    state = await _get_state(request, project_id)
    sids = _student_ids(state)

    # 1. Reviewer must exist
    if payload.reviewer_id not in sids:
        raise HTTPException(status_code=400, detail="Unknown reviewer_id")

    # 2. No prior submission
    if any(r.reviewer_id == payload.reviewer_id for r in state.peer_review_data):
        raise HTTPException(status_code=409, detail="Reviewer has already submitted")

    expected_reviewees = sids - {payload.reviewer_id}

    # Solo / no teammates
    if not expected_reviewees:
        raise HTTPException(status_code=400, detail="No teammates to review")

    seen_reviewees: set[str] = set()
    for item in payload.reviews:
        # 3. Cannot review self
        if item.reviewee_id == payload.reviewer_id:
            raise HTTPException(status_code=400, detail="Cannot review self")
        # 4. Reviewee must be a known teammate
        if item.reviewee_id not in expected_reviewees:
            raise HTTPException(
                status_code=400, detail=f"Unknown reviewee_id: {item.reviewee_id}"
            )
        if item.reviewee_id in seen_reviewees:
            raise HTTPException(
                status_code=400, detail=f"Duplicate review for {item.reviewee_id}"
            )
        seen_reviewees.add(item.reviewee_id)

        # 5. Dimension keys must be known
        unknown_dims = (set(item.ratings) | set(item.comments)) - DIMENSION_KEYS
        if unknown_dims:
            raise HTTPException(
                status_code=400, detail=f"Unknown dimension(s): {sorted(unknown_dims)}"
            )

        # 6. Must rate every dimension
        if set(item.ratings) != DIMENSION_KEYS:
            raise HTTPException(
                status_code=400, detail="All 5 dimensions must be rated"
            )

        # 7. Each rating in 1..5  (422 — validation)
        for dim, val in item.ratings.items():
            if not isinstance(val, int) or val < 1 or val > 5:
                raise HTTPException(
                    status_code=422,
                    detail=f"Rating for '{dim}' must be an integer 1..5",
                )

    # 8. All teammates must be reviewed
    if seen_reviewees != expected_reviewees:
        missing = expected_reviewees - seen_reviewees
        raise HTTPException(
            status_code=400, detail=f"Missing teammate review(s): {sorted(missing)}"
        )

    # Build PeerReview rows (sanitize free-text comment values)
    now = datetime.now(timezone.utc)
    new_rows = [
        PeerReview(
            reviewer_id=payload.reviewer_id,
            reviewee_id=item.reviewee_id,
            ratings=dict(item.ratings),
            comments={k: sanitize_text(v) for k, v in item.comments.items()},
            submitted_at=now,
        )
        for item in payload.reviews
    ]

    new_state = state.model_copy(
        update={"peer_review_data": list(state.peer_review_data) + new_rows}
    )
    store: InMemoryStateStore = request.app.state.state_store
    await store.save(project_id, new_state)
    return {"count": len(new_rows)}


@router.get("/{project_id}/status")
async def status(project_id: str, request: Request) -> dict[str, Any]:
    """Return submitted vs pending reviewer sets."""
    state = await _get_state(request, project_id)
    sids = _student_ids(state)
    submitted = {r.reviewer_id for r in state.peer_review_data}
    pending = sids - submitted
    return {
        "submitted": sorted(submitted),
        "pending": sorted(pending),
        "total": len(sids),
    }
