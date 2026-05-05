"""Student onboarding API routes.

Provides CRUD for student profiles, availability updates with significance
detection, project readiness checks, and OAuth linking stubs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator

from state.schema import AvailabilityChange, DateRange, StudentProfile, SyncUpState

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BlackoutPeriodInput(BaseModel):
    """Blackout period in request bodies. Mirrors DateRange validation."""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _start_before_end(self) -> BlackoutPeriodInput:
        if self.start >= self.end:
            raise ValueError("Blackout period start must be before end")
        return self


class CreateProfileRequest(BaseModel):
    """Request body for creating a student profile."""

    name: str
    email: str
    skills: dict[str, float] = Field(default_factory=dict)
    availability_hours_per_week: float = Field(default=0.0, ge=0)
    preferred_times: list[str] = Field(default_factory=list)
    blackout_periods: list[BlackoutPeriodInput] = Field(default_factory=list)
    timezone: str = "UTC"
    github_username: str = ""
    google_email: str = ""
    trello_username: str = ""

    @field_validator("skills")
    @classmethod
    def _validate_skill_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for skill, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Skill proficiency for '{skill}' must be between 0.0 and 1.0, got {score}"
                )
        return v


class AvailabilityUpdateRequest(BaseModel):
    """Request body for updating student availability."""

    availability_hours_per_week: float = Field(ge=0)
    blackout_periods: list[BlackoutPeriodInput] = Field(default_factory=list)


class AvailabilityUpdateResponse(BaseModel):
    """Response for availability update with significance info."""

    profile: StudentProfile
    triggered_redelegation: bool
    change_record: AvailabilityChange


class ProjectReadinessResponse(BaseModel):
    """Response for project readiness check."""

    project_id: str
    ready: bool
    total_expected: int
    onboarded_count: int
    missing_fields: dict[str, list[str]]


class OAuthCallbackRequest(BaseModel):
    """Stub request body for OAuth callback."""

    code: str


class OAuthCallbackResponse(BaseModel):
    """Stub response for OAuth callback."""

    provider: str
    linked: bool
    message: str


# ---------------------------------------------------------------------------
# In-memory student store (replaced by DB in later phases)
# ---------------------------------------------------------------------------


class StudentStore:
    """Simple in-memory store for student profiles.

    Thread-safe for single-worker async usage. Will be replaced by
    PostgreSQL in a later phase.
    """

    def __init__(self) -> None:
        self._students: dict[str, StudentProfile] = {}

    def create(self, profile: StudentProfile) -> StudentProfile:
        """Store a new student profile."""
        self._students[profile.student_id] = profile
        return profile

    def get(self, student_id: str) -> StudentProfile | None:
        """Retrieve a student profile by ID."""
        return self._students.get(student_id)

    def update(self, student_id: str, profile: StudentProfile) -> StudentProfile:
        """Replace a student profile."""
        self._students[student_id] = profile
        return profile

    def get_all(self) -> list[StudentProfile]:
        """Return all stored profiles."""
        return list(self._students.values())

    def count(self) -> int:
        """Return number of stored profiles."""
        return len(self._students)


_store = StudentStore()


def get_student_store() -> StudentStore:
    """FastAPI dependency returning the singleton StudentStore."""
    return _store


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/profile",
    response_model=StudentProfile,
    status_code=status.HTTP_201_CREATED,
)
def create_profile(
    body: CreateProfileRequest,
    store: StudentStore = Depends(get_student_store),
) -> StudentProfile:
    """Create a new student profile during onboarding."""
    student_id = uuid.uuid4().hex[:12]
    blackout_periods = [
        DateRange(start=bp.start, end=bp.end) for bp in body.blackout_periods
    ]
    profile = StudentProfile(
        student_id=student_id,
        name=body.name,
        email=body.email,
        skills=body.skills,
        availability_hours_per_week=body.availability_hours_per_week,
        preferred_times=body.preferred_times,
        blackout_periods=blackout_periods,
        timezone=body.timezone,
        github_username=body.github_username,
        google_email=body.google_email,
        trello_id=body.trello_username,
        onboarded_at=datetime.now(timezone.utc),
    )
    return store.create(profile)


@router.get("/profile/{student_id}", response_model=StudentProfile)
def get_profile(
    student_id: str,
    store: StudentStore = Depends(get_student_store),
) -> StudentProfile:
    """Retrieve a student profile by ID."""
    profile = store.get(student_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id}' not found",
        )
    return profile


@router.put(
    "/profile/{student_id}/availability",
    response_model=AvailabilityUpdateResponse,
)
async def update_availability(
    student_id: str,
    body: AvailabilityUpdateRequest,
    request: Request,
    store: StudentStore = Depends(get_student_store),
) -> AvailabilityUpdateResponse:
    """Update a student's availability mid-project.

    Detects significant changes (>=30% reduction in hours) and triggers
    re-delegation: searches for the project containing this student, runs
    the delegation agent + equity evaluator on the updated state, persists
    the result, and broadcasts a ``redelegation_triggered`` event over
    WebSocket.
    """
    profile = store.get(student_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id}' not found",
        )

    old_hours = profile.availability_hours_per_week
    new_hours = body.availability_hours_per_week
    old_blackouts = profile.blackout_periods
    new_blackouts = [
        DateRange(start=bp.start, end=bp.end) for bp in body.blackout_periods
    ]

    # Significance check: >=30% reduction in hours
    hours_reduction = (old_hours - new_hours) / max(old_hours, 1.0)
    triggered_redelegation = hours_reduction >= 0.30

    # Create audit record
    change_record = AvailabilityChange(
        student_id=student_id,
        timestamp=datetime.now(timezone.utc),
        old_hours=old_hours,
        new_hours=new_hours,
        old_blackouts=old_blackouts,
        new_blackouts=new_blackouts,
        triggered_redelegation=triggered_redelegation,
    )

    # Update the StudentStore profile
    updated_profile = profile.model_copy(
        update={
            "availability_hours_per_week": new_hours,
            "blackout_periods": new_blackouts,
        }
    )
    store.update(student_id, updated_profile)

    # Re-delegation side-effect (only on significant change)
    if triggered_redelegation:
        await _trigger_redelegation(request, student_id, updated_profile, change_record)

    return AvailabilityUpdateResponse(
        profile=updated_profile,
        triggered_redelegation=triggered_redelegation,
        change_record=change_record,
    )


async def _trigger_redelegation(
    request: Request,
    student_id: str,
    updated_profile: StudentProfile,
    change_record: AvailabilityChange,
) -> None:
    """Find the student's project, re-run delegation + equity, broadcast event.

    Silent no-op if the student is not part of any project in the state store.
    Errors during agent invocation are logged but do not propagate, so the
    availability update itself always succeeds.
    """
    state_store = getattr(request.app.state, "state_store", None)
    if state_store is None:
        logger.warning("No state_store on app.state; skipping re-delegation")
        return

    project_ids = await state_store.list_projects()
    target_id: str | None = None
    target_state: SyncUpState | None = None
    for pid in project_ids:
        s = await state_store.get(pid)
        if s is not None and any(sp.student_id == student_id for sp in s.student_profiles):
            target_id = pid
            target_state = s
            break

    if target_id is None or target_state is None:
        logger.info("Student %s not in any project; no re-delegation", student_id)
        return

    # Replace the student profile and append the audit record
    new_profiles = [
        updated_profile if sp.student_id == student_id else sp
        for sp in target_state.student_profiles
    ]
    affected_task_ids = [
        tid for tid, sid in target_state.delegation_matrix.items() if sid == student_id
    ]
    pre_state = target_state.model_copy(
        update={
            "student_profiles": new_profiles,
            "availability_updates": list(target_state.availability_updates) + [change_record],
            # Reset delegation so the agent re-runs cleanly
            "delegation_matrix": {},
        }
    )

    try:
        from agents.delegation import delegation as delegation_fn
        from evaluators.equity_evaluator import equity_evaluator as equity_fn

        delegation_result = await asyncio.to_thread(delegation_fn, pre_state)
        post_delegation = pre_state.model_copy(update=delegation_result)
        equity_result = await asyncio.to_thread(equity_fn, post_delegation)
        final_state = post_delegation.model_copy(update=equity_result)
        await state_store.save(target_id, final_state)
    except Exception as exc:
        logger.exception("Re-delegation failed for project %s: %s", target_id, exc)
        return

    try:
        from api.websockets import manager as ws_manager

        await ws_manager.broadcast(
            target_id,
            {
                "event": "redelegation_triggered",
                "data": {
                    "student_id": student_id,
                    "project_id": target_id,
                    "affected_task_ids": affected_task_ids,
                },
            },
        )
    except Exception:
        logger.exception("WebSocket broadcast failed for re-delegation")


@router.get(
    "/project/{project_id}/ready",
    response_model=ProjectReadinessResponse,
)
def check_project_readiness(
    project_id: str,
    store: StudentStore = Depends(get_student_store),
) -> ProjectReadinessResponse:
    """Check if all expected students have completed onboarding.

    Verifies that each profile has: name, email, at least one skill,
    and availability > 0.
    """
    students = store.get_all()
    missing_fields: dict[str, list[str]] = {}

    for student in students:
        missing: list[str] = []
        if not student.name:
            missing.append("name")
        if not student.email:
            missing.append("email")
        if not student.skills:
            missing.append("skills")
        if student.availability_hours_per_week <= 0:
            missing.append("availability_hours_per_week")
        if missing:
            missing_fields[student.student_id] = missing

    all_ready = len(students) > 0 and len(missing_fields) == 0

    return ProjectReadinessResponse(
        project_id=project_id,
        ready=all_ready,
        total_expected=len(students),
        onboarded_count=len(students) - len(missing_fields),
        missing_fields=missing_fields,
    )


# ---------------------------------------------------------------------------
# OAuth stubs
# ---------------------------------------------------------------------------


@router.post("/oauth/{provider}", response_model=OAuthCallbackResponse)
def oauth_callback(
    provider: str,
    body: OAuthCallbackRequest,
) -> OAuthCallbackResponse:
    """Stub for OAuth callback. Stores a placeholder token.

    Real OAuth redirect flows and token storage are added in a later phase.
    """
    valid_providers = {"google", "github", "trello"}
    if provider not in valid_providers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown OAuth provider '{provider}'. Must be one of: {', '.join(sorted(valid_providers))}",
        )

    return OAuthCallbackResponse(
        provider=provider,
        linked=True,
        message=f"OAuth stub: {provider} linked with placeholder token (code={body.code[:8]}...)",
    )
