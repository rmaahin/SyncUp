"""SyncUp state schema — all Pydantic models, enums, and the top-level SyncUpState."""

from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UrgencyLevel(str, Enum):
    """Task urgency classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Task lifecycle status."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"


class EventType(str, Enum):
    """Contribution event types tracked by the progress-tracking agent."""

    COMMIT = "commit"
    DOC_EDIT = "doc_edit"
    CARD_MOVE = "card_move"
    PR_REVIEW = "pr_review"


# ---------------------------------------------------------------------------
# Sub-models (dependency order)
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    """A start/end datetime range. Validates start < end."""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _start_before_end(self) -> DateRange:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


class RawMetrics(BaseModel):
    """Raw contribution metrics attached to each ContributionRecord."""

    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0
    commits_count: int = 0


class Milestone(BaseModel):
    """A named project milestone."""

    name: str
    target_date: datetime
    status: TaskStatus = TaskStatus.TODO
    description: str = ""


class BurnDownTarget(BaseModel):
    """A single data point on the burn-down curve."""

    date: datetime
    target_hours_remaining: float
    actual_hours_remaining: Optional[float] = None


class Task(BaseModel):
    """A decomposed unit of work within a project."""

    id: str
    title: str
    description: str = ""
    effort_hours: float = 0.0
    required_skills: list[str] = Field(default_factory=list)
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    dependencies: list[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    status: TaskStatus = TaskStatus.TODO


class StudentProfile(BaseModel):
    """A student's profile including skills, availability, and integration IDs."""

    student_id: str
    name: str
    email: str
    skills: dict[str, float] = Field(default_factory=dict)
    availability_hours_per_week: float = 0.0
    preferred_times: list[str] = Field(default_factory=list)
    blackout_periods: list[DateRange] = Field(default_factory=list)
    timezone: str = "UTC"
    github_username: str = ""
    google_email: str = ""
    trello_id: str = ""
    onboarded_at: Optional[datetime] = None


class ContributionRecord(BaseModel):
    """An immutable record of a student contribution event."""

    student_id: str
    timestamp: datetime
    event_type: EventType
    description: str = ""
    semantic_quality_score: float = Field(ge=0.0, le=1.0)
    raw_metrics: RawMetrics = Field(default_factory=RawMetrics)
    performed_by: str = ""  # Who actually performed the action (e.g. Trello memberCreator)


class MeetingRecord(BaseModel):
    """Record of a team meeting."""

    date: datetime
    attendees: list[str] = Field(default_factory=list)
    agenda: str = ""
    notes: str = ""
    action_items: list[str] = Field(default_factory=list)


class Intervention(BaseModel):
    """A conflict-resolution intervention directed at a student."""

    target_student_id: str
    trigger_reason: str
    message_text: str
    timestamp: datetime
    outcome: str = ""


class PeerReview(BaseModel):
    """A single peer-review submission."""

    reviewer_id: str
    reviewee_id: str
    ratings: dict[str, int] = Field(default_factory=dict)
    comments: dict[str, str] = Field(default_factory=dict)
    submitted_at: datetime


class PeerReviewSummary(BaseModel):
    """Aggregated peer-review statistics for one student."""

    student_id: str
    avg_per_dimension: dict[str, float] = Field(default_factory=dict)
    overall_avg: float = 0.0
    std_dev_per_dimension: dict[str, float] = Field(default_factory=dict)
    review_count: int = 0


class BiasFlag(BaseModel):
    """A detected anomaly in peer-review patterns."""

    flag_type: str  # "outlier_reviewer" | "targeted_low" | "inflation" | "retaliation"
    reviewer_id: str
    reviewee_id: Optional[str] = None
    description: str
    severity: str  # "low" | "medium" | "high"


class StudentReport(BaseModel):
    """Per-student section of the final TeamReport."""

    student_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    peer_summary: Optional[PeerReviewSummary] = None
    peer_bias_flags: list[BiasFlag] = Field(default_factory=list)
    narrative: str = ""
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)


class TeamReport(BaseModel):
    """Final synthesized end-of-project report."""

    project_id: str
    completion_pct: float = 0.0
    team_metrics: dict[str, Any] = Field(default_factory=dict)
    student_reports: list[StudentReport] = Field(default_factory=list)
    bias_flags: list[BiasFlag] = Field(default_factory=list)
    team_narrative: str = ""
    generated_at: datetime


class AvailabilityChange(BaseModel):
    """Audit record for a student availability update."""

    student_id: str
    timestamp: datetime
    old_hours: float
    new_hours: float
    old_blackouts: list[DateRange] = Field(default_factory=list)
    new_blackouts: list[DateRange] = Field(default_factory=list)
    triggered_redelegation: bool = False


class DraftIntervention(BaseModel):
    """A draft conflict-resolution message awaiting tone evaluation."""

    target_student_id: str
    message: str
    suggested_action: str = ""  # "extend_deadline", "redistribute_task", "schedule_check_in", "no_action"
    severity: str = "low"  # "low", "medium", "high"
    affected_teammates: list[str] = Field(default_factory=list)


class ToneResult(BaseModel):
    """Structured output from the Tone Evaluator (LLM-as-a-Judge)."""

    classification: str  # "constructive" or "punitive"
    reasoning: str = ""
    flagged_phrases: list[str] = Field(default_factory=list)


class EquityResult(BaseModel):
    """Structured output from the Workload Equity Evaluator."""

    balanced: bool
    reasoning: str = ""
    violations: list[str] = Field(default_factory=list)


class ProjectTimeline(BaseModel):
    """Project-level timeline data."""

    milestones: list[Milestone] = Field(default_factory=list)
    burn_down_targets: list[BurnDownTarget] = Field(default_factory=list)
    buffer_periods: list[DateRange] = Field(default_factory=list)


class PublishingStatus(BaseModel):
    """Tracks the outcome of each publishing target."""

    trello: str = "pending"
    calendar: str = "pending"
    docs: str = "pending"
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level LangGraph state
# ---------------------------------------------------------------------------


class SyncUpState(BaseModel):
    """Root state object flowing through the LangGraph graph.

    All fields have defaults so nodes can return partial dicts.
    Append-only fields use ``Annotated[list[X], operator.add]`` so that
    returning ``[new_item]`` appends rather than replaces.
    """

    project_id: str = ""
    project_brief: str = ""
    final_deadline: Optional[datetime] = None

    task_array: list[Task] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    student_profiles: list[StudentProfile] = Field(default_factory=list)
    delegation_matrix: dict[str, str] = Field(default_factory=dict)
    equity_result: Optional[EquityResult] = None
    equity_retries: int = 0

    # Append-only ledgers (LangGraph reducer = operator.add)
    contribution_ledger: Annotated[list[ContributionRecord], operator.add] = Field(
        default_factory=list
    )
    meeting_log: Annotated[list[MeetingRecord], operator.add] = Field(
        default_factory=list
    )
    intervention_history: Annotated[list[Intervention], operator.add] = Field(
        default_factory=list
    )
    availability_updates: Annotated[list[AvailabilityChange], operator.add] = Field(
        default_factory=list
    )

    peer_review_data: list[PeerReview] = Field(default_factory=list)
    peer_review_forms_generated: bool = False
    peer_review_form_template: dict[str, Any] = Field(default_factory=dict)
    team_report: Optional[TeamReport] = None
    project_timeline: ProjectTimeline = Field(default_factory=ProjectTimeline)

    # Publishing fields
    project_name: str = ""
    github_repo: str = ""  # e.g. "owner/repo" — routes GitHub webhooks to this project
    trello_board_id: Optional[str] = None
    trello_card_mapping: dict[str, str] = Field(default_factory=dict)
    calendar_event_mapping: dict[str, str] = Field(default_factory=dict)
    docs_task_matrix_id: Optional[str] = None
    webhook_configured: bool = False
    publishing_status: Optional[PublishingStatus] = None

    # Progress tracking fields
    pending_event: Optional[dict[str, Any]] = None
    student_progress: dict[str, str] = Field(default_factory=dict)

    # Conflict resolution / tone evaluation fields
    draft_intervention: Optional[DraftIntervention] = None
    tone_result: Optional[ToneResult] = None
    tone_rewrite_count: int = 0
    needs_redelegation: list[str] = Field(default_factory=list)  # task_ids flagged for re-delegation

    # Meeting coordinator fields
    next_meeting_scheduled: Optional[datetime] = None
    meeting_notes_doc_ids: list[str] = Field(default_factory=list)
    meeting_interval_days: int = 7
    meeting_mode: Optional[str] = None  # "schedule" | "ingest"
