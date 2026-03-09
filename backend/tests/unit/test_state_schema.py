"""Tests for the SyncUp state schema models and enums."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from state.schema import (
    AvailabilityChange,
    BurnDownTarget,
    ContributionRecord,
    DateRange,
    EventType,
    Intervention,
    MeetingRecord,
    Milestone,
    PeerReview,
    ProjectTimeline,
    RawMetrics,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
    UrgencyLevel,
)

NOW = datetime(2026, 3, 8, 12, 0, 0)
LATER = NOW + timedelta(days=7)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify enum members and string values."""

    def test_urgency_level_members(self) -> None:
        assert set(UrgencyLevel) == {
            UrgencyLevel.CRITICAL,
            UrgencyLevel.HIGH,
            UrgencyLevel.MEDIUM,
            UrgencyLevel.LOW,
        }

    def test_urgency_level_values(self) -> None:
        assert UrgencyLevel.CRITICAL.value == "critical"
        assert UrgencyLevel.LOW.value == "low"

    def test_task_status_members(self) -> None:
        assert set(TaskStatus) == {
            TaskStatus.TODO,
            TaskStatus.IN_PROGRESS,
            TaskStatus.REVIEW,
            TaskStatus.DONE,
        }

    def test_event_type_members(self) -> None:
        assert set(EventType) == {
            EventType.COMMIT,
            EventType.DOC_EDIT,
            EventType.CARD_MOVE,
            EventType.PR_REVIEW,
        }


# ---------------------------------------------------------------------------
# DateRange validation
# ---------------------------------------------------------------------------


class TestDateRange:
    """DateRange model validator tests."""

    def test_valid_range(self) -> None:
        dr = DateRange(start=NOW, end=LATER)
        assert dr.start == NOW
        assert dr.end == LATER

    def test_start_equals_end_raises(self) -> None:
        with pytest.raises(ValidationError, match="start must be before end"):
            DateRange(start=NOW, end=NOW)

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValidationError, match="start must be before end"):
            DateRange(start=LATER, end=NOW)


# ---------------------------------------------------------------------------
# Sub-model instantiation
# ---------------------------------------------------------------------------


class TestSubModels:
    """Verify all sub-models instantiate with valid data."""

    def test_raw_metrics_defaults(self) -> None:
        rm = RawMetrics()
        assert rm.lines_added == 0
        assert rm.commits_count == 0

    def test_milestone(self) -> None:
        m = Milestone(name="Alpha", target_date=NOW)
        assert m.status == TaskStatus.TODO

    def test_burn_down_target(self) -> None:
        bdt = BurnDownTarget(date=NOW, target_hours_remaining=100.0)
        assert bdt.actual_hours_remaining is None

    def test_task_defaults(self) -> None:
        t = Task(id="t1", title="Setup repo")
        assert t.status == TaskStatus.TODO
        assert t.urgency == UrgencyLevel.MEDIUM
        assert t.assigned_to is None
        assert t.dependencies == []

    def test_student_profile(self) -> None:
        sp = StudentProfile(student_id="s1", name="Alice", email="a@b.com")
        assert sp.skills == {}
        assert sp.timezone == "UTC"
        assert sp.github_username == ""

    def test_meeting_record(self) -> None:
        mr = MeetingRecord(date=NOW)
        assert mr.attendees == []
        assert mr.action_items == []

    def test_intervention(self) -> None:
        iv = Intervention(
            target_student_id="s1",
            trigger_reason="low activity",
            message_text="Please check in.",
            timestamp=NOW,
        )
        assert iv.outcome == ""

    def test_peer_review(self) -> None:
        pr = PeerReview(
            reviewer_id="s1", reviewee_id="s2", submitted_at=NOW
        )
        assert pr.ratings == {}

    def test_availability_change(self) -> None:
        ac = AvailabilityChange(
            student_id="s1",
            timestamp=NOW,
            old_hours=10.0,
            new_hours=5.0,
        )
        assert ac.triggered_redelegation is False

    def test_project_timeline_defaults(self) -> None:
        pt = ProjectTimeline()
        assert pt.milestones == []
        assert pt.burn_down_targets == []
        assert pt.buffer_periods == []


# ---------------------------------------------------------------------------
# ContributionRecord constraints
# ---------------------------------------------------------------------------


class TestContributionRecord:
    """Validate semantic_quality_score bounds."""

    def _make(self, score: float) -> ContributionRecord:
        return ContributionRecord(
            student_id="s1",
            timestamp=NOW,
            event_type=EventType.COMMIT,
            semantic_quality_score=score,
        )

    def test_valid_score_zero(self) -> None:
        assert self._make(0.0).semantic_quality_score == 0.0

    def test_valid_score_one(self) -> None:
        assert self._make(1.0).semantic_quality_score == 1.0

    def test_valid_score_mid(self) -> None:
        assert self._make(0.75).semantic_quality_score == 0.75

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(-0.1)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(1.1)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Models with required fields should raise ValidationError if omitted."""

    def test_task_missing_id(self) -> None:
        with pytest.raises(ValidationError):
            Task(title="No id")  # type: ignore[call-arg]

    def test_task_missing_title(self) -> None:
        with pytest.raises(ValidationError):
            Task(id="t1")  # type: ignore[call-arg]

    def test_student_profile_missing_email(self) -> None:
        with pytest.raises(ValidationError):
            StudentProfile(student_id="s1", name="Alice")  # type: ignore[call-arg]

    def test_contribution_record_missing_event_type(self) -> None:
        with pytest.raises(ValidationError):
            ContributionRecord(  # type: ignore[call-arg]
                student_id="s1",
                timestamp=NOW,
                semantic_quality_score=0.5,
            )


# ---------------------------------------------------------------------------
# SyncUpState
# ---------------------------------------------------------------------------


class TestSyncUpState:
    """Top-level state default construction."""

    def test_default_construction(self) -> None:
        state = SyncUpState()
        assert state.project_id == ""
        assert state.task_array == []
        assert state.contribution_ledger == []
        assert state.meeting_log == []
        assert state.intervention_history == []
        assert state.availability_updates == []
        assert state.delegation_matrix == {}
        assert isinstance(state.project_timeline, ProjectTimeline)

    def test_partial_construction(self) -> None:
        state = SyncUpState(project_id="p1", project_brief="A test project")
        assert state.project_id == "p1"
        assert state.task_array == []
