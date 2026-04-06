"""Tests for the guardrails state validator — integrity rules on state mutations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from state.schema import (
    AvailabilityChange,
    ContributionRecord,
    EventType,
    Intervention,
    MeetingRecord,
    RawMetrics,
    StudentProfile,
    SyncUpState,
    Task,
    UrgencyLevel,
)
from guardrails.state_validator import sanitize_state_update, validate_state_update


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)
_FINAL = datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc)


@pytest.fixture()
def base_state() -> SyncUpState:
    """State with 2 students, 3 tasks, delegation matrix, and a final deadline."""
    return SyncUpState(
        project_id="proj-001",
        project_brief="Build a web app",
        final_deadline=_FINAL,
        task_array=[
            Task(id="t1", title="Backend API", urgency=UrgencyLevel.HIGH),
            Task(id="t2", title="Frontend UI", urgency=UrgencyLevel.MEDIUM),
            Task(id="t3", title="Tests", urgency=UrgencyLevel.LOW),
        ],
        student_profiles=[
            StudentProfile(
                student_id="s1", name="Alice", email="alice@test.com"
            ),
            StudentProfile(
                student_id="s2", name="Bob", email="bob@test.com"
            ),
        ],
        delegation_matrix={"t1": "s1", "t2": "s2"},
        contribution_ledger=[
            ContributionRecord(
                student_id="s1",
                timestamp=_NOW,
                event_type=EventType.COMMIT,
                description="Initial commit",
                semantic_quality_score=0.8,
            ),
        ],
        meeting_log=[
            MeetingRecord(date=_NOW, attendees=["s1", "s2"], agenda="Kickoff"),
        ],
    )


def _make_record(
    student_id: str = "s1",
    score: float = 0.75,
    event_type: EventType = EventType.COMMIT,
) -> ContributionRecord:
    return ContributionRecord(
        student_id=student_id,
        timestamp=_NOW,
        event_type=event_type,
        description="Test record",
        semantic_quality_score=score,
    )


# ---------------------------------------------------------------------------
# Rule A — Append-only fields
# ---------------------------------------------------------------------------


class TestAppendOnly:
    """Append-only fields must not lose existing records."""

    def test_append_new_records_valid(self, base_state: SyncUpState) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 0.6),
            ]
        }
        valid, violations = validate_state_update(base_state, update)
        assert valid is True
        assert violations == []

    def test_shorter_list_rejected(self, base_state: SyncUpState) -> None:
        # Replace 1-item ledger with empty list
        update = {"contribution_ledger": []}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("Append-only" in v for v in violations)

    def test_empty_append_valid(self, base_state: SyncUpState) -> None:
        # Not including the field at all is fine
        update = {"project_name": "New Name"}
        valid, violations = validate_state_update(base_state, update)
        assert valid is True

    def test_non_list_type_rejected(self, base_state: SyncUpState) -> None:
        update = {"contribution_ledger": "not a list"}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("must be a list" in v for v in violations)

    def test_meeting_log_append_valid(self, base_state: SyncUpState) -> None:
        update = {
            "meeting_log": [
                *base_state.meeting_log,
                MeetingRecord(date=_NOW + timedelta(days=7), attendees=["s1"]),
            ]
        }
        valid, _ = validate_state_update(base_state, update)
        assert valid is True

    def test_meeting_log_truncation_rejected(self, base_state: SyncUpState) -> None:
        update = {"meeting_log": []}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False


# ---------------------------------------------------------------------------
# Rule B — Delegation integrity
# ---------------------------------------------------------------------------


class TestDelegationIntegrity:
    """delegation_matrix must reference valid task_ids and student_ids."""

    def test_valid_delegation_passes(self, base_state: SyncUpState) -> None:
        update = {"delegation_matrix": {"t1": "s1", "t2": "s2", "t3": "s1"}}
        valid, _ = validate_state_update(base_state, update)
        assert valid is True

    def test_unknown_task_id_rejected(self, base_state: SyncUpState) -> None:
        update = {"delegation_matrix": {"t1": "s1", "t-nonexistent": "s2"}}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("nonexistent task_id" in v for v in violations)

    def test_unknown_student_id_rejected(self, base_state: SyncUpState) -> None:
        update = {"delegation_matrix": {"t1": "s999"}}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("nonexistent student_id" in v for v in violations)

    def test_new_task_in_same_update_accepted(self, base_state: SyncUpState) -> None:
        new_task = Task(id="t4", title="Deploy")
        update = {
            "task_array": [*base_state.task_array, new_task],
            "delegation_matrix": {"t4": "s1"},
        }
        valid, _ = validate_state_update(base_state, update)
        assert valid is True


# ---------------------------------------------------------------------------
# Rule C — Score bounds
# ---------------------------------------------------------------------------


class TestScoreBounds:
    """semantic_quality_score must be in [0.0, 1.0]."""

    def test_valid_score_passes(self, base_state: SyncUpState) -> None:
        update = {"contribution_ledger": [_make_record(score=0.75)]}
        # Length check: this is a new list — needs to be >= current length
        # to pass append-only.  Wrap it in full ledger:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record(score=0.75),
            ]
        }
        valid, violations = validate_state_update(base_state, update)
        assert valid is True

    def test_score_at_boundaries_passes(self, base_state: SyncUpState) -> None:
        for score in [0.0, 1.0]:
            record = ContributionRecord(
                student_id="s1",
                timestamp=_NOW,
                event_type=EventType.COMMIT,
                description="test",
                semantic_quality_score=score,
            )
            update = {
                "contribution_ledger": [*base_state.contribution_ledger, record]
            }
            valid, _ = validate_state_update(base_state, update)
            assert valid is True, f"Score {score} should be valid"

    def test_score_above_1_rejected(self, base_state: SyncUpState) -> None:
        # Cannot create a ContributionRecord with score > 1.0 due to Pydantic
        # validation, so use a dict to bypass:
        update = {
            "contribution_ledger": [
                {
                    "student_id": "s1",
                    "timestamp": _NOW,
                    "event_type": "commit",
                    "semantic_quality_score": 1.5,
                }
            ]
        }
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("out of bounds" in v for v in violations)

    def test_score_below_0_rejected(self, base_state: SyncUpState) -> None:
        update = {
            "contribution_ledger": [
                {
                    "student_id": "s1",
                    "timestamp": _NOW,
                    "event_type": "commit",
                    "semantic_quality_score": -0.1,
                }
            ]
        }
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("out of bounds" in v for v in violations)


# ---------------------------------------------------------------------------
# Rule D — Deadline sanity
# ---------------------------------------------------------------------------


class TestDeadlineSanity:
    """No task deadline after final_deadline."""

    def test_valid_deadline_passes(self, base_state: SyncUpState) -> None:
        task = Task(
            id="t4",
            title="New Task",
            deadline=_FINAL - timedelta(days=1),
        )
        update = {"task_array": [task]}
        valid, _ = validate_state_update(base_state, update)
        assert valid is True

    def test_deadline_after_final_rejected(self, base_state: SyncUpState) -> None:
        task = Task(
            id="t4",
            title="Late Task",
            deadline=_FINAL + timedelta(days=1),
        )
        update = {"task_array": [task]}
        valid, violations = validate_state_update(base_state, update)
        assert valid is False
        assert any("after final_deadline" in v for v in violations)

    def test_no_final_deadline_any_task_deadline_valid(self) -> None:
        state = SyncUpState(project_id="proj-002", final_deadline=None)
        task = Task(
            id="t1",
            title="Whenever",
            deadline=datetime(2099, 12, 31, tzinfo=timezone.utc),
        )
        update = {"task_array": [task]}
        valid, _ = validate_state_update(state, update)
        assert valid is True

    def test_task_with_no_deadline_valid(self, base_state: SyncUpState) -> None:
        task = Task(id="t4", title="No Deadline")
        update = {"task_array": [task]}
        valid, _ = validate_state_update(base_state, update)
        assert valid is True


# ---------------------------------------------------------------------------
# Rule E — Cross-student protection
# ---------------------------------------------------------------------------


class TestCrossStudentProtection:
    """Student A's webhook cannot modify student B's data."""

    def test_student_modifies_own_data_valid(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 0.7),
            ]
        }
        valid, _ = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is True

    def test_student_modifies_other_student_rejected(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 0.9),
            ]
        }
        valid, violations = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is False
        assert any("Cross-student" in v for v in violations)

    def test_supervisor_can_modify_any_student(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 0.9),
            ]
        }
        valid, _ = validate_state_update(
            base_state,
            update,
            source_agent="supervisor",
            source_student_id="s1",
        )
        assert valid is True

    def test_delegation_agent_can_modify_any(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 0.5),
            ]
        }
        valid, _ = validate_state_update(
            base_state,
            update,
            source_agent="delegation",
            source_student_id="s1",
        )
        assert valid is True

    def test_no_source_student_no_restriction(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 0.6),
            ]
        }
        valid, _ = validate_state_update(base_state, update)
        assert valid is True

    def test_cross_student_availability_rejected(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "availability_updates": [
                AvailabilityChange(
                    student_id="s2",
                    timestamp=_NOW,
                    old_hours=10.0,
                    new_hours=5.0,
                )
            ]
        }
        valid, violations = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is False
        assert any("Cross-student" in v for v in violations)


# ---------------------------------------------------------------------------
# Rule F — Self-score prevention
# ---------------------------------------------------------------------------


class TestSelfScorePrevention:
    """Student cannot set own semantic_quality_score to 1.0."""

    def test_student_sets_own_score_to_1_rejected(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 1.0),
            ]
        }
        valid, violations = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is False
        assert any("Self-score" in v for v in violations)

    def test_student_sets_own_score_below_1_valid(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 0.95),
            ]
        }
        valid, _ = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is True

    def test_other_agent_sets_score_to_1_valid(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 1.0),
            ]
        }
        # No source_student_id → no self-score restriction
        valid, _ = validate_state_update(base_state, update)
        assert valid is True

    def test_student_sets_other_student_score_to_1_cross_student_rejected(
        self, base_state: SyncUpState
    ) -> None:
        """Setting another student's score to 1.0 — blocked by cross-student, not self-score."""
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s2", 1.0),
            ]
        }
        valid, violations = validate_state_update(
            base_state, update, source_student_id="s1"
        )
        assert valid is False
        assert any("Cross-student" in v for v in violations)


# ---------------------------------------------------------------------------
# sanitize_state_update
# ---------------------------------------------------------------------------


class TestSanitizeStateUpdate:
    """Test the fix-or-remove sanitisation path."""

    def test_clamps_score_to_bounds(self, base_state: SyncUpState) -> None:
        update = {
            "contribution_ledger": [
                {
                    "student_id": "s1",
                    "timestamp": _NOW,
                    "event_type": "commit",
                    "semantic_quality_score": 1.5,
                }
            ]
        }
        cleaned = sanitize_state_update(base_state, update)
        record = cleaned["contribution_ledger"][0]
        score = (
            record.semantic_quality_score
            if hasattr(record, "semantic_quality_score")
            else record["semantic_quality_score"]
        )
        assert score == 1.0

    def test_clamps_negative_score_to_zero(self, base_state: SyncUpState) -> None:
        update = {
            "contribution_ledger": [
                {
                    "student_id": "s1",
                    "timestamp": _NOW,
                    "event_type": "commit",
                    "semantic_quality_score": -0.5,
                }
            ]
        }
        cleaned = sanitize_state_update(base_state, update)
        record = cleaned["contribution_ledger"][0]
        score = (
            record.semantic_quality_score
            if hasattr(record, "semantic_quality_score")
            else record["semantic_quality_score"]
        )
        assert score == 0.0

    def test_clamps_self_score(self, base_state: SyncUpState) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 1.0),
            ]
        }
        cleaned = sanitize_state_update(
            base_state, update, source_student_id="s1"
        )
        # The last record (the new one) should be clamped to 0.99
        last_record = cleaned["contribution_ledger"][-1]
        score = last_record.semantic_quality_score
        assert score == pytest.approx(0.99)

    def test_removes_invalid_delegation_entries(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "delegation_matrix": {
                "t1": "s1",  # valid
                "t-bad": "s2",  # invalid task
                "t2": "s999",  # invalid student
            }
        }
        cleaned = sanitize_state_update(base_state, update)
        assert "t1" in cleaned["delegation_matrix"]
        assert "t-bad" not in cleaned["delegation_matrix"]
        assert "t2" not in cleaned["delegation_matrix"]

    def test_clamps_deadline_to_final(self, base_state: SyncUpState) -> None:
        late_task = Task(
            id="t4",
            title="Late",
            deadline=_FINAL + timedelta(days=30),
        )
        update = {"task_array": [late_task]}
        cleaned = sanitize_state_update(base_state, update)
        clamped_task = cleaned["task_array"][0]
        deadline = (
            clamped_task.deadline
            if isinstance(clamped_task, Task)
            else clamped_task["deadline"]
        )
        assert deadline == _FINAL

    def test_removes_cross_student_records(
        self, base_state: SyncUpState
    ) -> None:
        update = {
            "contribution_ledger": [
                *base_state.contribution_ledger,
                _make_record("s1", 0.7),
                _make_record("s2", 0.8),  # cross-student — should be removed
            ]
        }
        cleaned = sanitize_state_update(
            base_state, update, source_student_id="s1"
        )
        student_ids = [
            r.student_id
            if isinstance(r, ContributionRecord)
            else r["student_id"]
            for r in cleaned["contribution_ledger"]
        ]
        assert "s2" not in student_ids

    def test_drops_truncated_append_only_field(
        self, base_state: SyncUpState
    ) -> None:
        update = {"contribution_ledger": [], "project_name": "Updated"}
        cleaned = sanitize_state_update(base_state, update)
        assert "contribution_ledger" not in cleaned
        assert cleaned["project_name"] == "Updated"

    def test_preserves_valid_fields(self, base_state: SyncUpState) -> None:
        update = {
            "project_name": "New Name",
            "delegation_matrix": {"t1": "s2"},
        }
        cleaned = sanitize_state_update(base_state, update)
        assert cleaned["project_name"] == "New Name"
        assert cleaned["delegation_matrix"] == {"t1": "s2"}
