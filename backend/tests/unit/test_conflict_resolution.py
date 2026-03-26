"""Unit tests for the Conflict Resolution agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from agents.conflict_resolution import (
    _build_user_prompt,
    _find_first_behind_student,
    _gather_context,
    conflict_resolution,
)
from state.schema import (
    DateRange,
    DraftIntervention,
    Intervention,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
    ToneResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
_DEADLINE_PAST = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)  # 5 days ago
_DEADLINE_FUTURE = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)

_CONSTRUCTIVE_RESPONSE = json.dumps(
    {
        "message": "Hi Bob, I noticed the API endpoints task is a few days overdue. "
        "Are you running into any blockers? Let's set up a quick check-in to see "
        "how the team can help. Alice is waiting on this for the frontend integration.",
        "suggested_action": "schedule_check_in",
        "severity": "medium",
        "affected_teammates": ["Alice"],
    }
)

_MALFORMED_RESPONSE = "This is not valid JSON at all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_student(
    student_id: str = "s-1",
    name: str = "Alice",
    blackout_periods: list[DateRange] | None = None,
) -> StudentProfile:
    """Create a minimal StudentProfile for testing."""
    return StudentProfile(
        student_id=student_id,
        name=name,
        email=f"{name.lower()}@example.com",
        skills={"python": 0.8},
        availability_hours_per_week=10.0,
        github_username=f"{name.lower()}-dev",
        blackout_periods=blackout_periods or [],
    )


def _make_task(
    task_id: str = "task-1",
    title: str = "Build API endpoints",
    status: TaskStatus = TaskStatus.TODO,
    deadline: datetime | None = None,
    assigned_to: str | None = None,
    dependencies: list[str] | None = None,
) -> Task:
    """Create a minimal Task for testing."""
    return Task(
        id=task_id,
        title=title,
        status=status,
        deadline=deadline or _DEADLINE_PAST,
        effort_hours=10.0,
        assigned_to=assigned_to,
        dependencies=dependencies or [],
    )


def _make_state(
    student_progress: dict[str, str] | None = None,
    students: list[StudentProfile] | None = None,
    tasks: list[Task] | None = None,
    delegation_matrix: dict[str, str] | None = None,
    intervention_history: list[Intervention] | None = None,
    dependency_graph: dict[str, list[str]] | None = None,
    tone_result: ToneResult | None = None,
    tone_rewrite_count: int = 0,
) -> SyncUpState:
    """Create a SyncUpState for testing."""
    return SyncUpState(
        project_id="test-project",
        student_progress=student_progress or {},
        student_profiles=students or [_make_student(), _make_student("s-2", "Bob")],
        task_array=tasks or [],
        delegation_matrix=delegation_matrix or {},
        intervention_history=intervention_history or [],
        dependency_graph=dependency_graph or {},
        tone_result=tone_result,
        tone_rewrite_count=tone_rewrite_count,
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns the given responses in order."""
    llm = MagicMock()
    side_effects: list[MagicMock] = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        side_effects.append(msg)
    llm.invoke.side_effect = side_effects
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindFirstBehindStudent:
    """Tests for _find_first_behind_student."""

    def test_no_behind(self) -> None:
        state = _make_state(student_progress={"s-1": "on_track", "s-2": "at_risk"})
        assert _find_first_behind_student(state) is None

    def test_one_behind(self) -> None:
        state = _make_state(student_progress={"s-1": "on_track", "s-2": "behind"})
        assert _find_first_behind_student(state) == "s-2"


class TestGatherContext:
    """Tests for _gather_context."""

    def test_overdue_tasks_collected(self) -> None:
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
        )
        ctx = _gather_context("s-2", state, NOW)
        assert len(ctx["overdue_tasks"]) == 1
        assert ctx["overdue_tasks"][0]["days_overdue"] == 5

    def test_blocked_teammates_found(self) -> None:
        tasks = [
            _make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2"),
            _make_task(
                "t2",
                title="Frontend",
                deadline=_DEADLINE_FUTURE,
                assigned_to="s-1",
                dependencies=["t1"],
            ),
        ]
        state = _make_state(
            tasks=tasks,
            delegation_matrix={"t1": "s-2", "t2": "s-1"},
        )
        ctx = _gather_context("s-2", state, NOW)
        assert "Alice" in ctx["blocked_teammates"]


class TestConflictResolutionNode:
    """Tests for the conflict_resolution agent node function."""

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_no_behind_students(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """No behind students → return draft_intervention=None, no LLM call."""
        state = _make_state(student_progress={"s-1": "on_track", "s-2": "on_track"})
        result = conflict_resolution(state)
        assert result["draft_intervention"] is None
        mock_llm_factory.assert_not_called()

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_happy_path_generates_draft(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """One behind student → generates valid DraftIntervention."""
        mock_llm_factory.return_value = _make_mock_llm([_CONSTRUCTIVE_RESPONSE])

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            student_progress={"s-1": "on_track", "s-2": "behind"},
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
        )
        result = conflict_resolution(state)

        draft = result["draft_intervention"]
        assert isinstance(draft, DraftIntervention)
        assert draft.target_student_id == "s-2"
        assert draft.suggested_action == "schedule_check_in"
        assert draft.severity == "medium"
        assert "Alice" in draft.affected_teammates

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_calendar_blackout_biases_extend_deadline(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """Student has active blackout → suggested_action biased to extend_deadline."""
        response = json.dumps(
            {
                "message": "Hi Bob, we see you have exams right now.",
                "suggested_action": "schedule_check_in",
                "severity": "low",
                "affected_teammates": [],
            }
        )
        mock_llm_factory.return_value = _make_mock_llm([response])

        blackout = DateRange(
            start=NOW - timedelta(days=2),
            end=NOW + timedelta(days=3),
        )
        students = [
            _make_student("s-1", "Alice"),
            _make_student("s-2", "Bob", blackout_periods=[blackout]),
        ]
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            student_progress={"s-2": "behind"},
            students=students,
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
        )
        result = conflict_resolution(state)

        draft = result["draft_intervention"]
        assert draft.suggested_action == "extend_deadline"

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_third_intervention_escalates_severity(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """2 prior interventions for same student → severity forced to 'high'."""
        response = json.dumps(
            {
                "message": "Hi Bob, this is getting urgent.",
                "suggested_action": "redistribute_task",
                "severity": "medium",
                "affected_teammates": [],
            }
        )
        mock_llm_factory.return_value = _make_mock_llm([response])

        prior = [
            Intervention(
                target_student_id="s-2",
                trigger_reason="Task overdue",
                message_text="First nudge",
                timestamp=NOW - timedelta(days=3),
            ),
            Intervention(
                target_student_id="s-2",
                trigger_reason="Task overdue",
                message_text="Second nudge",
                timestamp=NOW - timedelta(days=1),
            ),
        ]
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            student_progress={"s-2": "behind"},
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
            intervention_history=prior,
        )
        result = conflict_resolution(state)

        draft = result["draft_intervention"]
        assert draft.severity == "high"

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_blocked_teammates_identified(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """Dependency graph shows teammate blocked → affected_teammates populated."""
        response = json.dumps(
            {
                "message": "Hi Bob, the API task needs attention.",
                "suggested_action": "schedule_check_in",
                "severity": "medium",
                "affected_teammates": ["Alice"],
            }
        )
        mock_llm_factory.return_value = _make_mock_llm([response])

        tasks = [
            _make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2"),
            _make_task(
                "t2",
                title="Frontend",
                deadline=_DEADLINE_FUTURE,
                assigned_to="s-1",
                dependencies=["t1"],
            ),
        ]
        state = _make_state(
            student_progress={"s-2": "behind"},
            tasks=tasks,
            delegation_matrix={"t1": "s-2", "t2": "s-1"},
        )
        result = conflict_resolution(state)

        draft = result["draft_intervention"]
        assert "Alice" in draft.affected_teammates

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_retry_on_malformed_json(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """First LLM call returns garbage, second returns valid JSON."""
        mock_llm_factory.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _CONSTRUCTIVE_RESPONSE]
        )

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            student_progress={"s-2": "behind"},
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
        )
        result = conflict_resolution(state)

        draft = result["draft_intervention"]
        assert isinstance(draft, DraftIntervention)
        # Verify 2 LLM invoke calls
        llm = mock_llm_factory.return_value
        assert llm.invoke.call_count == 2

    @patch("agents.conflict_resolution._get_now", return_value=NOW)
    @patch("agents.conflict_resolution.get_high_tier_llm")
    def test_rewrite_with_tone_feedback(
        self, mock_llm_factory: MagicMock, mock_now: MagicMock
    ) -> None:
        """State has punitive tone_result → prompt includes rewrite instructions."""
        mock_llm = _make_mock_llm([_CONSTRUCTIVE_RESPONSE])
        mock_llm_factory.return_value = mock_llm

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        tone = ToneResult(
            classification="punitive",
            reasoning="Uses blame language",
            flagged_phrases=["you failed to", "unacceptable"],
        )
        state = _make_state(
            student_progress={"s-2": "behind"},
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
            tone_result=tone,
            tone_rewrite_count=1,
        )
        result = conflict_resolution(state)

        # Verify draft was generated
        assert isinstance(result["draft_intervention"], DraftIntervention)

        # Verify the prompt included rewrite feedback
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "REWRITE REQUEST" in user_msg
        assert "you failed to" in user_msg
        assert "unacceptable" in user_msg
