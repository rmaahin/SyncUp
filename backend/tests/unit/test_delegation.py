"""Tests for the Delegation agent — mocks the LLM."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.delegation import (
    DelegationResponse,
    _compute_skill_scores,
    _extract_json,
    _parse_response,
    _validate_assignments,
    delegation,
)
from state.schema import (
    EquityResult,
    StudentProfile,
    SyncUpState,
    Task,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_DEADLINE = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _make_task(
    tid: str,
    effort: float = 5.0,
    skills: list[str] | None = None,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    deps: list[str] | None = None,
) -> Task:
    return Task(
        id=tid,
        title=tid,
        effort_hours=effort,
        required_skills=skills or [],
        urgency=urgency,
        dependencies=deps or [],
    )


def _make_student(
    sid: str,
    skills: dict[str, float] | None = None,
    hours: float = 20.0,
) -> StudentProfile:
    return StudentProfile(
        student_id=sid,
        name=sid,
        email=f"{sid}@test.com",
        skills=skills or {},
        availability_hours_per_week=hours,
    )


def _make_state(
    tasks: list[Task] | None = None,
    students: list[StudentProfile] | None = None,
    dep_graph: dict[str, list[str]] | None = None,
    equity_retries: int = 0,
    equity_result: EquityResult | None = None,
) -> SyncUpState:
    tasks = tasks or []
    return SyncUpState(
        project_id="test",
        project_brief="Test project",
        final_deadline=_DEADLINE,
        task_array=tasks,
        dependency_graph=dep_graph or {t.id: t.dependencies for t in tasks},
        student_profiles=students or [],
        equity_retries=equity_retries,
        equity_result=equity_result,
    )


def _valid_llm_response(assignments: list[dict[str, str]]) -> str:
    return json.dumps({"assignments": assignments})


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns the given responses in sequence."""
    mock = MagicMock()
    mock_responses = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        mock_responses.append(msg)
    mock.invoke.side_effect = mock_responses
    return mock


# ---------------------------------------------------------------------------
# Skill scoring
# ---------------------------------------------------------------------------


class TestSkillScoring:
    """Tests for _compute_skill_scores."""

    def test_full_match_scores_one(self) -> None:
        tasks = [_make_task("t1", skills=["python", "sql"])]
        students = [_make_student("s1", skills={"python": 1.0, "sql": 1.0})]
        scores = _compute_skill_scores(tasks, students)
        assert scores["t1"]["s1"] == 1.0

    def test_no_match_scores_zero(self) -> None:
        tasks = [_make_task("t1", skills=["python", "sql"])]
        students = [_make_student("s1", skills={"java": 1.0})]
        scores = _compute_skill_scores(tasks, students)
        assert scores["t1"]["s1"] == 0.0

    def test_partial_match(self) -> None:
        tasks = [_make_task("t1", skills=["python", "sql"])]
        students = [_make_student("s1", skills={"python": 0.8})]
        scores = _compute_skill_scores(tasks, students)
        assert scores["t1"]["s1"] == pytest.approx(0.4)  # (0.8 + 0.0) / 2

    def test_no_required_skills_scores_one(self) -> None:
        tasks = [_make_task("t1", skills=[])]
        students = [_make_student("s1")]
        scores = _compute_skill_scores(tasks, students)
        assert scores["t1"]["s1"] == 1.0

    def test_zero_availability_excluded(self) -> None:
        tasks = [_make_task("t1", skills=["python"])]
        students = [
            _make_student("s1", skills={"python": 1.0}, hours=20.0),
            _make_student("s2", skills={"python": 1.0}, hours=0.0),
        ]
        scores = _compute_skill_scores(tasks, students)
        assert "s1" in scores["t1"]
        assert "s2" not in scores["t1"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Tests for JSON extraction and response parsing."""

    def test_extract_json_plain(self) -> None:
        raw = '{"assignments": []}'
        assert _extract_json(raw) == raw

    def test_extract_json_with_fences(self) -> None:
        raw = '```json\n{"assignments": []}\n```'
        result = _extract_json(raw)
        assert '"assignments"' in result

    def test_extract_json_with_preamble(self) -> None:
        raw = 'Here is the result:\n{"assignments": []}'
        result = _extract_json(raw)
        assert result.startswith("{")

    def test_extract_json_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("no json here")

    def test_parse_response_valid(self) -> None:
        raw = json.dumps({
            "assignments": [
                {"task_id": "t1", "student_id": "s1"},
            ]
        })
        result = _parse_response(raw)
        assert len(result.assignments) == 1
        assert result.assignments[0].task_id == "t1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateAssignments:
    """Tests for _validate_assignments."""

    def test_valid_assignments(self) -> None:
        from agents.delegation import DelegationAssignment
        resp = DelegationResponse(
            assignments=[DelegationAssignment(task_id="t1", student_id="s1")]
        )
        result = _validate_assignments(resp, {"t1"}, {"s1"})
        assert result == {"t1": "s1"}

    def test_unknown_task_raises(self) -> None:
        from agents.delegation import DelegationAssignment
        resp = DelegationResponse(
            assignments=[DelegationAssignment(task_id="t99", student_id="s1")]
        )
        with pytest.raises(ValueError, match="Unknown task_id"):
            _validate_assignments(resp, {"t1"}, {"s1"})

    def test_unknown_student_raises(self) -> None:
        from agents.delegation import DelegationAssignment
        resp = DelegationResponse(
            assignments=[DelegationAssignment(task_id="t1", student_id="s99")]
        )
        with pytest.raises(ValueError, match="Unknown student_id"):
            _validate_assignments(resp, {"t1"}, {"s1"})

    def test_duplicate_task_raises(self) -> None:
        from agents.delegation import DelegationAssignment
        resp = DelegationResponse(
            assignments=[
                DelegationAssignment(task_id="t1", student_id="s1"),
                DelegationAssignment(task_id="t1", student_id="s2"),
            ]
        )
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_assignments(resp, {"t1"}, {"s1", "s2"})


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestDelegationEmptyState:
    """Delegation with empty state returns empty dict."""

    @patch("agents.delegation.get_high_tier_llm")
    def test_empty_task_array(self, mock_llm_factory: MagicMock) -> None:
        state = _make_state(tasks=[], students=[_make_student("s1")])
        result = delegation(state)
        assert result == {}
        mock_llm_factory.assert_not_called()

    @patch("agents.delegation.get_high_tier_llm")
    def test_no_available_students(self, mock_llm_factory: MagicMock) -> None:
        state = _make_state(
            tasks=[_make_task("t1")],
            students=[_make_student("s1", hours=0.0)],
        )
        result = delegation(state)
        assert result == {}
        mock_llm_factory.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestDelegationHappyPath:
    """Delegation with valid tasks and students."""

    @patch("agents.delegation.get_high_tier_llm")
    def test_produces_delegation_matrix(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [
            _make_task("t1", 5, ["python"]),
            _make_task("t2", 5, ["sql"]),
        ]
        students = [
            _make_student("s1", {"python": 0.9}, 20),
            _make_student("s2", {"sql": 0.8}, 15),
        ]

        llm_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
            {"task_id": "t2", "student_id": "s2"},
        ])
        mock_llm = _make_mock_llm([llm_response])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        assert "delegation_matrix" in result
        assert result["delegation_matrix"] == {"t1": "s1", "t2": "s2"}

    @patch("agents.delegation.get_high_tier_llm")
    def test_all_tasks_assigned(self, mock_llm_factory: MagicMock) -> None:
        tasks = [_make_task("t1", 5), _make_task("t2", 5), _make_task("t3", 5)]
        students = [_make_student("s1"), _make_student("s2")]

        llm_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
            {"task_id": "t2", "student_id": "s2"},
            {"task_id": "t3", "student_id": "s1"},
        ])
        mock_llm = _make_mock_llm([llm_response])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        assert set(result["delegation_matrix"].keys()) == {"t1", "t2", "t3"}

    @patch("agents.delegation.get_high_tier_llm")
    def test_tasks_have_assigned_to_and_deadline(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]

        llm_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
        ])
        mock_llm = _make_mock_llm([llm_response])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        updated_task = result["task_array"][0]
        assert updated_task.assigned_to == "s1"
        assert updated_task.deadline is not None

    @patch("agents.delegation.get_high_tier_llm")
    def test_project_timeline_populated(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]

        llm_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
        ])
        mock_llm = _make_mock_llm([llm_response])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        assert "project_timeline" in result
        assert len(result["project_timeline"].burn_down_targets) > 0


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestDelegationRetry:
    """Tests for LLM retry logic."""

    @patch("agents.delegation.get_high_tier_llm")
    def test_retry_on_malformed_json(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]

        valid_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
        ])
        mock_llm = _make_mock_llm([
            "not json at all",  # attempt 1 fails
            valid_response,     # attempt 2 succeeds
        ])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        assert "delegation_matrix" in result
        assert mock_llm.invoke.call_count == 2

    @patch("agents.delegation.get_high_tier_llm")
    def test_all_retries_exhausted(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]

        mock_llm = _make_mock_llm([
            "bad1",
            "bad2",
            "bad3",
        ])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students)
        result = delegation(state)

        assert result == {}
        assert mock_llm.invoke.call_count == 3


# ---------------------------------------------------------------------------
# Equity feedback
# ---------------------------------------------------------------------------


class TestDelegationWithEquityFeedback:
    """Tests that equity feedback is included in the prompt on re-delegation."""

    @patch("agents.delegation.get_high_tier_llm")
    def test_equity_feedback_in_prompt(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]

        llm_response = _valid_llm_response([
            {"task_id": "t1", "student_id": "s1"},
        ])
        mock_llm = _make_mock_llm([llm_response])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(
            tasks=tasks,
            students=students,
            equity_retries=1,
            equity_result=EquityResult(
                balanced=False,
                reasoning="Student s1 is overloaded",
                violations=["s1 has 50% above average"],
            ),
        )
        result = delegation(state)

        # Verify the LLM was called with equity feedback in the prompt.
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "PREVIOUS DELEGATION WAS REJECTED" in user_msg
        assert "overloaded" in user_msg
