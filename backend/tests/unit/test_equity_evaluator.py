"""Tests for the Equity Evaluator — mocks the LLM."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from evaluators.equity_evaluator import (
    _compute_effort_distribution,
    _find_deterministic_violations,
    _parse_response,
    equity_evaluator,
)
from state.schema import (
    EquityResult,
    StudentProfile,
    SyncUpState,
    Task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(tid: str, effort: float = 5.0) -> Task:
    return Task(id=tid, title=tid, effort_hours=effort)


def _make_student(sid: str, hours: float = 20.0) -> StudentProfile:
    return StudentProfile(
        student_id=sid,
        name=sid,
        email=f"{sid}@test.com",
        availability_hours_per_week=hours,
    )


def _make_state(
    tasks: list[Task] | None = None,
    students: list[StudentProfile] | None = None,
    delegation_matrix: dict[str, str] | None = None,
    equity_retries: int = 0,
) -> SyncUpState:
    return SyncUpState(
        project_id="test",
        project_brief="Test project",
        final_deadline=datetime(2026, 4, 1, tzinfo=timezone.utc),
        task_array=tasks or [],
        student_profiles=students or [],
        delegation_matrix=delegation_matrix or {},
        equity_retries=equity_retries,
    )


def _balanced_llm_response() -> str:
    return json.dumps({
        "balanced": True,
        "reasoning": "Workload is evenly distributed",
        "violations": [],
    })


def _imbalanced_llm_response() -> str:
    return json.dumps({
        "balanced": False,
        "reasoning": "Student s1 has 50% above average",
        "violations": ["s1 is overloaded with 15h vs 10h average"],
    })


def _make_mock_llm(responses: list[str]) -> MagicMock:
    mock = MagicMock()
    mock_responses = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        mock_responses.append(msg)
    mock.invoke.side_effect = mock_responses
    return mock


# ---------------------------------------------------------------------------
# Effort distribution
# ---------------------------------------------------------------------------


class TestEffortDistribution:
    """Tests for _compute_effort_distribution."""

    def test_sums_correctly(self) -> None:
        tasks = [_make_task("t1", 5), _make_task("t2", 10)]
        students = [_make_student("s1"), _make_student("s2")]
        matrix = {"t1": "s1", "t2": "s1"}
        dist = _compute_effort_distribution(tasks, matrix, students)
        assert dist["s1"] == 15.0
        assert dist["s2"] == 0.0

    def test_unassigned_tasks_excluded(self) -> None:
        tasks = [_make_task("t1", 5), _make_task("t2", 10)]
        students = [_make_student("s1")]
        matrix = {"t1": "s1"}  # t2 not assigned
        dist = _compute_effort_distribution(tasks, matrix, students)
        assert dist["s1"] == 5.0

    def test_empty_inputs(self) -> None:
        dist = _compute_effort_distribution([], {}, [])
        assert dist == {}


# ---------------------------------------------------------------------------
# Deterministic violations
# ---------------------------------------------------------------------------


class TestDeterministicViolations:
    """Tests for _find_deterministic_violations."""

    def test_balanced_no_violations(self) -> None:
        students = [_make_student("s1"), _make_student("s2")]
        effort = {"s1": 10.0, "s2": 10.0}
        violations = _find_deterministic_violations(effort, students)
        assert violations == []

    def test_imbalanced_flags_violation(self) -> None:
        students = [_make_student("s1"), _make_student("s2")]
        effort = {"s1": 20.0, "s2": 5.0}
        # avg = 12.5, threshold = 17.5. s1 (20) > 17.5
        violations = _find_deterministic_violations(effort, students)
        assert len(violations) == 1
        assert "s1" in violations[0]


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEquityEvaluatorEmptyState:
    """Evaluator with empty state returns balanced=True."""

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_no_tasks(self, mock_llm_factory: MagicMock) -> None:
        state = _make_state(tasks=[], students=[_make_student("s1")])
        result = equity_evaluator(state)
        assert result["equity_result"].balanced is True
        assert result["equity_retries"] == 1
        mock_llm_factory.assert_not_called()

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_no_delegation_matrix(self, mock_llm_factory: MagicMock) -> None:
        state = _make_state(
            tasks=[_make_task("t1")],
            students=[_make_student("s1")],
            delegation_matrix={},
        )
        result = equity_evaluator(state)
        assert result["equity_result"].balanced is True
        mock_llm_factory.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEquityEvaluatorHappyPath:
    """Evaluator with balanced workload."""

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_balanced_workload(self, mock_llm_factory: MagicMock) -> None:
        tasks = [_make_task("t1", 5), _make_task("t2", 5)]
        students = [_make_student("s1"), _make_student("s2")]
        matrix = {"t1": "s1", "t2": "s2"}

        mock_llm = _make_mock_llm([_balanced_llm_response()])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students, delegation_matrix=matrix)
        result = equity_evaluator(state)

        assert result["equity_result"].balanced is True
        assert result["equity_retries"] == 1

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_equity_retries_incremented(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]
        matrix = {"t1": "s1"}

        mock_llm = _make_mock_llm([_balanced_llm_response()])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(
            tasks=tasks, students=students,
            delegation_matrix=matrix, equity_retries=2,
        )
        result = equity_evaluator(state)
        assert result["equity_retries"] == 3

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_temperature_zero(self, mock_llm_factory: MagicMock) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]
        matrix = {"t1": "s1"}

        mock_llm = _make_mock_llm([_balanced_llm_response()])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students, delegation_matrix=matrix)
        equity_evaluator(state)

        mock_llm_factory.assert_called_once_with(temperature=0.0)


# ---------------------------------------------------------------------------
# Imbalanced
# ---------------------------------------------------------------------------


class TestEquityEvaluatorImbalanced:
    """Evaluator with unbalanced workload."""

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_imbalanced_returns_false(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 15), _make_task("t2", 5)]
        students = [_make_student("s1"), _make_student("s2")]
        matrix = {"t1": "s1", "t2": "s2"}

        mock_llm = _make_mock_llm([_imbalanced_llm_response()])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students, delegation_matrix=matrix)
        result = equity_evaluator(state)

        assert result["equity_result"].balanced is False
        assert len(result["equity_result"].violations) > 0


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestEquityEvaluatorRetry:
    """Tests for LLM retry logic."""

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_retry_on_malformed_output(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]
        matrix = {"t1": "s1"}

        mock_llm = _make_mock_llm([
            "not json",
            _balanced_llm_response(),
        ])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students, delegation_matrix=matrix)
        result = equity_evaluator(state)

        assert result["equity_result"].balanced is True
        assert mock_llm.invoke.call_count == 2

    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    def test_all_retries_exhausted_defaults_unbalanced(
        self, mock_llm_factory: MagicMock
    ) -> None:
        tasks = [_make_task("t1", 5)]
        students = [_make_student("s1")]
        matrix = {"t1": "s1"}

        mock_llm = _make_mock_llm(["bad1", "bad2", "bad3"])
        mock_llm_factory.return_value = mock_llm

        state = _make_state(tasks=tasks, students=students, delegation_matrix=matrix)
        result = equity_evaluator(state)

        assert result["equity_result"].balanced is False
        assert mock_llm.invoke.call_count == 3


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for _parse_response."""

    def test_valid_json(self) -> None:
        raw = json.dumps({
            "balanced": True, "reasoning": "ok", "violations": [],
        })
        result = _parse_response(raw)
        assert result.balanced is True

    def test_with_code_fences(self) -> None:
        inner = json.dumps({
            "balanced": False, "reasoning": "bad", "violations": ["x"],
        })
        raw = f"```json\n{inner}\n```"
        result = _parse_response(raw)
        assert result.balanced is False
        assert result.violations == ["x"]
