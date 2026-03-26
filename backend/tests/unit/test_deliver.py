"""Unit tests for the Deliver node."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.deliver import (
    DEADLINE_EXTENSION_DAYS,
    _compute_new_deadline,
    _extend_task_deadlines,
    _find_overdue_tasks,
    deliver,
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
_DEADLINE_PAST = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
_DEADLINE_FUTURE = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_draft(
    student_id: str = "s-2",
    message: str = "Hi Bob, let's check in on the API task.",
    suggested_action: str = "schedule_check_in",
    severity: str = "medium",
) -> DraftIntervention:
    """Create a minimal DraftIntervention for testing."""
    return DraftIntervention(
        target_student_id=student_id,
        message=message,
        suggested_action=suggested_action,
        severity=severity,
        affected_teammates=["Alice"],
    )


def _make_task(
    task_id: str = "t1",
    title: str = "Build API",
    status: TaskStatus = TaskStatus.TODO,
    deadline: datetime | None = None,
    assigned_to: str | None = None,
) -> Task:
    """Create a minimal Task for testing."""
    return Task(
        id=task_id,
        title=title,
        status=status,
        deadline=deadline or _DEADLINE_PAST,
        effort_hours=10.0,
        assigned_to=assigned_to,
    )


def _make_state(
    draft: DraftIntervention | None = None,
    tasks: list[Task] | None = None,
    delegation_matrix: dict[str, str] | None = None,
    trello_card_mapping: dict[str, str] | None = None,
    calendar_event_mapping: dict[str, str] | None = None,
    tone_result: ToneResult | None = None,
    tone_rewrite_count: int = 1,
) -> SyncUpState:
    """Create a SyncUpState for testing."""
    return SyncUpState(
        project_id="test-project",
        draft_intervention=draft,
        task_array=tasks or [],
        delegation_matrix=delegation_matrix or {},
        trello_card_mapping=trello_card_mapping or {},
        calendar_event_mapping=calendar_event_mapping or {},
        tone_result=tone_result,
        tone_rewrite_count=tone_rewrite_count,
        student_profiles=[
            StudentProfile(
                student_id="s-1", name="Alice", email="alice@example.com"
            ),
            StudentProfile(
                student_id="s-2", name="Bob", email="bob@example.com"
            ),
        ],
    )


def _setup_trello_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure a TrelloClient mock as async context manager."""
    mock_client = AsyncMock()
    mock_comment = MagicMock()
    mock_comment.id = "comment-1"
    mock_client.add_comment.return_value = mock_comment
    mock_client.update_card.return_value = MagicMock()

    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestFindOverdueTasks:
    """Tests for _find_overdue_tasks."""

    def test_finds_overdue(self) -> None:
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(tasks=tasks, delegation_matrix={"t1": "s-2"})
        result = _find_overdue_tasks("s-2", state, NOW)
        assert len(result) == 1
        assert result[0][0] == "t1"

    def test_ignores_done_tasks(self) -> None:
        tasks = [
            _make_task(
                "t1", deadline=_DEADLINE_PAST, assigned_to="s-2",
                status=TaskStatus.DONE,
            )
        ]
        state = _make_state(tasks=tasks, delegation_matrix={"t1": "s-2"})
        assert _find_overdue_tasks("s-2", state, NOW) == []

    def test_ignores_other_students(self) -> None:
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-1")]
        state = _make_state(tasks=tasks, delegation_matrix={"t1": "s-1"})
        assert _find_overdue_tasks("s-2", state, NOW) == []


class TestExtendTaskDeadlines:
    """Tests for _extend_task_deadlines."""

    def test_extends_overdue_only(self) -> None:
        t1 = _make_task("t1", deadline=_DEADLINE_PAST)
        t2 = _make_task("t2", deadline=_DEADLINE_FUTURE)

        updated = _extend_task_deadlines([t1, t2], [("t1", t1)], NOW)
        # t1 was overdue (deadline in past) → extended from now
        assert updated[0].deadline == NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)
        assert updated[1].deadline == _DEADLINE_FUTURE


class TestComputeNewDeadline:
    """Tests for _compute_new_deadline."""

    def test_adds_extension_from_now_when_overdue(self) -> None:
        """Overdue task (deadline in past) → base is now, not the old deadline."""
        result = _compute_new_deadline(NOW, original_deadline=_DEADLINE_PAST)
        assert result == NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)

    def test_adds_extension_from_original_when_future(self) -> None:
        """Future deadline → base is original deadline, not now."""
        result = _compute_new_deadline(NOW, original_deadline=_DEADLINE_FUTURE)
        assert result == _DEADLINE_FUTURE + timedelta(days=DEADLINE_EXTENSION_DAYS)

    def test_no_original_defaults_to_now(self) -> None:
        """No original deadline → falls back to now + extension."""
        result = _compute_new_deadline(NOW)
        assert result == NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)

    def test_avoids_blackout_period(self) -> None:
        """New deadline landing in a blackout → pushed past blackout end."""
        blackout_start = NOW + timedelta(days=2)
        blackout_end = NOW + timedelta(days=5)
        blackout = DateRange(start=blackout_start, end=blackout_end)

        # Without blackout: NOW + 3 days = inside blackout
        result = _compute_new_deadline(NOW, blackout_periods=[blackout])
        assert result == blackout_end + timedelta(days=1)

    def test_no_blackout_overlap_unchanged(self) -> None:
        """Blackout that doesn't overlap → deadline unchanged."""
        blackout = DateRange(
            start=NOW + timedelta(days=10),
            end=NOW + timedelta(days=15),
        )
        result = _compute_new_deadline(NOW, blackout_periods=[blackout])
        assert result == NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)


# ---------------------------------------------------------------------------
# Tests for deliver node
# ---------------------------------------------------------------------------


class TestDeliverNode:
    """Tests for the deliver node function."""

    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_creates_intervention_record(self, mock_now: MagicMock) -> None:
        """Verify Intervention is appended to intervention_history."""
        draft = _make_draft()
        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        state = _make_state(
            draft=draft,
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
        )

        result = await deliver(state)

        assert "intervention_history" in result
        assert len(result["intervention_history"]) == 1

        intervention = result["intervention_history"][0]
        assert isinstance(intervention, Intervention)
        assert intervention.target_student_id == "s-2"
        assert intervention.message_text == draft.message
        assert intervention.outcome == "schedule_check_in"
        assert intervention.timestamp == NOW

    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_clears_draft_state(self, mock_now: MagicMock) -> None:
        """Verify draft_intervention, tone_result, tone_rewrite_count are cleared."""
        state = _make_state(
            draft=_make_draft(),
            tone_result=ToneResult(
                classification="constructive", reasoning="ok"
            ),
            tone_rewrite_count=2,
        )
        result = await deliver(state)

        assert result["draft_intervention"] is None
        assert result["tone_result"] is None
        assert result["tone_rewrite_count"] == 0

    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    @patch("agents.deliver._find_overdue_card_id", return_value="card-abc")
    async def test_trello_comment_posted(
        self,
        mock_find_card: MagicMock,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
    ) -> None:
        """Verify add_comment is called with correct card_id and message."""
        mock_client = _setup_trello_mock(mock_trello_cls)

        draft = _make_draft()
        state = _make_state(
            draft=draft,
            tasks=[_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")],
            delegation_matrix={"t1": "s-2"},
            trello_card_mapping={"t1": "card-abc"},
        )
        result = await deliver(state)

        mock_client.add_comment.assert_called_once_with("card-abc", draft.message)
        assert len(result["intervention_history"]) == 1

    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    @patch("agents.deliver._find_overdue_card_id", return_value="card-abc")
    async def test_trello_failure_graceful(
        self,
        mock_find_card: MagicMock,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
    ) -> None:
        """Trello error -> intervention still logged, no exception raised."""
        mock_client = AsyncMock()
        mock_client.add_comment.side_effect = RuntimeError("Trello API down")
        mock_trello_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_trello_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        state = _make_state(draft=_make_draft())
        result = await deliver(state)

        assert len(result["intervention_history"]) == 1
        assert result["draft_intervention"] is None

    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_no_draft_is_noop(self, mock_now: MagicMock) -> None:
        """No draft_intervention -> empty result, no Trello call."""
        state = _make_state(draft=None)
        result = await deliver(state)

        assert result["draft_intervention"] is None
        assert result["tone_result"] is None
        assert result["tone_rewrite_count"] == 0
        assert "intervention_history" not in result

    # --- extend_deadline action tests ---

    @patch("agents.deliver._update_calendar_events", new_callable=AsyncMock)
    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_extend_deadline_updates_task_array(
        self,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
        mock_cal: AsyncMock,
    ) -> None:
        """extend_deadline -> task_array has updated deadlines."""
        mock_client = _setup_trello_mock(mock_trello_cls)

        tasks = [
            _make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2"),
            _make_task("t2", deadline=_DEADLINE_FUTURE, assigned_to="s-1"),
        ]
        draft = _make_draft(suggested_action="extend_deadline")
        state = _make_state(
            draft=draft,
            tasks=tasks,
            delegation_matrix={"t1": "s-2", "t2": "s-1"},
            trello_card_mapping={"t1": "card-1"},
        )
        result = await deliver(state)

        expected_deadline = NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)
        assert "task_array" in result
        updated_tasks = result["task_array"]
        # t1 should have new deadline
        t1 = next(t for t in updated_tasks if t.id == "t1")
        assert t1.deadline == expected_deadline
        # t2 should be unchanged
        t2 = next(t for t in updated_tasks if t.id == "t2")
        assert t2.deadline == _DEADLINE_FUTURE

    @patch("agents.deliver._update_calendar_events", new_callable=AsyncMock)
    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_extend_deadline_updates_trello_due_date(
        self,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
        mock_cal: AsyncMock,
    ) -> None:
        """extend_deadline -> TrelloClient.update_card called with new due date."""
        mock_client = _setup_trello_mock(mock_trello_cls)

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        draft = _make_draft(suggested_action="extend_deadline")
        state = _make_state(
            draft=draft,
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
            trello_card_mapping={"t1": "card-1"},
        )
        result = await deliver(state)

        expected_deadline = NOW + timedelta(days=DEADLINE_EXTENSION_DAYS)
        mock_client.update_card.assert_called_once_with("card-1", due=expected_deadline)

    @patch("agents.deliver._update_calendar_events", new_callable=AsyncMock)
    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_extend_deadline_calls_calendar_update(
        self,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
        mock_cal: AsyncMock,
    ) -> None:
        """extend_deadline -> _update_calendar_events called."""
        _setup_trello_mock(mock_trello_cls)

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        draft = _make_draft(suggested_action="extend_deadline")
        state = _make_state(
            draft=draft,
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
            calendar_event_mapping={"t1": "event-1"},
        )
        result = await deliver(state)

        mock_cal.assert_called_once()
        call_args = mock_cal.call_args
        assert call_args[0][1] == {"t1": "event-1"}  # calendar_event_mapping subset

    # --- redistribute_task action tests ---

    @patch("agents.deliver.TrelloClient")
    @patch("agents.deliver._get_now", return_value=NOW)
    async def test_redistribute_sets_redelegation_flag(
        self,
        mock_now: MagicMock,
        mock_trello_cls: MagicMock,
    ) -> None:
        """redistribute_task -> needs_redelegation contains overdue task IDs."""
        _setup_trello_mock(mock_trello_cls)

        tasks = [_make_task("t1", deadline=_DEADLINE_PAST, assigned_to="s-2")]
        draft = _make_draft(suggested_action="redistribute_task")
        state = _make_state(
            draft=draft,
            tasks=tasks,
            delegation_matrix={"t1": "s-2"},
            trello_card_mapping={"t1": "card-1"},
        )
        result = await deliver(state)

        assert "needs_redelegation" in result
        assert result["needs_redelegation"] == ["t1"]
        # Should NOT have task_array (no deadline change)
        assert "task_array" not in result
