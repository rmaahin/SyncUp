"""Tests for the Publishing agent — mocks Trello, Calendar, and Docs."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agents.publishing import (
    _REMINDER_MINUTES,
    _build_task_matrix_content,
    _resolve_project_name,
    publishing,
)
from integrations.trello import (
    TrelloAPIError,
    TrelloBoard,
    TrelloCard,
    TrelloChecklist,
    TrelloLabel,
    TrelloList,
)
from state.schema import (
    PublishingStatus,
    StudentProfile,
    SyncUpState,
    Task,
    UrgencyLevel,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
_DEADLINE_1 = _NOW + timedelta(days=7)
_DEADLINE_2 = _NOW + timedelta(days=14)

_LIST_NAMES = ["To Do", "In Progress", "Review", "Done"]
_URGENCY_COLORS = [
    ("critical", "red"),
    ("high", "orange"),
    ("medium", "yellow"),
    ("low", "green"),
]


def _make_task(
    task_id: str = "t-1",
    title: str = "Task 1",
    description: str = "Do something",
    effort_hours: float = 4.0,
    required_skills: list[str] | None = None,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    dependencies: list[str] | None = None,
    deadline: datetime | None = None,
) -> Task:
    return Task(
        id=task_id,
        title=title,
        description=description,
        effort_hours=effort_hours,
        required_skills=required_skills or [],
        urgency=urgency,
        dependencies=dependencies or [],
        deadline=deadline or _DEADLINE_1,
    )


def _make_student(
    student_id: str = "s-1",
    name: str = "Alice",
    trello_id: str = "trello-alice",
    google_email: str = "alice@uni.edu",
) -> StudentProfile:
    return StudentProfile(
        student_id=student_id,
        name=name,
        email=f"{name.lower()}@example.com",
        trello_id=trello_id,
        google_email=google_email,
        timezone="UTC",
    )


def _make_state(
    tasks: list[Task] | None = None,
    students: list[StudentProfile] | None = None,
    delegation_matrix: dict[str, str] | None = None,
    project_name: str = "Test Project",
) -> SyncUpState:
    tasks = tasks if tasks is not None else [_make_task()]
    students = students if students is not None else [_make_student()]
    delegation = delegation_matrix if delegation_matrix is not None else {"t-1": "s-1"}
    return SyncUpState(
        project_id="proj-1",
        project_name=project_name,
        task_array=tasks,
        student_profiles=students,
        delegation_matrix=delegation,
        final_deadline=_NOW + timedelta(days=30),
    )


# ---------------------------------------------------------------------------
# Mock setup helpers
# ---------------------------------------------------------------------------

_CARD_COUNTER = 0


def _mock_create_card(**kwargs: Any) -> TrelloCard:
    global _CARD_COUNTER
    _CARD_COUNTER += 1
    return TrelloCard(id=f"card-{_CARD_COUNTER}", name=kwargs.get("name", "card"))


def _setup_trello_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure a TrelloClient mock as async context manager."""
    global _CARD_COUNTER
    _CARD_COUNTER = 0

    mock_client = AsyncMock()
    mock_client.create_board.return_value = TrelloBoard(id="board-1", name="Test Board")
    mock_client.create_list.side_effect = [
        TrelloList(id=f"list-{i}", name=n) for i, n in enumerate(_LIST_NAMES)
    ]
    mock_client.add_label.side_effect = [
        TrelloLabel(id=f"lbl-{i}", name=u, color=c) for i, (u, c) in enumerate(_URGENCY_COLORS)
    ]
    mock_client.create_card.side_effect = _mock_create_card
    mock_client.add_checklist.return_value = TrelloChecklist(id="cl-1", name="Prerequisites")

    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _setup_mcp_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure a SyncUpMCPClient mock as async context manager."""
    mock_mcp = AsyncMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_mcp)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_mcp


def _setup_calendar_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure GoogleCalendarMCP mock."""
    mock_cal = AsyncMock()
    mock_cal.create_event.return_value = {"id": "evt-1"}
    mock_cls.return_value = mock_cal
    return mock_cal


def _setup_docs_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure GoogleDocsMCP mock."""
    mock_docs = AsyncMock()
    mock_docs.create_document.return_value = {"document_id": "doc-1"}
    mock_cls.return_value = mock_docs
    return mock_docs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveProjectName:
    """Tests for _resolve_project_name helper."""

    def test_uses_project_name(self) -> None:
        state = _make_state(project_name="My Project")
        assert _resolve_project_name(state) == "My Project"

    def test_falls_back_to_project_id(self) -> None:
        state = _make_state(project_name="")
        assert _resolve_project_name(state) == "proj-1"

    def test_falls_back_to_default(self) -> None:
        state = SyncUpState()
        assert _resolve_project_name(state) == "SyncUp Project"


class TestBuildTaskMatrixContent:
    """Tests for _build_task_matrix_content helper."""

    def test_contains_header_and_task(self) -> None:
        task = _make_task()
        student = _make_student()
        content = _build_task_matrix_content([task], {"t-1": "s-1"}, {"s-1": student})
        assert "Task" in content
        assert "Assignee" in content
        assert "Task 1" in content
        assert "Alice" in content

    def test_unassigned_task(self) -> None:
        task = _make_task()
        content = _build_task_matrix_content([task], {}, {})
        assert "Unassigned" in content


@pytest.mark.asyncio
class TestPublishingHappyPath:
    """All three integrations succeed."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_all_succeed(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        trello_client = _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)
        cal_mock = _setup_calendar_mock(MockCal)
        docs_mock = _setup_docs_mock(MockDocs)

        tasks = [
            _make_task("t-1", "Task 1", urgency=UrgencyLevel.CRITICAL, deadline=_DEADLINE_1),
            _make_task("t-2", "Task 2", urgency=UrgencyLevel.LOW, dependencies=["t-1"], deadline=_DEADLINE_2),
        ]
        students = [
            _make_student("s-1", "Alice"),
            _make_student("s-2", "Bob", trello_id="trello-bob", google_email="bob@uni.edu"),
        ]
        state = _make_state(
            tasks=tasks,
            students=students,
            delegation_matrix={"t-1": "s-1", "t-2": "s-2"},
        )

        result = await publishing(state)

        # Status checks
        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "success"
        assert status.calendar == "success"
        assert status.docs == "success"
        assert status.errors == []

        # Trello checks
        assert result["trello_board_id"] == "board-1"
        assert len(result["trello_card_mapping"]) == 2
        assert trello_client.create_board.call_count == 1
        assert trello_client.create_list.call_count == 4
        assert trello_client.add_label.call_count == 4
        assert trello_client.create_card.call_count == 2

        # Dependency checklist only for t-2 (which depends on t-1)
        assert trello_client.add_checklist.call_count == 1
        checklist_call = trello_client.add_checklist.call_args
        assert checklist_call.args[1] == "Prerequisites"
        assert "Task 1" in checklist_call.args[2]

        # Calendar checks
        assert cal_mock.create_event.call_count == 2
        assert len(result["calendar_event_mapping"]) == 2

        # Verify reminders
        for call in cal_mock.create_event.call_args_list:
            assert call.kwargs["reminders"] == _REMINDER_MINUTES

        # Docs checks
        assert docs_mock.create_document.call_count == 1
        assert result["docs_task_matrix_id"] == "doc-1"

        # Webhook stub
        assert result["webhook_configured"] is False


@pytest.mark.asyncio
class TestPublishingTrelloFails:
    """Trello fails but Calendar and Docs succeed."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_trello_error_graceful(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        # Trello raises on enter
        mock_client = AsyncMock()
        mock_client.create_board.side_effect = TrelloAPIError(500, "Server error")
        MockTrello.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockTrello.return_value.__aexit__ = AsyncMock(return_value=None)

        _setup_mcp_mock(MockMCP)
        _setup_calendar_mock(MockCal)
        _setup_docs_mock(MockDocs)

        state = _make_state()
        result = await publishing(state)

        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "failed"
        assert status.calendar == "success"
        assert status.docs == "success"
        assert any("Trello" in e for e in status.errors)
        assert "trello_board_id" not in result


@pytest.mark.asyncio
class TestPublishingCalendarFails:
    """Calendar fails but Trello and Docs succeed."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_calendar_error_graceful(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)

        mock_cal = AsyncMock()
        mock_cal.create_event.side_effect = Exception("Calendar API down")
        MockCal.return_value = mock_cal

        _setup_docs_mock(MockDocs)

        state = _make_state()
        result = await publishing(state)

        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "success"
        assert status.calendar == "failed"
        assert status.docs == "success"
        assert any("Calendar" in e for e in status.errors)


@pytest.mark.asyncio
class TestPublishingAllFail:
    """All three integrations fail — node should not crash."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_all_fail_no_crash(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        # Trello fails
        mock_client = AsyncMock()
        mock_client.create_board.side_effect = Exception("Trello boom")
        MockTrello.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockTrello.return_value.__aexit__ = AsyncMock(return_value=None)

        # MCP fails on Calendar
        mock_mcp = _setup_mcp_mock(MockMCP)
        mock_cal = AsyncMock()
        mock_cal.create_event.side_effect = Exception("Calendar boom")
        MockCal.return_value = mock_cal

        # Docs fails
        mock_docs = AsyncMock()
        mock_docs.create_document.side_effect = Exception("Docs boom")
        MockDocs.return_value = mock_docs

        state = _make_state()
        result = await publishing(state)

        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "failed"
        assert status.calendar == "failed"
        assert status.docs == "failed"
        assert len(status.errors) == 3


@pytest.mark.asyncio
class TestPublishingEmptyDelegation:
    """Empty delegation matrix returns early with all-failed status."""

    async def test_empty_delegation(self) -> None:
        state = _make_state(delegation_matrix={})
        result = await publishing(state)

        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "failed"
        assert status.calendar == "failed"
        assert status.docs == "failed"
        assert "Empty delegation matrix" in status.errors


@pytest.mark.asyncio
class TestPublishingEmptyTasks:
    """Empty task_array — board still created, no cards/events."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_empty_tasks(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        trello_client = _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)
        cal_mock = _setup_calendar_mock(MockCal)
        docs_mock = _setup_docs_mock(MockDocs)

        state = _make_state(
            tasks=[],
            delegation_matrix={"dummy": "s-1"},  # non-empty to pass early return
        )
        result = await publishing(state)

        status: PublishingStatus = result["publishing_status"]
        assert status.trello == "success"
        assert status.calendar == "success"
        assert status.docs == "success"

        # Board created but no cards
        assert trello_client.create_board.call_count == 1
        assert trello_client.create_card.call_count == 0

        # No calendar events
        assert cal_mock.create_event.call_count == 0

        # Doc still created
        assert docs_mock.create_document.call_count == 1


@pytest.mark.asyncio
class TestPublishingNoDependencies:
    """Task without dependencies should not trigger add_checklist."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_no_checklist_without_deps(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        trello_client = _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)
        _setup_calendar_mock(MockCal)
        _setup_docs_mock(MockDocs)

        task = _make_task(dependencies=[])  # no dependencies
        state = _make_state(tasks=[task])
        result = await publishing(state)

        trello_client.add_checklist.assert_not_called()


@pytest.mark.asyncio
class TestPublishingUrgencyLabels:
    """Each urgency level maps to the correct Trello label colour."""

    @patch.dict(os.environ, {"COURSE_CALENDAR_ID": "test-cal", "TRELLO_API_KEY": "k", "TRELLO_API_TOKEN": "t"})
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    async def test_urgency_label_colors(
        self,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        trello_client = _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)
        _setup_calendar_mock(MockCal)
        _setup_docs_mock(MockDocs)

        state = _make_state()
        await publishing(state)

        # Verify all 4 urgency labels created with correct colours
        label_calls = trello_client.add_label.call_args_list
        assert len(label_calls) == 4

        created_labels = {call.args[1]: call.args[2] for call in label_calls}
        assert created_labels["critical"] == "red"
        assert created_labels["high"] == "orange"
        assert created_labels["medium"] == "yellow"
        assert created_labels["low"] == "green"
