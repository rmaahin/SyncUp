"""Unit tests for the Meeting Coordinator agent — mock LLM + MCP."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.meeting_coordinator import (
    _build_context_summary,
    _generate_agenda,
    _handle_ingest,
    _handle_schedule,
    meeting_coordinator,
)
from state.schema import (
    DateRange,
    Intervention,
    MeetingRecord,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)

_AGENDA_RESPONSE = json.dumps(
    {
        "agenda_items": [
            "Review completed tasks since last meeting",
            "Discuss upcoming API integration deadline (Apr 15)",
            "Address blockers on the database migration task",
            "Assign action items for next sprint",
        ]
    }
)

_NOTES_RESPONSE = json.dumps(
    {
        "summary": "Team discussed progress on API integration and database migration. "
        "Decided to extend the DB deadline by 2 days.",
        "attendees": ["Alice", "Bob"],
        "action_items": [
            "Alice: finish API endpoint tests by Friday",
            "Bob: update DB schema documentation",
        ],
        "decisions": ["Extend DB migration deadline to Apr 17"],
        "blockers_discussed": ["CI pipeline flaky on Windows"],
    }
)

_MALFORMED_RESPONSE = "This is definitely not JSON"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_student(
    student_id: str = "s-1",
    name: str = "Alice",
    google_email: str = "alice@example.com",
) -> StudentProfile:
    """Create a minimal StudentProfile for testing."""
    return StudentProfile(
        student_id=student_id,
        name=name,
        email=f"{name.lower()}@example.com",
        skills={"python": 0.8},
        availability_hours_per_week=10.0,
        timezone="UTC",
        google_email=google_email,
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
        deadline=deadline,
        assigned_to=assigned_to,
        dependencies=dependencies or [],
    )


def _make_state(**overrides: Any) -> SyncUpState:
    """Create a minimal SyncUpState with sensible defaults."""
    defaults: dict[str, Any] = {
        "project_id": "proj-1",
        "project_name": "Test Project",
        "student_profiles": [
            _make_student("s-1", "Alice", "alice@example.com"),
            _make_student("s-2", "Bob", "bob@example.com"),
        ],
        "task_array": [
            _make_task("task-1", "Build API", TaskStatus.IN_PROGRESS, assigned_to="s-1"),
            _make_task("task-2", "Write tests", TaskStatus.TODO, assigned_to="s-2"),
        ],
        "final_deadline": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "meeting_mode": None,
        "meeting_interval_days": 7,
        "meeting_notes_doc_ids": [],
    }
    defaults.update(overrides)
    return SyncUpState(**defaults)


def _mock_mcp_context() -> AsyncMock:
    """Create a mock for the SyncUpMCPClient async context manager."""
    mcp_instance = AsyncMock()

    # GoogleCalendarMCP mock
    cal_mock = AsyncMock()
    cal_mock.get_events = AsyncMock(return_value=[])
    cal_mock.create_event = AsyncMock(return_value={"id": "evt-123"})

    # GoogleDocsMCP mock
    docs_mock = AsyncMock()
    docs_mock.create_document = AsyncMock(return_value={"document_id": "doc-456"})
    docs_mock.read_document = AsyncMock(
        return_value={"content": "Meeting notes: Alice and Bob discussed the API."}
    )

    return mcp_instance, cal_mock, docs_mock


# ---------------------------------------------------------------------------
# Tests: Mode dispatch
# ---------------------------------------------------------------------------


class TestModeDispatch:
    """Test that meeting_coordinator dispatches correctly based on meeting_mode."""

    @pytest.mark.asyncio
    async def test_none_mode_returns_empty(self) -> None:
        """meeting_mode=None → no-op."""
        state = _make_state(meeting_mode=None)
        result = await meeting_coordinator(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_unknown_mode_returns_empty(self) -> None:
        """Unknown meeting_mode → returns {} with warning."""
        state = _make_state(meeting_mode="unknown")
        result = await meeting_coordinator(state)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Schedule mode
# ---------------------------------------------------------------------------


class TestScheduleMode:
    """Tests for Mode 1 — schedule meeting."""

    @pytest.mark.asyncio
    @patch("agents.meeting_coordinator._get_now", return_value=NOW)
    @patch("agents.meeting_coordinator.get_low_tier_llm")
    @patch("agents.meeting_coordinator.find_optimal_meeting_slot")
    @patch("agents.meeting_coordinator.GoogleDocsMCP")
    @patch("agents.meeting_coordinator.GoogleCalendarMCP")
    @patch("agents.meeting_coordinator.SyncUpMCPClient")
    async def test_happy_path(
        self,
        mock_mcp_cls: MagicMock,
        mock_cal_cls: MagicMock,
        mock_docs_cls: MagicMock,
        mock_find_slot: MagicMock,
        mock_get_llm: MagicMock,
        mock_now: MagicMock,
    ) -> None:
        """Full schedule flow: slot found, agenda generated, event + doc created."""
        slot = datetime(2026, 4, 12, 14, 0, tzinfo=timezone.utc)
        mock_find_slot.return_value = slot

        # Mock MCP context manager
        mcp_instance = AsyncMock()
        mock_mcp_cls.return_value.__aenter__ = AsyncMock(return_value=mcp_instance)
        mock_mcp_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        # Mock Calendar MCP
        cal_instance = AsyncMock()
        cal_instance.get_events = AsyncMock(return_value=[])
        cal_instance.create_event = AsyncMock(return_value={"id": "evt-123"})
        mock_cal_cls.return_value = cal_instance

        # Mock Docs MCP
        docs_instance = AsyncMock()
        docs_instance.create_document = AsyncMock(return_value={"document_id": "doc-456"})
        mock_docs_cls.return_value = docs_instance

        # Mock LLM
        mock_get_llm.return_value = _make_mock_llm([_AGENDA_RESPONSE])

        state = _make_state(meeting_mode="schedule")
        result = await _handle_schedule(state)

        # Verify state update
        assert result["next_meeting_scheduled"] == slot
        assert result["meeting_mode"] is None
        assert len(result["meeting_log"]) == 1
        record = result["meeting_log"][0]
        assert isinstance(record, MeetingRecord)
        assert record.date == slot
        assert "doc-456" in result["meeting_notes_doc_ids"]

        # Verify MCP calls
        cal_instance.create_event.assert_called_once()
        docs_instance.create_document.assert_called_once()

    @pytest.mark.asyncio
    @patch("agents.meeting_coordinator._get_now", return_value=NOW)
    @patch("agents.meeting_coordinator.find_optimal_meeting_slot")
    @patch("agents.meeting_coordinator.SyncUpMCPClient")
    async def test_no_slot_available(
        self,
        mock_mcp_cls: MagicMock,
        mock_find_slot: MagicMock,
        mock_now: MagicMock,
    ) -> None:
        """No available slot → no event/doc created, next_meeting_scheduled=None."""
        mock_find_slot.return_value = None

        mcp_instance = AsyncMock()
        mock_mcp_cls.return_value.__aenter__ = AsyncMock(return_value=mcp_instance)
        mock_mcp_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        state = _make_state(meeting_mode="schedule")
        result = await _handle_schedule(state)

        assert result["next_meeting_scheduled"] is None
        assert result["meeting_mode"] is None
        assert "meeting_log" not in result

    @pytest.mark.asyncio
    @patch("agents.meeting_coordinator._get_now", return_value=NOW)
    @patch("agents.meeting_coordinator.get_low_tier_llm")
    @patch("agents.meeting_coordinator.find_optimal_meeting_slot")
    @patch("agents.meeting_coordinator.GoogleDocsMCP")
    @patch("agents.meeting_coordinator.GoogleCalendarMCP")
    @patch("agents.meeting_coordinator.SyncUpMCPClient")
    async def test_llm_failure_uses_fallback_agenda(
        self,
        mock_mcp_cls: MagicMock,
        mock_cal_cls: MagicMock,
        mock_docs_cls: MagicMock,
        mock_find_slot: MagicMock,
        mock_get_llm: MagicMock,
        mock_now: MagicMock,
    ) -> None:
        """LLM returns garbage → fallback agenda is used, flow completes."""
        slot = datetime(2026, 4, 12, 14, 0, tzinfo=timezone.utc)
        mock_find_slot.return_value = slot

        mcp_instance = AsyncMock()
        mock_mcp_cls.return_value.__aenter__ = AsyncMock(return_value=mcp_instance)
        mock_mcp_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        cal_instance = AsyncMock()
        cal_instance.get_events = AsyncMock(return_value=[])
        cal_instance.create_event = AsyncMock(return_value={"id": "evt-123"})
        mock_cal_cls.return_value = cal_instance

        docs_instance = AsyncMock()
        docs_instance.create_document = AsyncMock(return_value={"document_id": "doc-456"})
        mock_docs_cls.return_value = docs_instance

        # LLM returns garbage 3 times (MAX_RETRIES + 1)
        mock_get_llm.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _MALFORMED_RESPONSE, _MALFORMED_RESPONSE]
        )

        state = _make_state(meeting_mode="schedule")
        result = await _handle_schedule(state)

        # Should still succeed with fallback agenda
        assert result["next_meeting_scheduled"] == slot
        record = result["meeting_log"][0]
        assert "Status round" in record.agenda  # Fallback text


# ---------------------------------------------------------------------------
# Tests: Ingest mode
# ---------------------------------------------------------------------------


class TestIngestMode:
    """Tests for Mode 2 — ingest meeting notes."""

    @pytest.mark.asyncio
    @patch("agents.meeting_coordinator._get_now", return_value=NOW)
    @patch("agents.meeting_coordinator.get_low_tier_llm")
    @patch("agents.meeting_coordinator.GoogleDocsMCP")
    @patch("agents.meeting_coordinator.SyncUpMCPClient")
    async def test_happy_path(
        self,
        mock_mcp_cls: MagicMock,
        mock_docs_cls: MagicMock,
        mock_get_llm: MagicMock,
        mock_now: MagicMock,
    ) -> None:
        """Full ingest flow: reads doc, parses notes, creates MeetingRecord."""
        mcp_instance = AsyncMock()
        mock_mcp_cls.return_value.__aenter__ = AsyncMock(return_value=mcp_instance)
        mock_mcp_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        docs_instance = AsyncMock()
        docs_instance.read_document = AsyncMock(
            return_value={"content": "Meeting notes: Alice and Bob discussed the API."}
        )
        mock_docs_cls.return_value = docs_instance

        mock_get_llm.return_value = _make_mock_llm([_NOTES_RESPONSE])

        state = _make_state(
            meeting_mode="ingest",
            meeting_notes_doc_ids=["doc-123"],
        )
        result = await _handle_ingest(state)

        assert len(result["meeting_log"]) == 1
        record = result["meeting_log"][0]
        assert isinstance(record, MeetingRecord)
        assert "Alice" in record.attendees
        assert "Bob" in record.attendees
        assert len(record.action_items) == 2
        assert result["meeting_mode"] is None

    @pytest.mark.asyncio
    async def test_empty_doc_ids_returns_empty(self) -> None:
        """No doc IDs → returns {}."""
        state = _make_state(meeting_mode="ingest", meeting_notes_doc_ids=[])
        result = await _handle_ingest(state)
        assert result == {}

    @pytest.mark.asyncio
    @patch("agents.meeting_coordinator._get_now", return_value=NOW)
    @patch("agents.meeting_coordinator.get_low_tier_llm")
    @patch("agents.meeting_coordinator.GoogleDocsMCP")
    @patch("agents.meeting_coordinator.SyncUpMCPClient")
    async def test_llm_failure_uses_fallback(
        self,
        mock_mcp_cls: MagicMock,
        mock_docs_cls: MagicMock,
        mock_get_llm: MagicMock,
        mock_now: MagicMock,
    ) -> None:
        """LLM fails → raw text as notes, empty action_items."""
        mcp_instance = AsyncMock()
        mock_mcp_cls.return_value.__aenter__ = AsyncMock(return_value=mcp_instance)
        mock_mcp_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        docs_instance = AsyncMock()
        raw_notes = "We talked about the project. Alice will do the API."
        docs_instance.read_document = AsyncMock(return_value={"content": raw_notes})
        mock_docs_cls.return_value = docs_instance

        mock_get_llm.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _MALFORMED_RESPONSE, _MALFORMED_RESPONSE]
        )

        state = _make_state(
            meeting_mode="ingest",
            meeting_notes_doc_ids=["doc-123"],
        )
        result = await _handle_ingest(state)

        record = result["meeting_log"][0]
        # Fallback: raw text as notes
        assert raw_notes[:100] in record.notes
        assert record.action_items == []


# ---------------------------------------------------------------------------
# Tests: Context summary builder
# ---------------------------------------------------------------------------


class TestBuildContextSummary:
    """Tests for _build_context_summary."""

    def test_includes_overdue_tasks(self) -> None:
        """Overdue tasks appear in the summary."""
        state = _make_state(
            task_array=[
                _make_task(
                    "task-1",
                    "Build API",
                    TaskStatus.IN_PROGRESS,
                    deadline=datetime(2026, 4, 5, tzinfo=timezone.utc),
                    assigned_to="s-1",
                ),
            ],
        )
        summary = _build_context_summary(state, NOW)
        assert "OVERDUE" in summary
        assert "Build API" in summary

    def test_includes_behind_students(self) -> None:
        """Students flagged as behind appear in the summary."""
        state = _make_state(
            student_progress={"s-1": "behind", "s-2": "on_track"},
        )
        summary = _build_context_summary(state, NOW)
        assert "STUDENTS NEEDING ATTENTION" in summary
        assert "Alice" in summary

    def test_includes_last_meeting_action_items(self) -> None:
        """Action items from the last meeting appear in the summary."""
        state = _make_state(
            meeting_log=[
                MeetingRecord(
                    date=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    attendees=["Alice", "Bob"],
                    agenda="Review progress",
                    notes="Good meeting",
                    action_items=["Alice to finish API tests", "Bob to review PR"],
                )
            ],
        )
        summary = _build_context_summary(state, NOW)
        assert "ACTION ITEMS FROM LAST MEETING" in summary
        assert "Alice to finish API tests" in summary

    def test_empty_state_no_crash(self) -> None:
        """Empty state produces output without crashing."""
        state = _make_state(
            task_array=[],
            student_profiles=[],
            student_progress={},
        )
        summary = _build_context_summary(state, NOW)
        assert "PROJECT:" in summary


# ---------------------------------------------------------------------------
# Tests: Agenda generation
# ---------------------------------------------------------------------------


class TestGenerateAgenda:
    """Tests for the _generate_agenda helper."""

    @patch("agents.meeting_coordinator.get_low_tier_llm")
    def test_successful_generation(self, mock_get_llm: MagicMock) -> None:
        """LLM returns valid JSON → agenda items returned."""
        mock_get_llm.return_value = _make_mock_llm([_AGENDA_RESPONSE])
        state = _make_state()
        items = _generate_agenda(state, NOW)
        assert len(items) == 4
        assert "Review completed tasks" in items[0]

    @patch("agents.meeting_coordinator.get_low_tier_llm")
    def test_fallback_on_failure(self, mock_get_llm: MagicMock) -> None:
        """LLM returns garbage → fallback agenda."""
        mock_get_llm.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _MALFORMED_RESPONSE, _MALFORMED_RESPONSE]
        )
        state = _make_state()
        items = _generate_agenda(state, NOW)
        assert len(items) == 4
        assert "Status round" in items[0]
