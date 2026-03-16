"""Integration test — Phase 1 through Phase 5 end-to-end flow.

Simulates the full pipeline:
  1. State schema construction (Phase 1)
  2. Task Decomposition via mocked LLM (Phase 3)
  3. Delegation + pacing via mocked LLM (Phase 4)
  4. Equity Evaluator via mocked LLM (Phase 4)
  5. Publishing to Trello/Calendar/Docs via mocked externals (Phase 5)

All LLM calls are mocked with deterministic canned responses.
All external services (Trello API, MCP servers) are mocked.
No real API keys, network calls, or LLMs are needed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.delegation import delegation
from agents.publishing import publishing
from agents.task_decomposition import task_decomposition
from evaluators.equity_evaluator import equity_evaluator
from integrations.trello import (
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
    TaskStatus,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
_FINAL_DEADLINE = _NOW + timedelta(days=30)

_PROJECT_BRIEF = """\
Build a web application for a university course registration system.
Students should be able to browse courses, register, and view their schedule.
The system needs a REST API backend, a React frontend, and a PostgreSQL database.
"""

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------

_DECOMPOSITION_RESPONSE = json.dumps({
    "tasks": [
        {
            "id": "task-setup-repo",
            "title": "Set up project repository",
            "description": "Initialize Git repo, CI/CD pipeline, project structure",
            "effort_hours": 3.0,
            "required_skills": ["git", "devops"],
            "urgency": "critical",
            "dependencies": [],
        },
        {
            "id": "task-design-db",
            "title": "Design database schema",
            "description": "Create PostgreSQL schema for courses, students, registrations",
            "effort_hours": 5.0,
            "required_skills": ["sql", "database"],
            "urgency": "critical",
            "dependencies": ["task-setup-repo"],
        },
        {
            "id": "task-build-api",
            "title": "Build REST API",
            "description": "Implement FastAPI endpoints for course CRUD and registration",
            "effort_hours": 12.0,
            "required_skills": ["python", "api"],
            "urgency": "high",
            "dependencies": ["task-design-db"],
        },
        {
            "id": "task-build-frontend",
            "title": "Build React frontend",
            "description": "Create React UI for course browsing, registration, schedule view",
            "effort_hours": 15.0,
            "required_skills": ["react", "javascript"],
            "urgency": "high",
            "dependencies": ["task-build-api"],
        },
        {
            "id": "task-write-tests",
            "title": "Write integration tests",
            "description": "End-to-end tests covering registration flow",
            "effort_hours": 6.0,
            "required_skills": ["python", "testing"],
            "urgency": "medium",
            "dependencies": ["task-build-api", "task-build-frontend"],
        },
    ]
})

_DELEGATION_RESPONSE = json.dumps({
    "assignments": [
        {"task_id": "task-setup-repo", "student_id": "s-alice"},
        {"task_id": "task-design-db", "student_id": "s-alice"},
        {"task_id": "task-build-api", "student_id": "s-bob"},
        {"task_id": "task-build-frontend", "student_id": "s-carol"},
        {"task_id": "task-write-tests", "student_id": "s-bob"},
    ]
})

_EQUITY_RESPONSE = json.dumps({
    "balanced": True,
    "reasoning": "Workload is within acceptable range for all students.",
    "violations": [],
})

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_students() -> list[StudentProfile]:
    """Create three test students with complementary skills."""
    return [
        StudentProfile(
            student_id="s-alice",
            name="Alice",
            email="alice@uni.edu",
            skills={"git": 0.9, "devops": 0.8, "sql": 0.7, "database": 0.8},
            availability_hours_per_week=10.0,
            timezone="UTC",
            github_username="alice-gh",
            google_email="alice@uni.edu",
            trello_id="trello-alice",
            onboarded_at=_NOW - timedelta(days=1),
        ),
        StudentProfile(
            student_id="s-bob",
            name="Bob",
            email="bob@uni.edu",
            skills={"python": 0.9, "api": 0.8, "testing": 0.7, "sql": 0.5},
            availability_hours_per_week=15.0,
            timezone="UTC",
            github_username="bob-gh",
            google_email="bob@uni.edu",
            trello_id="trello-bob",
            onboarded_at=_NOW - timedelta(days=1),
        ),
        StudentProfile(
            student_id="s-carol",
            name="Carol",
            email="carol@uni.edu",
            skills={"react": 0.9, "javascript": 0.95, "css": 0.8},
            availability_hours_per_week=12.0,
            timezone="UTC",
            github_username="carol-gh",
            google_email="carol@uni.edu",
            trello_id="trello-carol",
            onboarded_at=_NOW - timedelta(days=1),
        ),
    ]


def _make_initial_state() -> SyncUpState:
    """Create the initial state as if Phase 1 (schema) + Phase 2 (onboarding) are done."""
    return SyncUpState(
        project_id="proj-course-reg",
        project_name="Course Registration System",
        project_brief=_PROJECT_BRIEF,
        final_deadline=_FINAL_DEADLINE,
        student_profiles=_make_students(),
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_llm_mock(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns canned responses in sequence."""
    mock_llm = MagicMock()
    mock_responses = []
    for text in responses:
        resp = MagicMock()
        resp.content = text
        mock_responses.append(resp)
    mock_llm.invoke.side_effect = mock_responses
    return mock_llm


_CARD_COUNTER = 0
_LIST_NAMES = ["To Do", "In Progress", "Review", "Done"]
_URGENCY_COLORS = [
    ("critical", "red"),
    ("high", "orange"),
    ("medium", "yellow"),
    ("low", "green"),
]


def _mock_create_card(**kwargs: Any) -> TrelloCard:
    global _CARD_COUNTER
    _CARD_COUNTER += 1
    return TrelloCard(id=f"card-{_CARD_COUNTER}", name=kwargs.get("name", "card"))


def _setup_trello_mock(mock_cls: MagicMock) -> AsyncMock:
    """Configure TrelloClient mock."""
    global _CARD_COUNTER
    _CARD_COUNTER = 0

    mock_client = AsyncMock()
    mock_client.create_board.return_value = TrelloBoard(id="board-1", name="Test")
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
    mock_mcp = AsyncMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_mcp)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_mcp


def _setup_calendar_mock(mock_cls: MagicMock) -> AsyncMock:
    mock_cal = AsyncMock()
    mock_cal.create_event.return_value = {"id": "evt-1"}
    mock_cls.return_value = mock_cal
    return mock_cal


def _setup_docs_mock(mock_cls: MagicMock) -> AsyncMock:
    mock_docs = AsyncMock()
    mock_docs.create_document.return_value = {"document_id": "doc-1"}
    mock_cls.return_value = mock_docs
    return mock_docs


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestPhase1To5Flow:
    """Runs each phase's agent in sequence with mocked externals.

    This validates that state flows correctly from one phase to the next:
      initial state -> task_decomposition -> delegation -> equity_evaluator -> publishing
    """

    @patch.dict(os.environ, {
        "GROQ_API_KEY": "test-key",
        "COURSE_CALENDAR_ID": "test-cal",
        "TRELLO_API_KEY": "k",
        "TRELLO_API_TOKEN": "t",
    })
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    @patch("agents.delegation.get_high_tier_llm")
    @patch("agents.task_decomposition.get_high_tier_llm")
    async def test_full_pipeline(
        self,
        mock_decomp_llm_factory: MagicMock,
        mock_deleg_llm_factory: MagicMock,
        mock_equity_llm_factory: MagicMock,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        # --- Set up LLM mocks ---
        mock_decomp_llm_factory.return_value = _make_llm_mock([_DECOMPOSITION_RESPONSE])
        mock_deleg_llm_factory.return_value = _make_llm_mock([_DELEGATION_RESPONSE])
        mock_equity_llm_factory.return_value = _make_llm_mock([_EQUITY_RESPONSE])

        # --- Set up external service mocks ---
        trello_client = _setup_trello_mock(MockTrello)
        _setup_mcp_mock(MockMCP)
        cal_mock = _setup_calendar_mock(MockCal)
        docs_mock = _setup_docs_mock(MockDocs)

        # =====================================================================
        # PHASE 1: State Schema — verify initial state is valid
        # =====================================================================
        state = _make_initial_state()

        assert state.project_id == "proj-course-reg"
        assert state.project_brief == _PROJECT_BRIEF
        assert state.final_deadline == _FINAL_DEADLINE
        assert len(state.student_profiles) == 3
        assert state.task_array == []
        assert state.delegation_matrix == {}
        assert state.publishing_status is None

        print("\n[Phase 1] State schema constructed successfully")
        print(f"  - Project: {state.project_name}")
        print(f"  - Students: {[s.name for s in state.student_profiles]}")
        print(f"  - Deadline: {state.final_deadline}")

        # =====================================================================
        # PHASE 3: Task Decomposition — project brief -> task_array
        # =====================================================================
        decomp_result = task_decomposition(state)

        assert "task_array" in decomp_result
        assert "dependency_graph" in decomp_result
        assert len(decomp_result["task_array"]) == 5

        # Merge result into state
        state = state.model_copy(update=decomp_result)

        assert len(state.task_array) == 5
        assert all(t.status == TaskStatus.TODO for t in state.task_array)
        assert all(t.id.startswith("task-") for t in state.task_array)

        # Verify dependency graph
        dep_graph = state.dependency_graph
        assert dep_graph["task-setup-repo"] == []
        assert dep_graph["task-design-db"] == ["task-setup-repo"]
        assert dep_graph["task-build-api"] == ["task-design-db"]
        assert "task-build-api" in dep_graph["task-write-tests"]

        # Verify urgency levels
        task_by_id = {t.id: t for t in state.task_array}
        assert task_by_id["task-setup-repo"].urgency == UrgencyLevel.CRITICAL
        assert task_by_id["task-build-api"].urgency == UrgencyLevel.HIGH
        assert task_by_id["task-write-tests"].urgency == UrgencyLevel.MEDIUM

        print("\n[Phase 3] Task Decomposition completed")
        for t in state.task_array:
            print(f"  - {t.id}: {t.title} ({t.effort_hours}h, {t.urgency.value})")
        print(f"  - Total effort: {sum(t.effort_hours for t in state.task_array)}h")

        # =====================================================================
        # PHASE 4a: Delegation — assign tasks to students + pacing
        # =====================================================================
        deleg_result = delegation(state)

        assert "delegation_matrix" in deleg_result
        assert "task_array" in deleg_result
        assert "project_timeline" in deleg_result

        matrix = deleg_result["delegation_matrix"]
        assert len(matrix) == 5
        assert matrix["task-setup-repo"] == "s-alice"
        assert matrix["task-build-api"] == "s-bob"
        assert matrix["task-build-frontend"] == "s-carol"

        # Verify tasks got deadlines and assignees from pacing
        updated_tasks = deleg_result["task_array"]
        for task in updated_tasks:
            assert task.assigned_to is not None, f"Task {task.id} has no assignee"
            assert task.deadline is not None, f"Task {task.id} has no deadline"
            assert task.deadline <= _FINAL_DEADLINE, (
                f"Task {task.id} deadline {task.deadline} exceeds final {_FINAL_DEADLINE}"
            )

        # Verify dependency ordering: each task's deadline is after its prereqs
        task_dl = {t.id: t.deadline for t in updated_tasks}
        for task in updated_tasks:
            for dep_id in task.dependencies:
                assert task_dl[dep_id] <= task_dl[task.id], (
                    f"Dependency violation: {dep_id} ({task_dl[dep_id]}) "
                    f"should be before {task.id} ({task_dl[task.id]})"
                )

        # Verify burn-down targets exist
        timeline = deleg_result["project_timeline"]
        assert len(timeline.burn_down_targets) > 0

        # Merge into state
        state = state.model_copy(update=deleg_result)

        print("\n[Phase 4a] Delegation completed")
        for tid, sid in state.delegation_matrix.items():
            student = next(s for s in state.student_profiles if s.student_id == sid)
            task = next(t for t in state.task_array if t.id == tid)
            print(f"  - {task.title} -> {student.name} (due {task.deadline:%Y-%m-%d})")

        # =====================================================================
        # PHASE 4b: Equity Evaluator — validate workload fairness
        # =====================================================================
        equity_result = equity_evaluator(state)

        assert "equity_result" in equity_result
        assert "equity_retries" in equity_result
        assert equity_result["equity_result"].balanced is True
        assert equity_result["equity_result"].violations == []
        assert equity_result["equity_retries"] == state.equity_retries + 1

        # Merge into state
        state = state.model_copy(update=equity_result)

        print("\n[Phase 4b] Equity Evaluator passed")
        print(f"  - Balanced: {state.equity_result.balanced}")
        print(f"  - Reasoning: {state.equity_result.reasoning}")

        # =====================================================================
        # PHASE 5: Publishing — push to Trello, Calendar, Docs
        # =====================================================================
        pub_result = await publishing(state)

        # Verify overall status
        status: PublishingStatus = pub_result["publishing_status"]
        assert status.trello == "success", f"Trello failed: {status.errors}"
        assert status.calendar == "success", f"Calendar failed: {status.errors}"
        assert status.docs == "success", f"Docs failed: {status.errors}"
        assert status.errors == []

        # Verify Trello artifacts
        assert pub_result["trello_board_id"] == "board-1"
        assert len(pub_result["trello_card_mapping"]) == 5
        assert trello_client.create_board.call_count == 1
        assert trello_client.create_list.call_count == 4
        assert trello_client.add_label.call_count == 4
        assert trello_client.create_card.call_count == 5

        # Tasks with dependencies should have checklists
        # task-design-db depends on task-setup-repo
        # task-build-api depends on task-design-db
        # task-build-frontend depends on task-build-api
        # task-write-tests depends on task-build-api and task-build-frontend
        assert trello_client.add_checklist.call_count == 4  # 4 tasks have deps

        # Verify calendar events created for all tasks with deadlines
        assert cal_mock.create_event.call_count == 5
        assert len(pub_result["calendar_event_mapping"]) == 5

        # Verify reminders on every calendar event
        for call in cal_mock.create_event.call_args_list:
            assert call.kwargs["reminders"] == [2880, 1440, 120]

        # Verify Google Doc created
        assert docs_mock.create_document.call_count == 1
        assert pub_result["docs_task_matrix_id"] == "doc-1"
        doc_call = docs_mock.create_document.call_args
        assert "Course Registration System" in doc_call.kwargs["title"]
        assert "Task Breakdown Matrix" in doc_call.kwargs["title"]

        # Verify webhook stub
        assert pub_result["webhook_configured"] is False

        # Merge into state
        state = state.model_copy(update=pub_result)

        print("\n[Phase 5] Publishing completed")
        print(f"  - Trello board: {state.trello_board_id}")
        print(f"  - Trello cards: {len(state.trello_card_mapping)}")
        print(f"  - Calendar events: {len(state.calendar_event_mapping)}")
        print(f"  - Google Doc: {state.docs_task_matrix_id}")
        print(f"  - Webhook: {state.webhook_configured}")

        # =====================================================================
        # FINAL: Verify complete state integrity
        # =====================================================================
        assert state.project_id == "proj-course-reg"
        assert len(state.task_array) == 5
        assert len(state.delegation_matrix) == 5
        assert len(state.student_profiles) == 3
        assert state.equity_result is not None
        assert state.equity_result.balanced is True
        assert state.publishing_status is not None
        assert state.publishing_status.trello == "success"
        assert state.publishing_status.calendar == "success"
        assert state.publishing_status.docs == "success"
        assert state.trello_board_id is not None
        assert state.docs_task_matrix_id is not None

        # Every task should be assigned, have a deadline, and have a Trello card
        for task in state.task_array:
            assert task.assigned_to in {s.student_id for s in state.student_profiles}
            assert task.deadline is not None
            assert task.id in state.trello_card_mapping
            assert task.id in state.delegation_matrix
            assert task.id in state.calendar_event_mapping

        print("\n[PASS] All phases 1-5 verified successfully!")
        print(f"  - {len(state.task_array)} tasks decomposed, delegated, and published")
        print(f"  - {len(state.student_profiles)} students with fair workload")
        print(f"  - All external integrations (Trello, Calendar, Docs) succeeded")


class TestPhase1To5PartialFailure:
    """Verifies the pipeline handles a publishing failure gracefully.

    Phases 1-4 succeed, but Trello fails in Phase 5. Calendar and Docs
    should still succeed — the system degrades gracefully.
    """

    @patch.dict(os.environ, {
        "GROQ_API_KEY": "test-key",
        "COURSE_CALENDAR_ID": "test-cal",
        "TRELLO_API_KEY": "k",
        "TRELLO_API_TOKEN": "t",
    })
    @patch("agents.publishing.GoogleDocsMCP")
    @patch("agents.publishing.GoogleCalendarMCP")
    @patch("agents.publishing.SyncUpMCPClient")
    @patch("agents.publishing.TrelloClient")
    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    @patch("agents.delegation.get_high_tier_llm")
    @patch("agents.task_decomposition.get_high_tier_llm")
    async def test_trello_fails_others_succeed(
        self,
        mock_decomp_llm_factory: MagicMock,
        mock_deleg_llm_factory: MagicMock,
        mock_equity_llm_factory: MagicMock,
        MockTrello: MagicMock,
        MockMCP: MagicMock,
        MockCal: MagicMock,
        MockDocs: MagicMock,
    ) -> None:
        mock_decomp_llm_factory.return_value = _make_llm_mock([_DECOMPOSITION_RESPONSE])
        mock_deleg_llm_factory.return_value = _make_llm_mock([_DELEGATION_RESPONSE])
        mock_equity_llm_factory.return_value = _make_llm_mock([_EQUITY_RESPONSE])

        # Trello explodes
        mock_client = AsyncMock()
        mock_client.create_board.side_effect = Exception("Trello is down!")
        MockTrello.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockTrello.return_value.__aexit__ = AsyncMock(return_value=None)

        _setup_mcp_mock(MockMCP)
        _setup_calendar_mock(MockCal)
        _setup_docs_mock(MockDocs)

        # Run phases 1-4
        state = _make_initial_state()
        state = state.model_copy(update=task_decomposition(state))
        state = state.model_copy(update=delegation(state))
        state = state.model_copy(update=equity_evaluator(state))

        # Phase 5 — Trello fails
        pub_result = await publishing(state)
        state = state.model_copy(update=pub_result)

        assert state.publishing_status is not None
        assert state.publishing_status.trello == "failed"
        assert state.publishing_status.calendar == "success"
        assert state.publishing_status.docs == "success"
        assert any("Trello" in e for e in state.publishing_status.errors)
        assert state.trello_board_id is None
        assert state.docs_task_matrix_id == "doc-1"

        print("\n[PASS] Graceful degradation: Trello failed, Calendar+Docs succeeded")


class TestPhase1To5EquityRebalance:
    """Verifies the equity evaluator can reject an unfair delegation.

    The first delegation is rejected by the equity evaluator, and the
    delegation agent re-runs with equity feedback to produce a balanced result.
    """

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("evaluators.equity_evaluator.get_high_tier_llm")
    @patch("agents.delegation.get_high_tier_llm")
    @patch("agents.task_decomposition.get_high_tier_llm")
    async def test_equity_rejection_triggers_redelegation(
        self,
        mock_decomp_llm_factory: MagicMock,
        mock_deleg_llm_factory: MagicMock,
        mock_equity_llm_factory: MagicMock,
    ) -> None:
        mock_decomp_llm_factory.return_value = _make_llm_mock([_DECOMPOSITION_RESPONSE])

        # First delegation call
        mock_deleg_llm_factory.return_value = _make_llm_mock([_DELEGATION_RESPONSE])

        # Equity evaluator: first rejects, second accepts
        unfair_response = json.dumps({
            "balanced": False,
            "reasoning": "Bob has 18h but Alice only has 8h. Imbalanced by 125%.",
            "violations": ["Bob overloaded (18h vs team avg 13.7h)"],
        })
        mock_equity_llm_factory.return_value = _make_llm_mock([unfair_response])

        state = _make_initial_state()
        state = state.model_copy(update=task_decomposition(state))
        state = state.model_copy(update=delegation(state))

        # First equity check — should reject
        eq_result_1 = equity_evaluator(state)
        state = state.model_copy(update=eq_result_1)

        assert state.equity_result is not None
        assert state.equity_result.balanced is False
        assert state.equity_retries == 1
        assert len(state.equity_result.violations) > 0

        print("\n[Phase 4b] First equity check rejected — triggering re-delegation")
        print(f"  - Balanced: {state.equity_result.balanced}")
        print(f"  - Violations: {state.equity_result.violations}")

        # Re-delegate with equity feedback
        # (In the real graph, after_equity_eval routes back to delegation)
        mock_deleg_llm_factory.return_value = _make_llm_mock([_DELEGATION_RESPONSE])
        state = state.model_copy(update=delegation(state))

        # Second equity check — should accept
        fair_response = json.dumps({
            "balanced": True,
            "reasoning": "Workload rebalanced within acceptable range.",
            "violations": [],
        })
        mock_equity_llm_factory.return_value = _make_llm_mock([fair_response])
        eq_result_2 = equity_evaluator(state)
        state = state.model_copy(update=eq_result_2)

        assert state.equity_result.balanced is True
        assert state.equity_retries == 2

        print("\n[Phase 4b] Second equity check passed after re-delegation")
        print(f"  - Balanced: {state.equity_result.balanced}")
        print(f"  - Retries used: {state.equity_retries}")
        print("\n[PASS] Equity rebalance flow verified!")
