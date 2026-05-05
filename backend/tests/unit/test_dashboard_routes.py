"""Tests for the dashboard read endpoints (student + professor)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from state.schema import (
    ContributionRecord,
    EventType,
    Intervention,
    MeetingRecord,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
)
from state.store import InMemoryStateStore

client = TestClient(app, raise_server_exceptions=False)


def _populated_state(project_id: str = "proj-d") -> SyncUpState:
    """Build a fully populated SyncUpState fixture for dashboard tests."""
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(days=14)

    students = [
        StudentProfile(
            student_id="s1", name="Alice", email="a@x.com", availability_hours_per_week=10.0
        ),
        StudentProfile(
            student_id="s2", name="Bob", email="b@x.com", availability_hours_per_week=10.0
        ),
        StudentProfile(
            student_id="s3", name="Carol", email="c@x.com", availability_hours_per_week=10.0
        ),
    ]

    tasks = [
        Task(id="t1", title="Design", status=TaskStatus.DONE, deadline=deadline),
        Task(id="t2", title="Backend", status=TaskStatus.DONE, deadline=deadline),
        Task(id="t3", title="Frontend", status=TaskStatus.IN_PROGRESS, deadline=deadline),
        Task(id="t4", title="Tests", status=TaskStatus.TODO, deadline=deadline),
        Task(id="t5", title="Docs", status=TaskStatus.TODO, deadline=deadline),
    ]

    delegation = {"t1": "s1", "t2": "s1", "t3": "s2", "t4": "s2", "t5": "s3"}

    contributions = [
        ContributionRecord(
            student_id="s1",
            timestamp=now - timedelta(days=2),
            event_type=EventType.COMMIT,
            description="commit A",
            semantic_quality_score=0.8,
        ),
        ContributionRecord(
            student_id="s1",
            timestamp=now - timedelta(days=1),
            event_type=EventType.COMMIT,
            description="commit B",
            semantic_quality_score=0.6,
        ),
        ContributionRecord(
            student_id="s2",
            timestamp=now - timedelta(hours=5),
            event_type=EventType.COMMIT,
            description="commit C",
            semantic_quality_score=0.9,
        ),
        ContributionRecord(
            student_id="s3",
            timestamp=now - timedelta(days=3),
            event_type=EventType.DOC_EDIT,
            description="doc edit",
            semantic_quality_score=0.5,
        ),
    ]

    interventions = [
        Intervention(
            target_student_id="s2",
            trigger_reason="behind",
            message_text="Check in?",
            timestamp=now - timedelta(days=1),
        ),
        Intervention(
            target_student_id="s3",
            trigger_reason="inactive",
            message_text="Are you ok?",
            timestamp=now - timedelta(hours=2),
        ),
    ]

    meetings = [
        MeetingRecord(
            date=now - timedelta(days=5),
            attendees=["s1", "s2", "s3"],
            agenda="kickoff",
            notes="went well",
        ),
    ]

    return SyncUpState(
        project_id=project_id,
        project_name="Demo",
        project_brief="A project",
        final_deadline=deadline,
        student_profiles=students,
        task_array=tasks,
        delegation_matrix=delegation,
        contribution_ledger=contributions,
        intervention_history=interventions,
        meeting_log=meetings,
        student_progress={"s1": "on_track", "s2": "behind", "s3": "at_risk"},
        next_meeting_scheduled=now + timedelta(days=2),
    )


@pytest.fixture(autouse=True)
def _setup_state() -> Any:
    """Inject a populated state into a fresh store per test."""
    app.state.state_store = InMemoryStateStore()
    state = _populated_state()
    asyncio.get_event_loop().run_until_complete(
        app.state.state_store.save("proj-d", state)
    )
    yield state


# ---------------------------------------------------------------------------
# Student endpoints
# ---------------------------------------------------------------------------


class TestStudentTasks:
    def test_returns_only_assigned(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s1/tasks")
        assert resp.status_code == 200
        ids = {t["id"] for t in resp.json()["tasks"]}
        assert ids == {"t1", "t2"}

    def test_404_for_missing_project(self) -> None:
        resp = client.get("/api/dashboard/student/nope/s1/tasks")
        assert resp.status_code == 404

    def test_404_for_missing_student(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/missing/tasks")
        assert resp.status_code == 404


class TestStudentProgress:
    def test_returns_correct_completion_pct(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s1/progress")
        assert resp.status_code == 200
        data = resp.json()
        # s1 has t1 (done) and t2 (done) → 100%
        assert data["completion_pct"] == 100.0
        assert data["progress_status"] == "on_track"

    def test_partial_completion(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s2/progress")
        data = resp.json()
        # s2 has t3 (in_progress) and t4 (todo) → 0%
        assert data["completion_pct"] == 0.0
        assert data["progress_status"] == "behind"


class TestStudentMeetings:
    def test_includes_past_and_upcoming(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s1/meetings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["past_meetings"]) == 1
        assert data["upcoming"] is not None


class TestStudentNotifications:
    def test_returns_only_for_student(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s2/notifications")
        assert resp.status_code == 200
        notifs = resp.json()["notifications"]
        assert len(notifs) == 1
        assert notifs[0]["target_student_id"] == "s2"

    def test_empty_for_uninvolved_student(self) -> None:
        resp = client.get("/api/dashboard/student/proj-d/s1/notifications")
        assert resp.json()["notifications"] == []


# ---------------------------------------------------------------------------
# Professor endpoints
# ---------------------------------------------------------------------------


class TestProfessorOverview:
    def test_returns_aggregate_stats(self) -> None:
        resp = client.get("/api/dashboard/professor/proj-d/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 5
        assert data["completed_tasks"] == 2
        assert data["completion_pct"] == 40.0
        assert data["student_count"] == 3
        assert data["on_track_count"] == 1
        assert data["at_risk_count"] == 1
        assert data["behind_count"] == 1


class TestProfessorStudents:
    def test_returns_per_student_summaries(self) -> None:
        resp = client.get("/api/dashboard/professor/proj-d/students")
        assert resp.status_code == 200
        students = resp.json()["students"]
        assert len(students) == 3
        s1 = next(s for s in students if s["student_id"] == "s1")
        assert s1["tasks_assigned"] == 2
        assert s1["tasks_completed"] == 2
        assert s1["contribution_count"] == 2


class TestProfessorContributions:
    def test_pagination_and_sort(self) -> None:
        resp = client.get(
            "/api/dashboard/professor/proj-d/student/s1/contributions?limit=1&offset=0"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 1
        # newest first → "commit B" (timestamp = now - 1 day)
        assert data["items"][0]["description"] == "commit B"


class TestProfessorInterventions:
    def test_sorted_newest_first(self) -> None:
        resp = client.get("/api/dashboard/professor/proj-d/interventions")
        data = resp.json()["interventions"]
        assert len(data) == 2
        # newest first → s3 intervention (2 hours ago)
        assert data[0]["target_student_id"] == "s3"


class TestProfessorTimeline:
    def test_returns_timeline(self) -> None:
        resp = client.get("/api/dashboard/professor/proj-d/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "milestones" in data
        assert "burn_down_targets" in data


class TestEdgeCases:
    def test_empty_project_returns_zeros(self) -> None:
        empty = SyncUpState(project_id="empty", project_name="E")
        asyncio.get_event_loop().run_until_complete(
            app.state.state_store.save("empty", empty)
        )
        resp = client.get("/api/dashboard/professor/empty/overview")
        data = resp.json()
        assert data["total_tasks"] == 0
        assert data["completion_pct"] == 0.0
        assert data["student_count"] == 0
