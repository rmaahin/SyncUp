"""Tests for the final-report generator service."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from services.report_generator import (
    collect_student_metrics,
    generate_student_report,
    generate_team_report,
)
from state.schema import (
    ContributionRecord,
    EventType,
    Intervention,
    MeetingRecord,
    PeerReviewSummary,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
)

NOW = datetime.now(timezone.utc)
DEADLINE = NOW + timedelta(days=14)


def _state() -> SyncUpState:
    students = [
        StudentProfile(student_id="s1", name="Alice", email="a@x.com"),
        StudentProfile(student_id="s2", name="Bob", email="b@x.com"),
    ]
    tasks = [
        Task(id="t1", title="A", status=TaskStatus.DONE, deadline=DEADLINE),
        Task(id="t2", title="B", status=TaskStatus.DONE, deadline=DEADLINE),
        Task(id="t3", title="C", status=TaskStatus.DONE, deadline=DEADLINE),
    ]
    delegation = {"t1": "s1", "t2": "s1", "t3": "s2"}
    contributions = [
        ContributionRecord(
            student_id="s1", timestamp=NOW, event_type=EventType.COMMIT,
            description="c1", semantic_quality_score=0.8,
        ),
        ContributionRecord(
            student_id="s1", timestamp=NOW, event_type=EventType.COMMIT,
            description="c2", semantic_quality_score=0.2,  # low quality
        ),
        ContributionRecord(
            student_id="s2", timestamp=NOW, event_type=EventType.PR_REVIEW,
            description="pr1", semantic_quality_score=0.9,
        ),
    ]
    meetings = [MeetingRecord(date=NOW, attendees=["s1", "s2"])]
    interventions = [
        Intervention(
            target_student_id="s1",
            trigger_reason="behind",
            message_text="nudge",
            timestamp=NOW,
            outcome="acknowledged",
        )
    ]
    return SyncUpState(
        project_id="proj-r",
        student_profiles=students,
        task_array=tasks,
        delegation_matrix=delegation,
        contribution_ledger=contributions,
        meeting_log=meetings,
        intervention_history=interventions,
    )


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------


class TestCollectMetrics:
    def test_s1_metrics(self) -> None:
        m = collect_student_metrics(_state(), "s1")
        assert m["tasks_assigned"] == 2
        assert m["tasks_completed"] == 2
        assert m["commits"] == 2
        assert m["prs"] == 0
        assert m["low_quality_contribution_count"] == 1  # 0.2 < 0.3
        assert m["meeting_attendance_rate"] == 1.0
        assert m["interventions_received"] == 1
        assert m["intervention_outcomes"] == {"acknowledged": 1}

    def test_s2_metrics(self) -> None:
        m = collect_student_metrics(_state(), "s2")
        assert m["tasks_assigned"] == 1
        assert m["tasks_completed"] == 1
        assert m["prs"] == 1
        assert m["commits"] == 0
        assert m["low_quality_contribution_count"] == 0


# ---------------------------------------------------------------------------
# Student report (LLM patched)
# ---------------------------------------------------------------------------


def _make_llm() -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(
        content=json.dumps({
            "narrative": "Solid contributor.",
            "strengths": ["meets deadlines"],
            "areas_for_improvement": ["communication"],
        })
    )
    return fake


class TestStudentReport:
    def test_no_peer_summary_omitted_from_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _make_llm()
        monkeypatch.setattr(
            "services.report_generator.get_high_tier_llm",
            lambda *a, **k: fake,
        )
        rep = generate_student_report("s1", _state(), None, [])
        assert rep.peer_summary is None
        # Inspect call_args for absence of peer fields
        call_args = fake.invoke.call_args
        msgs = call_args.args[0]
        user_content = msgs[1]["content"]
        assert "peer_summary" not in user_content
        assert rep.narrative == "Solid contributor."

    def test_with_peer_summary_included_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _make_llm()
        monkeypatch.setattr(
            "services.report_generator.get_high_tier_llm",
            lambda *a, **k: fake,
        )
        summary = PeerReviewSummary(
            student_id="s1",
            avg_per_dimension={"contribution_quality": 4.5},
            overall_avg=4.5,
            review_count=3,
        )
        rep = generate_student_report("s1", _state(), summary, [])
        assert rep.peer_summary is not None
        user_content = fake.invoke.call_args.args[0][1]["content"]
        assert "peer_summary" in user_content

    def test_llm_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = MagicMock()
        fake.invoke.side_effect = RuntimeError("groq down")
        monkeypatch.setattr(
            "services.report_generator.get_high_tier_llm",
            lambda *a, **k: fake,
        )
        rep = generate_student_report("s1", _state(), None, [])
        assert "Report generation failed" in rep.narrative
        assert rep.metrics["tasks_assigned"] == 2  # metrics still authoritative


# ---------------------------------------------------------------------------
# Team report
# ---------------------------------------------------------------------------


class TestTeamReport:
    def test_calls_llm_n_plus_one_times(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _make_llm()
        monkeypatch.setattr(
            "services.report_generator.get_high_tier_llm",
            lambda *a, **k: fake,
        )
        report = generate_team_report(_state())
        # 2 students + 1 team narrative
        assert fake.invoke.call_count == 3
        assert len(report.student_reports) == 2
        assert report.completion_pct == 1.0
