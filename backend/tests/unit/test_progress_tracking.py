"""Tests for the Progress Tracking agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents.progress_tracking import (
    INACTIVITY_THRESHOLD_DAYS,
    SemanticAnalysisResult,
    _analyze_quality,
    _build_description,
    _build_raw_metrics,
    _evaluate_progress,
    _extract_json,
    _map_event_type,
    _resolve_student,
    _resolve_student_from_github,
    _resolve_student_from_trello,
    _truncate_diff,
    progress_tracking,
)
from state.schema import (
    BurnDownTarget,
    ContributionRecord,
    EventType,
    ProjectTimeline,
    RawMetrics,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)

QUALITY_RESPONSE_HIGH = json.dumps(
    {"quality_score": 0.8, "is_gaming": False, "reasoning": "Substantive code changes"}
)
QUALITY_RESPONSE_GAMING = json.dumps(
    {"quality_score": 0.1, "is_gaming": True, "reasoning": "Trivial whitespace changes only"}
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


def _make_student(
    student_id: str = "student-1",
    github_username: str = "octocat",
    trello_id: str = "trello-1",
) -> StudentProfile:
    """Create a minimal StudentProfile for testing."""
    return StudentProfile(
        student_id=student_id,
        name="Test Student",
        email="test@example.com",
        github_username=github_username,
        trello_id=trello_id,
    )


def _make_task(
    task_id: str = "task-1",
    status: TaskStatus = TaskStatus.TODO,
    deadline: datetime | None = None,
    effort_hours: float = 10.0,
    assigned_to: str | None = "student-1",
) -> Task:
    """Create a minimal Task for testing."""
    return Task(
        id=task_id,
        title=f"Task {task_id}",
        status=status,
        deadline=deadline,
        effort_hours=effort_hours,
        assigned_to=assigned_to,
    )


def _make_github_push_event(
    username: str = "octocat",
    repo: str = "owner/repo",
) -> dict:
    """Create a sample GitHub push pending_event."""
    return {
        "event_type": "github_push",
        "github_username": username,
        "repository_full_name": repo,
        "commits": [
            {
                "sha": "abc123",
                "message": "Add authentication module",
                "added": ["src/auth.py", "tests/test_auth.py"],
                "removed": [],
                "modified": ["src/main.py"],
                "diff_summary": "+def authenticate(user):\n+    return True",
            }
        ],
        "timestamp": NOW.isoformat(),
    }


def _make_trello_event(card_id: str = "card-1") -> dict:
    """Create a sample Trello card move pending_event."""
    return {
        "event_type": "trello_card_update",
        "trello_card_id": card_id,
        "trello_card_name": "Set up repository",
        "trello_list_before": "list-todo",
        "trello_list_after": "list-in-progress",
        "timestamp": NOW.isoformat(),
    }


def _make_pr_event(username: str = "octocat") -> dict:
    """Create a sample GitHub PR pending_event."""
    return {
        "event_type": "github_pr",
        "github_username": username,
        "repository_full_name": "owner/repo",
        "pr_title": "Add user authentication",
        "pr_action": "opened",
        "pr_number": 42,
        "pr_files_changed": 5,
        "timestamp": NOW.isoformat(),
    }


def _make_state(
    pending_event: dict | None = None,
    students: list[StudentProfile] | None = None,
    tasks: list[Task] | None = None,
    delegation_matrix: dict[str, str] | None = None,
    trello_card_mapping: dict[str, str] | None = None,
    contribution_ledger: list[ContributionRecord] | None = None,
    project_timeline: ProjectTimeline | None = None,
) -> SyncUpState:
    """Create a SyncUpState for testing."""
    return SyncUpState(
        project_id="test-project",
        pending_event=pending_event,
        student_profiles=students or [_make_student()],
        task_array=tasks or [],
        delegation_matrix=delegation_matrix or {},
        trello_card_mapping=trello_card_mapping or {},
        contribution_ledger=contribution_ledger or [],
        project_timeline=project_timeline or ProjectTimeline(),
    )


# =========================================================================
# Helper function tests
# =========================================================================


class TestResolveStudentFromGitHub:
    """Tests for _resolve_student_from_github."""

    def test_match_found(self) -> None:
        profiles = [_make_student(student_id="s1", github_username="octocat")]
        assert _resolve_student_from_github("octocat", profiles) == "s1"

    def test_no_match(self) -> None:
        profiles = [_make_student(student_id="s1", github_username="octocat")]
        assert _resolve_student_from_github("unknown-user", profiles) is None

    def test_empty_profiles(self) -> None:
        assert _resolve_student_from_github("octocat", []) is None


class TestResolveStudentFromTrello:
    """Tests for _resolve_student_from_trello."""

    def test_card_maps_to_student(self) -> None:
        card_mapping = {"task-1": "card-abc"}
        delegation = {"task-1": "student-1"}
        assert _resolve_student_from_trello("card-abc", card_mapping, delegation) == "student-1"

    def test_unknown_card(self) -> None:
        card_mapping = {"task-1": "card-abc"}
        delegation = {"task-1": "student-1"}
        assert _resolve_student_from_trello("card-unknown", card_mapping, delegation) is None

    def test_card_found_but_task_not_delegated(self) -> None:
        card_mapping = {"task-1": "card-abc"}
        delegation: dict[str, str] = {}
        assert _resolve_student_from_trello("card-abc", card_mapping, delegation) is None


class TestResolveStudent:
    """Tests for _resolve_student (dispatcher)."""

    def test_github_push_resolves(self) -> None:
        event = _make_github_push_event(username="octocat")
        state = _make_state()
        assert _resolve_student(event, state) == "student-1"

    def test_trello_resolves(self) -> None:
        event = _make_trello_event(card_id="card-1")
        state = _make_state(
            trello_card_mapping={"task-1": "card-1"},
            delegation_matrix={"task-1": "student-1"},
        )
        assert _resolve_student(event, state) == "student-1"

    def test_unknown_event_type(self) -> None:
        event = {"event_type": "unknown_type"}
        state = _make_state()
        assert _resolve_student(event, state) is None


class TestBuildRawMetrics:
    """Tests for _build_raw_metrics."""

    def test_github_push_metrics(self) -> None:
        event = _make_github_push_event()
        metrics = _build_raw_metrics(event)
        assert metrics.commits_count == 1
        assert metrics.lines_added == 2  # 2 files in "added"
        assert metrics.files_changed == 3  # 2 added + 1 modified

    def test_github_pr_metrics(self) -> None:
        event = _make_pr_event()
        metrics = _build_raw_metrics(event)
        assert metrics.files_changed == 5
        assert metrics.commits_count == 1

    def test_trello_metrics_are_zero(self) -> None:
        event = _make_trello_event()
        metrics = _build_raw_metrics(event)
        assert metrics == RawMetrics()


class TestMapEventType:
    """Tests for _map_event_type."""

    def test_github_push(self) -> None:
        assert _map_event_type("github_push") == EventType.COMMIT

    def test_github_pr(self) -> None:
        assert _map_event_type("github_pr") == EventType.PR_REVIEW

    def test_trello(self) -> None:
        assert _map_event_type("trello_card_update") == EventType.CARD_MOVE

    def test_unknown_defaults_to_commit(self) -> None:
        assert _map_event_type("something_else") == EventType.COMMIT


class TestBuildDescription:
    """Tests for _build_description."""

    def test_single_commit_push(self) -> None:
        event = _make_github_push_event()
        desc = _build_description(event)
        assert "Push to owner/repo" in desc
        assert "Add authentication module" in desc

    def test_pr_description(self) -> None:
        event = _make_pr_event()
        desc = _build_description(event)
        assert "PR opened" in desc
        assert "Add user authentication" in desc

    def test_trello_card_move(self) -> None:
        event = _make_trello_event()
        desc = _build_description(event)
        assert "Card moved" in desc


class TestExtractJson:
    """Tests for _extract_json."""

    def test_plain_json(self) -> None:
        raw = '{"quality_score": 0.8, "is_gaming": false}'
        assert json.loads(_extract_json(raw))["quality_score"] == 0.8

    def test_json_with_fences(self) -> None:
        raw = '```json\n{"quality_score": 0.9}\n```'
        assert json.loads(_extract_json(raw))["quality_score"] == 0.9

    def test_json_with_preamble(self) -> None:
        raw = 'Here is the result:\n{"quality_score": 0.7}'
        assert json.loads(_extract_json(raw))["quality_score"] == 0.7

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("no json here")


class TestTruncateDiff:
    """Tests for _truncate_diff."""

    def test_short_diff_unchanged(self) -> None:
        assert _truncate_diff("short diff", 100) == "short diff"

    def test_long_diff_truncated(self) -> None:
        long_diff = "x" * 600
        result = _truncate_diff(long_diff, 500)
        assert len(result) == 500 + len("... [truncated]")
        assert result.endswith("... [truncated]")


# =========================================================================
# LLM quality analysis tests
# =========================================================================


class TestAnalyzeQuality:
    """Tests for _analyze_quality."""

    def test_high_quality_score(self) -> None:
        llm = _make_mock_llm([QUALITY_RESPONSE_HIGH])
        event = _make_github_push_event()
        metrics = RawMetrics(lines_added=50, lines_removed=10, files_changed=3, commits_count=1)
        result = _analyze_quality(llm, event, metrics)
        assert result.quality_score == 0.8
        assert result.is_gaming is False
        llm.invoke.assert_called_once()

    def test_gaming_detected(self) -> None:
        llm = _make_mock_llm([QUALITY_RESPONSE_GAMING])
        event = _make_github_push_event()
        metrics = RawMetrics(lines_added=1, files_changed=1)
        result = _analyze_quality(llm, event, metrics)
        assert result.quality_score == 0.1
        assert result.is_gaming is True

    def test_llm_failure_defaults_to_half(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        event = _make_github_push_event()
        metrics = RawMetrics()
        result = _analyze_quality(llm, event, metrics)
        assert result.quality_score == 0.5
        assert result.is_gaming is False

    def test_retry_on_first_failure(self) -> None:
        """First call fails, second succeeds."""
        llm = MagicMock()
        fail_msg = MagicMock()
        fail_msg.content = "not json"
        success_msg = MagicMock()
        success_msg.content = QUALITY_RESPONSE_HIGH
        llm.invoke.side_effect = [fail_msg, success_msg]

        event = _make_github_push_event()
        metrics = RawMetrics()
        result = _analyze_quality(llm, event, metrics)
        assert result.quality_score == 0.8
        assert llm.invoke.call_count == 2


# =========================================================================
# Progress evaluation tests
# =========================================================================


class TestEvaluateProgress:
    """Tests for _evaluate_progress."""

    def test_empty_delegation_returns_empty(self) -> None:
        state = _make_state()
        assert _evaluate_progress(state, NOW) == {}

    def test_student_on_track(self) -> None:
        """Student has 1/2 tasks done at 50% of timeline."""
        tasks = [
            _make_task("t1", status=TaskStatus.DONE, effort_hours=10.0),
            _make_task("t2", status=TaskStatus.IN_PROGRESS, effort_hours=10.0,
                       deadline=NOW + timedelta(days=14)),
        ]
        delegation = {"t1": "student-1", "t2": "student-1"}
        # Burn-down target: at NOW, expect 10 hours remaining out of 20 total
        timeline = ProjectTimeline(
            burn_down_targets=[BurnDownTarget(date=NOW, target_hours_remaining=10.0)]
        )
        ledger = [
            ContributionRecord(
                student_id="student-1",
                timestamp=NOW - timedelta(hours=1),
                event_type=EventType.COMMIT,
                semantic_quality_score=0.8,
            )
        ]
        state = _make_state(
            tasks=tasks,
            delegation_matrix=delegation,
            project_timeline=timeline,
            contribution_ledger=ledger,
        )
        result = _evaluate_progress(state, NOW)
        assert result["student-1"] == "on_track"

    def test_student_behind_overdue_task(self) -> None:
        """Student has a task past deadline and not done → behind."""
        tasks = [
            _make_task("t1", status=TaskStatus.TODO, deadline=NOW - timedelta(days=2)),
        ]
        delegation = {"t1": "student-1"}
        state = _make_state(tasks=tasks, delegation_matrix=delegation)
        result = _evaluate_progress(state, NOW)
        assert result["student-1"] == "behind"

    def test_student_behind_burn_down(self) -> None:
        """Student has 0% done but burn-down expects 50% → behind."""
        tasks = [
            _make_task("t1", status=TaskStatus.TODO, effort_hours=10.0,
                       deadline=NOW + timedelta(days=7)),
            _make_task("t2", status=TaskStatus.TODO, effort_hours=10.0,
                       deadline=NOW + timedelta(days=14)),
        ]
        delegation = {"t1": "student-1", "t2": "student-1"}
        # Burn-down says only 10 hours should remain (50% done)
        timeline = ProjectTimeline(
            burn_down_targets=[BurnDownTarget(date=NOW, target_hours_remaining=10.0)]
        )
        ledger = [
            ContributionRecord(
                student_id="student-1",
                timestamp=NOW - timedelta(hours=1),
                event_type=EventType.COMMIT,
                semantic_quality_score=0.5,
            )
        ]
        state = _make_state(
            tasks=tasks,
            delegation_matrix=delegation,
            project_timeline=timeline,
            contribution_ledger=ledger,
        )
        result = _evaluate_progress(state, NOW)
        assert result["student-1"] == "behind"

    def test_inactive_student_at_risk(self) -> None:
        """Student with no contributions for >3 days → at_risk."""
        tasks = [
            _make_task("t1", status=TaskStatus.IN_PROGRESS,
                       deadline=NOW + timedelta(days=14)),
        ]
        delegation = {"t1": "student-1"}
        # Last activity was 5 days ago
        ledger = [
            ContributionRecord(
                student_id="student-1",
                timestamp=NOW - timedelta(days=5),
                event_type=EventType.COMMIT,
                semantic_quality_score=0.7,
            )
        ]
        state = _make_state(
            tasks=tasks,
            delegation_matrix=delegation,
            contribution_ledger=ledger,
        )
        result = _evaluate_progress(state, NOW)
        assert result["student-1"] == "at_risk"

    def test_no_contributions_ever_at_risk(self) -> None:
        """Student with zero contributions and tasks not done → at_risk."""
        tasks = [
            _make_task("t1", status=TaskStatus.TODO,
                       deadline=NOW + timedelta(days=14)),
        ]
        delegation = {"t1": "student-1"}
        state = _make_state(tasks=tasks, delegation_matrix=delegation)
        result = _evaluate_progress(state, NOW)
        assert result["student-1"] == "at_risk"

    def test_all_done_student_not_at_risk_even_if_inactive(self) -> None:
        """Student with all tasks done is on_track even with no recent activity."""
        tasks = [
            _make_task("t1", status=TaskStatus.DONE, effort_hours=10.0),
        ]
        delegation = {"t1": "student-1"}
        state = _make_state(tasks=tasks, delegation_matrix=delegation)
        result = _evaluate_progress(state, NOW)
        # All tasks done, inactivity check should not flag
        assert result["student-1"] == "on_track"


# =========================================================================
# Agent node tests
# =========================================================================


class TestProgressTrackingNode:
    """Tests for the progress_tracking agent node function."""

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_github_push_creates_contribution_record(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """A GitHub push event should produce a ContributionRecord."""
        mock_get_llm.return_value = _make_mock_llm([QUALITY_RESPONSE_HIGH])
        event = _make_github_push_event()
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        assert "contribution_ledger" in result
        assert len(result["contribution_ledger"]) == 1
        record = result["contribution_ledger"][0]
        assert isinstance(record, ContributionRecord)
        assert record.student_id == "student-1"
        assert record.event_type == EventType.COMMIT
        assert record.semantic_quality_score == 0.8
        assert result["pending_event"] is None

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_github_pr_creates_record(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """A GitHub PR event should produce a ContributionRecord."""
        mock_get_llm.return_value = _make_mock_llm([QUALITY_RESPONSE_HIGH])
        event = _make_pr_event()
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        assert len(result["contribution_ledger"]) == 1
        record = result["contribution_ledger"][0]
        assert record.event_type == EventType.PR_REVIEW
        assert record.semantic_quality_score == 0.8

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    def test_trello_card_move_creates_record(self, mock_now: MagicMock) -> None:
        """A Trello card move should produce a record with neutral quality score."""
        event = _make_trello_event(card_id="card-1")
        state = _make_state(
            pending_event=event,
            trello_card_mapping={"task-1": "card-1"},
            delegation_matrix={"task-1": "student-1"},
        )

        result = progress_tracking(state)

        assert len(result["contribution_ledger"]) == 1
        record = result["contribution_ledger"][0]
        assert record.event_type == EventType.CARD_MOVE
        assert record.semantic_quality_score == 0.5  # neutral for card moves

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_gaming_detection_low_score(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """Gaming detection should record the low quality score."""
        mock_get_llm.return_value = _make_mock_llm([QUALITY_RESPONSE_GAMING])
        event = _make_github_push_event()
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        record = result["contribution_ledger"][0]
        assert record.semantic_quality_score == 0.1

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_llm_failure_defaults_to_half(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """LLM failure should default quality_score to 0.5."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_get_llm.return_value = mock_llm
        event = _make_github_push_event()
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        record = result["contribution_ledger"][0]
        assert record.semantic_quality_score == 0.5

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    def test_pending_event_none_is_noop(self, mock_now: MagicMock) -> None:
        """No pending event should return progress only, no contribution record."""
        state = _make_state(pending_event=None)

        result = progress_tracking(state)

        assert result["pending_event"] is None
        assert "contribution_ledger" not in result
        assert "student_progress" in result

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_unknown_github_user_skipped(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """Unknown GitHub username should skip processing."""
        event = _make_github_push_event(username="unknown-user")
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        assert result["pending_event"] is None
        assert "contribution_ledger" not in result
        mock_get_llm.assert_not_called()

    @patch("agents.progress_tracking._get_now", return_value=NOW)
    @patch("agents.progress_tracking.get_low_tier_llm")
    def test_clears_pending_event_after_processing(
        self, mock_get_llm: MagicMock, mock_now: MagicMock
    ) -> None:
        """pending_event should be set to None after processing."""
        mock_get_llm.return_value = _make_mock_llm([QUALITY_RESPONSE_HIGH])
        event = _make_github_push_event()
        state = _make_state(pending_event=event)

        result = progress_tracking(state)

        assert result["pending_event"] is None
