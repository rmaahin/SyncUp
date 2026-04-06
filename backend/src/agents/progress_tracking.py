"""Progress Tracking agent — processes webhook events and monitors student contributions.

Event-driven agent that wakes on incoming webhook events (GitHub push/PR,
Trello card moves), scores contribution quality via the low-tier Groq LLM,
appends to the immutable contribution ledger, and evaluates whether each
student is on track against the burn-down curve.

Uses LOW-TIER LLM (llama-3.1-8b-instant via ``get_low_tier_llm()``) because
this agent handles high-frequency events and needs to be cheap/fast.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from guardrails.sanitizer import sanitize_text, wrap_untrusted
from llm import get_low_tier_llm
from state.schema import (
    ContributionRecord,
    EventType,
    RawMetrics,
    StudentProfile,
    SyncUpState,
    TaskStatus,
)

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 1  # 2 total attempts (1 initial + 1 retry)
INACTIVITY_THRESHOLD_DAYS: int = 3
BEHIND_THRESHOLD: float = 0.15  # 15% behind burn-down = "behind"
MAX_DIFF_CHARS: int = 500

# ---------------------------------------------------------------------------
# Pydantic model for LLM output
# ---------------------------------------------------------------------------


class SemanticAnalysisResult(BaseModel):
    """Structured output from the low-tier LLM quality analysis."""

    quality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    is_gaming: bool = False
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a code contribution quality analyzer for student group projects.
Rate the quality of this contribution from 0.0 (trivial/gaming) to 1.0 (substantial/meaningful).

Gaming indicators: auto-generated boilerplate, meaningless whitespace-only changes, \
adding/removing the same lines repeatedly, single-character renames, empty commits.

Respond with ONLY a JSON object:
{"quality_score": <float 0.0-1.0>, "is_gaming": <bool>, "reasoning": "<1 sentence>"}"""


def _build_user_prompt(event: dict[str, Any], raw_metrics: RawMetrics) -> str:
    """Build the user message for LLM quality analysis.

    Args:
        event: The pending_event dict containing commit/PR data.
        raw_metrics: Pre-computed raw metrics for the contribution.

    Returns:
        The formatted user prompt string.
    """
    event_type = event.get("event_type", "unknown")

    # Gather commit messages (sanitise untrusted external content)
    commit_messages = ""
    if event_type == "github_push":
        commits = event.get("commits", [])
        commit_messages = sanitize_text(
            "; ".join(c.get("message", "") for c in commits)
        )
    elif event_type == "github_pr":
        commit_messages = sanitize_text(event.get("pr_title", ""))

    # Gather diff summary (sanitise before truncation)
    diff_summary = ""
    if event_type == "github_push":
        diffs = [c.get("diff_summary", "") for c in event.get("commits", [])]
        diff_summary = sanitize_text("\n".join(d for d in diffs if d))

    diff_summary = _truncate_diff(diff_summary, MAX_DIFF_CHARS)

    return (
        f"Event: {event_type}\n"
        f"Commit messages: {commit_messages}\n"
        f"Files changed: {raw_metrics.files_changed}, "
        f"Lines added: {raw_metrics.lines_added}, "
        f"Lines removed: {raw_metrics.lines_removed}\n"
        f"Diff summary:\n"
        f"{wrap_untrusted(diff_summary, 'github')}"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_now() -> datetime:
    """Return current UTC time. Patchable in tests."""
    return datetime.now(timezone.utc)


def _truncate_diff(diff_text: str, max_chars: int = MAX_DIFF_CHARS) -> str:
    """Truncate diff text to a maximum number of characters.

    Args:
        diff_text: The raw diff text.
        max_chars: Maximum character count.

    Returns:
        Truncated diff text with ellipsis if truncated.
    """
    if len(diff_text) <= max_chars:
        return diff_text
    return diff_text[:max_chars] + "... [truncated]"


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and preamble text to isolate JSON.

    Args:
        raw: Raw LLM response text.

    Returns:
        A cleaned string that should be valid JSON.

    Raises:
        ValueError: If no JSON object can be located in the response.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    # Scan for the first '{' to skip any preamble text
    brace_idx = text.find("{")
    if brace_idx == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[brace_idx:]


def _resolve_student_from_github(
    github_username: str,
    profiles: list[StudentProfile],
) -> str | None:
    """Map a GitHub username to a student_id.

    Args:
        github_username: The GitHub username from the webhook event.
        profiles: List of student profiles in state.

    Returns:
        The ``student_id`` if found, ``None`` otherwise.
    """
    for profile in profiles:
        if profile.github_username == github_username:
            return profile.student_id
    return None


def _resolve_student_from_trello(
    card_id: str,
    trello_card_mapping: dict[str, str],
    delegation_matrix: dict[str, str],
) -> str | None:
    """Map a Trello card_id to a student_id via card→task→student.

    Args:
        card_id: The Trello card ID from the webhook event.
        trello_card_mapping: task_id → card_id mapping from state.
        delegation_matrix: task_id → student_id mapping from state.

    Returns:
        The ``student_id`` if found, ``None`` otherwise.
    """
    # Invert card mapping: card_id → task_id
    card_to_task = {v: k for k, v in trello_card_mapping.items()}
    task_id = card_to_task.get(card_id)
    if task_id is None:
        return None
    return delegation_matrix.get(task_id)


def _resolve_student(
    event: dict[str, Any],
    state: SyncUpState,
) -> str | None:
    """Resolve the student responsible for a webhook event.

    Args:
        event: The pending_event dict.
        state: The current SyncUpState.

    Returns:
        The ``student_id`` if resolved, ``None`` otherwise.
    """
    event_type = event.get("event_type", "")

    if event_type in ("github_push", "github_pr"):
        username = event.get("github_username", "")
        if not username:
            return None
        return _resolve_student_from_github(username, state.student_profiles)

    if event_type == "trello_card_update":
        card_id = event.get("trello_card_id", "")
        if not card_id:
            return None
        return _resolve_student_from_trello(
            card_id, state.trello_card_mapping, state.delegation_matrix
        )

    return None


def _build_raw_metrics(event: dict[str, Any]) -> RawMetrics:
    """Extract raw metrics from a pending event.

    Args:
        event: The pending_event dict.

    Returns:
        A ``RawMetrics`` instance with lines added/removed/files changed.
    """
    event_type = event.get("event_type", "")

    if event_type == "github_push":
        commits = event.get("commits", [])
        lines_added = 0
        lines_removed = 0
        files_changed_set: set[str] = set()
        for commit in commits:
            lines_added += len(commit.get("added", []))
            lines_removed += len(commit.get("removed", []))
            files_changed_set.update(commit.get("added", []))
            files_changed_set.update(commit.get("removed", []))
            files_changed_set.update(commit.get("modified", []))
        return RawMetrics(
            lines_added=lines_added,
            lines_removed=lines_removed,
            files_changed=len(files_changed_set),
            commits_count=len(commits),
        )

    if event_type == "github_pr":
        return RawMetrics(
            files_changed=event.get("pr_files_changed", 0),
            commits_count=1,
        )

    # Trello card moves have no code metrics
    return RawMetrics()


def _map_event_type(event_type: str) -> EventType:
    """Map a pending_event event_type string to the EventType enum.

    Args:
        event_type: The string event type from pending_event.

    Returns:
        The corresponding ``EventType`` enum value.
    """
    mapping = {
        "github_push": EventType.COMMIT,
        "github_pr": EventType.PR_REVIEW,
        "trello_card_update": EventType.CARD_MOVE,
    }
    return mapping.get(event_type, EventType.COMMIT)


def _build_description(event: dict[str, Any]) -> str:
    """Build a human-readable description of the event.

    Args:
        event: The pending_event dict.

    Returns:
        A short description string.
    """
    event_type = event.get("event_type", "")

    if event_type == "github_push":
        commits = event.get("commits", [])
        count = len(commits)
        repo = event.get("repository_full_name", "unknown")
        if count == 1 and commits:
            msg = sanitize_text(commits[0].get("message", ""))
            return f"Push to {repo}: {msg}"
        return f"Push to {repo}: {count} commits"

    if event_type == "github_pr":
        action = event.get("pr_action", "")
        title = sanitize_text(event.get("pr_title", ""))
        return f"PR {action}: {title}"

    if event_type == "trello_card_update":
        card_name = sanitize_text(event.get("trello_card_name", ""))
        list_after = event.get("trello_list_after", "")
        return f"Card moved: {card_name} → {list_after}"

    return "Unknown event"


# ---------------------------------------------------------------------------
# LLM quality analysis
# ---------------------------------------------------------------------------


def _analyze_quality(
    llm: Any,
    event: dict[str, Any],
    raw_metrics: RawMetrics,
) -> SemanticAnalysisResult:
    """Call the low-tier LLM to analyze contribution quality.

    Args:
        llm: The ChatGroq LLM instance.
        event: The pending_event dict.
        raw_metrics: Pre-computed raw metrics.

    Returns:
        A ``SemanticAnalysisResult`` with quality score and gaming flag.
        Defaults to score=0.5 if the LLM call fails.
    """
    user_msg = _build_user_prompt(event, raw_metrics)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            cleaned = _extract_json(response.content)
            data = json.loads(cleaned)
            return SemanticAnalysisResult.model_validate(data)
        except Exception as exc:
            logger.warning(
                "Quality analysis attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error("Quality analysis failed after %d attempts — using default", MAX_RETRIES + 1)
    return SemanticAnalysisResult(quality_score=0.5, is_gaming=False, reasoning="LLM analysis failed")


# ---------------------------------------------------------------------------
# Progress evaluation
# ---------------------------------------------------------------------------


def _evaluate_progress(
    state: SyncUpState,
    now: datetime,
) -> dict[str, str]:
    """Evaluate each student's progress against the burn-down curve.

    For each student with assigned tasks, determines one of:
    - ``"on_track"`` — meeting or ahead of burn-down targets
    - ``"at_risk"`` — slightly behind or inactive
    - ``"behind"`` — missed deadline(s) or significantly behind

    Args:
        state: The current SyncUpState.
        now: The current UTC datetime.

    Returns:
        A dict mapping student_id → progress status string.
    """
    if not state.delegation_matrix:
        return {}

    # Build per-student task lists
    student_tasks: dict[str, list[str]] = {}
    for task_id, student_id in state.delegation_matrix.items():
        student_tasks.setdefault(student_id, []).append(task_id)

    # Index tasks by ID for fast lookup
    task_index = {t.id: t for t in state.task_array}

    # Compute total project effort for burn-down comparison
    total_effort = sum(t.effort_hours for t in state.task_array) or 1.0

    # Find latest contribution per student
    last_activity: dict[str, datetime] = {}
    for record in state.contribution_ledger:
        existing = last_activity.get(record.student_id)
        if existing is None or record.timestamp > existing:
            last_activity[record.student_id] = record.timestamp

    progress: dict[str, str] = {}

    for student_id, task_ids in student_tasks.items():
        tasks = [task_index[tid] for tid in task_ids if tid in task_index]
        if not tasks:
            continue

        total_count = len(tasks)
        done_count = sum(1 for t in tasks if t.status == TaskStatus.DONE)
        completion_ratio = done_count / total_count if total_count > 0 else 0.0

        # Check for overdue tasks (deadline passed, not done)
        has_overdue = any(
            t.deadline is not None
            and t.deadline < now
            and t.status != TaskStatus.DONE
            for t in tasks
        )
        if has_overdue:
            progress[student_id] = "behind"
            continue

        # Compare against burn-down curve
        status = "on_track"
        if state.project_timeline.burn_down_targets:
            # Find the nearest target to now
            nearest_target = min(
                state.project_timeline.burn_down_targets,
                key=lambda t: abs((t.date - now).total_seconds()),
            )
            expected_remaining_ratio = nearest_target.target_hours_remaining / total_effort
            expected_completion = 1.0 - expected_remaining_ratio
            if completion_ratio < expected_completion - BEHIND_THRESHOLD:
                status = "behind"

        # Check inactivity
        last = last_activity.get(student_id)
        inactive = (
            last is None or (now - last) > timedelta(days=INACTIVITY_THRESHOLD_DAYS)
        )
        if inactive and done_count < total_count:
            # Inactivity upgrades on_track to at_risk, doesn't downgrade behind
            if status == "on_track":
                status = "at_risk"

        progress[student_id] = status

    return progress


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def progress_tracking(state: SyncUpState) -> dict[str, Any]:
    """Progress Tracking agent node.

    Processes a pending webhook event (if any), scores the contribution
    quality via the low-tier LLM, appends a ``ContributionRecord`` to the
    ledger, and evaluates each student's progress against the burn-down curve.

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``contribution_ledger`` (list to append), ``pending_event``
        (cleared to None), and ``student_progress`` (updated mapping).
    """
    now = _get_now()

    if state.pending_event is None:
        # No event to process — just evaluate current progress
        progress = _evaluate_progress(state, now)
        return {"student_progress": progress, "pending_event": None}

    event = state.pending_event
    event_type = event.get("event_type", "")

    # 1. Resolve student
    student_id = _resolve_student(event, state)
    if student_id is None:
        logger.warning(
            "Could not resolve student for event type=%s — skipping",
            event_type,
        )
        progress = _evaluate_progress(state, now)
        return {"student_progress": progress, "pending_event": None}

    # 2. Build raw metrics
    raw_metrics = _build_raw_metrics(event)

    # 3. Determine event type enum
    event_type_enum = _map_event_type(event_type)

    # 4. Semantic quality analysis (LLM for commits/PRs, neutral for card moves)
    if event_type_enum in (EventType.COMMIT, EventType.PR_REVIEW):
        llm = get_low_tier_llm(temperature=0.1)
        analysis = _analyze_quality(llm, event, raw_metrics)
        quality_score = analysis.quality_score
    else:
        quality_score = 0.5  # neutral for card moves

    # 5. Create ContributionRecord
    # Use event timestamp if available, otherwise now
    event_timestamp_str = event.get("timestamp", "")
    try:
        event_timestamp = datetime.fromisoformat(event_timestamp_str)
    except (ValueError, TypeError):
        event_timestamp = now

    # Capture who performed the action (for Trello: memberCreator)
    performed_by = ""
    if event.get("event_type") == "trello_card_update":
        creator_name = event.get("member_creator_full_name", "")
        creator_user = event.get("member_creator_username", "")
        performed_by = creator_name or creator_user
    elif event.get("event_type") == "github_push":
        performed_by = event.get("github_username", "")
    elif event.get("event_type") == "github_pr":
        performed_by = event.get("github_username", "")

    record = ContributionRecord(
        student_id=student_id,
        timestamp=event_timestamp,
        event_type=event_type_enum,
        description=_build_description(event),
        semantic_quality_score=quality_score,
        raw_metrics=raw_metrics,
        performed_by=performed_by,
    )

    # 6. Evaluate progress
    progress = _evaluate_progress(state, now)

    return {
        "contribution_ledger": [record],  # appended via operator.add reducer
        "pending_event": None,
        "student_progress": progress,
    }
