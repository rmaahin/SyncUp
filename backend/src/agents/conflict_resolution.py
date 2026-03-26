"""Conflict Resolution agent — drafts empathetic intervention nudges for behind students.

When Progress Tracking marks a student as "behind", this agent gathers full
context (overdue tasks, contribution recency, blackout periods, blocked
teammates, past interventions) and calls the HIGH-TIER Groq LLM to draft
a constructive nudge message.

Uses HIGH-TIER LLM (llama-3.3-70b-versatile via ``get_high_tier_llm()``) because
this agent needs nuanced, empathetic writing — the 8B model is too blunt.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from llm import get_high_tier_llm
from state.schema import (
    DraftIntervention,
    Intervention,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
    ToneResult,
)

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts
MAX_TONE_REWRITES: int = 3


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a supportive project coordinator for a student team.
A team member is falling behind. Draft a brief, constructive intervention \
message (3-5 sentences) that:
- Acknowledges the specific overdue task by name
- Asks about potential roadblocks (don't assume laziness)
- Offers a concrete next step
- Mentions downstream impact on teammates WITHOUT blaming
- If blackout/constraint info is provided, acknowledge it
- If this is a repeat intervention, escalate gently (not punitively)
DO NOT be passive-aggressive, accusatory, or use guilt tactics.

You MUST respond with ONLY a JSON object — no markdown fences, no \
explanation, no text before or after the JSON:
{
  "message": "the nudge message text",
  "suggested_action": "extend_deadline" or "redistribute_task" or \
"schedule_check_in" or "no_action",
  "severity": "low" or "medium" or "high",
  "affected_teammates": ["list of teammate names blocked by this"]
}
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_now() -> datetime:
    """Return current UTC time. Patchable in tests."""
    return datetime.now(timezone.utc)


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
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    brace_idx = text.find("{")
    if brace_idx == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[brace_idx:]


def _find_first_behind_student(state: SyncUpState) -> str | None:
    """Return the first student_id with 'behind' status, or None.

    Args:
        state: The current SyncUpState.

    Returns:
        A student_id string or None if no student is behind.
    """
    for student_id, status in state.student_progress.items():
        if status == "behind":
            return student_id
    return None


def _gather_context(
    student_id: str,
    state: SyncUpState,
    now: datetime,
) -> dict[str, Any]:
    """Collect full context for a behind student.

    Gathers overdue tasks, last contribution, intervention history,
    blocked teammates, and active blackout periods.

    Args:
        student_id: The behind student's ID.
        state: The current SyncUpState.
        now: The current UTC datetime.

    Returns:
        A dict of context information for prompt building.
    """
    student_map = {s.student_id: s for s in state.student_profiles}
    task_map = {t.id: t for t in state.task_array}
    student = student_map.get(student_id)

    # Overdue tasks assigned to this student
    assigned_task_ids = [
        tid for tid, sid in state.delegation_matrix.items() if sid == student_id
    ]
    overdue_tasks: list[dict[str, Any]] = []
    for tid in assigned_task_ids:
        task = task_map.get(tid)
        if (
            task
            and task.deadline is not None
            and task.deadline < now
            and task.status != TaskStatus.DONE
        ):
            days_overdue = (now - task.deadline).days
            overdue_tasks.append(
                {"task_id": tid, "title": task.title, "days_overdue": days_overdue}
            )

    # Last contribution from this student
    last_contribution: datetime | None = None
    for record in state.contribution_ledger:
        if record.student_id == student_id:
            if last_contribution is None or record.timestamp > last_contribution:
                last_contribution = record.timestamp

    days_since_last = None
    if last_contribution is not None:
        days_since_last = (now - last_contribution).days

    # Past interventions for this student
    past_interventions = [
        i for i in state.intervention_history if i.target_student_id == student_id
    ]

    # Blocked teammates: who depends on this student's overdue tasks?
    blocked_teammates: list[str] = []
    overdue_task_ids = {t["task_id"] for t in overdue_tasks}
    for task in state.task_array:
        if task.assigned_to and task.assigned_to != student_id:
            # Check if any of this task's dependencies are overdue for our student
            for dep_id in task.dependencies:
                if dep_id in overdue_task_ids:
                    teammate = student_map.get(task.assigned_to)
                    name = teammate.name if teammate else task.assigned_to
                    if name not in blocked_teammates:
                        blocked_teammates.append(name)

    # Active blackout periods
    active_blackouts: list[str] = []
    if student:
        for bp in student.blackout_periods:
            if bp.start <= now <= bp.end:
                active_blackouts.append(
                    f"{bp.start.strftime('%Y-%m-%d')} to {bp.end.strftime('%Y-%m-%d')}"
                )

    return {
        "student_name": student.name if student else student_id,
        "student_id": student_id,
        "overdue_tasks": overdue_tasks,
        "days_since_last_contribution": days_since_last,
        "past_intervention_count": len(past_interventions),
        "blocked_teammates": blocked_teammates,
        "active_blackouts": active_blackouts,
    }


def _build_user_prompt(
    context: dict[str, Any],
    rewrite_feedback: list[str] | None = None,
) -> str:
    """Build the user message for the conflict resolution LLM call.

    Args:
        context: The gathered student context dict.
        rewrite_feedback: Flagged phrases from a punitive tone evaluation,
            if this is a rewrite attempt. None on first attempt.

    Returns:
        The formatted user prompt string.
    """
    lines: list[str] = [f"STUDENT: {context['student_name']}"]

    # Overdue tasks
    if context["overdue_tasks"]:
        lines.append("\nOVERDUE TASKS:")
        for t in context["overdue_tasks"]:
            lines.append(f"  - {t['title']} (overdue by {t['days_overdue']} days)")
    else:
        lines.append("\nNo specific overdue tasks identified.")

    # Last contribution
    if context["days_since_last_contribution"] is not None:
        lines.append(
            f"\nLAST CONTRIBUTION: {context['days_since_last_contribution']} days ago"
        )
    else:
        lines.append("\nLAST CONTRIBUTION: No contributions recorded")

    # Blocked teammates
    if context["blocked_teammates"]:
        lines.append(
            f"\nBLOCKED TEAMMATES: {', '.join(context['blocked_teammates'])}"
        )

    # Blackout periods
    if context["active_blackouts"]:
        lines.append("\nACTIVE BLACKOUT PERIODS (student has declared constraints):")
        for bp in context["active_blackouts"]:
            lines.append(f"  - {bp}")

    # Prior interventions
    count = context["past_intervention_count"]
    if count > 0:
        lines.append(f"\nPRIOR INTERVENTIONS: {count} previous nudge(s) sent")
        if count >= 2:
            lines.append("  This is a repeat situation — escalate gently.")

    # Rewrite feedback
    if rewrite_feedback:
        lines.append("\nREWRITE REQUEST: Your previous message was flagged as punitive.")
        lines.append(
            f"Avoid these phrases: {', '.join(repr(p) for p in rewrite_feedback)}"
        )
        lines.append("Rephrase to be constructive and supportive.")

    lines.append("\nRespond with ONLY the JSON object.")
    return "\n".join(lines)


def _parse_response(raw: str, student_id: str) -> DraftIntervention:
    """Parse raw LLM text into a validated ``DraftIntervention``.

    Args:
        raw: Raw LLM response content string.
        student_id: The target student's ID.

    Returns:
        A validated ``DraftIntervention`` instance.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
        ValueError: If Pydantic validation fails or no JSON found.
    """
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return DraftIntervention(
        target_student_id=student_id,
        message=data.get("message", ""),
        suggested_action=data.get("suggested_action", "no_action"),
        severity=data.get("severity", "low"),
        affected_teammates=data.get("affected_teammates", []),
    )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def conflict_resolution(state: SyncUpState) -> dict[str, Any]:
    """Conflict Resolution agent node.

    Identifies the first student with 'behind' status, gathers context,
    and calls the high-tier LLM to draft a constructive intervention nudge.

    If tone evaluation previously flagged the draft as punitive, rewrites
    using the flagged phrases as feedback. Stops rewriting after
    ``MAX_TONE_REWRITES`` attempts (force-accepts).

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``draft_intervention`` and ``tone_rewrite_count``
        to be merged into state.
    """
    # Force-accept after max rewrites
    if state.tone_rewrite_count >= MAX_TONE_REWRITES:
        logger.warning(
            "Max tone rewrites (%d) reached — force-accepting current draft",
            MAX_TONE_REWRITES,
        )
        return {}

    # Find the first behind student
    student_id = _find_first_behind_student(state)
    if student_id is None:
        return {"draft_intervention": None}

    now = _get_now()

    # Gather context
    context = _gather_context(student_id, state, now)

    # Force severity to "high" on 3rd+ intervention
    force_high_severity = context["past_intervention_count"] >= 2

    # Check for rewrite feedback from punitive tone result
    rewrite_feedback: list[str] | None = None
    if (
        state.tone_result is not None
        and state.tone_result.classification == "punitive"
        and state.tone_result.flagged_phrases
    ):
        rewrite_feedback = state.tone_result.flagged_phrases

    # Call high-tier LLM
    llm = get_high_tier_llm()
    user_msg = _build_user_prompt(context, rewrite_feedback)

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            draft = _parse_response(response.content, student_id)

            # Override severity if this is a repeat offender
            if force_high_severity:
                draft = draft.model_copy(update={"severity": "high"})

            # Bias toward extend_deadline if student has active blackout
            if context["active_blackouts"] and draft.suggested_action not in (
                "extend_deadline",
                "no_action",
            ):
                draft = draft.model_copy(
                    update={"suggested_action": "extend_deadline"}
                )

            return {
                "draft_intervention": draft,
                "tone_rewrite_count": state.tone_rewrite_count,
            }
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Conflict resolution attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    # All retries exhausted — return a safe fallback draft
    logger.error(
        "Conflict resolution failed after %d attempts: %s",
        MAX_RETRIES + 1,
        last_error,
    )
    return {
        "draft_intervention": DraftIntervention(
            target_student_id=student_id,
            message=(
                f"Hi {context['student_name']}, we noticed some tasks may need "
                "attention. Could you share a quick status update with the team? "
                "We're here to help if you're running into any blockers."
            ),
            suggested_action="schedule_check_in",
            severity="high" if force_high_severity else "medium",
            affected_teammates=context["blocked_teammates"],
        ),
        "tone_rewrite_count": state.tone_rewrite_count,
    }
