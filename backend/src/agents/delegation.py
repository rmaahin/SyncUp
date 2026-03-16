"""Delegation agent — assigns tasks to students based on skills and availability.

Reads ``task_array``, ``student_profiles``, and ``dependency_graph`` from state.
Uses the high-tier Groq LLM for reasoning about skill matching and workload
balancing, and the pacing service for deterministic deadline distribution.
Returns ``task_array`` (updated), ``delegation_matrix``, and ``project_timeline``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from llm import get_high_tier_llm
from services.pacing import (
    calculate_burn_down_curve,
    distribute_deadlines_for_student,
    validate_pacing,
)
from state.schema import (
    BurnDownTarget,
    ProjectTimeline,
    StudentProfile,
    SyncUpState,
    Task,
)

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts (1 initial + 2 retries)


# ---------------------------------------------------------------------------
# Pydantic models for LLM output parsing
# ---------------------------------------------------------------------------


class DelegationAssignment(BaseModel):
    """A single task-to-student assignment from the LLM."""

    task_id: str
    student_id: str


class DelegationResponse(BaseModel):
    """Top-level schema for the LLM delegation response."""

    assignments: list[DelegationAssignment]


# ---------------------------------------------------------------------------
# Skill-match scoring (deterministic)
# ---------------------------------------------------------------------------


def _compute_skill_scores(
    tasks: list[Task],
    students: list[StudentProfile],
) -> dict[str, dict[str, float]]:
    """Compute a skill-match score for every (task, student) pair.

    For each task, the score for a student is the mean of their proficiency
    ratings across the task's ``required_skills``.  If a student lacks a
    required skill, it contributes 0.  Tasks with no required skills get
    a score of 1.0 for all students (anyone can do them).

    Only students with ``availability_hours_per_week > 0`` are included.

    Args:
        tasks: The project tasks.
        students: All student profiles.

    Returns:
        A dict of ``{task_id: {student_id: score}}`` where score is in [0, 1].
    """
    available = [s for s in students if s.availability_hours_per_week > 0]
    scores: dict[str, dict[str, float]] = {}

    for task in tasks:
        task_scores: dict[str, float] = {}
        for student in available:
            if not task.required_skills:
                task_scores[student.student_id] = 1.0
            else:
                total = sum(
                    student.skills.get(skill, 0.0)
                    for skill in task.required_skills
                )
                task_scores[student.student_id] = total / len(
                    task.required_skills
                )
        scores[task.id] = task_scores

    return scores


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a project delegation expert. Given a list of tasks with skill \
requirements and effort estimates, and a list of students with skill \
proficiencies and weekly availability, assign each task to exactly one student.

Goals:
1. Prefer students with higher skill-match scores for each task.
2. Balance total effort hours across students proportionally to their \
available hours per week. A student with 5 hrs/week should get roughly \
1/4 the effort of a student with 20 hrs/week.
3. Every task must be assigned to exactly one student.
4. Students with 0 availability hours are excluded and must not be assigned.

You MUST respond with ONLY a JSON object matching this exact schema — no \
markdown fences, no explanation, no text before or after the JSON:
{
  "assignments": [
    {"task_id": "task-example", "student_id": "student-1"}
  ]
}
"""


def _build_user_prompt(
    tasks: list[Task],
    students: list[StudentProfile],
    skill_scores: dict[str, dict[str, float]],
    equity_feedback: str | None = None,
) -> str:
    """Build the user message for the delegation LLM call.

    Args:
        tasks: Project tasks to assign.
        students: Available student profiles.
        skill_scores: Pre-computed skill-match scores.
        equity_feedback: Optional feedback from a prior equity evaluation
            indicating violations to address in this re-delegation attempt.

    Returns:
        The formatted user prompt string.
    """
    task_info = json.dumps(
        [
            {
                "id": t.id,
                "title": t.title,
                "effort_hours": t.effort_hours,
                "required_skills": t.required_skills,
                "urgency": t.urgency.value,
                "dependencies": t.dependencies,
            }
            for t in tasks
        ],
        indent=2,
    )

    student_info = json.dumps(
        [
            {
                "student_id": s.student_id,
                "name": s.name,
                "skills": s.skills,
                "availability_hours_per_week": s.availability_hours_per_week,
            }
            for s in students
        ],
        indent=2,
    )

    score_lines: list[str] = []
    for tid, student_scores in skill_scores.items():
        for sid, score in student_scores.items():
            score_lines.append(f"  {tid} x {sid}: {score:.2f}")
    score_table = "\n".join(score_lines)

    prompt = (
        f"TASKS:\n{task_info}\n\n"
        f"STUDENTS:\n{student_info}\n\n"
        f"SKILL-MATCH SCORES:\n{score_table}\n\n"
        f"Assign each task to exactly one student. "
        f"Respond with ONLY the JSON object."
    )

    if equity_feedback:
        prompt += (
            f"\n\nIMPORTANT — PREVIOUS DELEGATION WAS REJECTED:\n"
            f"{equity_feedback}\n"
            f"Adjust the assignments to fix the violations above."
        )

    return prompt


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


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


def _parse_response(raw: str) -> DelegationResponse:
    """Parse raw LLM text into a validated ``DelegationResponse``.

    Args:
        raw: Raw LLM response content string.

    Returns:
        A validated ``DelegationResponse`` instance.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
        ValueError: If Pydantic validation fails or no JSON found.
    """
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return DelegationResponse.model_validate(data)


def _validate_assignments(
    response: DelegationResponse,
    task_ids: set[str],
    student_ids: set[str],
) -> dict[str, str]:
    """Validate and convert assignments to a delegation matrix.

    Args:
        response: The parsed LLM response.
        task_ids: Valid task IDs from state.
        student_ids: Valid (available) student IDs.

    Returns:
        A dict mapping task_id to student_id.

    Raises:
        ValueError: If an assignment references an unknown task or student,
            or if a task is assigned multiple times.
    """
    matrix: dict[str, str] = {}
    for assignment in response.assignments:
        if assignment.task_id not in task_ids:
            raise ValueError(f"Unknown task_id: {assignment.task_id}")
        if assignment.student_id not in student_ids:
            raise ValueError(f"Unknown student_id: {assignment.student_id}")
        if assignment.task_id in matrix:
            raise ValueError(
                f"Duplicate assignment for task: {assignment.task_id}"
            )
        matrix[assignment.task_id] = assignment.student_id

    return matrix


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def delegation(state: SyncUpState) -> dict[str, Any]:
    """Delegation agent node.

    Assigns each task to a student using LLM-based reasoning informed by
    pre-computed skill-match scores, then applies the pacing service to
    distribute deadlines.

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``task_array``, ``delegation_matrix``, and
        ``project_timeline`` to be merged into state.
    """
    if not state.task_array:
        return {}

    available_students = [
        s for s in state.student_profiles
        if s.availability_hours_per_week > 0
    ]
    if not available_students:
        logger.warning("No students with availability > 0; skipping delegation")
        return {}

    task_ids = {t.id for t in state.task_array}
    student_ids = {s.student_id for s in available_students}

    skill_scores = _compute_skill_scores(state.task_array, available_students)

    # Build equity feedback if this is a re-delegation attempt.
    equity_feedback: str | None = None
    if state.equity_retries > 0 and state.equity_result is not None:
        parts: list[str] = []
        if state.equity_result.reasoning:
            parts.append(f"Reason: {state.equity_result.reasoning}")
        if state.equity_result.violations:
            parts.append(
                "Violations:\n"
                + "\n".join(f"- {v}" for v in state.equity_result.violations)
            )
        equity_feedback = "\n".join(parts) if parts else None

    llm = get_high_tier_llm(temperature=0.7)
    system_msg = SYSTEM_PROMPT
    user_msg = _build_user_prompt(
        state.task_array, available_students, skill_scores, equity_feedback
    )

    last_error: Exception | None = None
    delegation_matrix: dict[str, str] = {}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )
            parsed = _parse_response(response.content)
            delegation_matrix = _validate_assignments(
                parsed, task_ids, student_ids
            )
            break
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Delegation attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )
    else:
        logger.error(
            "Delegation failed after %d attempts: %s",
            MAX_RETRIES + 1,
            last_error,
        )
        return {}

    # --- Apply pacing ---
    project_start = datetime.now(tz=timezone.utc)
    final_deadline = state.final_deadline or (
        project_start + timedelta(days=30)
    )

    base_deadlines = calculate_burn_down_curve(
        final_deadline,
        project_start,
        state.task_array,
        state.dependency_graph,
    )

    # Update tasks with assignments and base deadlines first.
    updated_tasks: list[Task] = []
    for task in state.task_array:
        assigned_to = delegation_matrix.get(task.id)
        deadline = base_deadlines.get(task.id)
        updated_tasks.append(
            task.model_copy(
                update={"assigned_to": assigned_to, "deadline": deadline}
            )
        )

    # Per-student deadline adjustments.
    student_map = {s.student_id: s for s in available_students}
    for sid in student_ids:
        student = student_map[sid]
        adjusted = distribute_deadlines_for_student(
            student_id=sid,
            tasks=updated_tasks,
            availability_hours_per_week=student.availability_hours_per_week,
            blackout_periods=student.blackout_periods,
            task_deadlines=base_deadlines,
            final_deadline=final_deadline,
        )
        for i, task in enumerate(updated_tasks):
            if task.id in adjusted:
                updated_tasks[i] = task.model_copy(
                    update={"deadline": adjusted[task.id]}
                )

    # Collect final deadlines for validation.
    final_deadlines = {t.id: t.deadline for t in updated_tasks if t.deadline}
    is_valid, violations = validate_pacing(
        final_deadlines, final_deadline, updated_tasks, state.dependency_graph
    )
    if not is_valid:
        for v in violations:
            logger.warning("Pacing violation: %s", v)

    # Build project timeline.
    total_effort = sum(t.effort_hours for t in updated_tasks)
    burn_down_targets: list[BurnDownTarget] = []
    remaining = total_effort
    for task in sorted(updated_tasks, key=lambda t: t.deadline or final_deadline):
        remaining -= task.effort_hours
        if task.deadline:
            burn_down_targets.append(
                BurnDownTarget(
                    date=task.deadline,
                    target_hours_remaining=max(remaining, 0.0),
                )
            )

    project_timeline = ProjectTimeline(
        burn_down_targets=burn_down_targets,
    )

    return {
        "task_array": updated_tasks,
        "delegation_matrix": delegation_matrix,
        "project_timeline": project_timeline,
    }
