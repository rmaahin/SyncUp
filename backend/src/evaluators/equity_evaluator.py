"""Workload Equity Evaluator — LLM-as-a-Judge for delegation fairness.

Validates that the delegation matrix distributes effort equitably across
students.  Combines a deterministic pre-check (>40% above average) with
an LLM-based single-criterion evaluation at near-zero temperature.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm import get_high_tier_llm
from state.schema import (
    EquityResult,
    StudentProfile,
    SyncUpState,
    Task,
)

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts


# ---------------------------------------------------------------------------
# Effort distribution (deterministic)
# ---------------------------------------------------------------------------


def _compute_effort_distribution(
    tasks: list[Task],
    delegation_matrix: dict[str, str],
    students: list[StudentProfile],
) -> dict[str, float]:
    """Sum assigned effort hours per student.

    Args:
        tasks: All project tasks.
        delegation_matrix: task_id → student_id mapping.
        students: All student profiles.

    Returns:
        A dict of student_id → total assigned effort hours.
    """
    effort: dict[str, float] = {s.student_id: 0.0 for s in students}
    task_map = {t.id: t for t in tasks}

    for task_id, student_id in delegation_matrix.items():
        task = task_map.get(task_id)
        if task and student_id in effort:
            effort[student_id] += task.effort_hours

    return effort


def _find_deterministic_violations(
    effort_dist: dict[str, float],
    students: list[StudentProfile],
) -> list[str]:
    """Flag students whose effort exceeds 40% above the group average.

    Args:
        effort_dist: student_id → total effort hours.
        students: All student profiles.

    Returns:
        A list of violation descriptions (may be empty).
    """
    assigned = {sid: hrs for sid, hrs in effort_dist.items() if hrs > 0}
    if not assigned:
        return []

    avg = sum(assigned.values()) / len(assigned)
    if avg <= 0:
        return []

    threshold = avg * 1.40
    student_map = {s.student_id: s for s in students}
    violations: list[str] = []

    for sid, hrs in assigned.items():
        if hrs > threshold:
            name = student_map.get(sid, StudentProfile(
                student_id=sid, name=sid, email=""
            )).name
            pct = ((hrs - avg) / avg) * 100
            violations.append(
                f"{name} ({sid}) has {hrs:.1f}h assigned, "
                f"{pct:.0f}% above average of {avg:.1f}h"
            )

    return violations


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a workload equity evaluator for student team projects.

Your SINGLE evaluation criterion: Is the workload distribution equitable \
across all students?

A distribution is INEQUITABLE if:
- Any student bears more than 40% above the average effort-hours.
- A student with low weekly availability is assigned disproportionately \
high-effort tasks relative to their available hours.
- The distribution ignores skill alignment to the point of unfairness.

You MUST respond with ONLY a JSON object — no markdown fences, no \
explanation, no text before or after the JSON:
{
  "balanced": true or false,
  "reasoning": "One sentence explaining your judgment",
  "violations": ["list of specific violations, empty if balanced"]
}

Use near-zero temperature. Be deterministic. If the distribution is \
reasonable and no student is significantly overloaded, return balanced=true.
"""


def _build_user_prompt(
    effort_dist: dict[str, float],
    students: list[StudentProfile],
    delegation_matrix: dict[str, str],
    tasks: list[Task],
    deterministic_violations: list[str],
) -> str:
    """Build the user message for the equity evaluation LLM call.

    Args:
        effort_dist: student_id → total effort hours.
        students: All student profiles.
        delegation_matrix: task_id → student_id.
        tasks: All project tasks.
        deterministic_violations: Pre-computed violations from the
            deterministic check (may be empty).

    Returns:
        The formatted user prompt string.
    """
    task_map = {t.id: t for t in tasks}
    student_map = {s.student_id: s for s in students}
    assigned_count = sum(1 for hrs in effort_dist.values() if hrs > 0)
    avg = sum(effort_dist.values()) / max(assigned_count, 1)

    lines: list[str] = ["WORKLOAD DISTRIBUTION:"]
    for sid, hrs in sorted(effort_dist.items()):
        student = student_map.get(sid)
        name = student.name if student else sid
        avail = student.availability_hours_per_week if student else 0
        # List tasks assigned to this student.
        assigned_tasks = [
            tid for tid, s in delegation_matrix.items() if s == sid
        ]
        task_names = [
            f"{tid} ({task_map[tid].effort_hours}h)"
            for tid in assigned_tasks if tid in task_map
        ]
        lines.append(
            f"  {name} ({sid}): {hrs:.1f}h total, "
            f"{avail:.0f}h/week available, "
            f"tasks: {', '.join(task_names) or 'none'}"
        )

    lines.append(f"\nAverage effort per student: {avg:.1f}h")

    if deterministic_violations:
        lines.append("\nDETERMINISTIC VIOLATIONS DETECTED:")
        for v in deterministic_violations:
            lines.append(f"  - {v}")

    lines.append(
        "\nIs this distribution equitable? "
        "Respond with ONLY the JSON object."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and preamble text to isolate JSON."""
    text = raw.strip()
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    brace_idx = text.find("{")
    if brace_idx == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[brace_idx:]


def _parse_response(raw: str) -> EquityResult:
    """Parse raw LLM text into a validated ``EquityResult``.

    Args:
        raw: Raw LLM response content string.

    Returns:
        A validated ``EquityResult`` instance.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
        ValueError: If Pydantic validation fails or no JSON found.
    """
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return EquityResult.model_validate(data)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def equity_evaluator(state: SyncUpState) -> dict[str, Any]:
    """Workload Equity Evaluator node.

    Validates the delegation matrix for equitable workload distribution.
    Combines a deterministic check (>40% above average) with an LLM-based
    single-criterion evaluation.

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``equity_result`` (EquityResult) and
        ``equity_retries`` (int) to be merged into state.
    """
    if not state.task_array or not state.delegation_matrix:
        return {
            "equity_result": EquityResult(
                balanced=True,
                reasoning="No tasks or delegation to evaluate",
            ),
            "equity_retries": state.equity_retries + 1,
        }

    effort_dist = _compute_effort_distribution(
        state.task_array, state.delegation_matrix, state.student_profiles
    )
    det_violations = _find_deterministic_violations(
        effort_dist, state.student_profiles
    )

    llm = get_high_tier_llm(temperature=0.0)
    user_msg = _build_user_prompt(
        effort_dist,
        state.student_profiles,
        state.delegation_matrix,
        state.task_array,
        det_violations,
    )

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            result = _parse_response(response.content)
            return {
                "equity_result": result,
                "equity_retries": state.equity_retries + 1,
            }
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Equity evaluation attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error(
        "Equity evaluation failed after %d attempts: %s",
        MAX_RETRIES + 1,
        last_error,
    )
    # Default to unbalanced on failure to trigger re-delegation.
    return {
        "equity_result": EquityResult(
            balanced=False,
            reasoning="Evaluation failed — defaulting to unbalanced",
            violations=["LLM evaluation could not be completed"],
        ),
        "equity_retries": state.equity_retries + 1,
    }
