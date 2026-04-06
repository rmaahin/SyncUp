"""Task Decomposition agent — decomposes a project brief into structured tasks.

Reads ``project_brief`` from state, calls the high-tier Groq LLM to break it
into granular subtasks with effort estimates, skill requirements, urgency
levels, and inter-task dependencies.  Returns ``task_array`` and
``dependency_graph`` as a state update dict.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from guardrails.sanitizer import sanitize_document
from llm import get_high_tier_llm
from state.schema import SyncUpState, Task, TaskStatus, UrgencyLevel

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts (1 initial + 2 retries)

# ---------------------------------------------------------------------------
# Pydantic models for LLM output parsing
# ---------------------------------------------------------------------------


class TaskOutput(BaseModel):
    """Schema for a single task in the LLM JSON response."""

    id: str
    title: str
    description: str = ""
    effort_hours: float = 0.0
    required_skills: list[str] = Field(default_factory=list)
    urgency: str = "medium"
    dependencies: list[str] = Field(default_factory=list)


class DecompositionResponse(BaseModel):
    """Top-level schema for the full LLM JSON response."""

    tasks: list[TaskOutput]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a project management expert specializing in decomposing project briefs \
into actionable tasks for student teams.

Given a project brief, you must:
1. Break it into discrete, well-scoped tasks (aim for 5-15 tasks).
2. Estimate effort in hours for each task (realistic for university students, \
typically 1-20 hours per task).
3. Identify required skills for each task.
4. Assign an urgency level to each task based on these criteria:
   - critical: On the critical path — blocks multiple downstream tasks.
   - high: Has a tight implicit deadline or blocks at least one other task.
   - medium: Standard work with no special time pressure.
   - low: Nice-to-have or flexible timing; nothing depends on it.
5. Identify dependencies between tasks (which tasks must complete first).

Rules:
- Generate a unique semantic ID for each task (e.g., "task-setup-repo", \
"task-design-db"). IDs must be lowercase, hyphenated, and prefixed with "task-".
- Dependencies must reference other task IDs from your output.
- Do NOT create circular dependencies.
- Effort hours should be realistic for university students (1-20 hours per task).

You MUST respond with ONLY a JSON object matching this exact schema — no \
markdown fences, no explanation, no text before or after the JSON:
{
  "tasks": [
    {
      "id": "task-example",
      "title": "Example Task",
      "description": "What this task involves",
      "effort_hours": 5.0,
      "required_skills": ["python", "sql"],
      "urgency": "medium",
      "dependencies": ["task-other"]
    }
  ]
}

Valid urgency values: "critical", "high", "medium", "low"
"""


def _build_user_prompt(project_brief: str, available_skills: list[str]) -> str:
    """Build the user message for the LLM call.

    Args:
        project_brief: The sanitized project brief from state.
        available_skills: Known skill names from student profiles (may be empty).

    Returns:
        The formatted user prompt string.
    """
    skills_section = ""
    if available_skills:
        skills_list = ", ".join(available_skills)
        skills_section = (
            f"\nThe team has members with the following skills "
            f"(match required_skills to these when possible):\n{skills_list}\n"
        )

    return (
        f"Decompose the following project brief into tasks.\n\n"
        f"PROJECT BRIEF:\n{project_brief}\n"
        f"{skills_section}\n"
        f"Respond with ONLY the JSON object. No additional text."
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_URGENCY_MAP: dict[str, UrgencyLevel] = {
    "critical": UrgencyLevel.CRITICAL,
    "high": UrgencyLevel.HIGH,
    "medium": UrgencyLevel.MEDIUM,
    "low": UrgencyLevel.LOW,
}


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


def _parse_response(raw: str) -> DecompositionResponse:
    """Parse raw LLM text into a validated ``DecompositionResponse``.

    Args:
        raw: Raw LLM response content string.

    Returns:
        A validated ``DecompositionResponse`` instance.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
        ValueError: If Pydantic validation fails or no JSON found.
    """
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return DecompositionResponse.model_validate(data)


def _to_tasks(parsed: DecompositionResponse) -> list[Task]:
    """Convert parsed LLM output into state ``Task`` models.

    Args:
        parsed: The validated decomposition response.

    Returns:
        A list of ``Task`` model instances.

    Raises:
        ValueError: If duplicate task IDs are detected.
    """
    seen_ids: set[str] = set()
    tasks: list[Task] = []

    for t in parsed.tasks:
        if t.id in seen_ids:
            raise ValueError(f"Duplicate task ID: {t.id}")
        seen_ids.add(t.id)

        urgency = _URGENCY_MAP.get(t.urgency.lower(), UrgencyLevel.MEDIUM)

        tasks.append(
            Task(
                id=t.id,
                title=t.title,
                description=t.description,
                effort_hours=t.effort_hours,
                required_skills=t.required_skills,
                urgency=urgency,
                dependencies=t.dependencies,
                assigned_to=None,
                deadline=None,
                status=TaskStatus.TODO,
            )
        )

    return tasks


def _build_dependency_graph(tasks: list[Task]) -> dict[str, list[str]]:
    """Build a dependency graph from the task list.

    Each key is a task ID; the value is the list of prerequisite task IDs.
    Dangling references (dependencies pointing to non-existent tasks) are
    silently dropped with a warning log.

    Args:
        tasks: The list of decomposed tasks.

    Returns:
        A dict mapping each task ID to its list of valid prerequisite IDs.
    """
    valid_ids: set[str] = {t.id for t in tasks}
    graph: dict[str, list[str]] = {}

    for task in tasks:
        valid_deps: list[str] = []
        for dep_id in task.dependencies:
            if dep_id in valid_ids:
                valid_deps.append(dep_id)
            else:
                logger.warning(
                    "Task %s has dangling dependency %s — dropping",
                    task.id,
                    dep_id,
                )
        graph[task.id] = valid_deps

    return graph


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def task_decomposition(state: SyncUpState) -> dict[str, Any]:
    """Task Decomposition agent node.

    Reads ``project_brief`` from state, calls the high-tier LLM to decompose
    it into a list of ``Task`` models with effort estimates, skill
    requirements, urgency levels, and inter-task dependencies.

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``task_array`` (list[Task]) and ``dependency_graph``
        (dict[str, list[str]]) to be merged into state.
    """
    if not state.project_brief or not state.project_brief.strip():
        return {"task_array": [], "dependency_graph": {}}

    # Collect known skills from student profiles
    available_skills: list[str] = sorted(
        {skill for profile in state.student_profiles for skill in profile.skills}
    )

    llm = get_high_tier_llm(temperature=0.7)

    sanitized_brief = sanitize_document(state.project_brief, "google_docs")

    system_msg = SYSTEM_PROMPT
    user_msg = _build_user_prompt(sanitized_brief, available_skills)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )
            parsed = _parse_response(response.content)
            tasks = _to_tasks(parsed)
            dep_graph = _build_dependency_graph(tasks)
            return {"task_array": tasks, "dependency_graph": dep_graph}
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Task decomposition attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error(
        "Task decomposition failed after %d attempts: %s",
        MAX_RETRIES + 1,
        last_error,
    )
    return {"task_array": [], "dependency_graph": {}}
