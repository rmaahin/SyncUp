"""Peer Review form generator — builds project-tailored review forms.

Invoked at project close (not part of the LangGraph routing). Calls the
low-tier LLM to tailor short dimension descriptions to the specific project,
then assembles per-student form payloads listing each teammate plus the tasks
they were assigned (so reviewers have concrete context).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm import get_low_tier_llm
from state.schema import SyncUpState

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

DIMENSIONS: list[dict[str, str]] = [
    {
        "key": "contribution_quality",
        "question": "How would you rate the quality of this teammate's contributions?",
    },
    {
        "key": "communication",
        "question": "How effectively did this teammate communicate with the team?",
    },
    {
        "key": "reliability",
        "question": "How reliable was this teammate in meeting commitments?",
    },
    {
        "key": "collaboration",
        "question": "How well did this teammate collaborate with others?",
    },
    {
        "key": "technical_competency",
        "question": "How would you rate this teammate's technical competency on this project?",
    },
]

DIMENSION_KEYS: set[str] = {d["key"] for d in DIMENSIONS}


SYSTEM_PROMPT: str = """\
You write short rubric descriptions for student peer-review dimensions.
Given a project brief and dimension keys, return a JSON object of the form:
{"dimension_descriptions": {"<dim_key>": "<<=25 word description>", ...}}
Use ONLY the dimension keys provided. Do not invent new keys.
Respond with ONLY the JSON object — no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Helpers
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


def _generate_descriptions(project_brief: str) -> dict[str, str]:
    """Call the low-tier LLM to tailor dimension descriptions to the project.

    Falls back to empty strings on any failure — peer-review form generation
    must never break at end-of-project.
    """
    keys_list = sorted(DIMENSION_KEYS)
    user_prompt = (
        f"Project brief:\n{project_brief or '(no brief provided)'}\n\n"
        f"Dimension keys: {keys_list}\n\n"
        "Return tailored ≤25-word descriptions for each key."
    )

    try:
        llm = get_low_tier_llm(temperature=0.3)
    except Exception as exc:  # missing API key, etc.
        logger.warning("Could not initialize low-tier LLM for peer-review: %s", exc)
        return {k: "" for k in DIMENSION_KEYS}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = getattr(response, "content", "")
            if not isinstance(content, str):
                content = str(content)
            parsed = json.loads(_extract_json(content))
            descriptions = parsed.get("dimension_descriptions", {})
            if not isinstance(descriptions, dict):
                raise ValueError("dimension_descriptions is not a dict")
            # Keep only known keys; fill in missing ones with empty strings.
            return {
                k: str(descriptions.get(k, ""))[:200]
                for k in DIMENSION_KEYS
            }
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Peer-review description attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error("Peer-review description generation failed — using blank fallback")
    return {k: "" for k in DIMENSION_KEYS}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_peer_review_form(state: SyncUpState) -> dict[str, Any]:
    """Generate the peer-review form template for a project.

    Args:
        state: Current ``SyncUpState`` (must contain student_profiles).

    Returns:
        A partial-state dict to be merged into the project state.
    """
    descriptions = _generate_descriptions(state.project_brief)

    dimensions_payload = [
        {
            "key": d["key"],
            "question": d["question"],
            "description": descriptions.get(d["key"], ""),
        }
        for d in DIMENSIONS
    ]

    # Inverse delegation: student_id → [task title for tasks they own]
    tasks_by_student: dict[str, list[str]] = {}
    task_titles = {t.id: t.title for t in state.task_array}
    for task_id, sid in state.delegation_matrix.items():
        title = task_titles.get(task_id)
        if title:
            tasks_by_student.setdefault(sid, []).append(title)

    forms_by_student: dict[str, dict[str, Any]] = {}
    for reviewer in state.student_profiles:
        teammates = []
        for other in state.student_profiles:
            if other.student_id == reviewer.student_id:
                continue
            teammates.append({
                "id": other.student_id,
                "name": other.name,
                "assigned_tasks": tasks_by_student.get(other.student_id, []),
            })
        forms_by_student[reviewer.student_id] = {"teammates": teammates}

    template: dict[str, Any] = {
        "dimensions": dimensions_payload,
        "forms_by_student": forms_by_student,
    }

    return {
        "peer_review_forms_generated": True,
        "peer_review_form_template": template,
    }
