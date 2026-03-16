"""Publishing agent — deterministic (no LLM) node that pushes the approved
delegation matrix to Trello, Google Calendar, and Google Docs.

Runs after ``human_review`` approves the delegation.  Each integration
block is independent: if one fails the others still execute.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from integrations.trello import TrelloClient
from mcp_layer.client import SyncUpMCPClient
from mcp_layer.google_calendar import GoogleCalendarMCP
from mcp_layer.google_docs import GoogleDocsMCP
from state.schema import PublishingStatus, StudentProfile, SyncUpState, Task, UrgencyLevel

logger = logging.getLogger(__name__)

# Trello list names in display order
_LIST_NAMES: list[str] = ["To Do", "In Progress", "Review", "Done"]

# Urgency → Trello label colour
_URGENCY_COLORS: dict[UrgencyLevel, str] = {
    UrgencyLevel.CRITICAL: "red",
    UrgencyLevel.HIGH: "orange",
    UrgencyLevel.MEDIUM: "yellow",
    UrgencyLevel.LOW: "green",
}

# Calendar reminder offsets in minutes (48h, 24h, 2h)
_REMINDER_MINUTES: list[int] = [2880, 1440, 120]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_project_name(state: SyncUpState) -> str:
    """Return a human-friendly project name."""
    if state.project_name:
        return str(state.project_name)
    if state.project_id:
        return str(state.project_id)
    return "SyncUp Project"


def _build_task_matrix_content(
    tasks: list[Task],
    delegation: dict[str, str],
    student_map: dict[str, StudentProfile],
) -> str:
    """Build a plain-text table summarising the task breakdown."""
    header = (
        f"{'Task':<40} | {'Assignee':<20} | {'Deadline':<20} | "
        f"{'Urgency':<10} | {'Effort (hrs)':<12} | {'Dependencies':<30} | {'Status':<12}"
    )
    sep = "-" * len(header)
    lines: list[str] = [header, sep]

    for task in tasks:
        assignee_id = delegation.get(task.id, "")
        assignee_name = student_map[assignee_id].name if assignee_id in student_map else "Unassigned"
        deadline_str = task.deadline.strftime("%Y-%m-%d %H:%M") if task.deadline else "—"
        deps = ", ".join(task.dependencies) if task.dependencies else "None"
        lines.append(
            f"{task.title:<40} | {assignee_name:<20} | {deadline_str:<20} | "
            f"{task.urgency.value:<10} | {task.effort_hours:<12.1f} | {deps:<30} | {task.status.value:<12}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration blocks
# ---------------------------------------------------------------------------


async def _publish_trello(
    state: SyncUpState,
    student_map: dict[str, StudentProfile],
    project_name: str,
) -> tuple[str | None, dict[str, str]]:
    """Create a Trello board with lists, labels, and one card per task.

    Returns:
        ``(board_id, {task_id: card_id})``
    """
    task_map: dict[str, Task] = {t.id: t for t in state.task_array}

    async with TrelloClient() as client:
        board = await client.create_board(f"{project_name} - SyncUp Board")
        board_id = board.id

        # Create the four standard lists
        lists: dict[str, str] = {}
        for name in _LIST_NAMES:
            lst = await client.create_list(board_id, name)
            lists[name] = lst.id

        # Create urgency labels
        labels: dict[UrgencyLevel, str] = {}
        for urgency, color in _URGENCY_COLORS.items():
            label = await client.add_label(board_id, urgency.value, color)
            labels[urgency] = label.id

        todo_list_id = lists["To Do"]
        card_mapping: dict[str, str] = {}

        for task in state.task_array:
            assignee_id = state.delegation_matrix.get(task.id)
            member_id: str | None = None
            if assignee_id and assignee_id in student_map:
                trello_id = student_map[assignee_id].trello_id
                if trello_id:
                    member_id = trello_id

            desc_parts = [task.description]
            if task.effort_hours:
                desc_parts.append(f"Effort: {task.effort_hours}h")
            if task.required_skills:
                desc_parts.append(f"Skills: {', '.join(task.required_skills)}")
            desc = "\n".join(desc_parts)

            label_ids = [labels[task.urgency]] if task.urgency in labels else []

            card = await client.create_card(
                list_id=todo_list_id,
                name=task.title,
                desc=desc,
                due=task.deadline,
                member_id=member_id,
                labels=label_ids,
            )
            card_mapping[task.id] = card.id

            # Add dependency checklist if the task has prerequisites
            if task.dependencies:
                dep_titles = [
                    task_map[dep_id].title if dep_id in task_map else dep_id
                    for dep_id in task.dependencies
                ]
                await client.add_checklist(card.id, "Prerequisites", dep_titles)

    return board_id, card_mapping


async def _publish_calendar(
    state: SyncUpState,
    student_map: dict[str, StudentProfile],
) -> dict[str, str]:
    """Create Google Calendar deadline events for each assigned task.

    Returns:
        ``{task_id: event_id}``
    """
    calendar_id = os.environ.get("COURSE_CALENDAR_ID", "")
    if not calendar_id:
        raise ValueError("COURSE_CALENDAR_ID environment variable is required")

    event_mapping: dict[str, str] = {}

    async with SyncUpMCPClient() as mcp:
        cal = GoogleCalendarMCP(mcp)
        for task in state.task_array:
            if task.deadline is None:
                continue

            assignee_id = state.delegation_matrix.get(task.id)
            attendees: list[str] = []
            if assignee_id and assignee_id in student_map:
                email = student_map[assignee_id].google_email
                if email:
                    attendees.append(email)

            result = await cal.create_event(
                calendar_id=calendar_id,
                summary=f"SyncUp Deadline: {task.title}",
                start=task.deadline - timedelta(hours=1),
                end=task.deadline,
                attendees=attendees,
                reminders=_REMINDER_MINUTES,
            )
            event_id = result.get("id", result.get("event_id", ""))
            if event_id:
                event_mapping[task.id] = event_id

    return event_mapping


async def _publish_docs(
    state: SyncUpState,
    student_map: dict[str, StudentProfile],
    project_name: str,
) -> str:
    """Create a Google Doc containing the task breakdown matrix.

    Returns:
        The created document ID.
    """
    content = _build_task_matrix_content(
        state.task_array, state.delegation_matrix, student_map,
    )

    async with SyncUpMCPClient() as mcp:
        docs = GoogleDocsMCP(mcp)
        result = await docs.create_document(
            title=f"{project_name} - Task Breakdown Matrix",
            content=content,
        )
        return str(result.get("document_id", ""))


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


async def publishing(state: SyncUpState) -> dict[str, Any]:
    """Publishing agent node — pushes delegation to Trello, Calendar, and Docs.

    This is a deterministic node (no LLM). Each integration block is
    independent: a failure in one does not prevent the others from executing.
    """
    status = PublishingStatus()
    result: dict[str, Any] = {}

    if not state.delegation_matrix:
        logger.warning("Publishing: empty delegation matrix — nothing to publish")
        status.trello = "failed"
        status.calendar = "failed"
        status.docs = "failed"
        status.errors.append("Empty delegation matrix")
        return {"publishing_status": status}

    student_map: dict[str, StudentProfile] = {
        sp.student_id: sp for sp in state.student_profiles
    }
    project_name = _resolve_project_name(state)

    # -- Trello ---------------------------------------------------------------
    try:
        board_id, card_mapping = await _publish_trello(state, student_map, project_name)
        result["trello_board_id"] = board_id
        result["trello_card_mapping"] = card_mapping
        status.trello = "success"
        logger.info("Publishing: Trello board created (%s)", board_id)
    except Exception as exc:
        logger.error("Publishing: Trello failed — %s", exc)
        status.trello = "failed"
        status.errors.append(f"Trello: {exc}")

    # -- Google Calendar ------------------------------------------------------
    try:
        event_mapping = await _publish_calendar(state, student_map)
        result["calendar_event_mapping"] = event_mapping
        status.calendar = "success"
        logger.info("Publishing: %d calendar events created", len(event_mapping))
    except Exception as exc:
        logger.error("Publishing: Calendar failed — %s", exc)
        status.calendar = "failed"
        status.errors.append(f"Calendar: {exc}")

    # -- Google Docs ----------------------------------------------------------
    try:
        doc_id = await _publish_docs(state, student_map, project_name)
        result["docs_task_matrix_id"] = doc_id
        status.docs = "success"
        logger.info("Publishing: Google Doc created (%s)", doc_id)
    except Exception as exc:
        logger.error("Publishing: Docs failed — %s", exc)
        status.docs = "failed"
        status.errors.append(f"Docs: {exc}")

    # -- GitHub webhook (stub) ------------------------------------------------
    logger.info("Publishing: GitHub webhook setup not yet implemented")
    result["webhook_configured"] = False

    result["publishing_status"] = status
    result["project_name"] = project_name
    return result
