"""Deliver node — publishes approved intervention nudges and acts on them.

Deterministic node (no LLM) that takes an approved ``DraftIntervention``
from state, creates an ``Intervention`` record, posts a Trello comment
on the overdue task card, and executes the suggested action:

- ``extend_deadline``: pushes the overdue task deadline by 3 days from now,
  updates the Trello card due date, and reschedules the Calendar event.
- ``redistribute_task``: sets a flag so the Delegation agent re-runs for
  just the affected task.
- ``schedule_check_in`` / ``no_action``: no additional side effects.

This is an async node (like publishing.py) because it uses the async
TrelloClient and MCP clients.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from integrations.trello import TrelloClient
from mcp_layer.client import SyncUpMCPClient
from mcp_layer.google_calendar import GoogleCalendarMCP
from state.schema import (
    DateRange,
    Intervention,
    StudentProfile,
    SyncUpState,
    Task,
    TaskStatus,
)

logger = logging.getLogger(__name__)

DEADLINE_EXTENSION_DAYS: int = 3


def _get_now() -> datetime:
    """Return current UTC time. Patchable in tests."""
    return datetime.now(timezone.utc)


def _find_overdue_tasks(
    student_id: str,
    state: SyncUpState,
    now: datetime,
) -> list[tuple[str, Task]]:
    """Find overdue tasks assigned to a student.

    Args:
        student_id: The behind student's ID.
        state: The current SyncUpState.
        now: The current UTC datetime.

    Returns:
        List of (task_id, Task) tuples for overdue tasks.
    """
    task_map = {t.id: t for t in state.task_array}
    overdue: list[tuple[str, Task]] = []

    for task_id, sid in state.delegation_matrix.items():
        if sid != student_id:
            continue
        task = task_map.get(task_id)
        if (
            task
            and task.deadline is not None
            and task.deadline < now
            and task.status != TaskStatus.DONE
        ):
            overdue.append((task_id, task))

    return overdue


def _find_overdue_card_id(
    student_id: str,
    state: SyncUpState,
) -> str | None:
    """Find the Trello card ID for an overdue task assigned to the student.

    Returns the first matching card ID, or None if no match found.

    Args:
        student_id: The behind student's ID.
        state: The current SyncUpState.

    Returns:
        A Trello card_id string or None.
    """
    now = _get_now()
    overdue = _find_overdue_tasks(student_id, state, now)
    for task_id, _ in overdue:
        card_id = state.trello_card_mapping.get(task_id)
        if card_id:
            return card_id
    return None


def _build_trigger_reason(
    student_id: str,
    state: SyncUpState,
    now: datetime,
) -> str:
    """Build a human-readable trigger reason string.

    Args:
        student_id: The behind student's ID.
        state: The current SyncUpState.
        now: The current UTC datetime.

    Returns:
        A short string describing why the intervention was triggered.
    """
    overdue = _find_overdue_tasks(student_id, state, now)
    if overdue:
        _, task = overdue[0]
        days = (now - task.deadline).days if task.deadline else 0
        return f"Task '{task.title}' overdue by {days} day(s)"
    return "Student marked as behind schedule"


def _compute_new_deadline(
    now: datetime,
    original_deadline: datetime | None = None,
    blackout_periods: list[DateRange] | None = None,
) -> datetime:
    """Compute an extended deadline that never pulls a date backward and avoids blackouts.

    Logic:
    1. Base = max(original_deadline, now) — never move a future deadline earlier.
    2. Add DEADLINE_EXTENSION_DAYS to the base.
    3. If the result falls within a student's blackout period, push past the blackout end.

    Args:
        now: The current UTC datetime.
        original_deadline: The task's current deadline (may be in the future).
        blackout_periods: The assigned student's blackout periods to avoid.

    Returns:
        The new deadline datetime.
    """
    base = now
    if original_deadline is not None and original_deadline > now:
        base = original_deadline

    new_dl = base + timedelta(days=DEADLINE_EXTENSION_DAYS)

    # Push past any blackout that overlaps the new deadline
    if blackout_periods:
        for bo in blackout_periods:
            if bo.start <= new_dl <= bo.end:
                # Move to 1 day after the blackout ends
                new_dl = bo.end + timedelta(days=1)

    return new_dl


def _get_student_blackouts(
    student_id: str, state: SyncUpState,
) -> list[DateRange]:
    """Return blackout periods for a student, or empty list if not found."""
    for sp in state.student_profiles:
        if sp.student_id == student_id:
            return sp.blackout_periods
    return []


def _extend_task_deadlines(
    task_array: list[Task],
    overdue_tasks: list[tuple[str, Task]],
    now: datetime,
    blackout_periods: list[DateRange] | None = None,
) -> list[Task]:
    """Return an updated task_array with extended deadlines for overdue tasks.

    Each overdue task gets its own computed deadline based on its original
    deadline, the current time, and blackout periods.

    Args:
        task_array: The full task list from state.
        overdue_tasks: List of (task_id, Task) tuples to extend.
        now: The current UTC datetime.
        blackout_periods: The student's blackout periods to avoid.

    Returns:
        A new task list with updated deadlines.
    """
    overdue_map = {tid: task for tid, task in overdue_tasks}
    updated: list[Task] = []
    for task in task_array:
        if task.id in overdue_map:
            new_dl = _compute_new_deadline(now, task.deadline, blackout_periods)
            updated.append(task.model_copy(update={"deadline": new_dl}))
        else:
            updated.append(task)
    return updated


async def _update_trello_due_dates(
    overdue_tasks: list[tuple[str, Task]],
    trello_card_mapping: dict[str, str],
    new_deadline: datetime,
    client: TrelloClient,
) -> None:
    """Update Trello card due dates for overdue tasks.

    Best-effort: logs errors but does not raise.

    Args:
        overdue_tasks: List of (task_id, Task) tuples.
        trello_card_mapping: task_id → card_id mapping.
        new_deadline: The new due date.
        client: An active TrelloClient instance.
    """
    for task_id, task in overdue_tasks:
        card_id = trello_card_mapping.get(task_id)
        if not card_id:
            continue
        try:
            await client.update_card(card_id, due=new_deadline)
            logger.info(
                "Trello card %s due date updated to %s for task '%s'",
                card_id,
                new_deadline.isoformat(),
                task.title,
            )
        except Exception as exc:
            logger.warning(
                "Failed to update Trello card %s due date: %s", card_id, exc
            )


async def _update_calendar_events(
    overdue_tasks: list[tuple[str, Task]],
    calendar_event_mapping: dict[str, str],
    new_deadline: datetime,
) -> None:
    """Update Google Calendar events for overdue tasks.

    Best-effort: logs errors but does not raise. Silently skips if
    the Calendar MCP server is not configured or the update_event
    tool is not available.

    Args:
        overdue_tasks: List of (task_id, Task) tuples.
        calendar_event_mapping: task_id → event_id mapping.
        new_deadline: The new deadline datetime.
    """
    # Check if any overdue task has a calendar event
    event_ids = {
        tid: calendar_event_mapping[tid]
        for tid, _ in overdue_tasks
        if tid in calendar_event_mapping
    }
    if not event_ids:
        return

    calendar_id = os.environ.get("COURSE_CALENDAR_ID", "")
    if not calendar_id:
        logger.info("COURSE_CALENDAR_ID not set — skipping calendar update")
        return

    try:
        async with SyncUpMCPClient() as mcp:
            cal = GoogleCalendarMCP(mcp)
            for task_id, event_id in event_ids.items():
                try:
                    # Event runs for 1 hour before the deadline
                    event_start = new_deadline - timedelta(hours=1)
                    await cal.update_event(
                        calendar_id=calendar_id,
                        event_id=event_id,
                        start=event_start,
                        end=new_deadline,
                    )
                    logger.info(
                        "Calendar event %s updated to %s",
                        event_id,
                        new_deadline.isoformat(),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to update calendar event %s: %s", event_id, exc
                    )
    except Exception as exc:
        logger.warning("Calendar MCP connection failed — skipping: %s", exc)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


async def deliver(state: SyncUpState) -> dict[str, Any]:
    """Deliver node — publishes approved intervention nudges and acts on them.

    Creates an ``Intervention`` record from the draft, attempts to post
    a Trello comment on the overdue task card, and executes the suggested
    action:

    - ``extend_deadline``: extends task deadline by 3 days, updates Trello
      card due date, updates Calendar event.
    - ``redistribute_task``: sets ``needs_redelegation`` flag for the
      Delegation agent to re-run on affected tasks.
    - Other actions: logged but no additional side effects.

    Args:
        state: The current ``SyncUpState``.

    Returns:
        A dict with ``intervention_history`` (list to append),
        ``draft_intervention`` (None), ``tone_result`` (None),
        ``tone_rewrite_count`` (0), and optionally ``task_array``
        (updated deadlines) to be merged into state.
    """
    cleanup: dict[str, Any] = {
        "draft_intervention": None,
        "tone_result": None,
        "tone_rewrite_count": 0,
    }

    if state.draft_intervention is None:
        logger.info("No draft intervention to deliver — clearing state")
        return cleanup

    draft = state.draft_intervention
    now = _get_now()

    # Create Intervention record
    intervention = Intervention(
        target_student_id=draft.target_student_id,
        trigger_reason=_build_trigger_reason(draft.target_student_id, state, now),
        message_text=draft.message,
        timestamp=now,
        outcome=draft.suggested_action,
    )

    logger.info(
        "Delivering intervention to %s: severity=%s, action=%s",
        draft.target_student_id,
        draft.severity,
        draft.suggested_action,
    )

    result: dict[str, Any] = {
        "intervention_history": [intervention],  # appended via operator.add
        "draft_intervention": None,
        "tone_result": None,
        "tone_rewrite_count": 0,
    }

    # Find overdue tasks for this student
    overdue_tasks = _find_overdue_tasks(draft.target_student_id, state, now)

    # Best-effort Trello comment
    card_id = None
    if overdue_tasks:
        first_task_id = overdue_tasks[0][0]
        card_id = state.trello_card_mapping.get(first_task_id)

    if card_id:
        try:
            async with TrelloClient() as client:
                await client.add_comment(card_id, draft.message)
            logger.info("Trello comment posted on card %s", card_id)
        except Exception as exc:
            logger.warning(
                "Failed to post Trello comment on card %s: %s", card_id, exc
            )

    # --- Execute suggested action ---

    if draft.suggested_action == "extend_deadline" and overdue_tasks:
        blackouts = _get_student_blackouts(draft.target_student_id, state)

        # Compute per-task deadlines (respects original deadline + blackouts)
        result["task_array"] = _extend_task_deadlines(
            state.task_array, overdue_tasks, now, blackouts
        )

        # For Trello/Calendar updates, use the first task's new deadline as reference
        first_task = overdue_tasks[0][1]
        new_deadline = _compute_new_deadline(now, first_task.deadline, blackouts)

        logger.info(
            "Extending deadline for %d overdue task(s) to %s (original: %s, blackouts: %d)",
            len(overdue_tasks),
            new_deadline.isoformat(),
            first_task.deadline.isoformat() if first_task.deadline else "none",
            len(blackouts),
        )

        # Update Trello card due dates
        try:
            async with TrelloClient() as client:
                await _update_trello_due_dates(
                    overdue_tasks,
                    state.trello_card_mapping,
                    new_deadline,
                    client,
                )
        except Exception as exc:
            logger.warning("Trello due date update failed: %s", exc)

        # Update Calendar events
        await _update_calendar_events(
            overdue_tasks,
            state.calendar_event_mapping,
            new_deadline,
        )

    elif draft.suggested_action == "redistribute_task" and overdue_tasks:
        # Flag for the Delegation agent to re-run on these tasks
        overdue_ids = [tid for tid, _ in overdue_tasks]
        result["needs_redelegation"] = overdue_ids
        logger.info(
            "Flagging %d task(s) for re-delegation: %s",
            len(overdue_ids),
            overdue_ids,
        )

    return result
