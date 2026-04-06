"""Meeting Coordinator agent — schedules team meetings and ingests meeting notes.

Two modes controlled by ``state.meeting_mode``:

* **"schedule"**: Find an optimal slot via the meeting scheduler service,
  generate an agenda with the low-tier LLM, create a Google Calendar event
  and a Google Doc for the agenda.
* **"ingest"**: Read an existing meeting-notes Google Doc, parse it with the
  low-tier LLM, and append a structured ``MeetingRecord`` to state.

Uses LOW-TIER Groq LLM (llama-3.1-8b-instant via ``get_low_tier_llm()``) —
agenda generation is templated, not complex reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from guardrails.sanitizer import sanitize_document
from llm import get_low_tier_llm
from mcp_layer.client import SyncUpMCPClient
from mcp_layer.google_calendar import GoogleCalendarMCP
from mcp_layer.google_docs import GoogleDocsMCP
from services.meeting_scheduler import find_optimal_meeting_slot
from state.schema import MeetingRecord, SyncUpState, TaskStatus

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2  # 3 total attempts


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

AGENDA_SYSTEM_PROMPT: str = """\
You are a meeting facilitator for a student project team. Generate a focused \
meeting agenda as bullet points.

Cover: progress review, upcoming deadlines, blocked items, items needing \
group decision, and follow-ups from last meeting. Be specific — reference \
task names and student names.

Respond with ONLY a JSON object — no markdown fences, no explanation, no \
text before or after the JSON:
{"agenda_items": ["item 1", "item 2", ...]}
"""

NOTES_SYSTEM_PROMPT: str = """\
You are a meeting notes parser. Extract structured information from raw \
meeting notes.

Respond with ONLY a JSON object — no markdown fences, no explanation, no \
text before or after the JSON:
{
  "summary": "2-3 sentence summary",
  "attendees": ["name1", "name2"],
  "action_items": ["specific action with owner if mentioned"],
  "decisions": ["decisions made"],
  "blockers_discussed": ["blockers raised"]
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


def _build_context_summary(state: SyncUpState, now: datetime) -> str:
    """Build a plain-text project summary for the agenda-generation prompt.

    Collects task statuses, upcoming deadlines, blocked tasks, student
    progress, recent interventions, and last meeting action items.
    """
    lines: list[str] = []

    # Project header
    lines.append(f"PROJECT: {state.project_name or state.project_id}")
    if state.final_deadline:
        lines.append(f"FINAL DEADLINE: {state.final_deadline.strftime('%Y-%m-%d')}")

    # Student names
    names = [sp.name for sp in state.student_profiles]
    lines.append(f"TEAM MEMBERS: {', '.join(names) if names else 'N/A'}")
    lines.append("")

    # Task counts by status
    counts: dict[str, int] = {"todo": 0, "in_progress": 0, "review": 0, "done": 0}
    for t in state.task_array:
        counts[t.status.value] = counts.get(t.status.value, 0) + 1
    lines.append(
        f"TASK STATUS: TODO={counts['todo']}, IN_PROGRESS={counts['in_progress']}, "
        f"REVIEW={counts['review']}, DONE={counts['done']}"
    )
    lines.append("")

    # Upcoming deadlines (next 7 days, not done)
    seven_days = now + timedelta(days=7)
    upcoming = [
        t
        for t in state.task_array
        if t.deadline and t.status != TaskStatus.DONE and t.deadline <= seven_days
    ]
    upcoming.sort(key=lambda t: t.deadline or now)
    if upcoming:
        lines.append("UPCOMING DEADLINES (next 7 days):")
        for t in upcoming[:10]:
            assignee = t.assigned_to or "unassigned"
            dl = t.deadline.strftime("%Y-%m-%d") if t.deadline else "N/A"
            lines.append(f"  - {t.title} (assigned to {assignee}, due {dl})")
    else:
        lines.append("UPCOMING DEADLINES: none in the next 7 days")
    lines.append("")

    # Overdue tasks
    overdue = [
        t
        for t in state.task_array
        if t.deadline and t.status != TaskStatus.DONE and t.deadline < now
    ]
    if overdue:
        lines.append("OVERDUE TASKS:")
        for t in overdue:
            days = (now - t.deadline).days if t.deadline else 0
            lines.append(
                f"  - {t.title} (assigned to {t.assigned_to or 'unassigned'}, "
                f"{days} day(s) overdue)"
            )
        lines.append("")

    # Blocked tasks (dependencies not all done)
    done_ids = {t.id for t in state.task_array if t.status == TaskStatus.DONE}
    blocked = []
    for t in state.task_array:
        if t.status == TaskStatus.DONE:
            continue
        deps = state.dependency_graph.get(t.id, [])
        if deps and not all(d in done_ids for d in deps):
            blocked.append(t)
    if blocked:
        lines.append("BLOCKED TASKS:")
        for t in blocked:
            lines.append(f"  - {t.title} (waiting on dependencies)")
        lines.append("")

    # Student progress
    if state.student_progress:
        at_risk = [
            sid
            for sid, status in state.student_progress.items()
            if status in ("at_risk", "behind")
        ]
        if at_risk:
            lines.append("STUDENTS NEEDING ATTENTION:")
            for sid in at_risk:
                sp = next(
                    (s for s in state.student_profiles if s.student_id == sid), None
                )
                name = sp.name if sp else sid
                lines.append(
                    f"  - {name}: {state.student_progress[sid]}"
                )
            lines.append("")

    # Recent interventions (last 7 days)
    recent_interventions = [
        i for i in state.intervention_history if i.timestamp >= now - timedelta(days=7)
    ]
    if recent_interventions:
        lines.append(f"RECENT INTERVENTIONS: {len(recent_interventions)} in last 7 days")
        lines.append("")

    # Last meeting action items
    if state.meeting_log:
        last_meeting = state.meeting_log[-1]
        if last_meeting.action_items:
            lines.append("ACTION ITEMS FROM LAST MEETING:")
            for item in last_meeting.action_items:
                lines.append(f"  - {item}")
            lines.append("")

    return "\n".join(lines)


def _format_agenda(agenda_items: list[str], project_name: str, date: str) -> str:
    """Format agenda items into readable text for a Google Doc."""
    lines = [f"Meeting Agenda — {project_name} — {date}", ""]
    for i, item in enumerate(agenda_items, 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append("--- Notes ---")
    lines.append("(Add meeting notes below)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


async def _handle_schedule(state: SyncUpState) -> dict[str, Any]:
    """Mode 1: Find a slot, generate agenda, create calendar event + doc."""
    now = _get_now()
    earliest = now + timedelta(days=1)
    latest = now + timedelta(days=state.meeting_interval_days)

    # --- Fetch calendar events per student via MCP ---
    calendar_events: dict[str, list[dict[str, Any]]] = {}
    doc_id: str | None = None

    try:
        async with SyncUpMCPClient() as mcp:
            cal = GoogleCalendarMCP(mcp)
            docs = GoogleDocsMCP(mcp)

            # Step 1: Fetch existing events for each student
            for sp in state.student_profiles:
                if not sp.google_email:
                    continue
                try:
                    events = await cal.get_events(
                        calendar_id=sp.google_email,
                        time_min=earliest,
                        time_max=latest,
                    )
                    calendar_events[sp.student_id] = events
                except Exception:
                    logger.warning(
                        "Failed to fetch calendar events for %s", sp.student_id,
                        exc_info=True,
                    )
                    calendar_events[sp.student_id] = []

            # Step 2: Find optimal slot
            slot = find_optimal_meeting_slot(
                student_profiles=state.student_profiles,
                duration_minutes=60,
                earliest=earliest,
                latest=latest,
                calendar_events=calendar_events,
            )

            if slot is None:
                logger.warning(
                    "No available meeting slot found for project %s. "
                    "Consider an async check-in instead.",
                    state.project_name or state.project_id,
                )
                return {"next_meeting_scheduled": None, "meeting_mode": None}

            # Step 3: Generate agenda via LLM
            agenda_items = _generate_agenda(state, now)
            slot_date = slot.strftime("%Y-%m-%d %H:%M UTC")
            project_name = state.project_name or state.project_id
            agenda_text = _format_agenda(agenda_items, project_name, slot_date)

            # Step 4: Create Google Doc for agenda/notes
            try:
                doc_result = await docs.create_document(
                    title=f"Meeting Notes — {project_name} — {slot.strftime('%Y-%m-%d')}",
                    content=agenda_text,
                )
                doc_id = doc_result.get("document_id")
            except Exception:
                logger.warning(
                    "Failed to create meeting notes doc", exc_info=True
                )

            # Step 5: Create Calendar event
            attendee_emails = [
                sp.google_email
                for sp in state.student_profiles
                if sp.google_email
            ]
            # First student's email is the calendar host
            host_email = attendee_emails[0] if attendee_emails else ""
            if host_email:
                try:
                    await cal.create_event(
                        calendar_id=host_email,
                        summary=f"SyncUp Team Meeting — {project_name}",
                        start=slot,
                        end=slot + timedelta(minutes=60),
                        attendees=attendee_emails,
                        reminders=[1440, 60],  # 24h and 1h before
                    )
                except Exception:
                    logger.warning(
                        "Failed to create calendar event", exc_info=True
                    )

    except Exception:
        # MCP connection failure — still find slot and generate agenda locally
        logger.warning("MCP connection failed, scheduling without MCP", exc_info=True)
        slot = find_optimal_meeting_slot(
            student_profiles=state.student_profiles,
            duration_minutes=60,
            earliest=earliest,
            latest=latest,
        )
        if slot is None:
            return {"next_meeting_scheduled": None, "meeting_mode": None}
        agenda_items = _generate_agenda(state, now)
        slot_date = slot.strftime("%Y-%m-%d %H:%M UTC")
        project_name = state.project_name or state.project_id
        agenda_text = _format_agenda(agenda_items, project_name, slot_date)

    # Build MeetingRecord
    record = MeetingRecord(
        date=slot,
        attendees=[sp.name for sp in state.student_profiles],
        agenda=agenda_text,
        notes="",
        action_items=[],
    )

    # Build updated doc IDs list
    updated_doc_ids = list(state.meeting_notes_doc_ids)
    if doc_id:
        updated_doc_ids.append(doc_id)

    return {
        "meeting_log": [record],
        "next_meeting_scheduled": slot,
        "meeting_notes_doc_ids": updated_doc_ids,
        "meeting_mode": None,
    }


def _generate_agenda(state: SyncUpState, now: datetime) -> list[str]:
    """Generate agenda items using the low-tier LLM with retry + fallback."""
    context = _build_context_summary(state, now)
    llm = get_low_tier_llm()

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": AGENDA_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ])
            raw_json = _extract_json(response.content)
            parsed = json.loads(raw_json)
            items = parsed.get("agenda_items", [])
            if isinstance(items, list) and items:
                return [str(item) for item in items]
            raise ValueError("agenda_items is empty or not a list")
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Agenda generation attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error("Agenda generation failed after %d attempts: %s", MAX_RETRIES + 1, last_error)
    return [
        "Status round — each member shares progress since last meeting",
        "Review upcoming deadlines and blocked tasks",
        "Discuss any blockers or items needing group decision",
        "Assign action items and next steps",
    ]


async def _handle_ingest(state: SyncUpState) -> dict[str, Any]:
    """Mode 2: Read meeting notes doc, parse with LLM, create MeetingRecord."""
    if not state.meeting_notes_doc_ids:
        logger.warning("No meeting notes doc IDs available for ingestion")
        return {}

    doc_id = state.meeting_notes_doc_ids[-1]
    raw_text = ""

    # Read the meeting notes document
    try:
        async with SyncUpMCPClient() as mcp:
            docs = GoogleDocsMCP(mcp)
            result = await docs.read_document(doc_id)
            raw_text = result.get("content", "") if isinstance(result, dict) else str(result)
    except Exception:
        logger.warning("Failed to read meeting notes doc %s", doc_id, exc_info=True)
        return {}

    if not raw_text.strip():
        logger.warning("Meeting notes doc %s is empty", doc_id)
        return {}

    # Parse notes with LLM
    parsed = _parse_meeting_notes(state, raw_text)

    now = _get_now()
    record = MeetingRecord(
        date=now,
        attendees=parsed.get("attendees", []),
        agenda="",
        notes=parsed.get("summary", raw_text[:500]),
        action_items=parsed.get("action_items", []),
    )

    return {
        "meeting_log": [record],
        "meeting_mode": None,
    }


def _parse_meeting_notes(state: SyncUpState, raw_text: str) -> dict[str, Any]:
    """Parse meeting notes with the low-tier LLM. Returns parsed dict or fallback."""
    names = ", ".join(sp.name for sp in state.student_profiles)
    project_name = state.project_name or state.project_id

    safe_text = sanitize_document(raw_text, "google_docs")

    user_prompt = (
        f"PROJECT: {project_name}\n"
        f"TEAM MEMBERS: {names}\n\n"
        f"MEETING NOTES:\n{safe_text}\n\n"
        "Extract the structured data from these meeting notes."
    )

    llm = get_low_tier_llm()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            raw_json = _extract_json(response.content)
            parsed = json.loads(raw_json)
            # Validate expected keys exist
            if "summary" not in parsed:
                raise ValueError("Missing 'summary' in parsed notes")
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Notes parsing attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )

    logger.error(
        "Meeting notes parsing failed after %d attempts: %s",
        MAX_RETRIES + 1,
        last_error,
    )
    return {
        "summary": raw_text[:500],
        "attendees": [sp.name for sp in state.student_profiles],
        "action_items": [],
        "decisions": [],
        "blockers_discussed": [],
    }


# ---------------------------------------------------------------------------
# Main agent node
# ---------------------------------------------------------------------------


async def meeting_coordinator(state: SyncUpState) -> dict[str, Any]:
    """Meeting Coordinator agent node.

    Dispatches to schedule or ingest mode based on ``state.meeting_mode``.
    Returns partial state update dict.
    """
    if state.meeting_mode is None:
        return {}

    if state.meeting_mode == "schedule":
        return await _handle_schedule(state)

    if state.meeting_mode == "ingest":
        return await _handle_ingest(state)

    logger.warning("Unknown meeting_mode: %s", state.meeting_mode)
    return {}
