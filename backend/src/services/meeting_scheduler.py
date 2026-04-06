"""Meeting scheduling service — pure Python, no LLM.

Provides slot-finding and recurring-schedule generation for the
Meeting Coordinator agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from state.schema import StudentProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preferred-time parsing helpers
# ---------------------------------------------------------------------------

DAY_MAP: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

# Business hours in local time (inclusive start, exclusive end)
_BIZ_START_HOUR = 9
_BIZ_END_HOUR = 21


def _parse_preferred_time(pref: str) -> tuple[int, int, int] | None:
    """Parse a cron-like preferred time string.

    Format: ``"Day HH:MM-HH:MM"`` e.g. ``"Mon 14:00-16:00"``.

    Returns:
        ``(weekday, start_minutes, end_minutes)`` where weekday is
        0=Monday … 6=Sunday and minutes are since midnight, or
        ``None`` if the string cannot be parsed.
    """
    parts = pref.strip().split()
    if len(parts) != 2:
        return None

    day_str = parts[0].lower()[:3]
    day_num = DAY_MAP.get(day_str)
    if day_num is None:
        return None

    time_parts = parts[1].split("-")
    if len(time_parts) != 2:
        return None

    try:
        s_h, s_m = (int(x) for x in time_parts[0].split(":"))
        e_h, e_m = (int(x) for x in time_parts[1].split(":"))
    except (ValueError, IndexError):
        return None

    return (day_num, s_h * 60 + s_m, e_h * 60 + e_m)


def _student_tz(student: StudentProfile) -> ZoneInfo:
    """Resolve a student's timezone, falling back to UTC on error."""
    try:
        return ZoneInfo(student.timezone)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning(
            "Invalid timezone %r for student %s — defaulting to UTC",
            student.timezone,
            student.student_id,
        )
        return ZoneInfo("UTC")


def _ranges_overlap(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> bool:
    """Return True if [start_a, end_a) overlaps [start_b, end_b)."""
    return start_a < end_b and end_a > start_b


def _parse_iso(value: str | datetime) -> datetime:
    """Parse an ISO datetime string, passthrough if already datetime."""
    if isinstance(value, datetime):
        return value
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_optimal_meeting_slot(
    student_profiles: list[StudentProfile],
    duration_minutes: int = 60,
    earliest: datetime | None = None,
    latest: datetime | None = None,
    calendar_events: dict[str, list[dict[str, Any]]] | None = None,
) -> datetime | None:
    """Find the best meeting slot where the most students are available.

    Args:
        student_profiles: Team members with availability info.
        duration_minutes: Length of the meeting in minutes.
        earliest: Start of the search window (UTC). Defaults to now.
        latest: End of the search window (UTC). Defaults to earliest + 7 days.
        calendar_events: ``student_id → list[{start, end, …}]`` of existing
            calendar events. Events are checked for time conflicts.

    Returns:
        The optimal meeting start time (UTC-aware) or ``None`` if no valid
        slot exists.
    """
    now = datetime.now(timezone.utc)
    if earliest is None:
        earliest = now
    if latest is None:
        latest = earliest + timedelta(days=7)
    if calendar_events is None:
        calendar_events = {}

    # Ensure timezone-aware
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)

    if not student_profiles:
        return earliest

    duration = timedelta(minutes=duration_minutes)

    # Pre-parse preferred times per student
    parsed_prefs: dict[str, list[tuple[int, int, int]]] = {}
    for sp in student_profiles:
        prefs: list[tuple[int, int, int]] = []
        for pt in sp.preferred_times:
            parsed = _parse_preferred_time(pt)
            if parsed is not None:
                prefs.append(parsed)
            else:
                logger.warning(
                    "Ignoring malformed preferred_time %r for student %s",
                    pt,
                    sp.student_id,
                )
        parsed_prefs[sp.student_id] = prefs

    # Pre-parse calendar events per student
    parsed_events: dict[str, list[tuple[datetime, datetime]]] = {}
    for sp in student_profiles:
        events = calendar_events.get(sp.student_id, [])
        parsed: list[tuple[datetime, datetime]] = []
        for evt in events:
            try:
                evt_start = _parse_iso(evt["start"])
                evt_end = _parse_iso(evt["end"])
                parsed.append((evt_start, evt_end))
            except (KeyError, ValueError, TypeError):
                logger.warning("Skipping unparseable calendar event: %s", evt)
        parsed_events[sp.student_id] = parsed

    best_slot: datetime | None = None
    best_score: float = -1.0

    slot = earliest
    while slot + duration <= latest:
        slot_end = slot + duration

        available_count = 0
        preference_bonus = 0.0

        for sp in student_profiles:
            tz = _student_tz(sp)
            local_start = slot.astimezone(tz)
            local_end = slot_end.astimezone(tz)

            # Business hours check
            if local_start.hour < _BIZ_START_HOUR:
                continue
            if (
                local_end.hour > _BIZ_END_HOUR
                or (local_end.hour == _BIZ_END_HOUR and local_end.minute > 0)
            ):
                continue

            # Blackout period check
            in_blackout = False
            for bp in sp.blackout_periods:
                bp_start = bp.start if bp.start.tzinfo else bp.start.replace(
                    tzinfo=timezone.utc
                )
                bp_end = bp.end if bp.end.tzinfo else bp.end.replace(
                    tzinfo=timezone.utc
                )
                if _ranges_overlap(slot, slot_end, bp_start, bp_end):
                    in_blackout = True
                    break
            if in_blackout:
                continue

            # Calendar conflict check
            has_conflict = False
            for evt_start, evt_end in parsed_events.get(sp.student_id, []):
                if _ranges_overlap(slot, slot_end, evt_start, evt_end):
                    has_conflict = True
                    break
            if has_conflict:
                continue

            # Student is available
            available_count += 1

            # Preference bonus (max 0.5 per student)
            for day_num, pref_start_min, pref_end_min in parsed_prefs.get(
                sp.student_id, []
            ):
                if local_start.weekday() == day_num:
                    local_min = local_start.hour * 60 + local_start.minute
                    if pref_start_min <= local_min < pref_end_min:
                        preference_bonus += 0.5
                        break  # max once per student

        score = available_count + preference_bonus

        # Must have at least one student available
        if available_count > 0 and score > best_score:
            best_score = score
            best_slot = slot

        slot += timedelta(minutes=30)

    return best_slot


def generate_recurring_schedule(
    first_meeting: datetime,
    interval_days: int,
    project_end: datetime,
) -> list[datetime]:
    """Generate a recurring meeting schedule, skipping weekends.

    Args:
        first_meeting: The first meeting datetime.
        interval_days: Days between consecutive meetings.
        project_end: No meetings scheduled after this datetime.

    Returns:
        List of meeting datetimes from *first_meeting* to *project_end*.
    """
    if project_end < first_meeting:
        return []

    schedule: list[datetime] = []
    current = first_meeting

    while current <= project_end:
        # Skip weekends: Saturday=5, Sunday=6
        if current.weekday() < 5:
            schedule.append(current)
        current += timedelta(days=interval_days)

    return schedule
