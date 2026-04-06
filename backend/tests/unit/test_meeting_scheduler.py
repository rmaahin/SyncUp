"""Unit tests for the meeting scheduler service — pure logic, NO mocking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from services.meeting_scheduler import (
    _parse_preferred_time,
    find_optimal_meeting_slot,
    generate_recurring_schedule,
)
from state.schema import DateRange, StudentProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_student(
    student_id: str = "s-1",
    name: str = "Alice",
    timezone_str: str = "UTC",
    preferred_times: list[str] | None = None,
    blackout_periods: list[DateRange] | None = None,
) -> StudentProfile:
    """Create a minimal StudentProfile for testing."""
    return StudentProfile(
        student_id=student_id,
        name=name,
        email=f"{name.lower()}@example.com",
        skills={"python": 0.8},
        availability_hours_per_week=10.0,
        timezone=timezone_str,
        preferred_times=preferred_times or [],
        blackout_periods=blackout_periods or [],
        google_email=f"{name.lower()}@example.com",
    )


# Use a known Wednesday at 10:00 UTC as our anchor
_WEDNESDAY_10AM = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)  # Wednesday


# ---------------------------------------------------------------------------
# _parse_preferred_time tests
# ---------------------------------------------------------------------------


class TestParsePreferredTime:
    """Tests for the cron-like preferred time parser."""

    def test_valid_format(self) -> None:
        result = _parse_preferred_time("Mon 14:00-16:00")
        assert result == (0, 840, 960)  # Monday, 14*60=840, 16*60=960

    def test_case_insensitive_day(self) -> None:
        result = _parse_preferred_time("FRIDAY 09:00-12:00")
        assert result is not None
        assert result[0] == 4  # Friday

    def test_invalid_day(self) -> None:
        assert _parse_preferred_time("Xyz 09:00-12:00") is None

    def test_missing_time_range(self) -> None:
        assert _parse_preferred_time("Mon") is None

    def test_missing_dash(self) -> None:
        assert _parse_preferred_time("Mon 14:00") is None

    def test_empty_string(self) -> None:
        assert _parse_preferred_time("") is None


# ---------------------------------------------------------------------------
# find_optimal_meeting_slot tests
# ---------------------------------------------------------------------------


class TestFindOptimalMeetingSlot:
    """Tests for the slot-finding algorithm."""

    def test_three_students_overlapping_availability(self) -> None:
        """3 students all in UTC, all available during business hours."""
        students = [
            _make_student("s-1", "Alice", "UTC"),
            _make_student("s-2", "Bob", "UTC"),
            _make_student("s-3", "Carol", "UTC"),
        ]
        # Search window: a single day (Wednesday) 9am-6pm UTC
        earliest = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        # Should be within business hours
        assert result.hour >= 9
        assert (result + timedelta(minutes=60)).hour <= 21

    def test_calendar_conflicts_avoids_busy_slot(self) -> None:
        """Student has a calendar event; algorithm picks a different slot."""
        students = [_make_student("s-1", "Alice", "UTC")]
        earliest = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 13, 0, tzinfo=timezone.utc)

        # Alice busy 9:00-11:00
        events = {
            "s-1": [
                {
                    "start": "2026-04-01T09:00:00+00:00",
                    "end": "2026-04-01T11:00:00+00:00",
                }
            ]
        }

        result = find_optimal_meeting_slot(
            students,
            duration_minutes=60,
            earliest=earliest,
            latest=latest,
            calendar_events=events,
        )
        assert result is not None
        # Must be 11:00 or later (after conflict)
        assert result >= datetime(2026, 4, 1, 11, 0, tzinfo=timezone.utc)

    def test_all_students_busy_returns_none(self) -> None:
        """All slots blocked → returns None."""
        students = [_make_student("s-1", "Alice", "UTC")]
        earliest = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)

        # Alice busy the entire window
        events = {
            "s-1": [
                {
                    "start": "2026-04-01T09:00:00+00:00",
                    "end": "2026-04-01T13:00:00+00:00",
                }
            ]
        }

        result = find_optimal_meeting_slot(
            students,
            duration_minutes=60,
            earliest=earliest,
            latest=latest,
            calendar_events=events,
        )
        assert result is None

    def test_blackout_period_avoidance(self) -> None:
        """Slots in a blackout period are skipped."""
        blackout = DateRange(
            start=datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
        )
        students = [_make_student("s-1", "Alice", "UTC", blackout_periods=[blackout])]
        earliest = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        assert result >= datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)

    def test_preferred_times_scoring(self) -> None:
        """Students preferring Wed afternoon → algorithm picks Wed afternoon slot."""
        students = [
            _make_student("s-1", "Alice", "UTC", preferred_times=["Wed 14:00-16:00"]),
            _make_student("s-2", "Bob", "UTC", preferred_times=["Wed 14:00-16:00"]),
        ]
        # Wide window covering morning and afternoon
        earliest = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)  # Wednesday
        latest = datetime(2026, 4, 1, 20, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        # Should prefer the 14:00 slot due to preference bonus
        assert result.hour == 14

    def test_timezone_handling(self) -> None:
        """Students in different timezones — respects business hours per tz."""
        students = [
            # Tokyo (UTC+9): 9am-9pm local = 0:00-12:00 UTC
            _make_student("s-1", "Yuki", "Asia/Tokyo"),
            # UTC: 9am-9pm local = 9:00-21:00 UTC
            _make_student("s-2", "Alice", "UTC"),
        ]
        # Overlap in business hours for both: 9:00-12:00 UTC
        # (9am-12pm UTC = 6pm-9pm Tokyo, which is within 9-21 local)
        earliest = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 20, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        # Verify valid for both timezones
        tokyo_tz = ZoneInfo("Asia/Tokyo")
        local_tokyo = result.astimezone(tokyo_tz)
        assert 9 <= local_tokyo.hour < 21

    def test_empty_profiles_returns_earliest(self) -> None:
        """No students → return earliest."""
        earliest = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        result = find_optimal_meeting_slot(
            [], duration_minutes=60, earliest=earliest, latest=earliest + timedelta(days=1)
        )
        assert result == earliest

    def test_tie_break_prefers_earlier_slot(self) -> None:
        """Equal scores → earlier slot wins."""
        students = [_make_student("s-1", "Alice", "UTC")]
        earliest = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        # First valid slot should win
        assert result == earliest

    def test_boundary_slot_starts_when_blackout_ends(self) -> None:
        """Slot starting exactly when blackout ends should be valid."""
        blackout = DateRange(
            start=datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
        )
        students = [_make_student("s-1", "Alice", "UTC", blackout_periods=[blackout])]
        earliest = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None
        # Should be exactly 12:00 (right after blackout ends)
        assert result == datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)

    def test_invalid_timezone_falls_back_to_utc(self) -> None:
        """Invalid timezone string → no crash, falls back to UTC."""
        students = [_make_student("s-1", "Alice", "Invalid/Timezone")]
        earliest = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)

        # Should not raise
        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        assert result is not None

    def test_cross_midnight_outside_business_hours(self) -> None:
        """A UTC late-night slot that is before 9am in Tokyo → filtered."""
        # 20:00 UTC = 05:00+1 Tokyo (outside 9-21 local)
        # 21:00 UTC = 06:00+1 Tokyo (outside 9-21 local)
        # 22:00 UTC = 07:00+1 Tokyo (outside 9-21 local)
        # 23:00 UTC = 08:00+1 Tokyo (outside 9-21 local, end would be 09:00)
        # So all slots 20:00-23:00 UTC should fail for Tokyo student
        students = [
            _make_student("s-1", "Yuki", "Asia/Tokyo"),
        ]
        earliest = datetime(2026, 4, 1, 20, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 4, 1, 23, 0, tzinfo=timezone.utc)

        result = find_optimal_meeting_slot(
            students, duration_minutes=60, earliest=earliest, latest=latest
        )
        # All slots map to pre-9am in Tokyo → None
        assert result is None


# ---------------------------------------------------------------------------
# generate_recurring_schedule tests
# ---------------------------------------------------------------------------


class TestGenerateRecurringSchedule:
    """Tests for the recurring schedule generator."""

    def test_weekly_schedule(self) -> None:
        """7-day interval generates weekly meetings."""
        first = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)  # Wednesday
        end = datetime(2026, 4, 22, 23, 59, tzinfo=timezone.utc)

        schedule = generate_recurring_schedule(first, interval_days=7, project_end=end)
        assert len(schedule) == 4  # Apr 1, 8, 15, 22
        for dt in schedule:
            assert dt.weekday() < 5  # No weekends

    def test_skips_weekends(self) -> None:
        """Meetings landing on weekends are skipped."""
        # Start on Friday, interval=2 → next would be Sunday (skip), then Tue
        first = datetime(2026, 4, 3, 10, 0, tzinfo=timezone.utc)  # Friday
        end = datetime(2026, 4, 10, 23, 59, tzinfo=timezone.utc)

        schedule = generate_recurring_schedule(first, interval_days=2, project_end=end)
        for dt in schedule:
            assert dt.weekday() < 5, f"{dt} is a weekend"

    def test_daily_meetings_skip_weekends(self) -> None:
        """interval_days=1 → daily meetings, still skips weekends."""
        first = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)  # Wednesday
        end = datetime(2026, 4, 7, 23, 59, tzinfo=timezone.utc)  # Tuesday

        schedule = generate_recurring_schedule(first, interval_days=1, project_end=end)
        # Wed, Thu, Fri, Mon, Tue = 5 (skip Sat+Sun)
        assert len(schedule) == 5
        for dt in schedule:
            assert dt.weekday() < 5

    def test_project_end_before_first_meeting(self) -> None:
        """project_end < first_meeting → empty list."""
        first = datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc)

        schedule = generate_recurring_schedule(first, interval_days=7, project_end=end)
        assert schedule == []

    def test_single_meeting_when_end_equals_first(self) -> None:
        """project_end == first_meeting → one meeting."""
        first = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)  # Wednesday
        schedule = generate_recurring_schedule(first, interval_days=7, project_end=first)
        assert len(schedule) == 1
        assert schedule[0] == first
