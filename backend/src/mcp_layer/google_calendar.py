"""Thin wrapper around Google Calendar MCP tools.

Tool names are defined as constants — update them when the actual MCP server
is connected and tool names are known.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mcp_layer.client import SyncUpMCPClient

# MCP tool name constants
TOOL_CHECK_AVAILABILITY = "google_calendar_check_availability"
TOOL_CREATE_EVENT = "google_calendar_create_event"
TOOL_GET_EVENTS = "google_calendar_get_events"
TOOL_UPDATE_EVENT = "google_calendar_update_event"


class GoogleCalendarMCP:
    """Typed interface to Google Calendar MCP tools.

    All methods delegate to the underlying MCP tool via ``ainvoke``.
    """

    def __init__(self, mcp_client: SyncUpMCPClient) -> None:
        self._mcp = mcp_client

    async def check_availability(
        self,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> list[dict[str, Any]]:
        """Query free/busy data for a calendar.

        Args:
            calendar_id: The calendar (usually a user email).
            time_min: Start of the window.
            time_max: End of the window.

        Returns:
            List of busy time ranges.
        """
        tool = self._mcp.require_tool(TOOL_CHECK_AVAILABILITY)
        result = await tool.ainvoke({
            "calendar_id": calendar_id,
            "time_min": time_min.isoformat(),
            "time_max": time_max.isoformat(),
        })
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
        return result if isinstance(result, list) else [result]

    async def create_event(
        self,
        calendar_id: str,
        summary: str,
        start: datetime,
        end: datetime,
        attendees: list[str],
        reminders: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event.

        Args:
            calendar_id: Target calendar.
            summary: Event title.
            start: Event start time.
            end: Event end time.
            attendees: List of attendee emails.
            reminders: Reminder offsets in minutes before the event.

        Returns:
            Created event data including event ID.
        """
        tool = self._mcp.require_tool(TOOL_CREATE_EVENT)
        params: dict[str, Any] = {
            "calendar_id": calendar_id,
            "summary": summary,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "attendees": attendees,
        }
        if reminders is not None:
            params["reminders"] = reminders
        result = await tool.ainvoke(params)
        if isinstance(result, str):
            try:
                return json.loads(result)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, TypeError):
                return {"result": result}
        return result if isinstance(result, dict) else {"result": result}

    async def update_event(
        self,
        calendar_id: str,
        event_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        summary: str | None = None,
        reminders: list[int] | None = None,
    ) -> dict[str, Any]:
        """Update an existing calendar event.

        Only non-None arguments are sent to the MCP tool.

        Args:
            calendar_id: The calendar containing the event.
            event_id: The event to update.
            start: New event start time.
            end: New event end time.
            summary: New event title.
            reminders: New reminder offsets in minutes.

        Returns:
            Updated event data.
        """
        tool = self._mcp.require_tool(TOOL_UPDATE_EVENT)
        params: dict[str, Any] = {
            "calendar_id": calendar_id,
            "event_id": event_id,
        }
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        if summary is not None:
            params["summary"] = summary
        if reminders is not None:
            params["reminders"] = reminders
        result = await tool.ainvoke(params)
        if isinstance(result, str):
            try:
                return json.loads(result)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, TypeError):
                return {"result": result}
        return result if isinstance(result, dict) else {"result": result}

    async def get_events(
        self,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> list[dict[str, Any]]:
        """List events in a calendar within a time range.

        Args:
            calendar_id: The calendar to query.
            time_min: Start of the window.
            time_max: End of the window.

        Returns:
            List of event data.
        """
        tool = self._mcp.require_tool(TOOL_GET_EVENTS)
        result = await tool.ainvoke({
            "calendar_id": calendar_id,
            "time_min": time_min.isoformat(),
            "time_max": time_max.isoformat(),
        })
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                pass
        return result if isinstance(result, list) else [result]
