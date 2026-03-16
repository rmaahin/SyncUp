"""Google Calendar MCP Server for SyncUp.

Exposes three tools over SSE transport:
  - google_calendar_create_event
  - google_calendar_get_events
  - google_calendar_check_availability

First run opens a browser for Google OAuth consent. The token is saved
to token.json for subsequent runs.

Usage:
    # First time — will open browser for OAuth:
    python server.py

    # With custom port:
    python server.py --port 3001
    # Or via env var:
    MCP_PORT=3001 python server.py

    # To re-authenticate:
    python server.py --reauth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gcal-mcp")

# ---------------------------------------------------------------------------
# Google Calendar API setup
# ---------------------------------------------------------------------------

# If modifying these scopes, delete token.json and re-authenticate.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Paths relative to this script
_DIR = Path(__file__).parent
TOKEN_PATH = _DIR / "token.json"
CREDENTIALS_PATH = _DIR / "credentials.json"


def get_calendar_service():
    """Build and return an authenticated Google Calendar API service."""
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                logger.error(
                    "credentials.json not found at %s\n"
                    "Download it from Google Cloud Console:\n"
                    "  1. Go to https://console.cloud.google.com/apis/credentials\n"
                    "  2. Create OAuth 2.0 Client ID (Desktop app)\n"
                    "  3. Download JSON and save as %s",
                    CREDENTIALS_PATH,
                    CREDENTIALS_PATH,
                )
                sys.exit(1)

            logger.info("No valid token found — starting OAuth flow...")
            logger.info("A browser window will open for Google sign-in.")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for next run
        TOKEN_PATH.write_text(creds.to_json())
        logger.info("Token saved to %s", TOKEN_PATH)

    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

_port = int(os.environ.get("MCP_PORT", "3001"))

mcp = FastMCP(
    "Google Calendar MCP",
    instructions="MCP server for Google Calendar — create events, check availability, list events",
    host="0.0.0.0",
    port=_port,
)


@mcp.tool()
def google_calendar_create_event(
    calendar_id: str,
    summary: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    reminders: list[int] | None = None,
) -> dict:
    """Create a calendar event.

    Args:
        calendar_id: Target calendar ID (e.g., 'primary' or an email address).
        summary: Event title.
        start: Event start time in ISO 8601 format.
        end: Event end time in ISO 8601 format.
        attendees: List of attendee email addresses.
        reminders: List of reminder times in minutes before the event.
    """
    service = get_calendar_service()

    event_body: dict = {
        "summary": summary,
        "start": {"dateTime": start, "timeZone": "UTC"},
        "end": {"dateTime": end, "timeZone": "UTC"},
    }

    if attendees:
        event_body["attendees"] = [{"email": email} for email in attendees]

    if reminders:
        event_body["reminders"] = {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": m} for m in reminders
            ],
        }

    event = (
        service.events()
        .insert(calendarId=calendar_id, body=event_body, sendUpdates="all")
        .execute()
    )

    logger.info("Created event: %s (ID: %s)", summary, event.get("id"))

    return {
        "id": event["id"],
        "summary": event.get("summary", ""),
        "start": event["start"].get("dateTime", ""),
        "end": event["end"].get("dateTime", ""),
        "htmlLink": event.get("htmlLink", ""),
        "status": event.get("status", ""),
    }


@mcp.tool()
def google_calendar_get_events(
    calendar_id: str,
    time_min: str,
    time_max: str,
) -> list[dict]:
    """List events in a calendar within a time range.

    Args:
        calendar_id: The calendar to query.
        time_min: Start of the time window in ISO 8601 format.
        time_max: End of the time window in ISO 8601 format.
    """
    service = get_calendar_service()

    result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = result.get("items", [])
    logger.info("Found %d events in %s", len(events), calendar_id)

    return [
        {
            "id": ev["id"],
            "summary": ev.get("summary", ""),
            "start": ev["start"].get("dateTime", ev["start"].get("date", "")),
            "end": ev["end"].get("dateTime", ev["end"].get("date", "")),
            "status": ev.get("status", ""),
        }
        for ev in events
    ]


@mcp.tool()
def google_calendar_check_availability(
    calendar_id: str,
    time_min: str,
    time_max: str,
) -> list[dict]:
    """Check busy/free times for a calendar.

    Args:
        calendar_id: The calendar to check (email address or 'primary').
        time_min: Start of the window in ISO 8601 format.
        time_max: End of the window in ISO 8601 format.
    """
    service = get_calendar_service()

    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": [{"id": calendar_id}],
    }

    result = service.freebusy().query(body=body).execute()
    busy_times = result.get("calendars", {}).get(calendar_id, {}).get("busy", [])

    logger.info("Found %d busy blocks for %s", len(busy_times), calendar_id)

    return [
        {"start": block["start"], "end": block["end"]}
        for block in busy_times
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Calendar MCP Server")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 3001, or MCP_PORT env var)")
    parser.add_argument("--reauth", action="store_true", help="Force re-authentication")
    args = parser.parse_args()

    # Override port if passed via CLI
    if args.port is not None:
        mcp._tcp_port = args.port
        _port = args.port

    if args.reauth and TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        logger.info("Deleted token.json — will re-authenticate")

    # Validate credentials exist and auth works before starting server
    logger.info("Verifying Google Calendar authentication...")
    service = get_calendar_service()
    # Quick test — list 1 event from primary calendar
    try:
        now = datetime.now(tz=timezone.utc).isoformat()
        service.events().list(
            calendarId="primary", timeMin=now, maxResults=1
        ).execute()
        logger.info("Authentication verified successfully!")
    except Exception as e:
        logger.error("Calendar API test failed: %s", e)
        sys.exit(1)

    logger.info("Starting Google Calendar MCP server on port %d...", _port)
    logger.info("SSE endpoint: http://localhost:%d/sse", _port)

    mcp.run(transport="sse")
