"""Google Docs MCP Server for SyncUp.

Exposes three tools over SSE transport:
  - google_docs_create_document
  - google_docs_read_document
  - google_docs_search_documents

First run opens a browser for Google OAuth consent. The token is saved
to token.json for subsequent runs.

Usage:
    # First time — will open browser for OAuth:
    python server.py

    # With custom port:
    python server.py --port 3002
    # Or via env var:
    MCP_PORT=3002 python server.py

    # To re-authenticate:
    python server.py --reauth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gdocs-mcp")

# ---------------------------------------------------------------------------
# Google API setup
# ---------------------------------------------------------------------------

# Scopes: Docs for read/create, Drive for search
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

# Paths relative to this script
_DIR = Path(__file__).parent
TOKEN_PATH = _DIR / "token.json"
CREDENTIALS_PATH = _DIR / "credentials.json"


def get_credentials() -> Credentials:
    """Load or create Google OAuth credentials."""
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

    return creds


def get_docs_service():
    """Build and return an authenticated Google Docs API service."""
    return build("docs", "v1", credentials=get_credentials())


def get_drive_service():
    """Build and return an authenticated Google Drive API service."""
    return build("drive", "v3", credentials=get_credentials())


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

_port = int(os.environ.get("MCP_PORT", "3002"))

mcp = FastMCP(
    "Google Docs MCP",
    instructions="MCP server for Google Docs — create documents, read content, search documents",
    host="0.0.0.0",
    port=_port,
)


@mcp.tool()
def google_docs_create_document(
    title: str,
    content: str = "",
) -> dict:
    """Create a new Google Doc with optional initial content.

    Args:
        title: Document title.
        content: Initial body text to insert into the document.
    """
    docs_service = get_docs_service()

    # Create the document
    doc = docs_service.documents().create(body={"title": title}).execute()
    document_id = doc["documentId"]
    logger.info("Created document: %s (ID: %s)", title, document_id)

    # Insert content if provided
    if content:
        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": content,
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()
        logger.info("Inserted %d chars of content into %s", len(content), document_id)

    return {
        "document_id": document_id,
        "title": doc.get("title", ""),
        "document_url": f"https://docs.google.com/document/d/{document_id}/edit",
    }


@mcp.tool()
def google_docs_read_document(document_id: str) -> dict:
    """Read the text content of a Google Doc.

    Args:
        document_id: The Google Docs document ID.
    """
    docs_service = get_docs_service()

    doc = docs_service.documents().get(documentId=document_id).execute()
    title = doc.get("title", "")

    # Extract plain text from the document body
    content_parts: list[str] = []
    body = doc.get("body", {})
    for element in body.get("content", []):
        paragraph = element.get("paragraph")
        if paragraph:
            for text_element in paragraph.get("elements", []):
                text_run = text_element.get("textRun")
                if text_run:
                    content_parts.append(text_run.get("content", ""))

    content = "".join(content_parts)
    logger.info("Read document %s (%d chars)", document_id, len(content))

    return {
        "document_id": document_id,
        "title": title,
        "content": content,
    }


@mcp.tool()
def google_docs_search_documents(query: str) -> list[dict]:
    """Search for Google Docs matching a query.

    Args:
        query: Search query string (searches document names/content via Google Drive).
    """
    drive_service = get_drive_service()

    # Search for Google Docs only
    drive_query = f"mimeType='application/vnd.google-apps.document' and name contains '{query}'"

    result = (
        drive_service.files()
        .list(
            q=drive_query,
            fields="files(id, name, modifiedTime, webViewLink)",
            pageSize=20,
            orderBy="modifiedTime desc",
        )
        .execute()
    )

    files = result.get("files", [])
    logger.info("Found %d documents matching '%s'", len(files), query)

    return [
        {
            "document_id": f["id"],
            "title": f.get("name", ""),
            "modified_time": f.get("modifiedTime", ""),
            "url": f.get("webViewLink", ""),
        }
        for f in files
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Docs MCP Server")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 3002, or MCP_PORT env var)")
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
    logger.info("Verifying Google Docs authentication...")
    creds = get_credentials()
    try:
        # Quick test — use Drive API to list 1 doc (read-only, no side effects)
        drive_service = build("drive", "v3", credentials=creds)
        drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=1,
        ).execute()
        logger.info("Authentication verified successfully!")
    except Exception as e:
        logger.error("Docs API test failed: %s", e)
        sys.exit(1)

    logger.info("Starting Google Docs MCP server on port %d...", _port)
    logger.info("SSE endpoint: http://localhost:%d/sse", _port)

    mcp.run(transport="sse")
