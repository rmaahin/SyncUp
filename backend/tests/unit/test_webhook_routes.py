"""Tests for webhook ingestion API routes."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import app
from state.schema import SyncUpState
from state.store import InMemoryStateStore

client = TestClient(app, raise_server_exceptions=False)

# Ensure app state has the required attributes (TestClient may skip lifespan)
if not hasattr(app.state, "state_store"):
    app.state.state_store = InMemoryStateStore()
if not hasattr(app.state, "board_project_map"):
    app.state.board_project_map = {}
if not hasattr(app.state, "repo_project_map"):
    app.state.repo_project_map = {}

WEBHOOK_SECRET = "test-webhook-secret"


def _sign_payload(payload: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Compute the GitHub HMAC-SHA256 signature for a payload."""
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _valid_push_payload() -> dict[str, Any]:
    """A minimal valid GitHub push webhook payload."""
    return {
        "ref": "refs/heads/main",
        "before": "0000000",
        "after": "abc1234",
        "commits": [
            {
                "id": "abc1234",
                "message": "Add feature X",
                "timestamp": "2026-04-01T12:00:00Z",
                "author": {"name": "octocat", "email": "octo@example.com"},
                "added": ["src/feature.py"],
                "removed": [],
                "modified": [],
            }
        ],
        "repository": {
            "id": 12345,
            "name": "my-repo",
            "full_name": "owner/my-repo",
            "html_url": "https://github.com/owner/my-repo",
        },
        "pusher": {"name": "octocat", "email": "octo@example.com"},
    }


def _valid_pr_payload() -> dict[str, Any]:
    """A minimal valid GitHub pull_request webhook payload."""
    return {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "id": 1,
            "number": 42,
            "title": "Add authentication",
            "state": "open",
            "user": {"login": "octocat"},
            "head": {"ref": "feature-auth"},
            "base": {"ref": "main"},
            "merged": False,
        },
    }


def _valid_trello_payload() -> dict[str, Any]:
    """A minimal valid Trello card update webhook payload."""
    return {
        "action": {
            "type": "updateCard",
            "data": {
                "card": {"id": "card-123", "name": "Setup repo"},
                "listBefore": {"id": "list-todo"},
                "listAfter": {"id": "list-in-progress"},
            },
        },
        "model": {"id": "board-1"},
    }


# =========================================================================
# GitHub webhook tests
# =========================================================================


class TestGitHubWebhook:
    """Tests for POST /api/webhooks/github."""

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_valid_push_returns_200(self) -> None:
        """Valid push event with correct signature → 200 accepted."""
        payload = _valid_push_payload()
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)

        # Mock MCP client on app state so diff pre-fetch doesn't fail
        app.state.mcp_client = None

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_valid_pr_returns_200(self) -> None:
        """Valid pull_request event with correct signature → 200 accepted."""
        payload = _valid_pr_payload()
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "pull_request",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_invalid_signature_returns_401(self) -> None:
        """Invalid HMAC signature → 401."""
        payload = _valid_push_payload()
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=invalid",
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 401

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_malformed_push_payload_returns_400(self) -> None:
        """Push event with missing required fields → 400."""
        # Missing 'repository' field which is required
        payload = {"ref": "refs/heads/main", "commits": []}
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 400

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_unknown_event_type_ignored(self) -> None:
        """Unknown X-GitHub-Event header → ignored (200)."""
        payload = {"action": "completed"}
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "check_run",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"


# =========================================================================
# Trello webhook tests
# =========================================================================


class TestTrelloWebhook:
    """Tests for POST /api/webhooks/trello."""

    def test_card_move_returns_200(self) -> None:
        """Valid card move event → 200 accepted."""
        payload = _valid_trello_payload()
        resp = client.post("/api/webhooks/trello", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_malformed_payload_returns_400(self) -> None:
        """Missing 'action' field → 400."""
        payload: dict[str, Any] = {"model": {"id": "board-1"}}
        resp = client.post("/api/webhooks/trello", json=payload)
        assert resp.status_code == 400

    def test_non_move_action_ignored(self) -> None:
        """Non-updateCard action → ignored."""
        payload = {
            "action": {
                "type": "createCard",
                "data": {"card": {"id": "card-1", "name": "New card"}},
            }
        }
        resp = client.post("/api/webhooks/trello", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_head_verification_returns_200(self) -> None:
        """Trello HEAD verification → 200."""
        resp = client.head("/api/webhooks/trello")
        assert resp.status_code == 200


# =========================================================================
# State-store integration tests (webhook → agent → persist)
# =========================================================================


def _bootstrap_project(
    project_id: str = "proj-1",
    github_repo: str = "owner/my-repo",
    trello_board_id: str = "board-1",
) -> None:
    """Bootstrap a project into the app state store via the API."""
    payload = {
        "project_id": project_id,
        "project_name": "Test Project",
        "github_repo": github_repo,
        "trello_board_id": trello_board_id,
    }
    resp = client.post(f"/api/projects/{project_id}/state", json=payload)
    assert resp.status_code == 200


class TestGitHubWebhookWithStore:
    """Tests for GitHub webhooks wired through the state store + agent."""

    @patch("api.routes.webhooks.progress_tracking")
    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_push_processes_with_state_store(self, mock_agent: MagicMock) -> None:
        """Push with matching project → agent invoked, contribution persisted."""
        _bootstrap_project("proj-gh", github_repo="owner/my-repo")

        mock_agent.return_value = {
            "contribution_ledger": [],
            "pending_event": None,
            "student_progress": {"alice": "on_track"},
        }

        payload = _valid_push_payload()
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)
        app.state.mcp_client = None

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processed"
        assert data["project_id"] == "proj-gh"
        mock_agent.assert_called_once()

    @patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}, clear=False)
    def test_push_no_matching_project(self) -> None:
        """Push from unregistered repo → accepted with no-match detail."""
        payload = _valid_push_payload()
        payload["repository"]["full_name"] = "unknown/no-project"
        body = json.dumps(payload).encode()
        signature = _sign_payload(body)
        app.state.mcp_client = None

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["detail"] == "no matching project"


class TestTrelloWebhookWithStore:
    """Tests for Trello webhooks wired through the state store + agent."""

    @patch("api.routes.webhooks.progress_tracking")
    def test_card_move_processes_with_state_store(self, mock_agent: MagicMock) -> None:
        """Card move with matching board → agent invoked, state saved."""
        _bootstrap_project("proj-tr", trello_board_id="board-1")

        mock_agent.return_value = {
            "contribution_ledger": [],
            "pending_event": None,
            "student_progress": {"bob": "on_track"},
        }

        payload = _valid_trello_payload()
        resp = client.post("/api/webhooks/trello", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processed"
        assert data["project_id"] == "proj-tr"
        mock_agent.assert_called_once()

    def test_trello_no_matching_board(self) -> None:
        """Card move from unregistered board → accepted with no-match detail."""
        payload = _valid_trello_payload()
        payload["model"]["id"] = "unknown-board-xyz"
        resp = client.post("/api/webhooks/trello", json=payload)
        assert resp.status_code == 200
        assert resp.json()["detail"] == "no matching project"
