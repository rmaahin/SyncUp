"""Tests for project management API routes."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

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


def _minimal_state_payload(
    project_id: str = "test-proj",
    trello_board_id: str = "",
    github_repo: str = "",
) -> dict[str, Any]:
    """Build a minimal SyncUpState JSON payload."""
    return {
        "project_id": project_id,
        "project_name": "Test Project",
        "trello_board_id": trello_board_id,
        "github_repo": github_repo,
    }


class TestBootstrapState:
    """Tests for POST /api/projects/{project_id}/state."""

    def test_bootstrap_state_returns_ok(self) -> None:
        """Valid state payload → 200 with summary."""
        payload = _minimal_state_payload("proj-1")
        resp = client.post("/api/projects/proj-1/state", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["project_id"] == "proj-1"

    def test_bootstrap_populates_board_map(self) -> None:
        """Bootstrap with trello_board_id populates board_project_map."""
        payload = _minimal_state_payload("proj-2", trello_board_id="board-abc")
        resp = client.post("/api/projects/proj-2/state", json=payload)
        assert resp.status_code == 200
        assert resp.json()["trello_board_id"] == "board-abc"
        # Verify the board map was populated on app state
        board_map = getattr(app.state, "board_project_map", {})
        assert board_map.get("board-abc") == "proj-2"

    def test_bootstrap_populates_repo_map(self) -> None:
        """Bootstrap with github_repo populates repo_project_map."""
        payload = _minimal_state_payload("proj-3", github_repo="owner/repo")
        resp = client.post("/api/projects/proj-3/state", json=payload)
        assert resp.status_code == 200
        assert resp.json()["github_repo"] == "owner/repo"
        repo_map = getattr(app.state, "repo_project_map", {})
        assert repo_map.get("owner/repo") == "proj-3"


class TestGetState:
    """Tests for GET /api/projects/{project_id}/state."""

    def test_get_state_returns_stored(self) -> None:
        """After bootstrap, GET returns the stored state."""
        payload = _minimal_state_payload("proj-get")
        client.post("/api/projects/proj-get/state", json=payload)
        resp = client.get("/api/projects/proj-get/state")
        assert resp.status_code == 200
        assert resp.json()["project_id"] == "proj-get"

    def test_get_state_not_found(self) -> None:
        """GET for nonexistent project → 404."""
        resp = client.get("/api/projects/nonexistent-xyz/state")
        assert resp.status_code == 404


class TestListProjects:
    """Tests for GET /api/projects/."""

    def test_list_projects(self) -> None:
        """After bootstrapping, project appears in the list."""
        payload = _minimal_state_payload("proj-list")
        client.post("/api/projects/proj-list/state", json=payload)
        resp = client.get("/api/projects/")
        assert resp.status_code == 200
        projects = resp.json()["projects"]
        assert "proj-list" in projects
