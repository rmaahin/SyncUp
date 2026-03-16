"""Tests for student onboarding API routes."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.routes.onboarding import StudentStore, get_student_store


# ---------------------------------------------------------------------------
# Test setup — inject a fresh store per test class
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_store() -> Any:
    """Override the student store dependency with a fresh instance per test."""
    store = StudentStore()
    app.dependency_overrides[get_student_store] = lambda: store
    yield store
    app.dependency_overrides.clear()


client = TestClient(app, raise_server_exceptions=False)


def _valid_profile_payload() -> dict[str, Any]:
    return {
        "name": "Alice Johnson",
        "email": "alice@university.edu",
        "skills": {"python": 0.9, "react": 0.6},
        "availability_hours_per_week": 10.0,
        "preferred_times": ["weekday_evenings"],
        "blackout_periods": [
            {"start": "2026-04-10T00:00:00", "end": "2026-04-15T00:00:00"}
        ],
        "timezone": "America/New_York",
        "github_username": "alicej",
        "google_email": "alice@gmail.com",
        "trello_username": "alicej_trello",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Health check endpoint."""

    def test_health(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Create profile
# ---------------------------------------------------------------------------


class TestCreateProfile:
    """POST /api/onboarding/profile"""

    def test_create_valid_profile(self) -> None:
        resp = client.post("/api/onboarding/profile", json=_valid_profile_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert "student_id" in data
        assert data["name"] == "Alice Johnson"
        assert data["email"] == "alice@university.edu"
        assert data["skills"] == {"python": 0.9, "react": 0.6}
        assert data["onboarded_at"] is not None

    def test_create_minimal_profile(self) -> None:
        resp = client.post(
            "/api/onboarding/profile",
            json={"name": "Bob", "email": "bob@uni.edu"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Bob"
        assert data["skills"] == {}
        assert data["timezone"] == "UTC"

    def test_create_missing_name(self) -> None:
        resp = client.post(
            "/api/onboarding/profile",
            json={"email": "noname@uni.edu"},
        )
        assert resp.status_code == 422

    def test_create_missing_email(self) -> None:
        resp = client.post(
            "/api/onboarding/profile",
            json={"name": "NoEmail"},
        )
        assert resp.status_code == 422

    def test_create_invalid_skill_score_too_high(self) -> None:
        payload = _valid_profile_payload()
        payload["skills"] = {"python": 1.5}
        resp = client.post("/api/onboarding/profile", json=payload)
        assert resp.status_code == 422

    def test_create_invalid_skill_score_negative(self) -> None:
        payload = _valid_profile_payload()
        payload["skills"] = {"python": -0.1}
        resp = client.post("/api/onboarding/profile", json=payload)
        assert resp.status_code == 422

    def test_create_invalid_blackout_period(self) -> None:
        payload = _valid_profile_payload()
        # start after end
        payload["blackout_periods"] = [
            {"start": "2026-04-15T00:00:00", "end": "2026-04-10T00:00:00"}
        ]
        resp = client.post("/api/onboarding/profile", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    """GET /api/onboarding/profile/{student_id}"""

    def test_get_existing(self) -> None:
        create_resp = client.post("/api/onboarding/profile", json=_valid_profile_payload())
        student_id = create_resp.json()["student_id"]
        get_resp = client.get(f"/api/onboarding/profile/{student_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["student_id"] == student_id

    def test_get_nonexistent(self) -> None:
        resp = client.get("/api/onboarding/profile/does_not_exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Availability update
# ---------------------------------------------------------------------------


class TestUpdateAvailability:
    """PUT /api/onboarding/profile/{student_id}/availability"""

    def _create_student(self, hours: float = 10.0) -> str:
        payload = _valid_profile_payload()
        payload["availability_hours_per_week"] = hours
        resp = client.post("/api/onboarding/profile", json=payload)
        return resp.json()["student_id"]

    def test_minor_change_no_redelegation(self) -> None:
        sid = self._create_student(hours=10.0)
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 9.0, "blackout_periods": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["triggered_redelegation"] is False
        assert data["profile"]["availability_hours_per_week"] == 9.0

    def test_significant_change_triggers_redelegation(self) -> None:
        sid = self._create_student(hours=10.0)
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 5.0, "blackout_periods": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["triggered_redelegation"] is True

    def test_exactly_30_percent_triggers_redelegation(self) -> None:
        sid = self._create_student(hours=10.0)
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 7.0, "blackout_periods": []},
        )
        assert resp.status_code == 200
        assert resp.json()["triggered_redelegation"] is True

    def test_29_percent_no_redelegation(self) -> None:
        sid = self._create_student(hours=10.0)
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 7.1, "blackout_periods": []},
        )
        assert resp.status_code == 200
        assert resp.json()["triggered_redelegation"] is False

    def test_nonexistent_student(self) -> None:
        resp = client.put(
            "/api/onboarding/profile/missing/availability",
            json={"availability_hours_per_week": 5.0, "blackout_periods": []},
        )
        assert resp.status_code == 404

    def test_change_record_created(self) -> None:
        sid = self._create_student(hours=10.0)
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 6.0, "blackout_periods": []},
        )
        data = resp.json()
        record = data["change_record"]
        assert record["student_id"] == sid
        assert record["old_hours"] == 10.0
        assert record["new_hours"] == 6.0
        assert record["triggered_redelegation"] is True

    def test_blackout_periods_updated(self) -> None:
        sid = self._create_student(hours=10.0)
        new_blackouts = [
            {"start": "2026-05-01T00:00:00", "end": "2026-05-05T00:00:00"}
        ]
        resp = client.put(
            f"/api/onboarding/profile/{sid}/availability",
            json={"availability_hours_per_week": 10.0, "blackout_periods": new_blackouts},
        )
        assert resp.status_code == 200
        profile = resp.json()["profile"]
        assert len(profile["blackout_periods"]) == 1


# ---------------------------------------------------------------------------
# Project readiness
# ---------------------------------------------------------------------------


class TestProjectReadiness:
    """GET /api/onboarding/project/{project_id}/ready"""

    def test_no_students(self) -> None:
        resp = client.get("/api/onboarding/project/proj1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False
        assert data["total_expected"] == 0

    def test_all_ready(self) -> None:
        client.post("/api/onboarding/profile", json=_valid_profile_payload())
        resp = client.get("/api/onboarding/project/proj1/ready")
        data = resp.json()
        assert data["ready"] is True
        assert data["onboarded_count"] == 1

    def test_missing_skills(self) -> None:
        # Create a profile with no skills
        payload = _valid_profile_payload()
        payload["skills"] = {}
        client.post("/api/onboarding/profile", json=payload)
        resp = client.get("/api/onboarding/project/proj1/ready")
        data = resp.json()
        assert data["ready"] is False
        # Find the student in missing_fields
        assert any("skills" in fields for fields in data["missing_fields"].values())

    def test_missing_availability(self) -> None:
        payload = _valid_profile_payload()
        payload["availability_hours_per_week"] = 0
        client.post("/api/onboarding/profile", json=payload)
        resp = client.get("/api/onboarding/project/proj1/ready")
        data = resp.json()
        assert data["ready"] is False


# ---------------------------------------------------------------------------
# OAuth stubs
# ---------------------------------------------------------------------------


class TestOAuthStubs:
    """POST /api/onboarding/oauth/{provider}"""

    def test_google_oauth_stub(self) -> None:
        resp = client.post(
            "/api/onboarding/oauth/google",
            json={"code": "auth_code_12345678"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "google"
        assert data["linked"] is True

    def test_github_oauth_stub(self) -> None:
        resp = client.post(
            "/api/onboarding/oauth/github",
            json={"code": "gh_code_12345678"},
        )
        assert resp.status_code == 200
        assert resp.json()["provider"] == "github"

    def test_trello_oauth_stub(self) -> None:
        resp = client.post(
            "/api/onboarding/oauth/trello",
            json={"code": "trello_code_12345678"},
        )
        assert resp.status_code == 200
        assert resp.json()["provider"] == "trello"

    def test_invalid_provider(self) -> None:
        resp = client.post(
            "/api/onboarding/oauth/invalid_provider",
            json={"code": "some_code_12345678"},
        )
        assert resp.status_code == 422
