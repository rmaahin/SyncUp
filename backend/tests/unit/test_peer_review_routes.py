"""Tests for the peer-review API routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from agents.peer_review import DIMENSION_KEYS
from state.schema import StudentProfile, SyncUpState, Task, TaskStatus
from state.store import InMemoryStateStore

client = TestClient(app, raise_server_exceptions=False)


def _state(project_id: str = "proj-pr") -> SyncUpState:
    deadline = datetime.now(timezone.utc) + timedelta(days=14)
    students = [
        StudentProfile(student_id="s1", name="Alice", email="a@x.com"),
        StudentProfile(student_id="s2", name="Bob", email="b@x.com"),
        StudentProfile(student_id="s3", name="Carol", email="c@x.com"),
        StudentProfile(student_id="s4", name="Dave", email="d@x.com"),
    ]
    tasks = [
        Task(id="t1", title="Design", status=TaskStatus.DONE, deadline=deadline),
        Task(id="t2", title="Backend", status=TaskStatus.DONE, deadline=deadline),
    ]
    delegation = {"t1": "s2", "t2": "s3"}
    return SyncUpState(
        project_id=project_id,
        project_brief="Build a web app for peer review.",
        student_profiles=students,
        task_array=tasks,
        delegation_matrix=delegation,
    )


@pytest.fixture(autouse=True)
def _setup() -> Any:
    app.state.state_store = InMemoryStateStore()
    asyncio.get_event_loop().run_until_complete(
        app.state.state_store.save("proj-pr", _state())
    )
    yield


def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the low-tier LLM in the peer-review agent to return canned JSON."""
    fake = MagicMock()
    canned = {k: f"Description for {k}" for k in DIMENSION_KEYS}
    fake.invoke.return_value = MagicMock(
        content='{"dimension_descriptions": ' + str(canned).replace("'", '"') + "}"
    )
    monkeypatch.setattr("agents.peer_review.get_low_tier_llm", lambda *a, **k: fake)


def _full_ratings(value: int = 4) -> dict[str, int]:
    return {k: value for k in DIMENSION_KEYS}


def _valid_payload(reviewer: str, others: list[str]) -> dict[str, Any]:
    return {
        "reviewer_id": reviewer,
        "reviews": [
            {
                "reviewee_id": o,
                "ratings": _full_ratings(4),
                "comments": {},
            }
            for o in others
        ],
    }


# ---------------------------------------------------------------------------
# Generate + form
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_populates_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_llm(monkeypatch)
        resp = client.post("/api/peer-review/proj-pr/generate")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["dimensions"]) == 5
        assert set(body["forms_by_student"].keys()) == {"s1", "s2", "s3", "s4"}

    def test_form_excludes_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_llm(monkeypatch)
        client.post("/api/peer-review/proj-pr/generate")
        resp = client.get("/api/peer-review/proj-pr/form/s1")
        assert resp.status_code == 200
        teammates = resp.json()["teammates"]
        ids = {t["id"] for t in teammates}
        assert "s1" not in ids
        assert ids == {"s2", "s3", "s4"}

    def test_form_includes_assigned_tasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_llm(monkeypatch)
        client.post("/api/peer-review/proj-pr/generate")
        resp = client.get("/api/peer-review/proj-pr/form/s1")
        teammates = {t["id"]: t for t in resp.json()["teammates"]}
        assert "Design" in teammates["s2"]["assigned_tasks"]
        assert "Backend" in teammates["s3"]["assigned_tasks"]

    def test_form_404_before_generate(self) -> None:
        resp = client.get("/api/peer-review/proj-pr/form/s1")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def _generate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_llm(monkeypatch)
        client.post("/api/peer-review/proj-pr/generate")

    def test_valid_submission_201(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        resp = client.post(
            "/api/peer-review/proj-pr/submit",
            json=_valid_payload("s1", ["s2", "s3", "s4"]),
        )
        assert resp.status_code == 201
        assert resp.json()["count"] == 3

    def test_out_of_range_rating_422(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        payload = _valid_payload("s1", ["s2", "s3", "s4"])
        payload["reviews"][0]["ratings"]["communication"] = 0
        resp = client.post("/api/peer-review/proj-pr/submit", json=payload)
        assert resp.status_code == 422

    def test_review_self_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        payload = _valid_payload("s1", ["s1", "s2", "s3", "s4"])
        resp = client.post("/api/peer-review/proj-pr/submit", json=payload)
        assert resp.status_code == 400

    def test_double_submission_409(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        payload = _valid_payload("s1", ["s2", "s3", "s4"])
        client.post("/api/peer-review/proj-pr/submit", json=payload)
        resp = client.post("/api/peer-review/proj-pr/submit", json=payload)
        assert resp.status_code == 409

    def test_missing_teammate_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        payload = _valid_payload("s1", ["s2", "s3"])  # missing s4
        resp = client.post("/api/peer-review/proj-pr/submit", json=payload)
        assert resp.status_code == 400

    def test_unknown_dimension_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._generate(monkeypatch)
        payload = _valid_payload("s1", ["s2", "s3", "s4"])
        payload["reviews"][0]["ratings"]["totally_made_up"] = 4
        resp = client.post("/api/peer-review/proj-pr/submit", json=payload)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_after_submission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_llm(monkeypatch)
        client.post("/api/peer-review/proj-pr/generate")
        client.post(
            "/api/peer-review/proj-pr/submit",
            json=_valid_payload("s1", ["s2", "s3", "s4"]),
        )
        resp = client.get("/api/peer-review/proj-pr/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["submitted"] == ["s1"]
        assert set(body["pending"]) == {"s2", "s3", "s4"}
        assert body["total"] == 4
