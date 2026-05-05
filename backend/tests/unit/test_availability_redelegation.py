"""Tests for the availability re-delegation trigger in the onboarding route."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.routes.onboarding import StudentStore, get_student_store
from state.schema import StudentProfile, SyncUpState, Task
from state.store import InMemoryStateStore

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _setup() -> Any:
    """Fresh state_store + student_store per test."""
    app.state.state_store = InMemoryStateStore()
    student_store = StudentStore()
    app.dependency_overrides[get_student_store] = lambda: student_store
    yield student_store
    app.dependency_overrides.clear()


def _seed_project_with_student(
    student_store: StudentStore, hours: float = 20.0
) -> str:
    """Create a profile in the StudentStore and a project state containing it."""
    profile = StudentProfile(
        student_id="s1",
        name="Alice",
        email="a@x.com",
        availability_hours_per_week=hours,
    )
    student_store.create(profile)
    state = SyncUpState(
        project_id="p1",
        project_name="P",
        project_brief="Brief",
        student_profiles=[profile],
        task_array=[Task(id="t1", title="T", effort_hours=5.0)],
        delegation_matrix={"t1": "s1"},
    )
    asyncio.get_event_loop().run_until_complete(
        app.state.state_store.save("p1", state)
    )
    return "s1"


class TestRedelegationTrigger:
    def test_significant_drop_triggers_delegation(
        self, _setup: StudentStore
    ) -> None:
        sid = _seed_project_with_student(_setup, hours=20.0)

        with (
            patch("agents.delegation.delegation") as mock_delegate,
            patch(
                "evaluators.equity_evaluator.equity_evaluator"
            ) as mock_equity,
            patch("api.websockets.manager") as mock_manager,
        ):
            mock_delegate.return_value = {"delegation_matrix": {"t1": "s1"}}
            mock_equity.return_value = {"equity_retries": 1}
            mock_manager.broadcast = AsyncMock()

            resp = client.put(
                f"/api/onboarding/profile/{sid}/availability",
                json={"availability_hours_per_week": 5.0, "blackout_periods": []},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["triggered_redelegation"] is True
        assert mock_delegate.called
        assert mock_equity.called
        assert mock_manager.broadcast.await_count >= 1

    def test_minor_change_does_not_trigger(self, _setup: StudentStore) -> None:
        sid = _seed_project_with_student(_setup, hours=20.0)

        with (
            patch("agents.delegation.delegation") as mock_delegate,
            patch(
                "evaluators.equity_evaluator.equity_evaluator"
            ) as mock_equity,
        ):
            resp = client.put(
                f"/api/onboarding/profile/{sid}/availability",
                json={"availability_hours_per_week": 18.0, "blackout_periods": []},
            )

        assert resp.status_code == 200
        assert resp.json()["triggered_redelegation"] is False
        assert not mock_delegate.called
        assert not mock_equity.called

    def test_no_project_silent_noop(self, _setup: StudentStore) -> None:
        # Create student but don't put them in a project
        profile = StudentProfile(
            student_id="orphan",
            name="X",
            email="x@x.com",
            availability_hours_per_week=20.0,
        )
        _setup.create(profile)

        with patch("agents.delegation.delegation") as mock_delegate:
            resp = client.put(
                "/api/onboarding/profile/orphan/availability",
                json={"availability_hours_per_week": 5.0, "blackout_periods": []},
            )

        assert resp.status_code == 200
        # triggered_redelegation flag is True (significance check) but no project found
        assert resp.json()["triggered_redelegation"] is True
        assert not mock_delegate.called

    def test_state_updated_after_redelegation(self, _setup: StudentStore) -> None:
        sid = _seed_project_with_student(_setup, hours=20.0)

        with (
            patch("agents.delegation.delegation") as mock_delegate,
            patch(
                "evaluators.equity_evaluator.equity_evaluator"
            ) as mock_equity,
            patch("api.websockets.manager") as mock_manager,
        ):
            mock_delegate.return_value = {"delegation_matrix": {"t1": "s1"}}
            mock_equity.return_value = {}
            mock_manager.broadcast = AsyncMock()

            client.put(
                f"/api/onboarding/profile/{sid}/availability",
                json={"availability_hours_per_week": 5.0, "blackout_periods": []},
            )

        # Verify state has the new availability and audit record
        state = asyncio.get_event_loop().run_until_complete(
            app.state.state_store.get("p1")
        )
        assert state is not None
        assert state.student_profiles[0].availability_hours_per_week == 5.0
        assert len(state.availability_updates) == 1
        assert state.availability_updates[0].triggered_redelegation is True
