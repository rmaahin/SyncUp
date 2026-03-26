"""Tests for the in-memory state store."""

from __future__ import annotations

import pytest

from state.schema import SyncUpState
from state.store import InMemoryStateStore


@pytest.fixture
def store() -> InMemoryStateStore:
    """Create a fresh store for each test."""
    return InMemoryStateStore()


def _make_state(project_id: str = "test-project") -> SyncUpState:
    """Create a minimal SyncUpState."""
    return SyncUpState(project_id=project_id, project_name="Test Project")


class TestInMemoryStateStore:
    """Tests for InMemoryStateStore."""

    async def test_save_and_get(self, store: InMemoryStateStore) -> None:
        state = _make_state("proj-1")
        await store.save("proj-1", state)
        result = await store.get("proj-1")
        assert result is not None
        assert result.project_id == "proj-1"

    async def test_get_missing_returns_none(self, store: InMemoryStateStore) -> None:
        result = await store.get("nonexistent")
        assert result is None

    async def test_overwrite_existing(self, store: InMemoryStateStore) -> None:
        state1 = _make_state("proj-1")
        await store.save("proj-1", state1)

        state2 = SyncUpState(project_id="proj-1", project_name="Updated")
        await store.save("proj-1", state2)

        result = await store.get("proj-1")
        assert result is not None
        assert result.project_name == "Updated"

    async def test_list_projects(self, store: InMemoryStateStore) -> None:
        await store.save("proj-a", _make_state("proj-a"))
        await store.save("proj-b", _make_state("proj-b"))
        projects = await store.list_projects()
        assert sorted(projects) == ["proj-a", "proj-b"]

    async def test_list_projects_empty(self, store: InMemoryStateStore) -> None:
        projects = await store.list_projects()
        assert projects == []

    async def test_delete(self, store: InMemoryStateStore) -> None:
        await store.save("proj-1", _make_state("proj-1"))
        deleted = await store.delete("proj-1")
        assert deleted is True
        assert await store.get("proj-1") is None

    async def test_delete_missing_returns_false(self, store: InMemoryStateStore) -> None:
        deleted = await store.delete("nonexistent")
        assert deleted is False
