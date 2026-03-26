"""In-memory state store for SyncUp.

Stores ``SyncUpState`` objects keyed by ``project_id``.  Uses an
``asyncio.Lock`` to guard concurrent writes from overlapping webhook
handlers.

.. note::
    This is a **temporary** implementation for development.  Phase 10
    will replace it with PostgreSQL-backed persistence (LangGraph
    ``PostgresSaver`` checkpointer).
"""

from __future__ import annotations

import asyncio
import logging

from state.schema import SyncUpState

logger = logging.getLogger(__name__)


class InMemoryStateStore:
    """Async-safe in-memory state store keyed by project_id.

    Thread-safety is provided by an ``asyncio.Lock`` — safe for
    concurrent FastAPI request handlers within the same event loop.
    """

    def __init__(self) -> None:
        self._data: dict[str, SyncUpState] = {}
        self._lock = asyncio.Lock()

    async def get(self, project_id: str) -> SyncUpState | None:
        """Retrieve the state for a project.

        Args:
            project_id: The project identifier.

        Returns:
            The ``SyncUpState`` if found, ``None`` otherwise.
        """
        async with self._lock:
            return self._data.get(project_id)

    async def save(self, project_id: str, state: SyncUpState) -> None:
        """Save (or overwrite) the state for a project.

        Args:
            project_id: The project identifier.
            state: The full ``SyncUpState`` to store.
        """
        async with self._lock:
            self._data[project_id] = state
            logger.debug("State saved for project %s", project_id)

    async def list_projects(self) -> list[str]:
        """Return all project IDs currently stored.

        Returns:
            A list of project ID strings.
        """
        async with self._lock:
            return list(self._data.keys())

    async def delete(self, project_id: str) -> bool:
        """Delete the state for a project.

        Args:
            project_id: The project identifier.

        Returns:
            ``True`` if the project existed and was deleted, ``False`` otherwise.
        """
        async with self._lock:
            if project_id in self._data:
                del self._data[project_id]
                logger.debug("State deleted for project %s", project_id)
                return True
            return False
