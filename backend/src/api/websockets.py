"""WebSocket layer for real-time dashboard notifications.

Provides a ``ConnectionManager`` singleton that tracks active WebSocket
connections per project and broadcasts state-change events to all
connected clients.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections grouped by project_id.

    Each connected dashboard client registers under a ``project_id``.
    ``broadcast`` sends a JSON message to every client watching that project.
    Dead connections are pruned automatically during broadcast.
    """

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, project_id: str, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection for a project."""
        await websocket.accept()
        self._connections.setdefault(project_id, []).append(websocket)
        logger.info("WebSocket connected for project %s", project_id)

    def disconnect(self, project_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection for a project."""
        conns = self._connections.get(project_id, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.info("WebSocket disconnected for project %s", project_id)

    async def broadcast(self, project_id: str, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients for a project.

        Dead connections are removed silently.

        Args:
            project_id: The project to broadcast to.
            message: Dict with ``event`` and ``data`` keys. A ``timestamp``
                field is added automatically.
        """
        payload: dict[str, Any] = {
            "event": message.get("event", "update"),
            "data": message.get("data", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        conns = self._connections.get(project_id, [])
        disconnected: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            conns.remove(ws)


manager = ConnectionManager()


@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str) -> None:
    """WebSocket endpoint for real-time project notifications.

    Clients connect per project. The server pushes state-change events
    (task updates, contributions, interventions, meetings, re-delegations).
    """
    await manager.connect(project_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(project_id, websocket)
