"""Tests for the WebSocket layer (ConnectionManager + endpoint)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.websockets import ConnectionManager

client = TestClient(app, raise_server_exceptions=False)


class TestConnectionManager:
    def test_connect_and_disconnect_track_state(self) -> None:
        mgr = ConnectionManager()
        ws = AsyncMock()
        # Bypass accept (mocked); simulate registration directly
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws))
        assert ws in mgr._connections["p1"]
        mgr.disconnect("p1", ws)
        assert ws not in mgr._connections["p1"]

    def test_broadcast_sends_to_all_on_project(self) -> None:
        mgr = ConnectionManager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws1))
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws2))
        asyncio.get_event_loop().run_until_complete(
            mgr.broadcast("p1", {"event": "x", "data": {"k": 1}})
        )
        assert ws1.send_json.await_count == 1
        assert ws2.send_json.await_count == 1
        # Validate message format
        msg = ws1.send_json.await_args.args[0]
        assert msg["event"] == "x"
        assert msg["data"] == {"k": 1}
        assert "timestamp" in msg

    def test_broadcast_does_not_cross_projects(self) -> None:
        mgr = ConnectionManager()
        ws_a, ws_b = AsyncMock(), AsyncMock()
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws_a))
        asyncio.get_event_loop().run_until_complete(mgr.connect("p2", ws_b))
        asyncio.get_event_loop().run_until_complete(
            mgr.broadcast("p1", {"event": "x", "data": {}})
        )
        assert ws_a.send_json.await_count == 1
        assert ws_b.send_json.await_count == 0

    def test_broadcast_prunes_dead_connections(self) -> None:
        mgr = ConnectionManager()
        ws_dead = AsyncMock()
        ws_dead.send_json.side_effect = Exception("connection closed")
        ws_live = AsyncMock()
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws_dead))
        asyncio.get_event_loop().run_until_complete(mgr.connect("p1", ws_live))
        asyncio.get_event_loop().run_until_complete(
            mgr.broadcast("p1", {"event": "x", "data": {}})
        )
        assert ws_dead not in mgr._connections["p1"]
        assert ws_live in mgr._connections["p1"]


class TestWebSocketEndpoint:
    def test_websocket_connect_disconnect(self) -> None:
        # TestClient's websocket_connect performs the WS handshake
        with client.websocket_connect("/api/ws/proj-w") as ws:
            # Connection accepted; close cleanly
            ws.close()
