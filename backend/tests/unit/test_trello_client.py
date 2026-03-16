"""Tests for the Trello REST API client."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from integrations.trello import (
    TrelloAPIError,
    TrelloBoard,
    TrelloCard,
    TrelloChecklist,
    TrelloClient,
    TrelloLabel,
    TrelloList,
    TrelloMember,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "test_key_123"
FAKE_TOKEN = "test_token_456"


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    reason_phrase: str = "OK",
) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.reason_phrase = reason_phrase
    resp.text = str(json_data)
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestTrelloClientInit:
    """Constructor validation."""

    def test_init_from_params(self) -> None:
        client = TrelloClient(api_key=FAKE_KEY, api_token=FAKE_TOKEN)
        assert client._api_key == FAKE_KEY
        assert client._api_token == FAKE_TOKEN

    def test_init_from_env(self) -> None:
        with patch.dict("os.environ", {"TRELLO_API_KEY": "ek", "TRELLO_API_TOKEN": "et"}):
            client = TrelloClient()
            assert client._api_key == "ek"
            assert client._api_token == "et"

    def test_init_missing_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                TrelloClient(api_token=FAKE_TOKEN)

    def test_init_missing_token_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API token"):
                TrelloClient(api_key=FAKE_KEY)


# ---------------------------------------------------------------------------
# Request tests
# ---------------------------------------------------------------------------


class TestTrelloClientRequests:
    """Test each client method with mocked HTTP responses."""

    async def _make_client(self) -> TrelloClient:
        client = TrelloClient(api_key=FAKE_KEY, api_token=FAKE_TOKEN)
        client._client = AsyncMock(spec=httpx.AsyncClient)
        return client

    async def test_create_board(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "board1", "name": "Test Board", "url": "https://trello.com/b/board1",
                "desc": "A test board", "closed": False,
            })
        )
        result = await client.create_board("Test Board", desc="A test board")
        assert isinstance(result, TrelloBoard)
        assert result.id == "board1"
        assert result.name == "Test Board"

    async def test_create_list(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "list1", "name": "To Do", "idBoard": "board1", "closed": False, "pos": 1.0,
            })
        )
        result = await client.create_list("board1", "To Do")
        assert isinstance(result, TrelloList)
        assert result.id == "list1"
        assert result.id_board == "board1"

    async def test_create_card(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "card1", "name": "Task 1", "desc": "Do something",
                "idList": "list1", "idBoard": "board1",
                "due": "2026-04-01T00:00:00Z", "idMembers": ["m1"],
                "idLabels": ["l1"], "url": "https://trello.com/c/card1", "closed": False,
            })
        )
        result = await client.create_card(
            list_id="list1", name="Task 1", desc="Do something",
            due=datetime(2026, 4, 1), member_id="m1", labels=["l1"],
        )
        assert isinstance(result, TrelloCard)
        assert result.id == "card1"
        assert result.id_list == "list1"

    async def test_move_card(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "card1", "name": "Task 1", "idList": "list2",
                "idBoard": "board1", "idMembers": [], "idLabels": [],
                "url": "", "closed": False,
            })
        )
        result = await client.move_card("card1", "list2")
        assert isinstance(result, TrelloCard)
        assert result.id_list == "list2"

    async def test_add_checklist(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        # Mock 3 calls: create checklist, add item, re-fetch
        client._client.request = AsyncMock(
            side_effect=[
                _mock_response(json_data={"id": "cl1", "name": "Deps", "idCard": "card1", "checkItems": []}),
                _mock_response(json_data={"id": "ci1", "name": "Dep A"}),
                _mock_response(json_data={
                    "id": "cl1", "name": "Deps", "idCard": "card1",
                    "checkItems": [{"id": "ci1", "name": "Dep A", "state": "incomplete"}],
                }),
            ]
        )
        result = await client.add_checklist("card1", "Deps", ["Dep A"])
        assert isinstance(result, TrelloChecklist)
        assert result.id == "cl1"
        assert len(result.check_items) == 1

    async def test_add_label(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "l1", "name": "Critical", "color": "red", "idBoard": "board1",
            })
        )
        result = await client.add_label("board1", "Critical", "red")
        assert isinstance(result, TrelloLabel)
        assert result.color == "red"

    async def test_get_board_cards(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data=[
                {"id": "c1", "name": "Card 1", "idList": "l1", "idBoard": "b1",
                 "idMembers": [], "idLabels": [], "url": "", "closed": False},
                {"id": "c2", "name": "Card 2", "idList": "l1", "idBoard": "b1",
                 "idMembers": [], "idLabels": [], "url": "", "closed": False},
            ])
        )
        result = await client.get_board_cards("b1")
        assert len(result) == 2
        assert all(isinstance(c, TrelloCard) for c in result)

    async def test_get_member_by_username(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "m1", "username": "alice", "fullName": "Alice Smith",
                "avatarUrl": "https://example.com/avatar.png",
            })
        )
        result = await client.get_member_by_username("alice")
        assert isinstance(result, TrelloMember)
        assert result.username == "alice"
        assert result.full_name == "Alice Smith"


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class TestTrelloClientErrors:
    """Verify error handling."""

    async def _make_client(self) -> TrelloClient:
        client = TrelloClient(api_key=FAKE_KEY, api_token=FAKE_TOKEN)
        client._client = AsyncMock(spec=httpx.AsyncClient)
        return client

    async def test_404_raises_trello_api_error(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(404, json_data=None, reason_phrase="Not Found")
        )
        with pytest.raises(TrelloAPIError) as exc_info:
            await client.create_board("x")
        assert exc_info.value.status_code == 404

    async def test_401_raises_trello_api_error(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(401, json_data=None, reason_phrase="Unauthorized")
        )
        with pytest.raises(TrelloAPIError) as exc_info:
            await client.create_board("x")
        assert exc_info.value.status_code == 401

    async def test_500_raises_trello_api_error(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(500, json_data=None, reason_phrase="Internal Server Error")
        )
        with pytest.raises(TrelloAPIError):
            await client.get_board_cards("b1")

    async def test_error_includes_status_code(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(429, json_data=None, reason_phrase="Rate Limited")
        )
        with pytest.raises(TrelloAPIError) as exc_info:
            await client.create_card("l1", "x")
        assert exc_info.value.status_code == 429
        assert "429" in str(exc_info.value)

    async def test_auth_params_always_included(self) -> None:
        client = await self._make_client()
        assert client._client is not None
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={
                "id": "b1", "name": "B", "url": "", "desc": "", "closed": False,
            })
        )
        await client.create_board("B")
        call_kwargs = client._client.request.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert params["key"] == FAKE_KEY
        assert params["token"] == FAKE_TOKEN
