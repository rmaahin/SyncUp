"""Async Trello REST API client for SyncUp.

Provides typed methods for board/list/card/label/checklist operations.
Auth credentials come from env vars or constructor parameters.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TrelloBoard(BaseModel):
    """Trello board representation."""

    id: str
    name: str
    url: str = ""
    desc: str = ""
    closed: bool = False


class TrelloList(BaseModel):
    """Trello list (column) representation."""

    id: str
    name: str
    id_board: str = Field(alias="idBoard", default="")
    closed: bool = False
    pos: float = 0.0

    model_config = {"populate_by_name": True}


class TrelloCard(BaseModel):
    """Trello card representation."""

    id: str
    name: str
    desc: str = ""
    id_list: str = Field(alias="idList", default="")
    id_board: str = Field(alias="idBoard", default="")
    due: Optional[str] = None
    id_members: list[str] = Field(alias="idMembers", default_factory=list)
    id_labels: list[str] = Field(alias="idLabels", default_factory=list)
    url: str = ""
    closed: bool = False

    model_config = {"populate_by_name": True}


class TrelloLabel(BaseModel):
    """Trello label representation."""

    id: str
    name: str
    color: str = ""
    id_board: str = Field(alias="idBoard", default="")

    model_config = {"populate_by_name": True}


class TrelloMember(BaseModel):
    """Trello member representation."""

    id: str
    username: str
    full_name: str = Field(alias="fullName", default="")
    avatar_url: Optional[str] = Field(alias="avatarUrl", default=None)

    model_config = {"populate_by_name": True}


class TrelloChecklist(BaseModel):
    """Trello checklist representation."""

    id: str
    name: str
    id_card: str = Field(alias="idCard", default="")
    check_items: list[dict[str, Any]] = Field(alias="checkItems", default_factory=list)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TrelloAPIError(Exception):
    """Raised when the Trello API returns a non-2xx response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"Trello API error {status_code}: {message}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.trello.com/1"


class TrelloClient:
    """Async Trello REST API client.

    Use as an async context manager::

        async with TrelloClient() as client:
            board = await client.create_board("My Board")
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("TRELLO_API_KEY", "")
        self._api_token = api_token or os.environ.get("TRELLO_API_TOKEN", "")
        if not self._api_key:
            raise ValueError("Trello API key is required (pass api_key or set TRELLO_API_KEY)")
        if not self._api_token:
            raise ValueError("Trello API token is required (pass api_token or set TRELLO_API_TOKEN)")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TrelloClient:
        self._client = httpx.AsyncClient(base_url=_BASE_URL)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- Internal helpers ---------------------------------------------------

    @property
    def _auth_params(self) -> dict[str, str]:
        return {"key": self._api_key, "token": self._api_token}

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Send an HTTP request to the Trello API.

        Merges auth params into query string. Raises ``TrelloAPIError`` on
        non-2xx responses.
        """
        if self._client is None:
            raise RuntimeError("TrelloClient must be used as an async context manager")

        merged_params = {**self._auth_params, **(params or {})}
        response = await self._client.request(
            method,
            path,
            params=merged_params,
            json=json_body,
        )
        if not response.is_success:
            raise TrelloAPIError(
                status_code=response.status_code,
                message=response.reason_phrase or "Unknown error",
                response_body=response.text,
            )
        return response.json()

    # -- Board operations ---------------------------------------------------

    async def create_board(self, name: str, desc: str = "") -> TrelloBoard:
        """Create a new Trello board."""
        data = await self._request("POST", "/boards", params={"name": name, "desc": desc})
        return TrelloBoard.model_validate(data)

    # -- List operations ----------------------------------------------------

    async def create_list(
        self, board_id: str, name: str, pos: str = "bottom"
    ) -> TrelloList:
        """Create a new list on a board."""
        data = await self._request(
            "POST",
            "/lists",
            params={"name": name, "idBoard": board_id, "pos": pos},
        )
        return TrelloList.model_validate(data)

    # -- Card operations ----------------------------------------------------

    async def create_card(
        self,
        list_id: str,
        name: str,
        desc: str = "",
        due: datetime | None = None,
        member_id: str | None = None,
        labels: list[str] | None = None,
    ) -> TrelloCard:
        """Create a card in a list."""
        params: dict[str, Any] = {
            "idList": list_id,
            "name": name,
            "desc": desc,
        }
        if due is not None:
            params["due"] = due.isoformat()
        if member_id is not None:
            params["idMembers"] = member_id
        if labels:
            params["idLabels"] = ",".join(labels)
        data = await self._request("POST", "/cards", params=params)
        return TrelloCard.model_validate(data)

    async def move_card(self, card_id: str, list_id: str) -> TrelloCard:
        """Move a card to a different list."""
        data = await self._request(
            "PUT", f"/cards/{card_id}", params={"idList": list_id}
        )
        return TrelloCard.model_validate(data)

    # -- Checklist operations -----------------------------------------------

    async def add_checklist(
        self, card_id: str, name: str, items: list[str]
    ) -> TrelloChecklist:
        """Add a checklist with items to a card."""
        data = await self._request(
            "POST", "/checklists", params={"idCard": card_id, "name": name}
        )
        checklist_id = data["id"]
        for item in items:
            await self._request(
                "POST",
                f"/checklists/{checklist_id}/checkItems",
                params={"name": item},
            )
        # Re-fetch to get the full checklist with items
        data = await self._request("GET", f"/checklists/{checklist_id}")
        return TrelloChecklist.model_validate(data)

    # -- Label operations ---------------------------------------------------

    async def add_label(
        self, board_id: str, name: str, color: str
    ) -> TrelloLabel:
        """Create a label on a board."""
        data = await self._request(
            "POST",
            f"/boards/{board_id}/labels",
            params={"name": name, "color": color},
        )
        return TrelloLabel.model_validate(data)

    # -- Query operations ---------------------------------------------------

    async def get_board_cards(self, board_id: str) -> list[TrelloCard]:
        """Get all cards on a board."""
        data = await self._request("GET", f"/boards/{board_id}/cards")
        return [TrelloCard.model_validate(card) for card in data]

    async def get_member_by_username(self, username: str) -> TrelloMember:
        """Look up a Trello member by username."""
        data = await self._request("GET", f"/members/{username}")
        return TrelloMember.model_validate(data)
