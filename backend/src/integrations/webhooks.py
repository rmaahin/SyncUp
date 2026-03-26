"""Webhook payload parsers for GitHub and Trello.

These are pure parsing functions — the FastAPI routes that receive webhooks
are defined separately (Phase 10). All external data is treated as UNTRUSTED.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Optional

from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class WebhookParseError(Exception):
    """Raised when a webhook payload cannot be parsed into the expected model."""

    def __init__(self, message: str, raw_payload: Any = None) -> None:
        self.raw_payload = raw_payload
        super().__init__(message)


# ---------------------------------------------------------------------------
# GitHub webhook models
# ---------------------------------------------------------------------------


class GitHubUser(BaseModel):
    """GitHub user (author/pusher)."""

    name: str = ""
    email: str = ""


class GitHubRepository(BaseModel):
    """GitHub repository metadata."""

    id: int
    name: str
    full_name: str = ""
    html_url: str = ""


class GitHubCommit(BaseModel):
    """A single commit in a push event."""

    id: str
    message: str = ""
    timestamp: str = ""
    author: dict[str, Any] = {}
    added: list[str] = []
    removed: list[str] = []
    modified: list[str] = []


class GitHubPushEvent(BaseModel):
    """Parsed GitHub push webhook event."""

    ref: str
    before: str = ""
    after: str = ""
    commits: list[GitHubCommit] = []
    repository: GitHubRepository
    pusher: GitHubUser = GitHubUser()


class GitHubPullRequest(BaseModel):
    """Pull request data within a PR event."""

    id: int
    number: int
    title: str = ""
    state: str = ""
    user: dict[str, Any] = {}
    head: dict[str, Any] = {}
    base: dict[str, Any] = {}
    merged: bool = False
    merged_at: Optional[str] = None


class GitHubPREvent(BaseModel):
    """Parsed GitHub pull_request webhook event."""

    action: str
    number: int
    pull_request: GitHubPullRequest


# ---------------------------------------------------------------------------
# Trello webhook models
# ---------------------------------------------------------------------------


class TrelloCardUpdateEvent(BaseModel):
    """Parsed Trello card update webhook event.

    Trello webhooks have deeply nested payloads, so we keep this flexible
    and provide helper properties to extract common fields.
    """

    action: dict[str, Any]
    model: dict[str, Any] = {}

    @property
    def action_type(self) -> str:
        """The type of action (e.g. 'updateCard')."""
        return str(self.action.get("type", ""))

    @property
    def card_id(self) -> str:
        """The ID of the affected card."""
        card = self.action.get("data", {}).get("card", {})
        return str(card.get("id", ""))

    @property
    def card_name(self) -> str:
        """The name of the affected card."""
        card = self.action.get("data", {}).get("card", {})
        return str(card.get("name", ""))

    @property
    def list_before_id(self) -> str:
        """The list the card was moved from (empty if not a move)."""
        list_before = self.action.get("data", {}).get("listBefore", {})
        return str(list_before.get("id", ""))

    @property
    def list_after_id(self) -> str:
        """The list the card was moved to (empty if not a move)."""
        list_after = self.action.get("data", {}).get("listAfter", {})
        return str(list_after.get("id", ""))

    @property
    def member_creator_username(self) -> str:
        """The Trello username of the member who performed this action."""
        creator = self.action.get("memberCreator", {})
        return str(creator.get("username", ""))

    @property
    def member_creator_full_name(self) -> str:
        """The full name of the member who performed this action."""
        creator = self.action.get("memberCreator", {})
        return str(creator.get("fullName", ""))


# ---------------------------------------------------------------------------
# GitHub signature validation
# ---------------------------------------------------------------------------


def validate_github_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Validate a GitHub webhook HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes.
        signature: Value of the ``X-Hub-Signature-256`` header (``sha256=<hex>``).
        secret: The webhook secret configured in GitHub.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    received = signature[len("sha256="):]
    return hmac.compare_digest(expected, received)


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def parse_github_push(payload: dict[str, Any]) -> GitHubPushEvent:
    """Parse a GitHub push webhook payload into a typed model.

    Raises:
        WebhookParseError: If the payload is missing required fields.
    """
    try:
        return GitHubPushEvent.model_validate(payload)
    except ValidationError as exc:
        raise WebhookParseError(
            f"Invalid GitHub push event payload: {exc}",
            raw_payload=payload,
        ) from exc


def parse_github_pr(payload: dict[str, Any]) -> GitHubPREvent:
    """Parse a GitHub pull_request webhook payload into a typed model.

    Raises:
        WebhookParseError: If the payload is missing required fields.
    """
    try:
        return GitHubPREvent.model_validate(payload)
    except ValidationError as exc:
        raise WebhookParseError(
            f"Invalid GitHub PR event payload: {exc}",
            raw_payload=payload,
        ) from exc


def parse_trello_card_update(payload: dict[str, Any]) -> TrelloCardUpdateEvent:
    """Parse a Trello webhook payload for card update events.

    Raises:
        WebhookParseError: If the payload is missing required fields.
    """
    try:
        return TrelloCardUpdateEvent.model_validate(payload)
    except ValidationError as exc:
        raise WebhookParseError(
            f"Invalid Trello card update payload: {exc}",
            raw_payload=payload,
        ) from exc
