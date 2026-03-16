"""Tests for webhook payload parsers (GitHub and Trello)."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from integrations.webhooks import (
    GitHubPREvent,
    GitHubPushEvent,
    TrelloCardUpdateEvent,
    WebhookParseError,
    parse_github_pr,
    parse_github_push,
    parse_trello_card_update,
    validate_github_signature,
)


# ---------------------------------------------------------------------------
# GitHub signature validation
# ---------------------------------------------------------------------------


class TestGitHubSignatureValidation:
    """Test HMAC-SHA256 signature verification."""

    SECRET = "my_webhook_secret"

    def _compute_signature(self, payload: bytes) -> str:
        digest = hmac.new(
            self.SECRET.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        return f"sha256={digest}"

    def test_valid_signature(self) -> None:
        payload = b'{"ref": "refs/heads/main"}'
        sig = self._compute_signature(payload)
        assert validate_github_signature(payload, sig, self.SECRET) is True

    def test_invalid_signature(self) -> None:
        payload = b'{"ref": "refs/heads/main"}'
        assert validate_github_signature(payload, "sha256=badhex", self.SECRET) is False

    def test_empty_payload(self) -> None:
        payload = b""
        sig = self._compute_signature(payload)
        assert validate_github_signature(payload, sig, self.SECRET) is True

    def test_malformed_signature_format(self) -> None:
        payload = b'{"data": true}'
        # Missing "sha256=" prefix
        assert validate_github_signature(payload, "md5=abcdef", self.SECRET) is False

    def test_empty_signature(self) -> None:
        payload = b'{"data": true}'
        assert validate_github_signature(payload, "", self.SECRET) is False


# ---------------------------------------------------------------------------
# GitHub push event parsing
# ---------------------------------------------------------------------------

_VALID_PUSH_PAYLOAD: dict[str, Any] = {
    "ref": "refs/heads/main",
    "before": "aaa111",
    "after": "bbb222",
    "commits": [
        {
            "id": "bbb222",
            "message": "feat: add feature",
            "timestamp": "2026-03-08T10:00:00Z",
            "author": {"name": "Alice", "email": "alice@example.com"},
            "added": ["src/new.py"],
            "removed": [],
            "modified": ["src/existing.py"],
        }
    ],
    "repository": {
        "id": 12345,
        "name": "syncup",
        "full_name": "team/syncup",
        "html_url": "https://github.com/team/syncup",
    },
    "pusher": {"name": "Alice", "email": "alice@example.com"},
}


class TestParseGitHubPush:
    """Test GitHub push event parsing."""

    def test_valid_push_event(self) -> None:
        event = parse_github_push(_VALID_PUSH_PAYLOAD)
        assert isinstance(event, GitHubPushEvent)
        assert event.ref == "refs/heads/main"
        assert len(event.commits) == 1
        assert event.commits[0].id == "bbb222"
        assert event.repository.name == "syncup"
        assert event.pusher.name == "Alice"

    def test_push_with_multiple_commits(self) -> None:
        payload = {
            **_VALID_PUSH_PAYLOAD,
            "commits": [
                {"id": "c1", "message": "commit 1"},
                {"id": "c2", "message": "commit 2"},
                {"id": "c3", "message": "commit 3"},
            ],
        }
        event = parse_github_push(payload)
        assert len(event.commits) == 3

    def test_malformed_push_raises(self) -> None:
        # Missing required 'repository' field
        with pytest.raises(WebhookParseError):
            parse_github_push({"ref": "refs/heads/main"})

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(WebhookParseError):
            parse_github_push({})


# ---------------------------------------------------------------------------
# GitHub PR event parsing
# ---------------------------------------------------------------------------

_VALID_PR_PAYLOAD: dict[str, Any] = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "id": 99999,
        "number": 42,
        "title": "Add task decomposition",
        "state": "open",
        "user": {"login": "alice"},
        "head": {"ref": "feature/task-decomp"},
        "base": {"ref": "main"},
        "merged": False,
        "merged_at": None,
    },
}


class TestParseGitHubPR:
    """Test GitHub PR event parsing."""

    def test_valid_pr_opened(self) -> None:
        event = parse_github_pr(_VALID_PR_PAYLOAD)
        assert isinstance(event, GitHubPREvent)
        assert event.action == "opened"
        assert event.number == 42
        assert event.pull_request.title == "Add task decomposition"
        assert event.pull_request.merged is False

    def test_valid_pr_merged(self) -> None:
        payload = {
            **_VALID_PR_PAYLOAD,
            "action": "closed",
            "pull_request": {
                **_VALID_PR_PAYLOAD["pull_request"],
                "state": "closed",
                "merged": True,
                "merged_at": "2026-03-09T15:00:00Z",
            },
        }
        event = parse_github_pr(payload)
        assert event.action == "closed"
        assert event.pull_request.merged is True
        assert event.pull_request.merged_at == "2026-03-09T15:00:00Z"

    def test_malformed_pr_raises(self) -> None:
        # Missing 'pull_request' key
        with pytest.raises(WebhookParseError):
            parse_github_pr({"action": "opened", "number": 1})

    def test_pr_missing_action_raises(self) -> None:
        with pytest.raises(WebhookParseError):
            parse_github_pr({"number": 1, "pull_request": {"id": 1, "number": 1}})


# ---------------------------------------------------------------------------
# Trello card update event parsing
# ---------------------------------------------------------------------------

_VALID_TRELLO_PAYLOAD: dict[str, Any] = {
    "action": {
        "type": "updateCard",
        "data": {
            "card": {"id": "card123", "name": "Setup CI"},
            "listBefore": {"id": "list_todo", "name": "To Do"},
            "listAfter": {"id": "list_doing", "name": "In Progress"},
        },
    },
    "model": {"id": "board1", "name": "Project Board"},
}


class TestParseTrelloCardUpdate:
    """Test Trello card update event parsing."""

    def test_valid_card_move(self) -> None:
        event = parse_trello_card_update(_VALID_TRELLO_PAYLOAD)
        assert isinstance(event, TrelloCardUpdateEvent)
        assert event.action_type == "updateCard"
        assert event.card_id == "card123"
        assert event.card_name == "Setup CI"
        assert event.list_before_id == "list_todo"
        assert event.list_after_id == "list_doing"

    def test_card_update_without_list_move(self) -> None:
        payload: dict[str, Any] = {
            "action": {
                "type": "updateCard",
                "data": {
                    "card": {"id": "card456", "name": "Write tests"},
                },
            },
        }
        event = parse_trello_card_update(payload)
        assert event.card_id == "card456"
        assert event.list_before_id == ""
        assert event.list_after_id == ""

    def test_malformed_event_raises(self) -> None:
        # Missing required 'action' field
        with pytest.raises(WebhookParseError):
            parse_trello_card_update({"model": {}})

    def test_preserves_raw_payload_on_error(self) -> None:
        bad_payload: dict[str, Any] = {"not_action": True}
        with pytest.raises(WebhookParseError) as exc_info:
            parse_trello_card_update(bad_payload)
        assert exc_info.value.raw_payload == bad_payload
