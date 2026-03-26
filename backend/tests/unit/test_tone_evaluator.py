"""Unit tests for the Tone Evaluator."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from evaluators.tone_evaluator import tone_evaluator
from state.schema import (
    DraftIntervention,
    SyncUpState,
    ToneResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONSTRUCTIVE_RESPONSE = json.dumps(
    {
        "classification": "constructive",
        "reasoning": "The message is supportive and asks about roadblocks.",
        "flagged_phrases": [],
    }
)

_PUNITIVE_RESPONSE = json.dumps(
    {
        "classification": "punitive",
        "reasoning": "The message uses blame language.",
        "flagged_phrases": ["you failed to deliver", "unacceptable behavior"],
    }
)

_MALFORMED_RESPONSE = "This is not valid JSON"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_draft(
    message: str = "Hi Bob, we noticed the API task is overdue. "
    "Are you running into any blockers?",
) -> DraftIntervention:
    """Create a minimal DraftIntervention for testing."""
    return DraftIntervention(
        target_student_id="s-2",
        message=message,
        suggested_action="schedule_check_in",
        severity="medium",
        affected_teammates=["Alice"],
    )


def _make_state(
    draft: DraftIntervention | None = None,
    tone_rewrite_count: int = 0,
) -> SyncUpState:
    """Create a SyncUpState for testing."""
    return SyncUpState(
        project_id="test-project",
        draft_intervention=draft,
        tone_rewrite_count=tone_rewrite_count,
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns the given responses in order."""
    llm = MagicMock()
    side_effects: list[MagicMock] = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        side_effects.append(msg)
    llm.invoke.side_effect = side_effects
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToneEvaluatorNode:
    """Tests for the tone_evaluator node function."""

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_constructive_passes(self, mock_llm_factory: MagicMock) -> None:
        """Constructive message → classification='constructive'."""
        mock_llm_factory.return_value = _make_mock_llm([_CONSTRUCTIVE_RESPONSE])

        state = _make_state(draft=_make_draft())
        result = tone_evaluator(state)

        tone = result["tone_result"]
        assert isinstance(tone, ToneResult)
        assert tone.classification == "constructive"
        assert tone.flagged_phrases == []
        # Should NOT increment rewrite count
        assert "tone_rewrite_count" not in result

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_punitive_flagged(self, mock_llm_factory: MagicMock) -> None:
        """Punitive message → classification='punitive' with flagged phrases."""
        mock_llm_factory.return_value = _make_mock_llm([_PUNITIVE_RESPONSE])

        state = _make_state(draft=_make_draft())
        result = tone_evaluator(state)

        tone = result["tone_result"]
        assert isinstance(tone, ToneResult)
        assert tone.classification == "punitive"
        assert len(tone.flagged_phrases) == 2
        assert "you failed to deliver" in tone.flagged_phrases
        # Should increment rewrite count
        assert result["tone_rewrite_count"] == 1

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_temperature_zero(self, mock_llm_factory: MagicMock) -> None:
        """Verify get_high_tier_llm is called with temperature=0.0."""
        mock_llm_factory.return_value = _make_mock_llm([_CONSTRUCTIVE_RESPONSE])

        state = _make_state(draft=_make_draft())
        tone_evaluator(state)

        mock_llm_factory.assert_called_once_with(temperature=0.0)

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_no_draft_returns_none(self, mock_llm_factory: MagicMock) -> None:
        """No draft_intervention → returns tone_result=None, no LLM call."""
        state = _make_state(draft=None)
        result = tone_evaluator(state)

        assert result["tone_result"] is None
        mock_llm_factory.assert_not_called()

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_retry_on_malformed_json(self, mock_llm_factory: MagicMock) -> None:
        """First attempt fails, second succeeds."""
        mock_llm_factory.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _CONSTRUCTIVE_RESPONSE]
        )

        state = _make_state(draft=_make_draft())
        result = tone_evaluator(state)

        tone = result["tone_result"]
        assert tone.classification == "constructive"
        llm = mock_llm_factory.return_value
        assert llm.invoke.call_count == 2

    @patch("evaluators.tone_evaluator.get_high_tier_llm")
    def test_all_retries_exhausted_defaults_constructive(
        self, mock_llm_factory: MagicMock
    ) -> None:
        """All 3 attempts fail → defaults to constructive to avoid blocking delivery."""
        mock_llm_factory.return_value = _make_mock_llm(
            [_MALFORMED_RESPONSE, _MALFORMED_RESPONSE, _MALFORMED_RESPONSE]
        )

        state = _make_state(draft=_make_draft())
        result = tone_evaluator(state)

        tone = result["tone_result"]
        assert tone.classification == "constructive"
        assert "failed" in tone.reasoning.lower()
