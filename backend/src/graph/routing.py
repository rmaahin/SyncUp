"""Conditional edge functions for the SyncUp LangGraph graph.

All functions are stubs that return a default route.  Real routing logic
will be implemented when the corresponding agents are built.
"""

from __future__ import annotations

from state.schema import SyncUpState


def supervisor_router(state: "SyncUpState") -> str:
    """Route from the supervisor to the appropriate worker node."""
    return "__end__"


def after_equity_eval(state: "SyncUpState") -> str:
    """Route after the equity evaluator — human review or re-delegation.

    Routes to ``human_review`` if the evaluation passed or the maximum
    number of re-delegation attempts (3) has been reached.  Otherwise
    routes back to ``delegation`` for re-optimization.
    """
    if state.equity_result is not None and state.equity_result.balanced:
        return "human_review"
    if state.equity_retries >= 3:
        return "human_review"
    return "delegation"


def after_tone_eval(state: "SyncUpState") -> str:
    """Route after the tone evaluator — end or escalate."""
    return "__end__"


def after_progress_check(state: "SyncUpState") -> str:
    """Route after progress tracking — conflict resolution or end."""
    return "__end__"


def after_availability_check(state: "SyncUpState") -> str:
    """Route after availability check — re-delegation or end."""
    return "__end__"
