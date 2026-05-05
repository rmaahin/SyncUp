"""Conditional edge functions for the SyncUp LangGraph graph.

All functions are stubs that return a default route.  Real routing logic
will be implemented when the corresponding agents are built.
"""

from __future__ import annotations

from state.schema import SyncUpState


def supervisor_router(state: "SyncUpState") -> str:
    """Route from the supervisor to the appropriate worker node.

    Inspects state fields to determine the next phase of the pipeline.
    Returns ``"__end__"`` when no further work is needed.
    """
    if state.pending_event is not None:
        return "progress_tracking"
    if state.meeting_mode is not None:
        return "meeting_coordinator"
    if state.project_brief and not state.task_array:
        return "task_decomposition"
    if state.task_array and not state.delegation_matrix:
        return "delegation"
    if state.needs_redelegation:
        return "delegation"
    if state.draft_intervention is not None:
        return "conflict_resolution"
    if (
        state.delegation_matrix
        and state.equity_result is not None
        and state.equity_result.balanced
        and state.publishing_status is None
    ):
        return "publishing"
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
    """Route after the tone evaluator — deliver or rewrite.

    Routes to ``deliver`` if the draft is constructive or the rewrite
    limit (3) has been reached.  Routes back to ``conflict_resolution``
    if the tone was punitive and rewrites remain.
    """
    # No draft → deliver handles cleanup
    if state.draft_intervention is None:
        return "deliver"

    # Rewrite limit reached → force-deliver
    if state.tone_rewrite_count >= 3:
        return "deliver"

    # Punitive → loop back for rewrite
    if (
        state.tone_result is not None
        and state.tone_result.classification == "punitive"
    ):
        return "conflict_resolution"

    # Constructive → deliver
    return "deliver"


def after_progress_check(state: "SyncUpState") -> str:
    """Route after progress tracking — conflict resolution if any student is behind.

    Checks ``student_progress`` for any student with status ``"behind"``.
    If found, routes to ``conflict_resolution`` for intervention.
    Otherwise routes to ``__end__``.
    """
    if not state.student_progress:
        return "__end__"
    for status in state.student_progress.values():
        if status == "behind":
            return "conflict_resolution"
    return "__end__"


def after_availability_check(state: "SyncUpState") -> str:
    """Route after availability check — re-delegation or end."""
    return "__end__"
