"""LangGraph StateGraph definition for SyncUp.

``build_graph()`` constructs and compiles the graph.  The module-level
``graph`` object is the compiled graph ready for ``.invoke()`` / ``.stream()``.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.delegation import delegation as _delegation
from agents.publishing import publishing as _publishing
from agents.task_decomposition import task_decomposition as _task_decomposition
from evaluators.equity_evaluator import equity_evaluator as _equity_evaluator
from graph.routing import (
    after_equity_eval,
    after_progress_check,
    after_tone_eval,
    supervisor_router,
)
from state.schema import SyncUpState


# ---------------------------------------------------------------------------
# Placeholder node functions
# ---------------------------------------------------------------------------


def supervisor(state: SyncUpState) -> dict[str, Any]:
    """Supervisor node — routes to the appropriate worker."""
    return {}


def progress_tracking(state: SyncUpState) -> dict[str, Any]:
    """Progress Tracking agent node."""
    return {}


def conflict_resolution(state: SyncUpState) -> dict[str, Any]:
    """Conflict Resolution agent node."""
    return {}


def meeting_coordinator(state: SyncUpState) -> dict[str, Any]:
    """Meeting Coordinator agent node."""
    return {}


def tone_evaluator(state: SyncUpState) -> dict[str, Any]:
    """Tone / Constructiveness Evaluator (LLM-as-a-Judge)."""
    return {}


def human_review(state: SyncUpState) -> dict[str, Any]:
    """Human-in-the-loop checkpoint — professor approves delegation matrix."""
    return {}


def report_generator(state: SyncUpState) -> dict[str, Any]:
    """Report Generator + Peer Review Generator node."""
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_NODE_FUNCTIONS = {
    "supervisor": supervisor,
    "task_decomposition": _task_decomposition,
    "delegation": _delegation,
    "progress_tracking": progress_tracking,
    "conflict_resolution": conflict_resolution,
    "meeting_coordinator": meeting_coordinator,
    "publishing": _publishing,
    "equity_evaluator": _equity_evaluator,
    "tone_evaluator": tone_evaluator,
    "human_review": human_review,
    "report_generator": report_generator,
}


def build_graph() -> Any:
    """Build and compile the SyncUp StateGraph.

    Returns the compiled graph object.
    """
    builder = StateGraph(SyncUpState)

    # -- Add all nodes --
    for name, fn in _NODE_FUNCTIONS.items():
        builder.add_node(name, fn)

    # -- Entry point --
    builder.set_entry_point("supervisor")

    # -- Edges from supervisor (conditional) --
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "task_decomposition": "task_decomposition",
            "delegation": "delegation",
            "progress_tracking": "progress_tracking",
            "conflict_resolution": "conflict_resolution",
            "meeting_coordinator": "meeting_coordinator",
            "publishing": "publishing",
            "report_generator": "report_generator",
            "__end__": END,
        },
    )

    # -- Task decomposition -> delegation --
    builder.add_edge("task_decomposition", "delegation")

    # -- Equity evaluator (conditional) -> delegation | human_review --
    builder.add_conditional_edges(
        "equity_evaluator",
        after_equity_eval,
        {
            "delegation": "delegation",
            "human_review": "human_review",
        },
    )

    # -- Human review -> supervisor --
    builder.add_edge("human_review", "supervisor")

    # -- Delegation -> equity evaluator --
    builder.add_edge("delegation", "equity_evaluator")

    # -- Progress tracking (conditional) -> conflict_resolution | end --
    builder.add_conditional_edges(
        "progress_tracking",
        after_progress_check,
        {
            "conflict_resolution": "conflict_resolution",
            "__end__": END,
        },
    )

    # -- Conflict resolution -> tone evaluator --
    builder.add_edge("conflict_resolution", "tone_evaluator")

    # -- Tone evaluator (conditional) -> end --
    builder.add_conditional_edges(
        "tone_evaluator",
        after_tone_eval,
        {
            "__end__": END,
        },
    )

    # -- Meeting coordinator -> supervisor --
    builder.add_edge("meeting_coordinator", "supervisor")

    # -- Publishing -> supervisor --
    builder.add_edge("publishing", "supervisor")

    # -- Report generator -> end --
    builder.add_edge("report_generator", END)

    return builder.compile()


# Module-level compiled graph
graph = build_graph()
