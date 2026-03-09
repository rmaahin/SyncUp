"""Tests for the LangGraph graph skeleton."""

from graph.main import (
    build_graph,
    conflict_resolution,
    delegation,
    equity_evaluator,
    graph,
    human_review,
    meeting_coordinator,
    progress_tracking,
    publishing,
    report_generator,
    supervisor,
    task_decomposition,
    tone_evaluator,
)
from graph.routing import (
    after_availability_check,
    after_equity_eval,
    after_progress_check,
    after_tone_eval,
    supervisor_router,
)
from state.schema import SyncUpState


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Verify the graph compiles and the module-level object exists."""

    def test_build_graph_returns_compiled(self) -> None:
        compiled = build_graph()
        assert compiled is not None

    def test_module_level_graph_exists(self) -> None:
        assert graph is not None


# ---------------------------------------------------------------------------
# Placeholder nodes
# ---------------------------------------------------------------------------


class TestPlaceholderNodes:
    """Each placeholder node should accept state and return an empty dict."""

    def test_supervisor(self) -> None:
        assert supervisor(SyncUpState()) == {}

    def test_task_decomposition(self) -> None:
        assert task_decomposition(SyncUpState()) == {}

    def test_delegation(self) -> None:
        assert delegation(SyncUpState()) == {}

    def test_progress_tracking(self) -> None:
        assert progress_tracking(SyncUpState()) == {}

    def test_conflict_resolution(self) -> None:
        assert conflict_resolution(SyncUpState()) == {}

    def test_meeting_coordinator(self) -> None:
        assert meeting_coordinator(SyncUpState()) == {}

    def test_publishing(self) -> None:
        assert publishing(SyncUpState()) == {}

    def test_equity_evaluator(self) -> None:
        assert equity_evaluator(SyncUpState()) == {}

    def test_tone_evaluator(self) -> None:
        assert tone_evaluator(SyncUpState()) == {}

    def test_human_review(self) -> None:
        assert human_review(SyncUpState()) == {}

    def test_report_generator(self) -> None:
        assert report_generator(SyncUpState()) == {}


# ---------------------------------------------------------------------------
# Routing stubs
# ---------------------------------------------------------------------------


class TestRoutingStubs:
    """Routing functions return valid string targets."""

    def test_supervisor_router(self) -> None:
        result = supervisor_router(SyncUpState())
        assert result == "__end__"

    def test_after_equity_eval(self) -> None:
        result = after_equity_eval(SyncUpState())
        assert result == "human_review"

    def test_after_tone_eval(self) -> None:
        result = after_tone_eval(SyncUpState())
        assert result == "__end__"

    def test_after_progress_check(self) -> None:
        result = after_progress_check(SyncUpState())
        assert result == "__end__"

    def test_after_availability_check(self) -> None:
        result = after_availability_check(SyncUpState())
        assert result == "__end__"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestGraphInvoke:
    """Smoke test: invoke the graph with default state."""

    def test_invoke_default_state(self) -> None:
        result = graph.invoke(SyncUpState())
        assert result is not None
