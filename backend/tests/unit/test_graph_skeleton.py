"""Tests for the LangGraph graph skeleton."""

from agents.delegation import delegation
from agents.publishing import publishing
from agents.task_decomposition import task_decomposition
from evaluators.equity_evaluator import equity_evaluator
from graph.main import (
    build_graph,
    conflict_resolution,
    graph,
    human_review,
    meeting_coordinator,
    progress_tracking,
    report_generator,
    supervisor,
    tone_evaluator,
)
from graph.routing import (
    after_availability_check,
    after_equity_eval,
    after_progress_check,
    after_tone_eval,
    supervisor_router,
)
from state.schema import EquityResult, SyncUpState


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
        # Real agent returns empty task_array for empty brief (not a placeholder)
        result = task_decomposition(SyncUpState())
        assert result == {"task_array": [], "dependency_graph": {}}

    def test_delegation(self) -> None:
        # Real agent returns empty dict when no tasks exist.
        assert delegation(SyncUpState()) == {}

    def test_progress_tracking(self) -> None:
        assert progress_tracking(SyncUpState()) == {}

    def test_conflict_resolution(self) -> None:
        assert conflict_resolution(SyncUpState()) == {}

    def test_meeting_coordinator(self) -> None:
        assert meeting_coordinator(SyncUpState()) == {}

    async def test_publishing(self) -> None:
        # Real agent returns early with failed status when delegation_matrix is empty.
        result = await publishing(SyncUpState())
        assert result["publishing_status"].trello == "failed"

    def test_equity_evaluator(self) -> None:
        # Real evaluator returns balanced=True when no tasks exist.
        result = equity_evaluator(SyncUpState())
        assert result["equity_result"].balanced is True
        assert result["equity_retries"] == 1

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

    def test_after_equity_eval_balanced(self) -> None:
        state = SyncUpState(
            equity_result=EquityResult(balanced=True, reasoning="ok"),
            equity_retries=1,
        )
        assert after_equity_eval(state) == "human_review"

    def test_after_equity_eval_max_retries(self) -> None:
        state = SyncUpState(
            equity_result=EquityResult(balanced=False, reasoning="bad"),
            equity_retries=3,
        )
        assert after_equity_eval(state) == "human_review"

    def test_after_equity_eval_retry(self) -> None:
        state = SyncUpState(
            equity_result=EquityResult(balanced=False, reasoning="bad"),
            equity_retries=1,
        )
        assert after_equity_eval(state) == "delegation"

    def test_after_equity_eval_no_result(self) -> None:
        # No equity_result yet — should route to delegation.
        state = SyncUpState(equity_retries=0)
        assert after_equity_eval(state) == "delegation"

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
