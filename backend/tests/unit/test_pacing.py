"""Tests for the pacing service — pure logic, no mocks needed."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from services.pacing import (
    _buffer_days,
    _topological_sort,
    calculate_burn_down_curve,
    distribute_deadlines_for_student,
    validate_pacing,
)
from state.schema import DateRange, Task, UrgencyLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = datetime(2026, 1, 1)
_END = datetime(2026, 4, 1)  # 90 days


def _task(
    tid: str,
    effort: float = 5.0,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    deps: list[str] | None = None,
    assigned_to: str | None = None,
) -> Task:
    return Task(
        id=tid,
        title=tid,
        effort_hours=effort,
        urgency=urgency,
        dependencies=deps or [],
        assigned_to=assigned_to,
    )


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Tests for Kahn's algorithm topological sort."""

    def test_linear_chain(self) -> None:
        tasks = [_task("a"), _task("b", deps=["a"]), _task("c", deps=["b"])]
        dep_graph = {"a": [], "b": ["a"], "c": ["b"]}
        result = _topological_sort(tasks, dep_graph)
        assert result.index("a") < result.index("b") < result.index("c")

    def test_parallel_tasks_no_deps(self) -> None:
        tasks = [_task("x"), _task("y"), _task("z")]
        dep_graph = {"x": [], "y": [], "z": []}
        result = _topological_sort(tasks, dep_graph)
        assert set(result) == {"x", "y", "z"}

    def test_parallel_tasks_critical_first(self) -> None:
        tasks = [
            _task("low", urgency=UrgencyLevel.LOW),
            _task("crit", urgency=UrgencyLevel.CRITICAL),
            _task("med", urgency=UrgencyLevel.MEDIUM),
        ]
        dep_graph = {"low": [], "crit": [], "med": []}
        result = _topological_sort(tasks, dep_graph)
        assert result.index("crit") < result.index("med")
        assert result.index("med") < result.index("low")

    def test_diamond_dependency(self) -> None:
        tasks = [
            _task("a"),
            _task("b", deps=["a"]),
            _task("c", deps=["a"]),
            _task("d", deps=["b", "c"]),
        ]
        dep_graph = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
        result = _topological_sort(tasks, dep_graph)
        assert result.index("a") < result.index("b")
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")

    def test_circular_dependency_raises(self) -> None:
        tasks = [_task("a", deps=["b"]), _task("b", deps=["a"])]
        dep_graph = {"a": ["b"], "b": ["a"]}
        with pytest.raises(ValueError, match="Circular dependency"):
            _topological_sort(tasks, dep_graph)

    def test_empty_list(self) -> None:
        result = _topological_sort([], {})
        assert result == []

    def test_single_task_no_deps(self) -> None:
        tasks = [_task("solo")]
        dep_graph = {"solo": []}
        result = _topological_sort(tasks, dep_graph)
        assert result == ["solo"]


# ---------------------------------------------------------------------------
# Burn-down curve
# ---------------------------------------------------------------------------


class TestCalculateBurnDownCurve:
    """Tests for calculate_burn_down_curve."""

    def test_chain_deadlines_monotonically_increasing(self) -> None:
        tasks = [_task("a", 5), _task("b", 5, deps=["a"]), _task("c", 5, deps=["b"])]
        dep_graph = {"a": [], "b": ["a"], "c": ["b"]}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        assert result["a"] < result["b"] < result["c"]

    def test_critical_earlier_than_low_same_level(self) -> None:
        tasks = [
            _task("crit", 5, UrgencyLevel.CRITICAL),
            _task("low", 5, UrgencyLevel.LOW),
        ]
        dep_graph = {"crit": [], "low": []}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        assert result["crit"] < result["low"]

    def test_buffer_between_dependent_tasks(self) -> None:
        tasks = [_task("a", 10), _task("b", 10, deps=["a"])]
        dep_graph = {"a": [], "b": ["a"]}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        gap = result["b"] - result["a"]
        min_buffer = timedelta(days=_buffer_days(tasks[1]))
        assert gap >= min_buffer

    def test_no_deadline_exceeds_final(self) -> None:
        tasks = [_task(f"t{i}", 10) for i in range(10)]
        dep_graph = {t.id: [] for t in tasks}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        for dl in result.values():
            assert dl <= _END

    def test_single_task(self) -> None:
        tasks = [_task("solo", 5)]
        dep_graph = {"solo": []}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        assert "solo" in result
        assert _START < result["solo"] <= _END

    def test_empty_tasks(self) -> None:
        result = calculate_burn_down_curve(_END, _START, [], {})
        assert result == {}

    def test_tasks_not_front_loaded(self) -> None:
        """No more than 40% of tasks should have deadlines in the first 20% of time."""
        tasks = [_task(f"t{i}", 5) for i in range(10)]
        dep_graph = {t.id: [] for t in tasks}
        result = calculate_burn_down_curve(_END, _START, tasks, dep_graph)
        duration = (_END - _START).total_seconds()
        first_20_cutoff = _START + timedelta(seconds=duration * 0.20)
        count_in_first_20 = sum(1 for dl in result.values() if dl <= first_20_cutoff)
        assert count_in_first_20 <= 4  # at most 40% of 10 tasks


# ---------------------------------------------------------------------------
# Per-student deadline adjustment
# ---------------------------------------------------------------------------


class TestDistributeDeadlinesForStudent:
    """Tests for distribute_deadlines_for_student."""

    def test_blackout_avoidance(self) -> None:
        base_deadline = datetime(2026, 2, 15)
        blackout = DateRange(
            start=datetime(2026, 2, 10),
            end=datetime(2026, 2, 20),
        )
        task = _task("t1", 5, assigned_to="s1")
        result = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=20.0,
            blackout_periods=[blackout],
            task_deadlines={"t1": base_deadline},
            final_deadline=_END,
        )
        assert result["t1"] > blackout.end

    def test_low_availability_extends_deadline(self) -> None:
        base_deadline = datetime(2026, 2, 1)
        task = _task("t1", 10, assigned_to="s1")

        high_avail = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=20.0,
            blackout_periods=[],
            task_deadlines={"t1": base_deadline},
            final_deadline=_END,
        )
        low_avail = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=5.0,
            blackout_periods=[],
            task_deadlines={"t1": base_deadline},
            final_deadline=_END,
        )
        # Low-availability student gets a later deadline.
        assert low_avail["t1"] > high_avail["t1"]

    def test_final_deadline_clamping(self) -> None:
        # Deadline very close to final with low availability — should clamp.
        task = _task("t1", 20, assigned_to="s1")
        result = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=2.0,
            blackout_periods=[],
            task_deadlines={"t1": datetime(2026, 3, 28)},
            final_deadline=_END,
        )
        assert result["t1"] <= _END

    def test_no_tasks_for_student(self) -> None:
        task = _task("t1", 5, assigned_to="s2")
        result = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=20.0,
            blackout_periods=[],
            task_deadlines={"t1": datetime(2026, 2, 1)},
            final_deadline=_END,
        )
        assert result == {}

    def test_high_availability_unchanged(self) -> None:
        """Student with >= reference hours should not get extensions."""
        base_deadline = datetime(2026, 2, 1)
        task = _task("t1", 5, assigned_to="s1")
        result = distribute_deadlines_for_student(
            student_id="s1",
            tasks=[task],
            availability_hours_per_week=20.0,
            blackout_periods=[],
            task_deadlines={"t1": base_deadline},
            final_deadline=_END,
        )
        assert result["t1"] == base_deadline


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidatePacing:
    """Tests for validate_pacing."""

    def test_valid_pacing(self) -> None:
        # 3 tasks spread across 3 weeks — each week has ~33% < 40%.
        tasks = [
            _task("a", 5),
            _task("b", 5, deps=["a"]),
            _task("c", 5, deps=["b"]),
        ]
        dep_graph = {"a": [], "b": ["a"], "c": ["b"]}
        deadlines = {
            "a": datetime(2026, 1, 15),
            "b": datetime(2026, 2, 15),
            "c": datetime(2026, 3, 15),
        }
        valid, violations = validate_pacing(deadlines, _END, tasks, dep_graph)
        assert valid is True
        assert violations == []

    def test_deadline_past_final(self) -> None:
        tasks = [_task("a", 5)]
        deadlines = {"a": datetime(2026, 5, 1)}
        valid, violations = validate_pacing(deadlines, _END, tasks, {"a": []})
        assert valid is False
        assert any("exceeds final deadline" in v for v in violations)

    def test_dependency_ordering_violation(self) -> None:
        tasks = [_task("a", 5), _task("b", 5, deps=["a"])]
        dep_graph = {"a": [], "b": ["a"]}
        deadlines = {
            "a": datetime(2026, 3, 1),
            "b": datetime(2026, 2, 1),  # before its dependency
        }
        valid, violations = validate_pacing(deadlines, _END, tasks, dep_graph)
        assert valid is False
        assert any("not after dependency" in v for v in violations)

    def test_weekly_concentration_violation(self) -> None:
        # All 5 tasks (50h total) in the same week → one week has 100%.
        tasks = [_task(f"t{i}", 10) for i in range(5)]
        dep_graph = {t.id: [] for t in tasks}
        same_day = datetime(2026, 2, 2)  # All deadlines in same week
        deadlines = {t.id: same_day for t in tasks}
        valid, violations = validate_pacing(deadlines, _END, tasks, dep_graph)
        assert valid is False
        assert any("exceeding 40% threshold" in v for v in violations)

    def test_missing_buffer_violation(self) -> None:
        tasks = [_task("a", 10), _task("b", 10, deps=["a"])]
        dep_graph = {"a": [], "b": ["a"]}
        # Only 2 hours apart — not enough buffer for a 10h task.
        deadlines = {
            "a": datetime(2026, 2, 1, 10, 0),
            "b": datetime(2026, 2, 1, 12, 0),
        }
        valid, violations = validate_pacing(deadlines, _END, tasks, dep_graph)
        assert valid is False
        assert any("insufficient buffer" in v for v in violations)
