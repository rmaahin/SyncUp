"""Pacing service — deterministic deadline distribution for SyncUp projects.

Pure Python, no LLM dependency.  Provides burn-down curve calculation,
per-student deadline adjustment, and pacing validation.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional

from state.schema import DateRange, Task, UrgencyLevel

logger = logging.getLogger(__name__)

# Urgency multipliers applied to the raw timeline fraction.
# Lower values push the deadline earlier in the project window.
_URGENCY_MULTIPLIER: dict[UrgencyLevel, float] = {
    UrgencyLevel.CRITICAL: 0.70,
    UrgencyLevel.HIGH: 0.85,
    UrgencyLevel.MEDIUM: 1.00,
    UrgencyLevel.LOW: 1.15,
}

# Priority ordering for topological sort tie-breaking (lower = earlier).
_URGENCY_PRIORITY: dict[UrgencyLevel, int] = {
    UrgencyLevel.CRITICAL: 0,
    UrgencyLevel.HIGH: 1,
    UrgencyLevel.MEDIUM: 2,
    UrgencyLevel.LOW: 3,
}

_MIN_BUFFER_DAYS: float = 1.0
_BUFFER_EFFORT_RATIO: float = 0.10  # 10% of effort_hours as buffer days


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def _topological_sort(
    tasks: list[Task],
    dependency_graph: dict[str, list[str]],
) -> list[str]:
    """Return task IDs in topological order using Kahn's algorithm.

    Within each BFS level tasks are sorted by urgency priority so that
    CRITICAL tasks appear before LOW ones at the same depth.

    Args:
        tasks: All project tasks.
        dependency_graph: Mapping of task_id → list of prerequisite task_ids.

    Returns:
        A list of task IDs in valid topological order.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    task_ids: set[str] = {t.id for t in tasks}
    task_map: dict[str, Task] = {t.id: t for t in tasks}

    # Build in-degree map (only consider edges within task_ids).
    in_degree: dict[str, int] = {tid: 0 for tid in task_ids}
    # Forward adjacency: prerequisite → list of dependents.
    forward: dict[str, list[str]] = defaultdict(list)

    for tid in task_ids:
        for dep in dependency_graph.get(tid, []):
            if dep in task_ids:
                in_degree[tid] += 1
                forward[dep].append(tid)

    # Seed the queue with zero-in-degree tasks, sorted by urgency.
    queue: deque[str] = deque(
        sorted(
            (tid for tid, deg in in_degree.items() if deg == 0),
            key=lambda tid: _URGENCY_PRIORITY.get(
                task_map[tid].urgency, 3
            ),
        )
    )

    result: list[str] = []
    while queue:
        # Process the current level (all items currently in queue).
        level_size = len(queue)
        level_items: list[str] = []
        for _ in range(level_size):
            level_items.append(queue.popleft())

        # Sort level by urgency priority so CRITICAL tasks come first.
        level_items.sort(
            key=lambda tid: _URGENCY_PRIORITY.get(
                task_map[tid].urgency, 3
            )
        )

        for tid in level_items:
            result.append(tid)
            for neighbor in forward[tid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    if len(result) != len(task_ids):
        raise ValueError("Circular dependency detected")

    return result


# ---------------------------------------------------------------------------
# Burn-down curve
# ---------------------------------------------------------------------------


def _buffer_days(task: Task) -> float:
    """Minimum buffer days required after prerequisites for *task*."""
    return max(_MIN_BUFFER_DAYS, task.effort_hours * _BUFFER_EFFORT_RATIO)


def calculate_burn_down_curve(
    final_deadline: datetime,
    project_start: datetime,
    tasks: list[Task],
    dependency_graph: dict[str, list[str]],
) -> dict[str, datetime]:
    """Compute a recommended deadline for every task.

    The algorithm topologically sorts tasks, distributes them proportionally
    across the project timeline, applies urgency-based adjustments, and
    ensures buffer gaps between dependent tasks.

    Args:
        final_deadline: The hard project deadline.
        project_start: When the project begins.
        tasks: All decomposed tasks.
        dependency_graph: task_id → list of prerequisite task_ids.

    Returns:
        A mapping of task_id → recommended deadline datetime.
    """
    if not tasks:
        return {}

    sorted_ids = _topological_sort(tasks, dependency_graph)
    task_map: dict[str, Task] = {t.id: t for t in tasks}

    total_effort = sum(t.effort_hours for t in tasks)
    if total_effort <= 0:
        total_effort = 1.0  # avoid division by zero

    total_duration = final_deadline - project_start
    if total_duration.total_seconds() <= 0:
        # Degenerate case: everything due immediately.
        return {tid: final_deadline for tid in sorted_ids}

    # --- Step 1: proportional positioning with urgency multiplier ---
    deadlines: dict[str, datetime] = {}
    cumulative = 0.0

    for tid in sorted_ids:
        task = task_map[tid]
        cumulative += task.effort_hours
        raw_fraction = cumulative / total_effort

        multiplier = _URGENCY_MULTIPLIER.get(task.urgency, 1.0)
        adjusted_fraction = min(raw_fraction * multiplier, 1.0)

        deadline = project_start + total_duration * adjusted_fraction
        deadlines[tid] = deadline

    # --- Step 2: enforce dependency ordering + buffer ---
    for tid in sorted_ids:
        task = task_map[tid]
        deps = dependency_graph.get(tid, [])
        if not deps:
            continue

        latest_dep = max(
            (deadlines[d] for d in deps if d in deadlines),
            default=project_start,
        )
        buffer = timedelta(days=_buffer_days(task))
        earliest_allowed = latest_dep + buffer

        if deadlines[tid] < earliest_allowed:
            deadlines[tid] = earliest_allowed

    # --- Step 3: clamp to final_deadline ---
    overflow = False
    for tid in sorted_ids:
        if deadlines[tid] > final_deadline:
            overflow = True
            break

    if overflow:
        # Scale all deadlines proportionally to fit within the window.
        max_offset = max(
            (deadlines[tid] - project_start).total_seconds()
            for tid in sorted_ids
        )
        if max_offset > 0:
            scale = total_duration.total_seconds() / max_offset
            for tid in sorted_ids:
                offset = (deadlines[tid] - project_start).total_seconds()
                deadlines[tid] = project_start + timedelta(
                    seconds=offset * scale
                )

    return deadlines


# ---------------------------------------------------------------------------
# Per-student deadline adjustment
# ---------------------------------------------------------------------------


def _falls_in_blackout(
    dt: datetime, blackouts: list[DateRange]
) -> Optional[DateRange]:
    """Return the blackout period containing *dt*, or ``None``."""
    for bp in blackouts:
        if bp.start <= dt <= bp.end:
            return bp
    return None


def distribute_deadlines_for_student(
    student_id: str,
    tasks: list[Task],
    availability_hours_per_week: float,
    blackout_periods: list[DateRange],
    task_deadlines: dict[str, datetime],
    final_deadline: datetime,
) -> dict[str, datetime]:
    """Adjust global deadlines for a specific student's constraints.

    Accounts for the student's available hours per week and blackout
    periods.  A student with fewer available hours gets more calendar time
    for the same effort.

    Args:
        student_id: The student whose deadlines to adjust.
        tasks: Tasks assigned to this student.
        availability_hours_per_week: Student's weekly availability.
        blackout_periods: Periods when the student is unavailable.
        task_deadlines: Base deadlines from ``calculate_burn_down_curve``.
        final_deadline: Hard project deadline (never exceeded).

    Returns:
        Adjusted task_id → deadline mapping for this student's tasks.
    """
    student_tasks = [t for t in tasks if t.assigned_to == student_id]
    if not student_tasks:
        return {}

    adjusted: dict[str, datetime] = {}

    # Reference availability: 20 hrs/week is "full speed".
    reference_hours = 20.0
    effective_avail = max(availability_hours_per_week, 1.0)  # floor at 1

    for task in student_tasks:
        base_deadline = task_deadlines.get(task.id)
        if base_deadline is None:
            continue

        deadline = base_deadline

        # If student has fewer hours, they need proportionally more time.
        if effective_avail < reference_hours:
            base_effort_days = task.effort_hours / (reference_hours / 7.0)
            extra_factor = reference_hours / effective_avail
            needed_days = base_effort_days * extra_factor
            extension_days = needed_days - base_effort_days
            if extension_days > 0:
                deadline = deadline + timedelta(days=extension_days)

        # Push past blackout periods.
        blackout = _falls_in_blackout(deadline, blackout_periods)
        if blackout is not None:
            deadline = blackout.end + timedelta(days=1)

        # Clamp to final deadline.
        adjusted[task.id] = min(deadline, final_deadline)

    return adjusted


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_pacing(
    task_deadlines: dict[str, datetime],
    final_deadline: datetime,
    tasks: list[Task],
    dependency_graph: dict[str, list[str]],
) -> tuple[bool, list[str]]:
    """Validate that a set of deadlines satisfies project constraints.

    Checks:
    1. No deadline exceeds ``final_deadline``.
    2. Dependency ordering is respected.
    3. No single ISO week has >40% of total effort assigned.
    4. Buffer exists between dependent tasks.

    Args:
        task_deadlines: task_id → deadline mapping.
        final_deadline: Hard project deadline.
        tasks: All project tasks.
        dependency_graph: task_id → list of prerequisite task_ids.

    Returns:
        A tuple of ``(is_valid, list_of_violation_descriptions)``.
    """
    violations: list[str] = []
    task_map: dict[str, Task] = {t.id: t for t in tasks}

    # Check 1: no deadline past final_deadline.
    for tid, dl in task_deadlines.items():
        if dl > final_deadline:
            violations.append(
                f"Task {tid} deadline {dl.isoformat()} exceeds "
                f"final deadline {final_deadline.isoformat()}"
            )

    # Check 2: dependency ordering.
    for tid, deps in dependency_graph.items():
        if tid not in task_deadlines:
            continue
        for dep in deps:
            if dep not in task_deadlines:
                continue
            if task_deadlines[tid] <= task_deadlines[dep]:
                violations.append(
                    f"Task {tid} deadline ({task_deadlines[tid].isoformat()}) "
                    f"is not after dependency {dep} "
                    f"({task_deadlines[dep].isoformat()})"
                )

    # Check 3: weekly effort concentration.
    total_effort = sum(t.effort_hours for t in tasks if t.id in task_deadlines)
    if total_effort > 0:
        week_effort: dict[tuple[int, int], float] = defaultdict(float)
        for tid, dl in task_deadlines.items():
            task = task_map.get(tid)
            if task:
                iso_cal = dl.isocalendar()
                week_effort[(iso_cal[0], iso_cal[1])] += task.effort_hours

        threshold = total_effort * 0.40
        for (year, week), effort in week_effort.items():
            if effort > threshold:
                violations.append(
                    f"Week {year}-W{week:02d} has {effort:.1f}h "
                    f"({effort / total_effort * 100:.0f}% of "
                    f"{total_effort:.1f}h total), exceeding 40% threshold"
                )

    # Check 4: buffer between dependent tasks.
    for tid, deps in dependency_graph.items():
        if tid not in task_deadlines:
            continue
        task = task_map.get(tid)
        if not task:
            continue
        required_buffer = timedelta(days=_buffer_days(task))
        for dep in deps:
            if dep not in task_deadlines:
                continue
            actual_gap = task_deadlines[tid] - task_deadlines[dep]
            if actual_gap < required_buffer:
                violations.append(
                    f"Task {tid} has insufficient buffer after "
                    f"dependency {dep}: {actual_gap.days}d < "
                    f"{required_buffer.days}d required"
                )

    return (len(violations) == 0, violations)
