"""State validator — enforces integrity rules on state mutations before commit.

Validates proposed updates to :class:`SyncUpState` against six rule categories:
A. Append-only field protection (contribution_ledger, meeting_log, etc.)
B. Delegation integrity (task_id and student_id references)
C. Score bounds (semantic_quality_score ∈ [0, 1], valid urgency enums)
D. Deadline sanity (no task deadline after final_deadline)
E. Cross-student protection (student A cannot modify student B's data)
F. Self-score prevention (student cannot give themselves a perfect score)

No LLM calls — purely deterministic validation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Final, Optional

from state.schema import (
    AvailabilityChange,
    ContributionRecord,
    Intervention,
    MeetingRecord,
    SyncUpState,
    Task,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPEND_ONLY_FIELDS: Final[list[str]] = [
    "contribution_ledger",
    "meeting_log",
    "intervention_history",
    "availability_updates",
]

_APPEND_ONLY_TYPES: Final[dict[str, type]] = {
    "contribution_ledger": ContributionRecord,
    "meeting_log": MeetingRecord,
    "intervention_history": Intervention,
    "availability_updates": AvailabilityChange,
}

PRIVILEGED_AGENTS: Final[set[str]] = {"supervisor", "delegation"}

# Tolerance for floating-point comparison
_EPSILON: Final[float] = 1e-9


# ---------------------------------------------------------------------------
# Private rule checkers
# ---------------------------------------------------------------------------


def _check_append_only(
    current_state: SyncUpState,
    proposed_update: dict[str, Any],
) -> list[str]:
    """Rule A: append-only fields must not lose existing records."""
    violations: list[str] = []

    for field in APPEND_ONLY_FIELDS:
        if field not in proposed_update:
            continue

        current_list = getattr(current_state, field)
        proposed_value = proposed_update[field]

        if not isinstance(proposed_value, list):
            violations.append(
                f"Append-only field '{field}' must be a list, "
                f"got {type(proposed_value).__name__}"
            )
            continue

        # In LangGraph reducer context, agents return [new_items] which get
        # appended.  But if someone passes a full replacement list that is
        # shorter than the current list, that's a deletion attempt.
        if len(proposed_value) < len(current_list):
            violations.append(
                f"Append-only violation on '{field}': proposed length "
                f"({len(proposed_value)}) < current length ({len(current_list)})"
            )

    return violations


def _check_delegation_integrity(
    current_state: SyncUpState,
    proposed_update: dict[str, Any],
) -> list[str]:
    """Rule B: delegation_matrix references must be valid."""
    violations: list[str] = []

    if "delegation_matrix" not in proposed_update:
        return violations

    delegation = proposed_update["delegation_matrix"]
    if not isinstance(delegation, dict):
        return [f"delegation_matrix must be a dict, got {type(delegation).__name__}"]

    # Build valid sets from current state + any proposed task/student updates
    valid_task_ids: set[str] = {t.id for t in current_state.task_array}
    if "task_array" in proposed_update:
        for t in proposed_update["task_array"]:
            tid = t.id if isinstance(t, Task) else t.get("id", "")
            if tid:
                valid_task_ids.add(tid)

    valid_student_ids: set[str] = {
        sp.student_id for sp in current_state.student_profiles
    }

    for task_id, student_id in delegation.items():
        if task_id not in valid_task_ids:
            violations.append(
                f"Delegation references nonexistent task_id '{task_id}'"
            )
        if student_id not in valid_student_ids:
            violations.append(
                f"Delegation references nonexistent student_id '{student_id}'"
            )

    return violations


def _check_score_bounds(
    proposed_update: dict[str, Any],
) -> list[str]:
    """Rule C: semantic_quality_score ∈ [0.0, 1.0], urgency must be valid."""
    violations: list[str] = []

    # Check contribution_ledger scores
    if "contribution_ledger" in proposed_update:
        ledger = proposed_update["contribution_ledger"]
        if isinstance(ledger, list):
            for idx, record in enumerate(ledger):
                score = (
                    record.semantic_quality_score
                    if isinstance(record, ContributionRecord)
                    else record.get("semantic_quality_score", 0.0)
                    if isinstance(record, dict)
                    else 0.0
                )
                if not (0.0 <= score <= 1.0):
                    violations.append(
                        f"contribution_ledger[{idx}]: semantic_quality_score "
                        f"{score} out of bounds [0.0, 1.0]"
                    )

    # Check task urgency values
    if "task_array" in proposed_update:
        valid_urgencies = {e.value for e in UrgencyLevel}
        for idx, task in enumerate(proposed_update["task_array"]):
            urgency = (
                task.urgency.value
                if isinstance(task, Task)
                else task.get("urgency", "medium")
            )
            if urgency not in valid_urgencies:
                violations.append(
                    f"task_array[{idx}]: invalid urgency '{urgency}'"
                )

    return violations


def _check_deadline_sanity(
    current_state: SyncUpState,
    proposed_update: dict[str, Any],
) -> list[str]:
    """Rule D: no task deadline after final_deadline."""
    violations: list[str] = []

    final = proposed_update.get("final_deadline", current_state.final_deadline)
    if final is None:
        return violations

    if "task_array" in proposed_update:
        for idx, task in enumerate(proposed_update["task_array"]):
            deadline = (
                task.deadline if isinstance(task, Task) else task.get("deadline")
            )
            if deadline is not None and deadline > final:
                violations.append(
                    f"task_array[{idx}]: deadline {deadline.isoformat()} "
                    f"is after final_deadline {final.isoformat()}"
                )

    return violations


def _check_cross_student(
    proposed_update: dict[str, Any],
    source_agent: Optional[str],
    source_student_id: Optional[str],
) -> list[str]:
    """Rule E: student A's webhook cannot modify student B's data."""
    violations: list[str] = []

    if source_student_id is None:
        return violations
    if source_agent in PRIVILEGED_AGENTS:
        return violations

    # Check contribution_ledger records
    if "contribution_ledger" in proposed_update:
        for idx, record in enumerate(proposed_update["contribution_ledger"]):
            sid = (
                record.student_id
                if isinstance(record, ContributionRecord)
                else record.get("student_id", "")
            )
            if sid and sid != source_student_id:
                violations.append(
                    f"Cross-student violation: source student '{source_student_id}' "
                    f"attempted to modify contribution_ledger[{idx}] "
                    f"belonging to student '{sid}'"
                )

    # Check availability_updates records
    if "availability_updates" in proposed_update:
        for idx, record in enumerate(proposed_update["availability_updates"]):
            sid = (
                record.student_id
                if isinstance(record, AvailabilityChange)
                else record.get("student_id", "")
            )
            if sid and sid != source_student_id:
                violations.append(
                    f"Cross-student violation: source student '{source_student_id}' "
                    f"attempted to modify availability_updates[{idx}] "
                    f"belonging to student '{sid}'"
                )

    return violations


def _check_self_score(
    proposed_update: dict[str, Any],
    source_agent: Optional[str],
    source_student_id: Optional[str],
) -> list[str]:
    """Rule F: student cannot set their own semantic_quality_score to 1.0."""
    violations: list[str] = []

    if source_student_id is None:
        return violations
    if source_agent in PRIVILEGED_AGENTS:
        return violations

    if "contribution_ledger" in proposed_update:
        for idx, record in enumerate(proposed_update["contribution_ledger"]):
            sid = (
                record.student_id
                if isinstance(record, ContributionRecord)
                else record.get("student_id", "")
            )
            score = (
                record.semantic_quality_score
                if isinstance(record, ContributionRecord)
                else record.get("semantic_quality_score", 0.0)
            )
            if sid == source_student_id and score >= 1.0 - _EPSILON:
                violations.append(
                    f"Self-score violation: student '{source_student_id}' "
                    f"attempted to set own score to {score} in "
                    f"contribution_ledger[{idx}]"
                )

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_state_update(
    current_state: SyncUpState,
    proposed_update: dict[str, Any],
    source_agent: Optional[str] = None,
    source_student_id: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Validate a proposed state update against integrity rules.

    Args:
        current_state: The current ``SyncUpState`` before the update.
        proposed_update: Dict of field names to new values (partial state update).
        source_agent: The agent name producing this update
            (e.g. ``"progress_tracking"``).
        source_student_id: If the update originates from a student webhook,
            the student_id.

    Returns:
        A tuple of ``(is_valid, violations)`` where *violations* lists
        human-readable strings describing each rule violation.
    """
    violations: list[str] = []

    violations.extend(_check_append_only(current_state, proposed_update))
    violations.extend(_check_delegation_integrity(current_state, proposed_update))
    violations.extend(_check_score_bounds(proposed_update))
    violations.extend(_check_deadline_sanity(current_state, proposed_update))
    violations.extend(
        _check_cross_student(proposed_update, source_agent, source_student_id)
    )
    violations.extend(
        _check_self_score(proposed_update, source_agent, source_student_id)
    )

    if violations:
        for v in violations:
            logger.warning("State validation violation: %s", v)

    return len(violations) == 0, violations


def sanitize_state_update(
    current_state: SyncUpState,
    proposed_update: dict[str, Any],
    source_agent: Optional[str] = None,
    source_student_id: Optional[str] = None,
) -> dict[str, Any]:
    """Sanitise a proposed state update by fixing or removing invalid data.

    Applies the same rules as :func:`validate_state_update` but attempts to
    fix violations rather than just reporting them.  Unfixable violations
    cause the offending field to be dropped from the update.

    Args:
        current_state: The current ``SyncUpState``.
        proposed_update: Dict of field names to new values.
        source_agent: The agent name producing this update.
        source_student_id: If from a student webhook, the student_id.

    Returns:
        A sanitised copy of *proposed_update* with violations fixed or removed.
    """
    cleaned = copy.deepcopy(proposed_update)

    # --- Rule A: append-only protection ---
    for field in APPEND_ONLY_FIELDS:
        if field not in cleaned:
            continue
        current_list = getattr(current_state, field)
        proposed_value = cleaned[field]
        if not isinstance(proposed_value, list):
            logger.warning(
                "Sanitise: dropping non-list '%s' (type %s)",
                field,
                type(proposed_value).__name__,
            )
            del cleaned[field]
        elif len(proposed_value) < len(current_list):
            logger.warning(
                "Sanitise: dropping truncated '%s' "
                "(proposed %d < current %d)",
                field,
                len(proposed_value),
                len(current_list),
            )
            del cleaned[field]

    # --- Rule B: delegation integrity ---
    if "delegation_matrix" in cleaned:
        delegation = cleaned["delegation_matrix"]
        if isinstance(delegation, dict):
            valid_task_ids = {t.id for t in current_state.task_array}
            if "task_array" in cleaned:
                for t in cleaned["task_array"]:
                    tid = t.id if isinstance(t, Task) else t.get("id", "")
                    if tid:
                        valid_task_ids.add(tid)
            valid_student_ids = {
                sp.student_id for sp in current_state.student_profiles
            }
            to_remove = [
                k
                for k, v in delegation.items()
                if k not in valid_task_ids or v not in valid_student_ids
            ]
            for k in to_remove:
                logger.warning(
                    "Sanitise: removing invalid delegation entry %s → %s",
                    k,
                    delegation[k],
                )
                del delegation[k]

    # --- Rule C: score bounds ---
    if "contribution_ledger" in cleaned:
        for idx, record in enumerate(cleaned["contribution_ledger"]):
            if isinstance(record, ContributionRecord):
                if record.semantic_quality_score > 1.0:
                    logger.warning(
                        "Sanitise: clamping score %.4f → 1.0 at ledger[%d]",
                        record.semantic_quality_score,
                        idx,
                    )
                    # ContributionRecord is a Pydantic model — rebuild
                    data = record.model_dump()
                    data["semantic_quality_score"] = 1.0
                    cleaned["contribution_ledger"][idx] = ContributionRecord(**data)
                elif record.semantic_quality_score < 0.0:
                    logger.warning(
                        "Sanitise: clamping score %.4f → 0.0 at ledger[%d]",
                        record.semantic_quality_score,
                        idx,
                    )
                    data = record.model_dump()
                    data["semantic_quality_score"] = 0.0
                    cleaned["contribution_ledger"][idx] = ContributionRecord(**data)
            elif isinstance(record, dict):
                score = record.get("semantic_quality_score", 0.0)
                if score > 1.0:
                    logger.warning(
                        "Sanitise: clamping score %.4f → 1.0 at ledger[%d]",
                        score,
                        idx,
                    )
                    record["semantic_quality_score"] = 1.0
                elif score < 0.0:
                    logger.warning(
                        "Sanitise: clamping score %.4f → 0.0 at ledger[%d]",
                        score,
                        idx,
                    )
                    record["semantic_quality_score"] = 0.0

    # --- Rule D: deadline sanity ---
    final = cleaned.get("final_deadline", current_state.final_deadline)
    if final is not None and "task_array" in cleaned:
        for idx, task in enumerate(cleaned["task_array"]):
            deadline = (
                task.deadline if isinstance(task, Task) else task.get("deadline")
            )
            if deadline is not None and deadline > final:
                logger.warning(
                    "Sanitise: clamping task[%d] deadline %s → final %s",
                    idx,
                    deadline.isoformat(),
                    final.isoformat(),
                )
                if isinstance(task, Task):
                    data = task.model_dump()
                    data["deadline"] = final
                    cleaned["task_array"][idx] = Task(**data)
                else:
                    task["deadline"] = final

    # --- Rule E: cross-student protection ---
    if source_student_id and source_agent not in PRIVILEGED_AGENTS:
        if "contribution_ledger" in cleaned:
            original = cleaned["contribution_ledger"]
            filtered = []
            for record in original:
                sid = (
                    record.student_id
                    if isinstance(record, ContributionRecord)
                    else record.get("student_id", "")
                )
                if sid and sid != source_student_id:
                    logger.warning(
                        "Sanitise: removing cross-student contribution "
                        "record for '%s' (source: '%s')",
                        sid,
                        source_student_id,
                    )
                else:
                    filtered.append(record)
            cleaned["contribution_ledger"] = filtered

        if "availability_updates" in cleaned:
            original = cleaned["availability_updates"]
            filtered = []
            for record in original:
                sid = (
                    record.student_id
                    if isinstance(record, AvailabilityChange)
                    else record.get("student_id", "")
                )
                if sid and sid != source_student_id:
                    logger.warning(
                        "Sanitise: removing cross-student availability "
                        "update for '%s' (source: '%s')",
                        sid,
                        source_student_id,
                    )
                else:
                    filtered.append(record)
            cleaned["availability_updates"] = filtered

    # --- Rule F: self-score prevention ---
    if source_student_id and source_agent not in PRIVILEGED_AGENTS:
        if "contribution_ledger" in cleaned:
            for idx, record in enumerate(cleaned["contribution_ledger"]):
                sid = (
                    record.student_id
                    if isinstance(record, ContributionRecord)
                    else record.get("student_id", "")
                )
                score = (
                    record.semantic_quality_score
                    if isinstance(record, ContributionRecord)
                    else record.get("semantic_quality_score", 0.0)
                )
                if sid == source_student_id and score >= 1.0 - _EPSILON:
                    logger.warning(
                        "Sanitise: clamping self-score %.4f → 0.99 "
                        "for student '%s' at ledger[%d]",
                        score,
                        source_student_id,
                        idx,
                    )
                    if isinstance(record, ContributionRecord):
                        data = record.model_dump()
                        data["semantic_quality_score"] = 0.99
                        cleaned["contribution_ledger"][idx] = ContributionRecord(
                            **data
                        )
                    elif isinstance(record, dict):
                        record["semantic_quality_score"] = 0.99

    return cleaned
