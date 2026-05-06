"""Final report generator — synthesizes a per-student + team-level project report.

Combines deterministic metrics from ``SyncUpState`` with peer-review summaries
and high-tier LLM narratives. Runs at project close; called from the
``/api/reports/{project_id}/generate`` endpoint, not from the LangGraph graph.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from guardrails.sanitizer import sanitize_text
from llm import get_high_tier_llm
from services.peer_review_analysis import aggregate_peer_reviews, detect_bias
from state.schema import (
    BiasFlag,
    EventType,
    PeerReviewSummary,
    StudentReport,
    SyncUpState,
    TaskStatus,
    TeamReport,
)

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 2
LOW_QUALITY_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


STUDENT_SYSTEM_PROMPT: str = """\
You write fair, evidence-based end-of-project reports for ONE student on a \
group project. You will be given the student's deterministic metrics and \
(optionally) a summary of how teammates rated them. Cite specific metrics; do \
not invent facts. If peer-bias flags are present, be cautious about peer \
ratings.
You MUST respond with ONLY a JSON object — no markdown fences, no preamble:
{
  "narrative": "<3-5 sentence balanced summary>",
  "strengths": ["<bullet>", ...],
  "areas_for_improvement": ["<bullet>", ...]
}
"""


TEAM_SYSTEM_PROMPT: str = """\
You write a brief end-of-project summary for an entire student team. You will \
receive aggregate team metrics. Keep it to <=4 sentences. Do not invent facts.
You MUST respond with ONLY a JSON object: {"narrative": "<text>"}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and preamble text to isolate JSON."""
    text = raw.strip()
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    brace_idx = text.find("{")
    if brace_idx == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[brace_idx:]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Deterministic metric collection
# ---------------------------------------------------------------------------


def collect_student_metrics(state: SyncUpState, student_id: str) -> dict[str, Any]:
    """Compute deterministic per-student metrics from project state.

    No LLM calls. Best-effort — fields with no signal return ``None`` so the
    frontend can render "n/a".
    """
    now = _now()
    sid = student_id

    assigned_ids = {
        tid for tid, owner in state.delegation_matrix.items() if owner == sid
    }
    assigned = [t for t in state.task_array if t.id in assigned_ids]
    completed = [t for t in assigned if t.status == TaskStatus.DONE]

    completed_with_dl = [t for t in completed if t.deadline is not None]
    if completed_with_dl:
        on_time = sum(
            1 for t in completed_with_dl
            if t.deadline is not None and t.deadline >= now
        )
        deadline_adherence_rate: Optional[float] = on_time / len(completed_with_dl)
    else:
        deadline_adherence_rate = None

    contributions = [c for c in state.contribution_ledger if c.student_id == sid]
    commits = sum(1 for c in contributions if c.event_type == EventType.COMMIT)
    prs = sum(1 for c in contributions if c.event_type == EventType.PR_REVIEW)
    doc_edits = sum(1 for c in contributions if c.event_type == EventType.DOC_EDIT)
    card_moves = sum(1 for c in contributions if c.event_type == EventType.CARD_MOVE)

    quality_scores = [c.semantic_quality_score for c in contributions]
    avg_quality: Optional[float] = (
        sum(quality_scores) / len(quality_scores) if quality_scores else None
    )
    low_quality_count = sum(1 for s in quality_scores if s < LOW_QUALITY_THRESHOLD)

    total_meetings = len(state.meeting_log)
    if total_meetings:
        attended = sum(1 for m in state.meeting_log if sid in m.attendees)
        meeting_attendance_rate: Optional[float] = attended / total_meetings
    else:
        meeting_attendance_rate = None

    interventions = [
        i for i in state.intervention_history if i.target_student_id == sid
    ]
    intervention_outcomes = dict(Counter(i.outcome for i in interventions if i.outcome))

    return {
        "tasks_assigned": len(assigned),
        "tasks_completed": len(completed),
        "deadline_adherence_rate": deadline_adherence_rate,
        "commits": commits,
        "prs": prs,
        "doc_edits": doc_edits,
        "card_moves": card_moves,
        "avg_semantic_quality": avg_quality,
        "low_quality_contribution_count": low_quality_count,
        "meeting_attendance_rate": meeting_attendance_rate,
        "interventions_received": len(interventions),
        "intervention_outcomes": intervention_outcomes,
    }


# ---------------------------------------------------------------------------
# LLM narrative helpers
# ---------------------------------------------------------------------------


def _sanitize_summary(summary: Optional[PeerReviewSummary]) -> Optional[dict[str, Any]]:
    if summary is None:
        return None
    return summary.model_dump(mode="json")


def _sanitize_bias_flags(flags: list[BiasFlag]) -> list[dict[str, Any]]:
    return [
        {
            "flag_type": f.flag_type,
            "reviewer_id": f.reviewer_id,
            "reviewee_id": f.reviewee_id,
            "description": sanitize_text(f.description),
            "severity": f.severity,
        }
        for f in flags
    ]


def generate_student_report(
    student_id: str,
    state: SyncUpState,
    peer_summary: Optional[PeerReviewSummary],
    bias_flags: list[BiasFlag],
) -> StudentReport:
    """Generate a single student's report — metrics + LLM narrative."""
    metrics = collect_student_metrics(state, student_id)

    student_flags = [
        f for f in bias_flags
        if f.reviewer_id == student_id or f.reviewee_id == student_id
    ]

    payload: dict[str, Any] = {
        "student_id": student_id,
        "metrics": metrics,
    }
    if peer_summary is not None:
        payload["peer_summary"] = _sanitize_summary(peer_summary)
    if student_flags:
        payload["bias_flags"] = _sanitize_bias_flags(student_flags)

    user_msg = json.dumps(payload, default=str)

    try:
        llm = get_high_tier_llm(temperature=0.2)
    except Exception as exc:
        logger.warning("Could not initialize LLM for student report: %s", exc)
        return StudentReport(
            student_id=student_id,
            metrics=metrics,
            peer_summary=peer_summary,
            peer_bias_flags=student_flags,
            narrative="Report generation failed; metrics shown below are authoritative.",
        )

    last_error: BaseException | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            content = getattr(response, "content", "")
            if not isinstance(content, str):
                content = str(content)
            parsed = json.loads(_extract_json(content))
            narrative = str(parsed.get("narrative", "")).strip()
            strengths = [str(s) for s in parsed.get("strengths", []) if s]
            improvements = [
                str(s) for s in parsed.get("areas_for_improvement", []) if s
            ]
            return StudentReport(
                student_id=student_id,
                metrics=metrics,
                peer_summary=peer_summary,
                peer_bias_flags=student_flags,
                narrative=narrative,
                strengths=strengths,
                areas_for_improvement=improvements,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Student report attempt %d/%d failed for %s: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                student_id,
                exc,
            )

    logger.error("Student report failed after retries for %s: %s", student_id, last_error)
    return StudentReport(
        student_id=student_id,
        metrics=metrics,
        peer_summary=peer_summary,
        peer_bias_flags=student_flags,
        narrative="Report generation failed; metrics shown below are authoritative.",
    )


def _compute_team_metrics(
    state: SyncUpState,
    summaries: dict[str, PeerReviewSummary],
) -> dict[str, Any]:
    total_tasks = len(state.task_array)
    done_tasks = sum(1 for t in state.task_array if t.status == TaskStatus.DONE)
    completion_pct = (done_tasks / total_tasks) if total_tasks else 0.0

    per_student_rates: list[float] = []
    total_meetings = len(state.meeting_log)
    if total_meetings:
        for sp in state.student_profiles:
            attended = sum(
                1 for m in state.meeting_log if sp.student_id in m.attendees
            )
            per_student_rates.append(attended / total_meetings)
    team_meeting_attendance_rate: Optional[float] = (
        sum(per_student_rates) / len(per_student_rates)
        if per_student_rates
        else None
    )

    reviewed_overalls = [
        s.overall_avg for s in summaries.values() if s.review_count > 0
    ]
    avg_peer_review_score: Optional[float] = (
        sum(reviewed_overalls) / len(reviewed_overalls)
        if reviewed_overalls
        else None
    )

    return {
        "completion_pct": completion_pct,
        "total_tasks": total_tasks,
        "completed_tasks": done_tasks,
        "team_meeting_attendance_rate": team_meeting_attendance_rate,
        "avg_peer_review_score": avg_peer_review_score,
        "total_interventions": len(state.intervention_history),
        "total_contributions": len(state.contribution_ledger),
        "student_count": len(state.student_profiles),
    }


def _generate_team_narrative(team_metrics: dict[str, Any]) -> str:
    try:
        llm = get_high_tier_llm(temperature=0.2)
    except Exception as exc:
        logger.warning("Could not initialize LLM for team narrative: %s", exc)
        return ""

    user_msg = json.dumps(team_metrics, default=str)
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": TEAM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            content = getattr(response, "content", "")
            if not isinstance(content, str):
                content = str(content)
            parsed = json.loads(_extract_json(content))
            return str(parsed.get("narrative", "")).strip()
        except Exception as exc:
            logger.warning(
                "Team narrative attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                exc,
            )
    return ""


def generate_team_report(state: SyncUpState) -> TeamReport:
    """Generate the final TeamReport for a project."""
    student_ids = [s.student_id for s in state.student_profiles]
    summaries = aggregate_peer_reviews(state.peer_review_data, student_ids)
    bias_flags = detect_bias(state.peer_review_data, student_ids)

    student_reports: list[StudentReport] = []
    for sid in student_ids:
        student_reports.append(
            generate_student_report(sid, state, summaries.get(sid), bias_flags)
        )

    team_metrics = _compute_team_metrics(state, summaries)
    team_narrative = _generate_team_narrative(team_metrics)

    return TeamReport(
        project_id=state.project_id,
        completion_pct=team_metrics["completion_pct"],
        team_metrics=team_metrics,
        student_reports=student_reports,
        bias_flags=bias_flags,
        team_narrative=team_narrative,
        generated_at=_now(),
    )
