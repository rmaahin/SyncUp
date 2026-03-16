"""Live end-to-end test — Phase 1 through 5 with REAL services.

This script calls each agent node in sequence using:
  - Real Groq LLM (task decomposition, delegation, equity evaluation)
  - Real Trello API (creates a board with cards)
  - Google Calendar via MCP (if configured, otherwise skipped gracefully)
  - Google Docs via MCP (if configured, otherwise skipped gracefully)

Usage:
    # From backend/ directory:
    python scripts/live_e2e_test.py

    # Inside Docker:
    docker compose exec backend python scripts/live_e2e_test.py

Prerequisites:
    - .env file with at minimum: GROQ_API_KEY, TRELLO_API_KEY, TRELLO_API_TOKEN
    - Optional: COURSE_CALENDAR_ID + Google Calendar MCP server running
    - Optional: Google Docs MCP server running

WARNING: This creates REAL resources (Trello boards, calendar events).
         You will need to manually delete the Trello board afterward.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Ensure src/ is on the path when running as a script
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

from dotenv import load_dotenv

load_dotenv()

from agents.delegation import delegation
from agents.publishing import publishing
from agents.task_decomposition import task_decomposition
from evaluators.equity_evaluator import equity_evaluator
from state.schema import StudentProfile, SyncUpState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_e2e_test")

# How far out the project deadline is (from now)
PROJECT_DURATION_DAYS = 14  # 2 weeks

# Your email — calendar events will be sent here
YOUR_EMAIL = os.environ.get("TEST_USER_EMAIL", "")


# ---------------------------------------------------------------------------
# Mock project data
# ---------------------------------------------------------------------------

PROJECT_BRIEF = """\
Build a simple personal to-do list web app. \
2-person team, 2-week timeline.

Features:
1. A single-page app where users can add, complete, and delete to-do items.
2. Items are stored in a PostgreSQL database via a Python FastAPI backend.
3. Basic HTML/CSS frontend (no framework needed).

That's it — keep it minimal.
"""


def _make_students() -> list[StudentProfile]:
    """Create mock students. Uses YOUR_EMAIL for the first student."""
    email = YOUR_EMAIL or "student1@example.com"
    return [
        StudentProfile(
            student_id="s-1",
            name="Alice",
            email=email,
            skills={"python": 0.9, "fastapi": 0.8, "sql": 0.7},
            availability_hours_per_week=10.0,
            timezone="America/New_York",
            github_username="alice-dev",
            google_email=email,
            trello_id="",
            onboarded_at=datetime.now(tz=timezone.utc),
        ),
        StudentProfile(
            student_id="s-2",
            name="Bob",
            email="bob@example.com",
            skills={"html": 0.8, "css": 0.7, "javascript": 0.6},
            availability_hours_per_week=10.0,
            timezone="America/New_York",
            github_username="bob-dev",
            google_email="bob@example.com",
            trello_id="",
            onboarded_at=datetime.now(tz=timezone.utc),
        ),
    ]


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight() -> dict[str, bool]:
    """Check which services are configured."""
    checks: dict[str, bool] = {}

    # Groq (required)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    checks["groq"] = bool(groq_key)
    if not groq_key:
        logger.error("GROQ_API_KEY is not set — LLM calls will fail")

    # Trello (required for publishing)
    trello_key = os.environ.get("TRELLO_API_KEY", "")
    trello_token = os.environ.get("TRELLO_API_TOKEN", "")
    checks["trello"] = bool(trello_key and trello_token)
    if not checks["trello"]:
        logger.warning("TRELLO_API_KEY or TRELLO_API_TOKEN not set — Trello publishing will fail")

    # Google Calendar MCP (optional)
    cal_mcp_url = os.environ.get("GOOGLE_CALENDAR_MCP_URL", "")
    cal_id = os.environ.get("COURSE_CALENDAR_ID", "")
    checks["calendar"] = bool(cal_mcp_url and cal_id)
    if not checks["calendar"]:
        logger.info("Google Calendar MCP not configured — calendar publishing will be skipped (graceful)")

    # Google Docs MCP (optional)
    docs_mcp_url = os.environ.get("GOOGLE_DOCS_MCP_URL", "")
    checks["docs"] = bool(docs_mcp_url)
    if not checks["docs"]:
        logger.info("Google Docs MCP not configured — docs publishing will be skipped (graceful)")

    return checks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline() -> None:
    """Execute the full Phase 1-5 pipeline with real services."""
    print("=" * 70)
    print("  SyncUp Live E2E Test — Phases 1 through 5")
    print("=" * 70)

    # Pre-flight
    print("\n--- Pre-flight checks ---")
    checks = preflight()
    print(f"  Groq LLM:        {'OK' if checks['groq'] else 'MISSING (required)'}")
    print(f"  Trello API:      {'OK' if checks['trello'] else 'MISSING (will fail gracefully)'}")
    print(f"  Google Calendar:  {'OK' if checks['calendar'] else 'not configured (will skip)'}")
    print(f"  Google Docs:      {'OK' if checks['docs'] else 'not configured (will skip)'}")

    if not checks["groq"]:
        print("\nERROR: GROQ_API_KEY is required. Add it to .env and retry.")
        sys.exit(1)

    # =========================================================================
    # PHASE 1: State Schema
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Constructing initial state")
    print("=" * 70)

    now = datetime.now(tz=timezone.utc)
    final_deadline = now + timedelta(days=PROJECT_DURATION_DAYS)
    students = _make_students()

    state = SyncUpState(
        project_id="proj-campuseats",
        project_name="CampusEats",
        project_brief=PROJECT_BRIEF,
        final_deadline=final_deadline,
        student_profiles=students,
    )

    print(f"  Project:  {state.project_name}")
    print(f"  Students: {[s.name for s in state.student_profiles]}")
    print(f"  Deadline: {state.final_deadline}")
    print(f"  Brief:    {state.project_brief[:80]}...")
    print("  [OK] State constructed")

    # =========================================================================
    # PHASE 3: Task Decomposition (real LLM)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Task Decomposition (calling Groq LLM)")
    print("=" * 70)

    decomp_result = task_decomposition(state)
    state = state.model_copy(update=decomp_result)

    if not state.task_array:
        print("  ERROR: Task decomposition returned no tasks!")
        sys.exit(1)

    print(f"  Decomposed into {len(state.task_array)} tasks:")
    total_effort = 0.0
    for t in state.task_array:
        deps = f" (depends on: {', '.join(t.dependencies)})" if t.dependencies else ""
        print(f"    - [{t.urgency.value:>8}] {t.title} ({t.effort_hours}h){deps}")
        total_effort += t.effort_hours
    print(f"  Total effort: {total_effort}h")
    print(f"  Dependency graph: {len(state.dependency_graph)} entries")
    print("  [OK] Task decomposition complete")

    # =========================================================================
    # PHASE 4a: Delegation (real LLM + real pacing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4a: Delegation (calling Groq LLM + pacing algorithm)")
    print("=" * 70)

    deleg_result = delegation(state)

    if "delegation_matrix" not in deleg_result:
        print("  ERROR: Delegation returned no matrix!")
        sys.exit(1)

    state = state.model_copy(update=deleg_result)

    print(f"  Assignments:")
    student_map = {s.student_id: s for s in state.student_profiles}
    for tid, sid in state.delegation_matrix.items():
        task = next((t for t in state.task_array if t.id == tid), None)
        student = student_map.get(sid)
        if task and student:
            dl = task.deadline.strftime("%Y-%m-%d") if task.deadline else "no deadline"
            print(f"    - {task.title} -> {student.name} (due {dl})")

    if state.project_timeline.burn_down_targets:
        print(f"  Burn-down targets: {len(state.project_timeline.burn_down_targets)} points")
    print("  [OK] Delegation complete")

    # =========================================================================
    # PHASE 4b: Equity Evaluator (real LLM)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4b: Equity Evaluator (calling Groq LLM)")
    print("=" * 70)

    equity_result = equity_evaluator(state)
    state = state.model_copy(update=equity_result)

    er = state.equity_result
    if er:
        print(f"  Balanced:   {er.balanced}")
        print(f"  Reasoning:  {er.reasoning[:100]}...")
        if er.violations:
            print(f"  Violations: {er.violations}")
    print(f"  Retries:    {state.equity_retries}")
    print("  [OK] Equity evaluation complete")

    # =========================================================================
    # PHASE 5: Publishing (real Trello, optional Calendar/Docs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 5: Publishing (Trello + Calendar + Docs)")
    print("=" * 70)

    pub_result = await publishing(state)
    state = state.model_copy(update=pub_result)

    ps = state.publishing_status
    if ps:
        print(f"  Trello:   {ps.trello}")
        print(f"  Calendar: {ps.calendar}")
        print(f"  Docs:     {ps.docs}")
        if ps.errors:
            print(f"  Errors:")
            for err in ps.errors:
                print(f"    - {err}")

    if state.trello_board_id:
        print(f"\n  Trello board ID: {state.trello_board_id}")
        print(f"  Trello cards created: {len(state.trello_card_mapping)}")
        # Construct the board URL (standard Trello URL pattern)
        print(f"  Board URL: https://trello.com/b/{state.trello_board_id}")

    if state.calendar_event_mapping:
        print(f"  Calendar events created: {len(state.calendar_event_mapping)}")

    if state.docs_task_matrix_id:
        print(f"  Google Doc ID: {state.docs_task_matrix_id}")

    print("  [OK] Publishing complete")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Tasks decomposed:     {len(state.task_array)}")
    print(f"  Tasks delegated:      {len(state.delegation_matrix)}")
    print(f"  Equity balanced:      {er.balanced if er else 'N/A'}")
    print(f"  Trello board:         {state.trello_board_id or 'not created'}")
    print(f"  Calendar events:      {len(state.calendar_event_mapping)}")
    print(f"  Google Doc:           {state.docs_task_matrix_id or 'not created'}")
    print(f"  Webhook configured:   {state.webhook_configured}")

    if state.trello_board_id:
        print(f"\n  NOTE: A Trello board was created. Delete it when done:")
        print(f"  https://trello.com/b/{state.trello_board_id}")

    successes = []
    failures = []
    if ps:
        for svc in ["trello", "calendar", "docs"]:
            (successes if getattr(ps, svc) == "success" else failures).append(svc)

    print(f"\n  Succeeded: {', '.join(successes) if successes else 'none'}")
    print(f"  Failed:    {', '.join(failures) if failures else 'none'}")
    print("\n  Done!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_pipeline())
