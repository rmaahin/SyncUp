"""Live end-to-end test — Phase 1 through 8 with REAL services.

This script calls each agent node in sequence using:
  - Real Groq LLM (task decomposition, delegation, equity evaluation,
    progress tracking, conflict resolution, tone evaluation,
    meeting agenda generation, meeting notes parsing)
  - Real Trello API (creates a board with cards, posts intervention comments)
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

import httpx

from agents.conflict_resolution import conflict_resolution
from agents.delegation import delegation
from agents.deliver import deliver
from agents.meeting_coordinator import meeting_coordinator
from agents.progress_tracking import progress_tracking
from agents.publishing import publishing
from agents.task_decomposition import task_decomposition
from evaluators.equity_evaluator import equity_evaluator
from evaluators.tone_evaluator import tone_evaluator
from services.meeting_scheduler import find_optimal_meeting_slot, generate_recurring_schedule
from state.schema import ContributionRecord, DateRange, EventType, StudentProfile, SyncUpState, TaskStatus

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
Build a single-page "Hello World" landing page. \
2-person team, 2-week timeline. Keep this VERY small — only 3 tasks max.

Tasks:
1. Write the HTML/CSS landing page (index.html + style.css).
2. Write a Python FastAPI backend that serves the page.
3. Deploy to a shared server.

That is the ENTIRE project — exactly 3 tasks, nothing more.
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
            preferred_times=["Mon 14:00-16:00", "Wed 14:00-16:00", "Fri 10:00-12:00"],
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
            preferred_times=["Mon 14:00-16:00", "Wed 10:00-12:00", "Thu 14:00-16:00"],
            timezone="America/New_York",
            github_username="bob-dev",
            google_email="bob@example.com",
            trello_id="",
            # Active blackout: exam week — triggers extend_deadline bias
            blackout_periods=[
                DateRange(
                    start=datetime.now(tz=timezone.utc) - timedelta(days=1),
                    end=datetime.now(tz=timezone.utc) + timedelta(days=4),
                )
            ],
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
    """Execute the full Phase 1-8 pipeline with real services."""
    print("=" * 70)
    print("  SyncUp Live E2E Test — Phases 1 through 8")
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
    # PHASE 6: Progress Tracking (real low-tier LLM)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 6: Progress Tracking (calling Groq low-tier LLM)")
    print("=" * 70)

    # --- 6a: Simulate a GitHub push event from Alice ---
    print("\n  --- 6a: Simulating GitHub push from Alice ---")
    first_task = state.task_array[0] if state.task_array else None
    push_event = {
        "event_type": "github_push",
        "github_username": "alice-dev",
        "repository_full_name": "team/campuseats",
        "commits": [
            {
                "sha": "a1b2c3d",
                "message": f"feat: implement {first_task.title if first_task else 'initial setup'}",
                "added": ["src/main.py", "src/models.py"],
                "removed": [],
                "modified": ["README.md"],
                "diff_summary": (
                    "+from fastapi import FastAPI\n"
                    "+app = FastAPI()\n"
                    "+\n"
                    "+@app.get('/health')\n"
                    "+def health():\n"
                    "+    return {'status': 'ok'}\n"
                ),
            }
        ],
        "timestamp": now.isoformat(),
    }

    state = state.model_copy(update={"pending_event": push_event})
    push_result = progress_tracking(state)
    state = state.model_copy(update=push_result)

    if state.contribution_ledger:
        rec = state.contribution_ledger[-1]
        print(f"  Event type:      {rec.event_type.value}")
        print(f"  Student:         {student_map.get(rec.student_id, rec.student_id)}")
        print(f"  Quality score:   {rec.semantic_quality_score:.2f}")
        print(f"  Description:     {rec.description[:80]}")
        print(f"  Raw metrics:     +{rec.raw_metrics.lines_added} -{rec.raw_metrics.lines_removed} files={rec.raw_metrics.files_changed}")
    print("  [OK] GitHub push event processed")

    # --- 6b: Simulate a Trello card move (if we have card mappings) ---
    if state.trello_card_mapping:
        print("\n  --- 6b: Simulating Trello card move ---")
        # Pick the first task's card
        first_task_id = list(state.trello_card_mapping.keys())[0]
        first_card_id = state.trello_card_mapping[first_task_id]
        task_name = next(
            (t.title for t in state.task_array if t.id == first_task_id),
            "Unknown task",
        )

        trello_event = {
            "event_type": "trello_card_update",
            "trello_card_id": first_card_id,
            "trello_card_name": task_name,
            "trello_list_before": "To Do",
            "trello_list_after": "In Progress",
            "timestamp": now.isoformat(),
        }

        state = state.model_copy(update={"pending_event": trello_event})
        trello_result = progress_tracking(state)
        state = state.model_copy(update=trello_result)

        if len(state.contribution_ledger) >= 2:
            rec = state.contribution_ledger[-1]
            print(f"  Event type:      {rec.event_type.value}")
            print(f"  Student:         {student_map.get(rec.student_id, rec.student_id)}")
            print(f"  Quality score:   {rec.semantic_quality_score:.2f} (neutral for card moves)")
            print(f"  Description:     {rec.description[:80]}")
        print("  [OK] Trello card move event processed")
    else:
        print("\n  --- 6b: Skipping Trello card move (no card mappings) ---")

    # --- 6c: Simulate a GitHub PR event from Bob ---
    print("\n  --- 6c: Simulating GitHub PR from Bob ---")
    pr_event = {
        "event_type": "github_pr",
        "github_username": "bob-dev",
        "repository_full_name": "team/campuseats",
        "pr_title": "Add frontend HTML/CSS layout",
        "pr_action": "opened",
        "pr_number": 1,
        "pr_files_changed": 4,
        "timestamp": now.isoformat(),
    }

    state = state.model_copy(update={"pending_event": pr_event})
    pr_result = progress_tracking(state)
    state = state.model_copy(update=pr_result)

    if state.contribution_ledger:
        rec = state.contribution_ledger[-1]
        print(f"  Event type:      {rec.event_type.value}")
        print(f"  Student:         {student_map.get(rec.student_id, rec.student_id)}")
        print(f"  Quality score:   {rec.semantic_quality_score:.2f}")
        print(f"  Description:     {rec.description[:80]}")
    print("  [OK] GitHub PR event processed")

    # --- 6d: Progress evaluation summary ---
    print("\n  --- 6d: Student progress evaluation ---")
    if state.student_progress:
        for sid, status in state.student_progress.items():
            name = student_map.get(sid, sid)
            print(f"    {name}: {status}")
    else:
        print("    No progress data (no tasks with deadlines yet)")

    print(f"\n  Contribution ledger: {len(state.contribution_ledger)} records total")
    print(f"  Pending event:      {state.pending_event}  (should be None)")
    print("  [OK] Progress tracking complete")

    # =========================================================================
    # PHASE 7: Conflict Resolution + Tone Evaluation + Delivery
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 7: Conflict Resolution → Tone Evaluation → Delivery")
    print("=" * 70)

    # --- 7a: Force Bob into "behind" status ---
    print("\n  --- 7a: Forcing Bob into 'behind' status ---")

    # Make Bob's first assigned task overdue by setting deadline to 1 day ago.
    # We store the original deadline so the deliver node can respect it when
    # computing the extension (max(original, now) + 3 days).
    overdue_tasks_updated = []
    bob_overdue_task_title = None
    bob_original_deadline = None
    for t in state.task_array:
        if state.delegation_matrix.get(t.id) == "s-2" and bob_overdue_task_title is None:
            bob_original_deadline = t.deadline
            # Set deadline to yesterday to trigger overdue detection
            updated = t.model_copy(
                update={
                    "deadline": now - timedelta(days=1),
                    "status": TaskStatus.TODO,
                }
            )
            overdue_tasks_updated.append(updated)
            bob_overdue_task_title = t.title
            print(f"  Task:              '{t.title}'")
            print(f"  Original deadline: {bob_original_deadline.strftime('%Y-%m-%d %H:%M') if bob_original_deadline else 'none'}")
            print(f"  Forced to:         {updated.deadline.strftime('%Y-%m-%d %H:%M')} (simulated overdue)")
        else:
            overdue_tasks_updated.append(t)

    if bob_overdue_task_title:
        state = state.model_copy(
            update={
                "task_array": overdue_tasks_updated,
                "student_progress": {**state.student_progress, "s-2": "behind"},
            }
        )
        print(f"  Bob's status:      {state.student_progress.get('s-2', 'unknown')}")
        bob_profile = next((s for s in state.student_profiles if s.student_id == "s-2"), None)
        if bob_profile and bob_profile.blackout_periods:
            bp = bob_profile.blackout_periods[0]
            print(f"  Bob's blackout:    {bp.start.strftime('%Y-%m-%d')} to {bp.end.strftime('%Y-%m-%d')} (exam week)")
    else:
        print("  WARN: No task assigned to Bob — skipping Phase 7")

    if state.student_progress.get("s-2") == "behind":
        # --- 7b: Conflict Resolution (real high-tier LLM) ---
        print("\n  --- 7b: Conflict Resolution (calling Groq high-tier LLM) ---")
        cr_result = conflict_resolution(state)
        state = state.model_copy(update=cr_result)

        draft = state.draft_intervention
        if draft:
            print(f"  Target:           {draft.target_student_id}")
            print(f"  Severity:         {draft.severity}")
            print(f"  Suggested action: {draft.suggested_action}")
            print(f"  Affected:         {draft.affected_teammates}")
            print(f"  Message:")
            for line in draft.message.split(". "):
                print(f"    {line.strip()}")
        else:
            print("  WARN: No draft intervention generated")

        # --- 7c: Tone Evaluation (real high-tier LLM at temp=0.0) ---
        print("\n  --- 7c: Tone Evaluation (calling Groq high-tier LLM, temp=0.0) ---")
        te_result = tone_evaluator(state)
        state = state.model_copy(update=te_result)

        tone = state.tone_result
        if tone:
            print(f"  Classification:   {tone.classification}")
            print(f"  Reasoning:        {tone.reasoning}")
            if tone.flagged_phrases:
                print(f"  Flagged phrases:  {tone.flagged_phrases}")
        else:
            print("  WARN: No tone result returned")

        # --- 7d: Handle result ---
        if tone and tone.classification == "punitive":
            print("\n  --- 7d: Message flagged as PUNITIVE — attempting one rewrite ---")
            state = state.model_copy(
                update={"tone_rewrite_count": state.tone_rewrite_count + 1}
            )

            # Rewrite
            cr_result2 = conflict_resolution(state)
            state = state.model_copy(update=cr_result2)
            draft2 = state.draft_intervention
            if draft2:
                print(f"  Rewritten message:")
                for line in draft2.message.split(". "):
                    print(f"    {line.strip()}")

            # Re-evaluate tone
            te_result2 = tone_evaluator(state)
            state = state.model_copy(update=te_result2)
            tone2 = state.tone_result
            if tone2:
                print(f"  Re-evaluation:    {tone2.classification}")

        # --- 7e: Deliver (posts Trello comment if board exists) ---
        if state.draft_intervention:
            print("\n  --- 7e: Delivering intervention ---")
            del_result = await deliver(state)
            state = state.model_copy(update=del_result)

            if state.intervention_history:
                last_intervention = state.intervention_history[-1]
                print(f"  Intervention logged: {last_intervention.trigger_reason}")
                print(f"  Outcome:            {last_intervention.outcome}")
                print(f"  Timestamp:          {last_intervention.timestamp}")
            print(f"  Draft cleared:      {state.draft_intervention is None}")
            print(f"  Rewrite count:      {state.tone_rewrite_count}")

            # Show action side effects
            if "task_array" in del_result:
                print(f"\n  --- Deadline extended! ---")
                print(f"  NOTE: In this test, the deadline was artificially set to the past.")
                print(f"  The system uses max(original_deadline, now) + {3} days, then avoids blackouts.")
                for t in state.task_array:
                    if state.delegation_matrix.get(t.id) == "s-2":
                        new_dl_str = t.deadline.strftime('%Y-%m-%d %H:%M') if t.deadline else 'none'
                        print(f"  Task '{t.title}' new deadline: {new_dl_str}")
            if "needs_redelegation" in del_result:
                print(f"\n  --- Tasks flagged for re-delegation: {del_result['needs_redelegation']} ---")

            print("  [OK] Intervention delivered")
        else:
            print("\n  --- 7e: No draft to deliver (was cleared) ---")

    print("  [OK] Phase 7 complete")

    # =========================================================================
    # PHASE 8: Meeting Coordinator (real low-tier LLM + optional MCP)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 8: Meeting Coordinator (scheduling + agenda generation)")
    print("=" * 70)

    # --- 8a: Meeting Scheduler service (pure Python, no LLM) ---
    print("\n  --- 8a: Finding optimal meeting slot ---")
    slot = find_optimal_meeting_slot(
        student_profiles=state.student_profiles,
        duration_minutes=60,
        earliest=now + timedelta(days=1),
        latest=now + timedelta(days=7),
    )
    if slot:
        print(f"  Best slot found:  {slot.strftime('%A %Y-%m-%d %H:%M %Z')}")
        # Show how the slot aligns with preferences
        for sp in state.student_profiles:
            prefs = ", ".join(sp.preferred_times) if sp.preferred_times else "none"
            print(f"    {sp.name} preferences: {prefs}")
    else:
        print("  No slot found (all students busy or blackout conflict)")

    # --- 8b: Generate recurring schedule ---
    print("\n  --- 8b: Generating recurring meeting schedule ---")
    if slot:
        schedule = generate_recurring_schedule(
            first_meeting=slot,
            interval_days=state.meeting_interval_days,
            project_end=state.final_deadline or now + timedelta(days=14),
        )
        print(f"  Recurring meetings ({state.meeting_interval_days}-day interval):")
        for i, dt in enumerate(schedule):
            print(f"    {i + 1}. {dt.strftime('%A %Y-%m-%d %H:%M %Z')}")
        print(f"  Total meetings planned: {len(schedule)}")
    else:
        print("  Skipping (no initial slot found)")

    # --- 8c: Schedule mode — full meeting coordinator agent ---
    print("\n  --- 8c: Meeting Coordinator — Schedule mode ---")
    print("  (Calls low-tier LLM for agenda generation, MCP for calendar/docs)")
    state = state.model_copy(update={"meeting_mode": "schedule"})
    mc_result = await meeting_coordinator(state)
    state = state.model_copy(update=mc_result)

    if state.next_meeting_scheduled:
        print(f"  Meeting scheduled: {state.next_meeting_scheduled.strftime('%A %Y-%m-%d %H:%M %Z')}")
    else:
        print("  No meeting scheduled (no available slot or MCP unavailable)")

    if state.meeting_log:
        last_meeting = state.meeting_log[-1]
        print(f"  Attendees:        {last_meeting.attendees}")
        print("  Agenda preview:")
        # Show first few lines of the agenda
        for line in last_meeting.agenda.split("\n")[:8]:
            if line.strip():
                print(f"    {line}")
        if len(last_meeting.agenda.split("\n")) > 8:
            print("    ...")

    if state.meeting_notes_doc_ids:
        print(f"  Meeting notes doc: {state.meeting_notes_doc_ids[-1]}")
    else:
        print("  Meeting notes doc: not created (Google Docs MCP not configured)")

    # --- 8d: Ingest mode — simulate meeting notes ---
    print("\n  --- 8d: Meeting Coordinator — Ingest mode ---")
    if state.meeting_notes_doc_ids:
        print("  (Reading meeting notes doc via MCP + parsing with low-tier LLM)")
        state = state.model_copy(update={"meeting_mode": "ingest"})
        ingest_result = await meeting_coordinator(state)
        state = state.model_copy(update=ingest_result)

        if len(state.meeting_log) >= 2:
            ingested = state.meeting_log[-1]
            print(f"  Notes summary:    {ingested.notes[:120]}{'...' if len(ingested.notes) > 120 else ''}")
            print(f"  Action items:     {ingested.action_items}")
            print(f"  Attendees parsed: {ingested.attendees}")
        print("  [OK] Meeting notes ingested")
    else:
        print("  Skipping ingest — no meeting notes doc to read")
        print("  (Google Docs MCP not configured; this is expected without MCP)")

    print(f"\n  Meeting log entries: {len(state.meeting_log)}")
    print("  [OK] Phase 8 complete")

    # =========================================================================
    # PHASE 6 LIVE: Bootstrap state + register Trello webhook
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 6 LIVE: Bootstrap state into API + register Trello webhook")
    print("=" * 70)

    # Set github_repo on state for webhook routing
    state = state.model_copy(update={"github_repo": "team/campuseats"})

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    project_id = state.project_id

    # Try to bootstrap state into the running FastAPI server
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            # Check if backend is reachable
            health = await http.get(f"{backend_url}/health")
            if health.status_code != 200:
                print("  Backend not reachable — skipping live webhook setup")
                raise ConnectionError()

            # Bootstrap state into the in-memory store
            print(f"  Bootstrapping state for project '{project_id}'...")
            resp = await http.post(
                f"{backend_url}/api/projects/{project_id}/state",
                json=state.model_dump(mode="json"),
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  [OK] State bootstrapped: {data.get('tasks', 0)} tasks, "
                      f"{data.get('students', 0)} students")
            else:
                print(f"  WARN: Bootstrap failed: {resp.status_code} {resp.text[:100]}")

            # Try to detect ngrok public URL
            # Inside Docker, ngrok is at "ngrok:4040"; outside, it's "localhost:4040"
            ngrok_url = None
            ngrok_api_urls = ["http://ngrok:4040/api/tunnels", "http://localhost:4040/api/tunnels"]
            for ngrok_api_url in ngrok_api_urls:
                try:
                    tunnels_resp = await http.get(ngrok_api_url)
                except Exception:
                    continue
                if tunnels_resp.status_code == 200:
                    tunnels = tunnels_resp.json().get("tunnels", [])
                    for tunnel in tunnels:
                        if tunnel.get("proto") == "https":
                            ngrok_url = tunnel["public_url"]
                            break
                    if not ngrok_url and tunnels:
                        ngrok_url = tunnels[0].get("public_url")
                if ngrok_url:
                    break

            if ngrok_url:
                print(f"  Detected ngrok URL: {ngrok_url}")

                # Register Trello webhook
                print("  Registering Trello webhook...")
                reg_resp = await http.post(
                    f"{backend_url}/api/projects/{project_id}/webhooks/register",
                    params={"ngrok_url": ngrok_url},
                )
                if reg_resp.status_code == 200:
                    reg_data = reg_resp.json()
                    trello_wh = reg_data.get("webhooks", {}).get("trello", {})
                    if "webhook_id" in trello_wh:
                        print(f"  [OK] Trello webhook registered: {trello_wh['webhook_id']}")
                        print(f"       Callback: {trello_wh['callback_url']}")
                        print("\n  >>> Move a card on your Trello board now! <<<")
                        print("  >>> The backend will process it and log the contribution. <<<")
                    else:
                        print(f"  WARN: Trello webhook failed: {trello_wh.get('error', 'unknown')}")
                else:
                    print(f"  WARN: Registration failed: {reg_resp.status_code}")
            else:
                print("  ngrok not detected (is the ngrok service running?)")
                print("  Skipping live webhook registration.")
                print("  To set up manually:")
                print("    1. Run: docker compose up ngrok")
                print("    2. Get URL from: http://localhost:4040")
                print(f"    3. Call: POST {backend_url}/api/projects/{project_id}/webhooks/register?ngrok_url=<URL>")

    except (httpx.ConnectError, ConnectionError):
        print("  Backend not reachable — skipping live webhook setup")
        print("  (This is normal when running the script outside Docker)")

    print("  [OK] Live webhook setup complete")

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
    print(f"  Contributions logged: {len(state.contribution_ledger)}")
    print(f"  Interventions sent:   {len(state.intervention_history)}")
    print(f"  Meetings logged:      {len(state.meeting_log)}")
    print(f"  Next meeting:         {state.next_meeting_scheduled.strftime('%Y-%m-%d %H:%M') if state.next_meeting_scheduled else 'none'}")
    print(f"  Meeting docs:         {len(state.meeting_notes_doc_ids)}")
    print(f"  Student progress:     {state.student_progress or 'N/A'}")

    if state.intervention_history:
        last = state.intervention_history[-1]
        print(f"\n  Last intervention:")
        print(f"    Target:  {last.target_student_id}")
        print(f"    Reason:  {last.trigger_reason}")
        print(f"    Action:  {last.outcome}")
        msg_preview = last.message_text[:100]
        print(f"    Message: {msg_preview}{'...' if len(last.message_text) > 100 else ''}")

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
