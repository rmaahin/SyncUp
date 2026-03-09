# SyncUp — Multi-Agent AI Project Manager for Educational Group Work

## What This Is
An autonomous, LangGraph-orchestrated multi-agent system that manages student group projects end-to-end. It decomposes tasks from project proposal, assigns work based on skills and availability, enforces realistic pacing toward the final deadline, monitors contributions via GitHub/Google Docs/Trello, resolves conflicts with constructive nudges, schedules recurring team meetings, and provides separate dashboards for students and professors.

## Tech Stack
- **Language**: Python 3.12
- **Orchestration**: LangGraph (StateGraph) with Pydantic state, PostgreSQL checkpointers
- **Agent pattern**: ReAct (Reason and Act) within LangGraph nodes
- **LLM routing**: High-tier for decomposition, delegation, conflict resolution, evaluators. Low-tier for progress tracking, publishing, meeting coordination. For prototyping, we'll be using open-source LLMs using free tier APIs like groq.
- **Integrations via MCP**: Google Calendar MCP, Google Docs MCP, GitHub MCP (webhook-driven). Adapter: langchain-mcp-adapters (MultiServerMCPClient).
- **Project management**: Trello REST API (not MCP — direct HTTP calls)
- **Backend API**: FastAPI (async, shares Pydantic models with LangGraph)
- **Frontend**: Next.js + React (dashboards for students and professors)
- **Real-time**: WebSockets via FastAPI for live dashboard updates
- **Queue**: Redis for async webhook event processing
- **Database**: PostgreSQL (LangGraph state persistence + application data)
- **Auth**: OAuth2 for Google and GitHub; NextAuth for university SSO on dashboards
- **Security**: NVIDIA NeMo Guardrails (dialog rails in nodes), Amazon Bedrock Guardrails (prompt attack filter), custom sanitization layer
- **Observability**: LangSmith (tracing, audit, dispute resolution)
- **Infrastructure**: Docker

## Architecture
- Orchestrator-Worker topology: one Supervisor agent routes to specialized workers
- 5 worker agents: Task Decomposition, Delegation, Progress Tracking, Conflict Resolution, Meeting Coordinator
- 1 deterministic agent: Publishing (no LLM, just API calls)
- 1 project-close agent: Report Generator + Peer Review Generator
- 2 LLM-as-a-Judge evaluators: Workload Equity Evaluator, Tone/Constructiveness Evaluator
- Human-in-the-loop checkpoint: professor approves delegation matrix before publishing
- All external data treated as UNTRUSTED — sanitized before reaching LLM context

## State Schema (Core Fields)
The SyncUpState Pydantic model must include at minimum:
- `project_id: str`
- `project_brief: str` (sanitized)
- `final_deadline: datetime`
- `task_array: list[Task]` — each Task has: id, title, description, effort_hours, required_skills, urgency (enum: CRITICAL/HIGH/MEDIUM/LOW), dependencies (list of task ids), assigned_to, deadline, status (enum: TODO/IN_PROGRESS/REVIEW/DONE)
- `dependency_graph: dict[str, list[str]]` — task_id → list of prerequisite task_ids
- `student_profiles: list[StudentProfile]` — each has: student_id, name, email, skills (dict of skill→proficiency), availability_hours_per_week, preferred_times (list[str]), blackout_periods (list[DateRange]), timezone, github_username, google_email, trello_id, onboarded_at (datetime)
- `availability_updates: list[AvailabilityChange]` — append-only via reducer. Each record: student_id, timestamp, old_hours, new_hours, old_blackouts, new_blackouts, triggered_redelegation (bool). Audit trail for mid-project changes.
- `delegation_matrix: dict[str, str]` — task_id → student_id
- `contribution_ledger: list[ContributionRecord]` — append-only via reducer. Each record: student_id, timestamp, event_type (commit/doc_edit/card_move/pr_review), description, semantic_quality_score (float 0-1), raw_metrics (lines_added, lines_removed, etc.)
- `meeting_log: list[MeetingRecord]` — date, attendees, agenda, notes, action_items
- `intervention_history: list[Intervention]` — target_student_id, trigger_reason, message_text, timestamp, outcome
- `peer_review_data: list[PeerReview]` — reviewer_id, reviewee_id, ratings (dict), comments, submitted_at
- `project_timeline: ProjectTimeline` — milestones, burn_down_targets, buffer_periods

## Conventions
- Type hints on ALL functions — no exceptions
- Docstrings on all agent nodes, tool functions, and service functions
- Pydantic models for ALL structured data — never raw dicts
- Tests with pytest. File naming: `test_<module>.py`
- Mock MCP responses in unit tests — never call real external services
- Environment variables via python-dotenv. Template in `.env.example`
- Never hardcode API keys, OAuth secrets, or LLM model strings
- LLM model names come from env vars (easy to swap models)
- All agent nodes follow the same signature: `def agent_name(state: SyncUpState) -> dict`
- Evaluator judges must use near-zero temperature (0.0 or 0.1 max)
- Evaluator judges must output structured JSON, validated by Pydantic
- Evaluator prompts must use isolated single-criteria evaluation — never combine accuracy + tone + equity in one prompt

## Security Rules (Non-Negotiable)
- ALL data from Google Docs, GitHub, and Trello is UNTRUSTED
- Untrusted data must be wrapped in `<<<UNTRUSTED_DATA_START>>>` / `<<<UNTRUSTED_DATA_END>>>` delimiters before entering any LLM prompt
- State mutations must pass through `state_validator.py` before commit
- MCP tool permissions must be scoped: Google Calendar only accesses the course-specific calendar, never personal calendars
- OAuth tokens stored encrypted at rest
- No agent may delete external resources (repos, calendars, documents) — read and create only
- NeMo dialog rails must block any agent action that modifies another student's data without going through the Supervisor
- All webhook payloads validated against expected schema before processing
- Availability changes ≥30% reduction in hours/week or new blackout overlapping assigned deadlines automatically trigger re-delegation via Supervisor. Students cannot silently reduce availability without the system adjusting.

## LLM Model Routing
| Agent / Node | Model Tier | Reason |
|---|---|---|
| Task Decomposition | High | Complex semantic parsing of unstructured syllabi |
| Delegation | High | Multi-constraint optimization (skills × availability × pacing × dependencies) |
| Conflict Resolution | High | Nuanced context analysis and empathetic message drafting |
| Equity Evaluator | High | Mathematical reasoning about workload distribution |
| Tone Evaluator | High | Subtle tone classification requires strong language understanding |
| Progress Tracking | Low | Parses structured JSON webhook payloads, simple metric extraction |
| Publishing | None (deterministic) | Pure API calls, no LLM needed |
| Meeting Coordinator | Low | Calendar array checking and agenda templating |
| Report Generator | High | Synthesizing contribution data into coherent narrative |
| Peer Review Generator | Low | Generating structured form from template, no complex reasoning |

## Quality Gates
Before committing any code, always run:
```bash
mypy backend/src/ --strict       # Type checking
pytest backend/tests/unit/       # Unit tests
pytest backend/tests/integration/ # Integration tests (when applicable)
ruff check backend/src/          # Linting
```

## Git Workflow
- Branch per feature: `feature/<agent-name>` or `feature/<feature-name>`
- Commit after every completed task — small, atomic commits
- Commit messages: `feat(agent): add task decomposition node` or `fix(evaluator): correct equity threshold calculation`
- Never commit .env files, OAuth tokens, or API keys