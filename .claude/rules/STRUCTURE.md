## Project Structure
```
syncup/
├── CLAUDE.md
├── DockerFile                  # FastAPI, PostgreSQL, Redis, Next.js
├── .env
├── backend/src/
│   ├── state/                  # schema.py (SyncUp Pydantic), reducers.py
│   ├── agents/                 # supervisor.py, task_decomposition.py, delegation.py,
│   │                           # progress_tracking.py, conflict_resolution.py,
│   │                           # meeting_coordinator.py, publishing.py (deterministic)
│   ├── evaluators/             # equity_evaluator.py, tone_evaluator.py
│   ├── mcp/                    # client.py (MultiServerMCPClient), google_calendar.py,
│   │                           # google_docs.py, github.py
│   ├── integrations/           # trello.py (REST client), webhooks.py (handlers)
│   ├── guardrails/             # sanitizer.py, state_validator.py, nemo_rails.py
│   ├── graph/                  # main.py (StateGraph definition), routing.py (edges)
│   ├── api/                    # FastAPI app, routes/ (projects, webhooks, dashboard,
│   │                           # auth, onboarding), websockets.py
│   ├── models/                 # student.py, task.py, contribution.py, peer_review.py
│   └── services/               # pacing.py (burn-down), meeting_scheduler.py, report_generator.py
├── backend/tests/
│   ├── unit/                   # test_<module>.py for each module above
│   └── integration/            # test_graph_flow.py, test_mcp_connections.py, test_webhook_pipeline.py
├── frontend/
│   ├── app/
│   │   ├── student/            # onboarding/page.tsx, dashboard/page.tsx,
│   │   │                       # settings/page.tsx (availability updates), peer-review/page.tsx
│   │   └── professor/          # dashboard/page.tsx, team/[id]/page.tsx, review/page.tsx
│   ├── components/             # TaskBoard, BurndownChart, ContributionTable,
│   │                           # PeerReviewForm, OnboardingForm, AvailabilityEditor
│   └── lib/                    # api.ts (FastAPI client), websocket.ts
└── docs/                       # ARCHITECTURE.md, API.md, DEPLOYMENT.md
```