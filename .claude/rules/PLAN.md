## Build Order (Recommended Implementation Phases)
1. State schema (including StudentProfile, AvailabilityChange models) + graph skeleton (placeholder nodes that compile)
2. MCP client configuration + Trello API client + **student onboarding API routes** (profile CRUD, OAuth flows)
3. Task Decomposition agent + tests
4. Pacing service + Delegation agent + Equity Evaluator + tests
5. Publishing agent (Trello + Calendar + Docs) + tests
6. Progress Tracking agent + webhook handlers + tests
7. Conflict Resolution agent + Tone Evaluator + tests
8. Meeting Coordinator agent + tests
9. Guardrails layer (sanitizer + state validator + NeMo rails) + adversarial tests
10. FastAPI routes (remaining: dashboard, websockets) + availability update endpoint + re-delegation trigger logic
11. Frontend: **onboarding UI first** (form + OAuth linking), then student dashboard, then student settings (availability editor), then professor dashboard
12. Peer Review Generator + Report Generator
13. Integration tests across the full graph