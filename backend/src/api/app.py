"""FastAPI application for SyncUp.

Sets up CORS, MCP client lifespan, health check, and API routers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcp_layer.client import SyncUpMCPClient
from state.store import InMemoryStateStore

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage MCP client lifecycle — connect on startup, disconnect on shutdown."""
    mcp_client = SyncUpMCPClient()
    try:
        await mcp_client.connect()
        application.state.mcp_client = mcp_client
    except Exception:
        logger.exception("MCP client failed to initialize — app will run without MCP")
        application.state.mcp_client = None

    # In-memory state store (replaced with PostgreSQL in Phase 10)
    application.state.state_store = InMemoryStateStore()
    application.state.board_project_map: dict[str, str] = {}  # trello_board_id → project_id
    application.state.repo_project_map: dict[str, str] = {}   # github repo full_name → project_id

    yield

    if getattr(application.state, "mcp_client", None) is not None:
        await application.state.mcp_client.disconnect()


app = FastAPI(
    title="SyncUp",
    description="Multi-Agent AI Project Manager for Educational Group Work",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
_frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from api.routes.onboarding import router as onboarding_router  # noqa: E402
from api.routes.projects import router as projects_router  # noqa: E402
from api.routes.webhooks import router as webhooks_router  # noqa: E402

app.include_router(onboarding_router, prefix="/api/onboarding", tags=["onboarding"])
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(webhooks_router, prefix="/api/webhooks", tags=["webhooks"])


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
