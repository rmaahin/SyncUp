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

app.include_router(onboarding_router, prefix="/api/onboarding", tags=["onboarding"])


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
