"""
FastAPI application factory for pipeline ops dashboard API.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="S4 Pipeline Ops",
        description="GPU monitoring, job scheduling, and pipeline health dashboard",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.routes import router
    app.include_router(router)

    # Multi-node agent endpoints (each node serves these)
    from src.multinode.agent import router as agent_router
    app.include_router(agent_router)

    return app
