"""FastAPI entry point.

    uvicorn backend.app.main:app --reload
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from .db import SessionLocal, init_db
from .models import MLModel
from .routers import incidents, knowledge, monitoring

_seed_lock = threading.Lock()


def _ensure_seeded() -> None:
    init_db()
    with SessionLocal() as db:
        empty = db.scalars(select(MLModel)).first() is None
    if empty and os.getenv("AUTO_SEED", "1") == "1":
        from .seed import seed

        seed()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_seeded()
    yield


app = FastAPI(
    title="ML Incident Copilot",
    description="Monitoring → incident workflow → explainable diagnosis → actions → two-audience summaries → "
                "feedback loop, for ML models running at a client.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(monitoring.router)
app.include_router(incidents.router)
app.include_router(knowledge.router)


@app.post("/api/admin/reseed", tags=["admin"])
def reseed() -> dict:
    """Rebuild the demo environment so the data ends 'now' (handy right before recording a demo)."""
    if not _seed_lock.acquire(blocking=False):
        raise HTTPException(409, "A reseed is already running.")
    try:
        from .seed import seed

        return seed(verbose=False)
    finally:
        _seed_lock.release()


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "ML Incident Copilot API", "docs": "/docs"}
