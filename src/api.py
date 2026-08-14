"""
api.py, FastAPI wrapper so the docs site widget can call the agent.

Run: uvicorn src.api:app --port 8080
POST /ask with {"question": "..."} → answer + sources/examples/gaps/used_inference.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(__file__))

from agent import answer
from config import QDRANT_DIR
from freshness import (
    load_kb_status,
    qdrant_lock_held,
    retire_source,
    unretire_source,
    utc_now,
    write_freshness_status,
)

DOCS_SITE_ORIGIN = os.environ.get("DOCS_SITE_ORIGIN", "*")
API_KEY = os.environ.get("API_KEY")


class AskRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    title: str
    url: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    scope: str = "in_scope"
    used_inference: bool = False
    timing: dict = Field(default_factory=dict)
    # Agent-loop visibility for chat UI / demos
    plan_facets: list[str] = Field(default_factory=list)
    plan_queries: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app):
    # one Qdrant file handle for the process; same path as CLI agent
    app.state.qdrant = QdrantClient(path=QDRANT_DIR)
    yield
    app.state.qdrant.close()


app = FastAPI(title="Fabrix Docs Agent", lifespan=lifespan)

_cors_origins = ["*"] if DOCS_SITE_ORIGIN == "*" else [DOCS_SITE_ORIGIN]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "Content-Type"],
)


def _check_api_key(x_api_key: Optional[str]):
    # optional; set API_KEY env to require X-API-Key header
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _require_admin_key(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY not configured for admin")
    _check_api_key(x_api_key)


class SourceBody(BaseModel):
    source: str


class RefreshBody(BaseModel):
    force: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    # thin HTTP layer over agent.answer()
    _check_api_key(x_api_key)
    result = answer(body.question, client=request.app.state.qdrant)
    # Sources only when grounded; already empty on abstain in agent
    sources = result.sources if result.sources else []
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceResponse(title=s["title"], url=s["url"], excerpt=s["excerpt"])
            for s in sources
        ],
        examples=list(result.examples or []),
        gaps=list(result.gaps or []),
        scope=result.scope or "in_scope",
        used_inference=bool(result.used_inference),
        timing=dict(result.timing or {}),
        plan_facets=list(getattr(result, "plan_facets", None) or []),
        plan_queries=list(getattr(result, "plan_queries", None) or []),
    )


@app.get("/admin/kb-status")
def kb_status(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_admin_key(x_api_key)
    return load_kb_status(request.app.state.qdrant)


@app.post("/admin/retire")
def admin_retire(
    body: SourceBody,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_admin_key(x_api_key)
    src = (body.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    return retire_source(src)


@app.post("/admin/unretire")
def admin_unretire(
    body: SourceBody,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_admin_key(x_api_key)
    src = (body.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    return unretire_source(src)


@app.post("/admin/refresh")
def admin_refresh(
    body: RefreshBody = RefreshBody(),
    x_api_key: Optional[str] = Header(default=None),
):
    _require_admin_key(x_api_key)
    locked = qdrant_lock_held(QDRANT_DIR)
    import subprocess

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cmd = [sys.executable, os.path.join(root, "scripts", "run_freshness_pipeline.py")]
    if body.force:
        cmd.append("--force")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(root, "src")
    write_freshness_status(
        {"running": True, "forced": bool(body.force), "started_at": utc_now()}
    )
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    out = {"accepted": True, "pid": proc.pid, "force": bool(body.force)}
    if locked:
        out["warning"] = (
            "Qdrant lock is held (this API). Scrape/audit can run; "
            "ingest/build_kb will fail until uvicorn is stopped."
        )
    return out
