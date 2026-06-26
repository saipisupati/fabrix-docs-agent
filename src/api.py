"""
api.py, FastAPI wrapper so the docs site widget can call the agent.

Run: uvicorn src.api:app --port 8080
POST /ask with {"question": "..."} → answer + source links. Keys stay on the server.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(__file__))

from agent import answer
from config import QDRANT_DIR

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
    sources: list[SourceResponse]


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
    allow_headers=["*"],
)


def _check_api_key(x_api_key: Optional[str]):
    # optional; set API_KEY env to require X-API-Key header
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceResponse(title=s["title"], url=s["url"], excerpt=s["excerpt"])
            for s in result.sources
        ],
    )
