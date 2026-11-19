"""HTTP API.

    uvicorn jobmatch.api:app --reload
    curl "localhost:8000/search?q=backend+engineer&location=india"

The index and the embedding model load once at startup, not per request:
loading them per call would add seconds to every query.
"""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from .agent import Agent
from .embed import DEFAULT_MODEL as EMBED_MODEL
from .embed import get_embedder
from .index import open_store
from .llm import GroqLLM
from .search import Filters, Searcher

state: dict = {}


def build_state() -> dict:
    """Everything the app needs, built once. Tests replace this to swap in a
    temporary index and a stub embedder."""
    conn = db.connect()
    return {"conn": conn, "agents": {},
            "searcher": Searcher(conn, open_store("numpy", EMBED_MODEL),
                                 get_embedder(EMBED_MODEL))}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.update(build_state())
    yield
    state["conn"].close()
    state.clear()


app = FastAPI(title="jobmatch", version="0.1.0", lifespan=lifespan)

STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = "default"


@app.get("/health")
def health() -> dict:
    """Readiness, not just liveness: an empty index means this instance is up
    but cannot answer anything, and that should not look healthy."""
    try:
        stats = db.stats(state["conn"])
        vectors = state["searcher"].store.count()
    except Exception as exc:
        raise HTTPException(503, f"not ready: {exc}")
    ready = stats["active"] > 0 and vectors > 0
    if not ready:
        raise HTTPException(503, "index is empty")
    return {"status": "ok", "jobs": stats["active"], "vectors": vectors}


@app.get("/stats")
def stats() -> dict:
    return db.stats(state["conn"])


@app.get("/search")
def search(
    q: str = Query(min_length=1, max_length=400),
    k: int = Query(10, ge=1, le=50),
    mode: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$"),
    location: str | None = None,
    remote: bool | None = None,
    seniority: list[str] = Query(default=[]),
    exclude_seniority: list[str] = Query(default=[]),
    max_years: int | None = Query(None, ge=0, le=25),
    diverse: int | None = Query(None, ge=1, le=10),
) -> dict:
    filters = Filters(location=location, remote=remote, seniority=seniority,
                      exclude_seniority=exclude_seniority, max_years=max_years)
    hits = state["searcher"].search(q, k=k, mode=mode, filters=filters,
                                    max_per_company=diverse)
    return {"query": q, "mode": mode, "count": len(hits), "results": [
        {"job_id": h.job_id, "title": h.job.title, "company": h.job.company,
         "location": h.job.location_canonical, "remote": h.job.remote,
         "seniority": h.job.seniority, "min_years": h.job.min_years,
         "url": h.job.url,
         "found_by": h.found_by} for h in hits]}


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    found = db.get_job(state["conn"], job_id)
    if not found:
        raise HTTPException(404, "no such job")
    return found.model_dump()


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Conversation state is per user and in memory, so it does not survive a
    restart or span replicas. Preferences do: those are in SQLite."""
    agents = state["agents"]
    if request.user_id not in agents:
        try:
            llm = GroqLLM()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        agents[request.user_id] = Agent(state["conn"], state["searcher"], llm,
                                        user_id=request.user_id)
    try:
        result = agents[request.user_id].run(request.message)
    except Exception as exc:
        raise HTTPException(502, f"model call failed: {exc}")
    return {"reply": result.reply, "grounded": result.grounded,
            "fell_back": result.fell_back, "remembered": result.remembered,
            "tool_calls": [{"tool": s.tool, "error": s.error} for s in result.steps]}
