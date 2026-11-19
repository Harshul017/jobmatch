"""The tools the agent may call.

Each one declares its arguments as a Pydantic model, which does double duty:
it generates the JSON schema sent to the model, and it validates what comes
back. A small open-weights model will sometimes invent a field or send a
string where a list belongs, and that should be a correctable error rather
than a crash.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from . import db
from .models import Preference
from .search import Filters, Searcher

MAX_RESULTS = 8
MAX_PER_COMPANY = 2   # a shortlist from one employer is not a shortlist

# Every tool result is replayed to the model on every subsequent turn, so the
# context -- and the latency -- grows with the square of how chatty they are.
# These caps are the difference between a fast loop and a slow one.
SNIPPET_CHARS = 220
DESCRIPTION_CHARS = 1400


class SearchJobsArgs(BaseModel):
    query: str = Field(description="What kind of role to look for, in plain words")
    location: str | None = Field(default=None, description="City or country, e.g. 'Bengaluru' or 'India'")
    remote: bool | None = Field(default=None, description="Only remote roles")
    seniority: list[str] = Field(
        default_factory=list,
        description="Only these levels. Any of: intern, junior, mid, senior, "
                    "staff, principal, lead, management. 'mid' means a title "
                    "with no seniority marker at all.")
    exclude_seniority: list[str] = Field(
        default_factory=list,
        description="Levels to rule out. Use this for 'nothing senior', which "
                    "means exclude senior, staff, principal and management "
                    "rather than asking only for junior roles.")
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS)


class GetJobArgs(BaseModel):
    job_id: str = Field(description="The id of a job returned by search_jobs")


class SavePreferenceArgs(BaseModel):
    skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_only: bool | None = None
    seniority: str | None = None
    avoid: list[str] = Field(default_factory=list, description="Companies to exclude")


class Tool(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable

    model_config = {"arbitrary_types_allowed": True}

    def schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": schema}}


def _search_jobs(ctx: "ToolContext", args: SearchJobsArgs) -> dict:
    pref = ctx.preference
    filters = Filters(
        location=args.location or (pref.locations[0] if pref.locations else None),
        remote=args.remote if args.remote is not None else pref.remote_only,
        seniority=args.seniority or ([pref.seniority] if pref.seniority else []),
        exclude_seniority=args.exclude_seniority,
        exclude_companies=pref.avoid,
    )
    # Remembered so the fallback can honour the same constraints instead of
    # answering a different question than the one that was asked.
    if not filters.is_empty():
        ctx.filters_used.append(filters)
    ctx.queries_used.append(args.query)
    hits = ctx.searcher.search(args.query, k=args.limit, filters=filters,
                               max_per_company=MAX_PER_COMPANY)
    if not hits and not filters.is_empty():
        # Better to show near-misses than nothing; say so rather than pretending
        # the filters were honoured.
        hits = ctx.searcher.search(args.query, k=args.limit,
                                   max_per_company=MAX_PER_COMPANY)
        relaxed = True
    else:
        relaxed = False
    ctx.seen.update(h.job_id for h in hits)
    return {
        "relaxed_filters": relaxed,
        "results": [{
            "job_id": h.job_id,
            "title": h.job.title,
            "company": h.job.company,
            "location": h.job.location_canonical or h.job.location,
            "remote": h.job.remote,
            "seniority": h.job.seniority,
            "url": h.job.url,
            "snippet": h.job.description[:SNIPPET_CHARS],
        } for h in hits],
    }


def _get_job(ctx: "ToolContext", args: GetJobArgs) -> dict:
    job = db.get_job(ctx.conn, args.job_id)
    if not job:
        return {"error": f"No job with id {args.job_id}. Use an id from search_jobs."}
    ctx.seen.add(args.job_id)
    return {"job_id": args.job_id, "title": job.title, "company": job.company,
            "location": job.location_canonical or job.location, "url": job.url,
            "description": job.description[:DESCRIPTION_CHARS]}


def _save_preference(ctx: "ToolContext", args: SavePreferenceArgs) -> dict:
    """Merge into what is already remembered, and report only what moved.

    The caller surfaces `changed` to the person. A preference that is written
    silently is one they cannot correct, and a wrong one quietly filters their
    results from then on.
    """
    pref = ctx.preference
    changed = {}
    for field_name in ("skills", "locations", "avoid"):
        new = [v for v in getattr(args, field_name) if v]
        if new:
            before = list(getattr(pref, field_name))
            merged = list(dict.fromkeys(before + new))
            if merged != before:
                setattr(pref, field_name, merged)
                changed[field_name] = merged
    if args.remote_only is not None and args.remote_only != pref.remote_only:
        pref.remote_only = args.remote_only
        changed["remote_only"] = args.remote_only
    if args.seniority and args.seniority != pref.seniority:
        pref.seniority = args.seniority
        changed["seniority"] = args.seniority

    if changed:
        db.save_preference(ctx.conn, pref)
        ctx.remembered.update(changed)
    return {"changed": changed, "now": pref.model_dump(exclude={"user_id", "updated_at"})}


class ToolContext:
    """What the tools are allowed to touch, and what they record."""

    def __init__(self, conn: sqlite3.Connection, searcher: Searcher, user_id: str):
        self.conn = conn
        self.searcher = searcher
        self.user_id = user_id
        self.preference: Preference = db.load_preference(conn, user_id)
        self.seen: set[str] = set()   # job ids the model is allowed to cite
        self.filters_used: list[Filters] = []
        self.queries_used: list[str] = []
        self.remembered: dict = {}    # surfaced to the person after each turn

    @property
    def best_query(self) -> str | None:
        """The model's own phrasing of what to look for.

        Better than the raw message for a fallback search: the person's
        sentence carries the location too, and "bengaluru" in the query text
        matches job *titles* containing the city, which is how a fallback for
        "backend roles in bengaluru" surfaced an Account Executive posting.
        """
        return self.queries_used[0] if self.queries_used else None

    @property
    def first_filters(self) -> Filters | None:
        """The model's first reading of the request.

        Deliberately not the most recent one. A model that cannot find enough
        results loosens its search, and if the fallback inherited that it would
        quietly drop the constraint the person actually stated -- which is how
        "nothing senior" came back with four senior roles.
        """
        return self.filters_used[0] if self.filters_used else None


TOOLS: list[Tool] = [
    Tool(name="search_jobs", args_model=SearchJobsArgs, run=_search_jobs,
         description="Search live job listings. Use this before recommending any role."),
    Tool(name="get_job", args_model=GetJobArgs, run=_get_job,
         description="Read the full description of one job returned by search_jobs."),
    Tool(name="save_preference", args_model=SavePreferenceArgs, run=_save_preference,
         description="Remember what this person wants, so later searches reflect it."),
]

BY_NAME = {t.name: t for t in TOOLS}


def schemas() -> list[dict]:
    return [t.schema() for t in TOOLS]


def execute(ctx: ToolContext, name: str, raw_args: dict) -> tuple[dict, str | None]:
    """Run a tool. Returns (result, error). An error is text for the model to
    read and correct, not an exception."""
    tool = BY_NAME.get(name)
    if tool is None:
        return {}, f"No tool named {name!r}. Available: {', '.join(BY_NAME)}."
    if "__malformed__" in raw_args:
        return {}, "Your arguments were not valid JSON. Send them again as a JSON object."
    try:
        args = tool.args_model(**raw_args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4]
        )
        return {}, f"Invalid arguments for {name}: {problems}"
    try:
        return tool.run(ctx, args), None
    except Exception as exc:  # a tool bug should not kill the conversation
        return {}, f"{name} failed: {type(exc).__name__}: {exc}"
