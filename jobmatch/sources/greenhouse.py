"""Greenhouse public job board API.

Endpoint: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
The description arrives as HTML-escaped HTML, which normalize.strip_html undoes.
"""
from .. import normalize
from ..models import Job
from .base import SourceError, first, get_json

NAME = "greenhouse"
BASE = "https://boards-api.greenhouse.io/v1/boards"


def probe(slug: str) -> int | None:
    """Return the job count if this slug exists on Greenhouse, else None."""
    try:
        data = get_json(f"{BASE}/{slug}/jobs", attempts=1)
    except Exception:
        return None
    jobs = data.get("jobs") if isinstance(data, dict) else None
    return len(jobs) if isinstance(jobs, list) else None


def fetch(slug: str, company: str | None = None) -> list[Job]:
    data = get_json(f"{BASE}/{slug}/jobs", params={"content": "true"})
    if not isinstance(data, dict) or "jobs" not in data:
        raise SourceError(f"unexpected greenhouse payload for {slug}")

    out = []
    for raw in data["jobs"]:
        loc = raw.get("location") or {}
        location = loc.get("name") if isinstance(loc, dict) else str(loc or "")
        description = normalize.strip_html(first(raw, "content", "description", default=""))
        if not description:
            continue
        title = raw.get("title") or ""
        out.append(Job(
            source=NAME,
            source_id=str(raw.get("id")),
            company=company or slug,
            title=title,
            location=location,
            remote=normalize.detect_remote(location, description),
            seniority=normalize.detect_seniority(title),
            description=description,
            url=first(raw, "absolute_url", "url"),
            posted_at=first(raw, "updated_at", "created_at"),
        ))
    return out
