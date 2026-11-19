"""Ashby public job posting API.

Endpoint: https://api.ashbyhq.com/posting-api/job-board/{slug}
Ashby gives a short teaser by default; descriptions come from descriptionPlain
when present, otherwise the HTML variant.
"""
from .. import normalize
from ..models import Job
from .base import SourceError, first, get_json

NAME = "ashby"
BASE = "https://api.ashbyhq.com/posting-api/job-board"


def probe(slug: str) -> int | None:
    try:
        data = get_json(f"{BASE}/{slug}", attempts=1)
    except Exception:
        return None
    jobs = data.get("jobs") if isinstance(data, dict) else None
    return len(jobs) if isinstance(jobs, list) else None


def fetch(slug: str, company: str | None = None) -> list[Job]:
    data = get_json(f"{BASE}/{slug}", params={"includeCompensation": "true"})
    if not isinstance(data, dict) or "jobs" not in data:
        raise SourceError(f"unexpected ashby payload for {slug}")

    out = []
    for raw in data["jobs"]:
        location = first(raw, "location", "locationName", default="") or ""
        if isinstance(location, dict):
            location = location.get("name", "")
        body = first(raw, "descriptionPlain", "descriptionHtml", "description", default="")
        description = normalize.strip_html(body)
        if not description:
            continue
        title = raw.get("title") or ""
        out.append(Job(
            source=NAME,
            source_id=str(first(raw, "id", "jobId", default="")),
            company=company or first(raw, "organizationName", default=slug),
            title=title,
            location=str(location),
            remote=normalize.detect_remote(str(location), description, raw.get("isRemote")),
            seniority=normalize.detect_seniority(title),
            description=description,
            url=first(raw, "jobUrl", "applyUrl", "url"),
            posted_at=first(raw, "publishedAt", "updatedAt"),
        ))
    return out
