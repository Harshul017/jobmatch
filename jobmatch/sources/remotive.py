"""Remotive public jobs API.

Remotive asks API users to link back and to attribute listings to them, and
delays its feed by 24 hours. We store the job url so every result can credit
the source, and we fetch at most a few times a day.
"""
from .. import normalize
from ..models import Job
from .base import SourceError, first, get_json

NAME = "remotive"
URL = "https://remotive.com/api/remote-jobs"
ATTRIBUTION = "Listings from Remotive (remotive.com)"


def fetch(category: str | None = "software-dev", limit: int | None = None) -> list[Job]:
    params = {}
    if category:
        params["category"] = category
    if limit:
        params["limit"] = limit
    data = get_json(URL, params=params)
    if not isinstance(data, dict) or "jobs" not in data:
        raise SourceError("unexpected remotive payload")

    out = []
    for raw in data["jobs"]:
        description = normalize.strip_html(raw.get("description"))
        if not description:
            continue
        title = raw.get("title") or ""
        location = raw.get("candidate_required_location") or ""
        out.append(Job(
            source=NAME,
            source_id=str(raw.get("id")),
            company=raw.get("company_name") or "unknown",
            title=title,
            location=location,
            remote=True,  # the whole board is remote roles
            seniority=normalize.detect_seniority(title),
            description=description,
            url=first(raw, "url", "job_url"),
            posted_at=raw.get("publication_date"),
        ))
    return out
