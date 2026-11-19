"""Arbeitnow free job board API (Europe and remote).

Arbeitnow offers the API free and asks users to link back, so we keep the
source url on every record and page politely.
"""
from datetime import datetime, timezone

from .. import normalize
from ..models import Job
from .base import SourceError, first, get_json

NAME = "arbeitnow"
URL = "https://www.arbeitnow.com/api/job-board-api"
ATTRIBUTION = "Listings from Arbeitnow (arbeitnow.com)"


def _iso(ts) -> str | None:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def fetch(pages: int = 3) -> list[Job]:
    out = []
    for page in range(1, pages + 1):
        data = get_json(URL, params={"page": page})
        if not isinstance(data, dict) or "data" not in data:
            raise SourceError("unexpected arbeitnow payload")
        rows = data["data"]
        if not rows:
            break
        for raw in rows:
            description = normalize.strip_html(raw.get("description"))
            if not description:
                continue
            title = raw.get("title") or ""
            location = raw.get("location") or ""
            out.append(Job(
                source=NAME,
                source_id=str(first(raw, "slug", "id", default=title)),
                company=raw.get("company_name") or "unknown",
                title=title,
                location=location,
                remote=normalize.detect_remote(location, description, raw.get("remote")),
                seniority=normalize.detect_seniority(title),
                description=description,
                url=raw.get("url"),
                posted_at=_iso(raw.get("created_at")),
            ))
    return out
