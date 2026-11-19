"""Lever postings API.

Endpoint: https://api.lever.co/v0/postings/{slug}?mode=json
Returns a bare list. Titles live in 'text', descriptions in 'descriptionPlain'.
"""
from datetime import datetime, timezone

from .. import normalize
from ..models import Job
from .base import SourceError, first, get_json

NAME = "lever"
BASE = "https://api.lever.co/v0/postings"


def probe(slug: str) -> int | None:
    try:
        data = get_json(f"{BASE}/{slug}", params={"mode": "json"}, attempts=1)
    except Exception:
        return None
    return len(data) if isinstance(data, list) else None


def _iso(ms) -> str | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def fetch(slug: str, company: str | None = None) -> list[Job]:
    data = get_json(f"{BASE}/{slug}", params={"mode": "json"})
    if not isinstance(data, list):
        raise SourceError(f"unexpected lever payload for {slug}")

    out = []
    for raw in data:
        cats = raw.get("categories") or {}
        location = cats.get("location") or ""
        body = first(raw, "descriptionPlain", "description", default="")
        description = normalize.strip_html(body)
        # Lever splits requirements into 'lists'; without them most postings
        # lose exactly the skill keywords we want to match on.
        for block in raw.get("lists") or []:
            description += "\n\n" + normalize.strip_html(
                f"{block.get('text', '')}\n{block.get('content', '')}"
            )
        if not description.strip():
            continue
        title = raw.get("text") or ""
        workplace = (raw.get("workplaceType") or "").lower() or None
        out.append(Job(
            source=NAME,
            source_id=str(raw.get("id")),
            company=company or slug,
            title=title,
            location=location,
            remote=normalize.detect_remote(
                location, description, True if workplace == "remote" else None
            ),
            seniority=normalize.detect_seniority(title),
            description=normalize.collapse(description),
            url=first(raw, "hostedUrl", "applyUrl"),
            posted_at=_iso(raw.get("createdAt")),
        ))
    return out
