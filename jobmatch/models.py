"""The one shape every source is flattened into."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from . import locations, normalize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job(BaseModel):
    source: str
    source_id: str
    company: str
    title: str
    description: str
    location: str | None = None
    remote: bool = False
    url: str | None = None
    posted_at: str | None = None
    seniority: str | None = None

    @field_validator("company", "title", "location", mode="before")
    @classmethod
    def _tidy(cls, v):
        return normalize.collapse(v) if isinstance(v, str) else v

    @property
    def id(self) -> str:
        return normalize.job_id(self.source, self.source_id)

    @property
    def location_canonical(self) -> str:
        """Stored alongside the raw string so filtering is cheap and exact."""
        return locations.canonical(self.location)

    @property
    def dedupe_key(self) -> str:
        return normalize.dedupe_key(self.company, self.title, self.location)

    def embedding_text(self, max_chars: int = 1200) -> str:
        """What gets embedded.

        Deliberately not the whole posting: these models have a few hundred
        tokens of context, and a 5,000-character description is mostly boilerplate
        about benefits and equal opportunity. The title carries most of the
        signal, and the opening of a description is where the role is actually
        described, so we keep those and drop the tail.
        """
        head = " ".join([
            self.title,
            f"at {self.company}",
            self.location or "",
            self.seniority or "",
            "remote" if self.remote else "",
        ])
        return f"{head}\n{self.description[:max_chars]}".strip()

    def search_text(self) -> str:
        """What gets embedded and indexed. Title and company repeated once so
        they carry weight in BM25 without needing per-field boosting."""
        parts = [self.title, self.company, self.location or "", self.description]
        return "\n".join(p for p in parts if p)


class Preference(BaseModel):
    """What the agent remembers about a user between turns."""
    user_id: str
    skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_only: bool | None = None
    seniority: str | None = None
    avoid: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_now)
