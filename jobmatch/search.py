"""Hybrid retrieval: keyword search, vector search, and a fusion of the two.

Neither half is sufficient on its own. BM25 finds the posting that literally
says "OpenSearch" but misses the one that says "Elasticsearch"; embeddings do
the opposite, and will happily rank a vaguely similar role above an exact
keyword hit. Reciprocal Rank Fusion merges the two rankings by position, so
neither engine's score scale has to be normalised against the other.
"""
from __future__ import annotations

import re
import sqlite3

from pydantic import BaseModel, Field

from . import db, locations
from .models import Job

RRF_K = 60  # dampening constant; 60 is the value from the original TREC paper
POOL = 50   # how deep to read from each engine before fusing


# A title with no seniority marker is a mid-level role. It is stored as NULL,
# so "mid" has to be translated rather than matched directly.
MID = "mid"
LEVELS = ("intern", "junior", MID, "senior", "staff", "principal", "lead", "management")


class Filters(BaseModel):
    """Hard constraints, applied before ranking rather than after."""
    location: str | None = None
    remote: bool | None = None
    seniority: list[str] = Field(default_factory=list)
    exclude_seniority: list[str] = Field(default_factory=list)
    company: str | None = None
    exclude_companies: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.location, self.remote is not None, self.seniority,
                        self.exclude_seniority, self.company, self.exclude_companies])


class Hit(BaseModel):
    job: Job
    job_id: str
    score: float
    rank: int
    found_by: list[str] = Field(default_factory=list)


def fts_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression.

    Raw user text cannot go in directly: quotes, hyphens and bare AND/OR/NOT
    are operators in FTS5 and a stray one is a syntax error. Every term is
    quoted and joined with OR, since requiring all terms makes a long natural
    sentence match nothing at all.
    """
    terms = re.findall(r"[A-Za-z0-9+#.]+", text or "")
    terms = [t for t in terms if len(t) > 1 or t.isdigit()]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:40])


def _location_clause(wanted: str) -> tuple[str, list]:
    parsed = locations.parse(wanted)
    if parsed["remote"] and not parsed["city"] and not parsed["country"]:
        return "remote = 1", []
    if parsed["city"]:
        return "(location_canonical = ? OR location_canonical LIKE ?)", [
            parsed["canonical"], f"{parsed['city']}%"]
    if parsed["country"]:
        cc = parsed["country"]
        return "(location_canonical = ? OR location_canonical LIKE ?)", [cc, f"%, {cc}"]
    return "location_canonical LIKE ?", [f"%{parsed['canonical']}%"]


def allowed_ids(conn: sqlite3.Connection, filters: Filters | None) -> set[str] | None:
    """Resolve filters to a set of ids. None means 'no restriction'."""
    if filters is None or filters.is_empty():
        return None

    where, params = ["expired_at IS NULL"], []
    if filters.location:
        clause, extra = _location_clause(filters.location)
        where.append(clause)
        params += extra
    if filters.remote is not None:
        where.append("remote = ?")
        params.append(int(filters.remote))
    if filters.seniority:
        named = [s for s in filters.seniority if s != MID]
        parts = []
        if named:
            parts.append(f"seniority IN ({','.join('?' * len(named))})")
            params += named
        if MID in filters.seniority:
            parts.append("seniority IS NULL")
        where.append("(" + " OR ".join(parts) + ")")
    if filters.exclude_seniority:
        named = [s for s in filters.exclude_seniority if s != MID]
        if named:
            # NULL means mid-level, and NOT IN never matches NULL, so keep it
            # explicitly or every mid-level role would be excluded too.
            where.append(f"(seniority IS NULL OR seniority NOT IN "
                         f"({','.join('?' * len(named))}))")
            params += named
        if MID in filters.exclude_seniority:
            where.append("seniority IS NOT NULL")
    if filters.company:
        where.append("LOWER(company) LIKE ?")
        params.append(f"%{filters.company.lower()}%")
    for name in filters.exclude_companies:
        where.append("LOWER(company) NOT LIKE ?")
        params.append(f"%{name.lower()}%")

    sql = f"SELECT id FROM jobs WHERE {' AND '.join(where)}"
    return {r["id"] for r in conn.execute(sql, params)}


def bm25_rank(conn: sqlite3.Connection, query: str, k: int = POOL,
              ids: set[str] | None = None) -> list[str]:
    """Keyword ranking from SQLite's FTS5. Title and company are weighted up,
    or a long description drowns out the one line that names the role."""
    match = fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        """
        SELECT job_id, bm25(jobs_fts, 0.0, 6.0, 3.0, 2.0, 1.0) AS score
        FROM jobs_fts WHERE jobs_fts MATCH ?
        ORDER BY score LIMIT ?
        """,
        (match, k if ids is None else max(k * 6, 400)),
    ).fetchall()
    out = [r["job_id"] for r in rows]
    if ids is not None:
        out = [j for j in out if j in ids]
    return out[:k]


def dense_rank(store, embedder, query: str, k: int = POOL,
               ids: set[str] | None = None) -> list[str]:
    vector = embedder.encode([query])[0]
    return [job_id for job_id, _ in store.search(vector, k, allowed_ids=ids)]


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, list[str]]]:
    """Reciprocal Rank Fusion: score(d) = sum over engines of 1/(k + rank(d)).

    Uses only positions, so BM25's negative log-ish scores and cosine
    similarities never have to be put on a common scale. A document found by
    both engines beats one found deep in a single ranking.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ranking in rankings.items():
        for position, job_id in enumerate(ranking, start=1):
            scores[job_id] = scores.get(job_id, 0.0) + 1.0 / (k + position)
            sources.setdefault(job_id, []).append(name)
    ordered = sorted(scores, key=lambda j: -scores[j])
    return [(j, scores[j], sources[j]) for j in ordered]


class Searcher:
    """Ties the index, the vector store and the embedder together."""

    def __init__(self, conn: sqlite3.Connection, store, embedder):
        self.conn, self.store, self.embedder = conn, store, embedder

    def rank_ids(self, query: str, k: int = 10, mode: str = "hybrid",
                 filters: Filters | None = None,
                 max_per_company: int | None = None) -> list[tuple[str, float, list[str]]]:
        ids = allowed_ids(self.conn, filters)
        pool = max(k, POOL)
        if mode == "bm25":
            ranking = bm25_rank(self.conn, query, pool, ids)
            ranked = [(j, 1.0 / (RRF_K + i), ["bm25"]) for i, j in enumerate(ranking, 1)]
        elif mode == "dense":
            ranking = dense_rank(self.store, self.embedder, query, pool, ids)
            ranked = [(j, 1.0 / (RRF_K + i), ["dense"]) for i, j in enumerate(ranking, 1)]
        else:
            ranked = rrf({
                "bm25": bm25_rank(self.conn, query, pool, ids),
                "dense": dense_rank(self.store, self.embedder, query, pool, ids),
            })
        if max_per_company:
            ranked = self._diversify(ranked, k, max_per_company)
        return ranked[:k]

    def _diversify(self, ranked, k: int, cap: int):
        """Limit how much of the list one employer can take.

        Pure relevance ranking has no notion of variety, and a large company
        with many similar postings will fill the whole page -- three of four
        results from one employer is a worse shortlist than three employers,
        even if it scores better. Capped entries are not discarded, just
        pushed below the ones that add a new company.
        """
        companies: dict[str, int] = {}
        kept, overflow = [], []
        for item in ranked:
            job = db.get_job(self.conn, item[0])
            name = (job.company if job else "").lower()
            if companies.get(name, 0) < cap:
                companies[name] = companies.get(name, 0) + 1
                kept.append(item)
            else:
                overflow.append(item)
            if len(kept) >= k:
                break
        return kept + overflow

    def search_diverse(self, query: str, k: int = 10, **kw) -> list["Hit"]:
        return self.search(query, k, max_per_company=2, **kw)

    def search(self, query: str, k: int = 10, mode: str = "hybrid",
               filters: Filters | None = None,
               max_per_company: int | None = None) -> list[Hit]:
        hits = []
        for rank, (job_id, score, found_by) in enumerate(
            self.rank_ids(query, k, mode, filters, max_per_company), start=1
        ):
            job = db.get_job(self.conn, job_id)
            if job:
                hits.append(Hit(job=job, job_id=job_id, score=score,
                                rank=rank, found_by=found_by))
        return hits
