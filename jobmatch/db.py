"""SQLite storage: the jobs table, a BM25 index over it, and preference memory.

The index is meant to be re-ingested repeatedly, so writes are upserts keyed on
a stable job id. Listings that stop appearing in a source's feed are marked
expired rather than deleted, which keeps history and makes the index 'changing'
rather than a one-off dump.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH
from .models import Job, Preference

SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT,
    location_canonical TEXT,
    remote      INTEGER NOT NULL DEFAULT 0,
    seniority   TEXT,
    description TEXT NOT NULL,
    url         TEXT,
    posted_at   TEXT,
    dedupe_key  TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    expired_at  TEXT
);
-- Standalone FTS5 table (not external-content) so upserts stay a simple
-- delete-then-insert and cannot drift out of sync with triggers.
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    job_id UNINDEXED,
    title,
    company,
    location,
    description,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS preferences (
    user_id    TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    started_at TEXT NOT NULL,
    fetched    INTEGER NOT NULL DEFAULT 0,
    inserted   INTEGER NOT NULL DEFAULT 0,
    updated    INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    expired    INTEGER NOT NULL DEFAULT 0,
    note       TEXT
);
"""

# Created after migration, because an index on a column that an older database
# has not grown yet would fail before we get the chance to add it.
SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_dedupe ON jobs(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs(expired_at);
CREATE INDEX IF NOT EXISTS idx_jobs_loc    ON jobs(location_canonical);
"""

SCHEMA = SCHEMA_TABLES + SCHEMA_INDEXES

SCHEMA_VERSION = 2

# Columns added after the first release. CREATE TABLE IF NOT EXISTS silently
# skips an existing table, so new columns have to be ALTERed in by hand.
ADDED_COLUMNS = {"location_canonical": "TEXT"}


def _migrate(conn: sqlite3.Connection) -> list[str]:
    """Bring an older database up to the current column set. Returns what changed."""
    present = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    if not present:
        return []
    added = []
    for column, decl in ADDED_COLUMNS.items():
        if column not in present:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")
            added.append(column)
    return added


def needs_rebuild(conn: sqlite3.Connection) -> bool:
    """True when rows predate a derived field and would be wrong if trusted.

    A migrated column is empty for existing rows, and dedupe keys computed
    under the old rules are stale, so the honest fix is a re-ingest.
    """
    row = conn.execute(
        """SELECT COUNT(*) c FROM jobs
           WHERE expired_at IS NULL
             AND (location_canonical IS NULL OR location_canonical = '')
             AND location IS NOT NULL AND location != ''"""
    ).fetchone()
    return bool(row["c"])


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    # check_same_thread=False because the API serves requests from a thread
    # pool while holding one connection. Python's sqlite3 reports
    # threadsafety=3 (serialized) on a standard build, so the library itself
    # serializes access; without this flag any threaded server raises
    # "SQLite objects created in a thread can only be used in that thread".
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_TABLES)
    added = _migrate(conn)
    conn.executescript(SCHEMA_INDEXES)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    if added:
        print(f"[db] migrated: added {', '.join(added)}. Run with --rebuild to refill it.")
    return conn


def upsert_job(conn: sqlite3.Connection, job: Job) -> str:
    """Insert, update, or recognise as a cross-source duplicate.

    Returns 'inserted', 'updated' or 'duplicate'.
    """
    ts = now()
    existing = conn.execute("SELECT id FROM jobs WHERE id = ?", (job.id,)).fetchone()

    if existing is None:
        # Same role already here from another board? Keep the first one seen and
        # just refresh its timestamp, so the index holds one row per real job.
        clash = conn.execute(
            "SELECT id FROM jobs WHERE dedupe_key = ? AND source != ? AND expired_at IS NULL",
            (job.dedupe_key, job.source),
        ).fetchone()
        if clash:
            conn.execute("UPDATE jobs SET last_seen = ? WHERE id = ?", (ts, clash["id"]))
            return "duplicate"

    conn.execute(
        """
        INSERT INTO jobs (id, source, source_id, company, title, location,
                          location_canonical, remote, seniority, description, url,
                          posted_at, dedupe_key, first_seen, last_seen, expired_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
        ON CONFLICT(id) DO UPDATE SET
            company = excluded.company,
            title = excluded.title,
            location = excluded.location,
            location_canonical = excluded.location_canonical,
            remote = excluded.remote,
            seniority = excluded.seniority,
            description = excluded.description,
            url = excluded.url,
            posted_at = excluded.posted_at,
            dedupe_key = excluded.dedupe_key,
            last_seen = excluded.last_seen,
            expired_at = NULL
        """,
        (job.id, job.source, job.source_id, job.company, job.title, job.location,
         job.location_canonical, int(job.remote), job.seniority, job.description,
         job.url, job.posted_at, job.dedupe_key, ts, ts),
    )

    conn.execute("DELETE FROM jobs_fts WHERE job_id = ?", (job.id,))
    conn.execute(
        "INSERT INTO jobs_fts (job_id, title, company, location, description) VALUES (?,?,?,?,?)",
        (job.id, job.title, job.company, job.location or "", job.description),
    )
    return "inserted" if existing is None else "updated"


def expire_missing(conn: sqlite3.Connection, source: str, seen_ids: set[str]) -> int:
    """Mark rows from this source that the latest fetch did not return."""
    rows = conn.execute(
        "SELECT id FROM jobs WHERE source = ? AND expired_at IS NULL", (source,)
    ).fetchall()
    gone = [r["id"] for r in rows if r["id"] not in seen_ids]
    if gone:
        ts = now()
        conn.executemany(
            "UPDATE jobs SET expired_at = ? WHERE id = ?", [(ts, i) for i in gone]
        )
        conn.executemany("DELETE FROM jobs_fts WHERE job_id = ?", [(i,) for i in gone])
    return len(gone)


def active_jobs(conn: sqlite3.Connection, limit: int | None = None) -> list[Job]:
    sql = "SELECT * FROM jobs WHERE expired_at IS NULL ORDER BY last_seen DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [_row_to_job(r) for r in conn.execute(sql)]


def get_job(conn: sqlite3.Connection, job_id: str) -> Job | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        source=row["source"], source_id=row["source_id"], company=row["company"],
        title=row["title"], location=row["location"], remote=bool(row["remote"]),
        seniority=row["seniority"], description=row["description"],
        url=row["url"], posted_at=row["posted_at"],
    )


def record_run(conn: sqlite3.Connection, source: str, counts: dict, note: str = "") -> None:
    conn.execute(
        """INSERT INTO ingest_runs (source, started_at, fetched, inserted, updated,
                                    duplicates, expired, note)
           VALUES (?,?,?,?,?,?,?,?)""",
        (source, now(), counts.get("fetched", 0), counts.get("inserted", 0),
         counts.get("updated", 0), counts.get("duplicate", 0),
         counts.get("expired", 0), note),
    )


def save_preference(conn: sqlite3.Connection, pref: Preference) -> None:
    conn.execute(
        """INSERT INTO preferences (user_id, payload, updated_at) VALUES (?,?,?)
           ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload,
                                              updated_at=excluded.updated_at""",
        (pref.user_id, pref.model_dump_json(), now()),
    )
    conn.commit()


def load_preference(conn: sqlite3.Connection, user_id: str) -> Preference:
    row = conn.execute(
        "SELECT payload FROM preferences WHERE user_id = ?", (user_id,)
    ).fetchone()
    if not row:
        return Preference(user_id=user_id)
    return Preference(**json.loads(row["payload"]))


def rebuild(conn: sqlite3.Connection) -> None:
    """Drop everything and recreate. Used when the schema or a derived field
    (dedupe key, canonical location) changes and the old rows are stale."""
    for table in ("jobs_fts", "jobs", "ingest_runs"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(SCHEMA_TABLES)
    conn.executescript(SCHEMA_INDEXES)
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]
    active = conn.execute(
        "SELECT COUNT(*) c FROM jobs WHERE expired_at IS NULL"
    ).fetchone()["c"]
    by_source = {
        r["source"]: r["c"]
        for r in conn.execute(
            "SELECT source, COUNT(*) c FROM jobs WHERE expired_at IS NULL GROUP BY source"
        )
    }
    companies = conn.execute(
        "SELECT COUNT(DISTINCT company) c FROM jobs WHERE expired_at IS NULL"
    ).fetchone()["c"]
    top_locations = {
        r["location_canonical"]: r["c"]
        for r in conn.execute(
            """SELECT location_canonical, COUNT(*) c FROM jobs
               WHERE expired_at IS NULL AND location_canonical != ''
               GROUP BY 1 ORDER BY 2 DESC LIMIT 8"""
        )
    }
    return {"total": total, "active": active, "expired": total - active,
            "companies": companies, "by_source": by_source,
            "top_locations": top_locations}
