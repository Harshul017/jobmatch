import pytest

from jobmatch import db
from jobmatch.models import Job, Preference


def make_job(**kw) -> Job:
    base = dict(source="greenhouse", source_id="1", company="Acme",
                title="Backend Engineer", description="Build APIs in Python.",
                location="Gurugram")
    base.update(kw)
    return Job(**base)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    yield c
    c.close()


def test_insert_then_update_same_job(conn):
    assert db.upsert_job(conn, make_job()) == "inserted"
    assert db.upsert_job(conn, make_job(title="Backend Engineer II")) == "updated"
    assert db.stats(conn)["active"] == 1
    assert db.active_jobs(conn)[0].title == "Backend Engineer II"


def test_same_role_on_two_boards_is_stored_once(conn):
    db.upsert_job(conn, make_job())
    result = db.upsert_job(conn, make_job(source="lever", source_id="xyz"))
    assert result == "duplicate"
    assert db.stats(conn)["active"] == 1


def test_fts_row_follows_the_job(conn):
    job = make_job()
    db.upsert_job(conn, job)
    hits = conn.execute(
        "SELECT job_id FROM jobs_fts WHERE jobs_fts MATCH 'python'"
    ).fetchall()
    assert [h["job_id"] for h in hits] == [job.id]


def test_update_does_not_duplicate_the_fts_row(conn):
    db.upsert_job(conn, make_job())
    db.upsert_job(conn, make_job(description="Build APIs in Python and Go."))
    count = conn.execute("SELECT COUNT(*) c FROM jobs_fts").fetchone()["c"]
    assert count == 1


def test_missing_job_is_expired_and_leaves_the_index(conn):
    job = make_job()
    db.upsert_job(conn, job)
    assert db.expire_missing(conn, "greenhouse", seen_ids=set()) == 1
    assert db.stats(conn)["active"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM jobs_fts").fetchone()["c"] == 0


def test_reappearing_job_is_revived(conn):
    job = make_job()
    db.upsert_job(conn, job)
    db.expire_missing(conn, "greenhouse", set())
    db.upsert_job(conn, job)
    assert db.stats(conn)["active"] == 1


def test_preferences_round_trip(conn):
    db.save_preference(conn, Preference(user_id="u1", skills=["python"], remote_only=True))
    loaded = db.load_preference(conn, "u1")
    assert loaded.skills == ["python"] and loaded.remote_only is True


def test_unknown_user_gets_empty_preferences(conn):
    assert db.load_preference(conn, "nobody").skills == []


OLD_JOBS_TABLE = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, source_id TEXT NOT NULL,
    company TEXT NOT NULL, title TEXT NOT NULL, location TEXT,
    remote INTEGER NOT NULL DEFAULT 0, seniority TEXT, description TEXT NOT NULL,
    url TEXT, posted_at TEXT, dedupe_key TEXT NOT NULL, first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL, expired_at TEXT
);
"""


def _old_database(path):
    """A database written before location_canonical existed."""
    import sqlite3
    raw = sqlite3.connect(path)
    raw.executescript(OLD_JOBS_TABLE)
    raw.execute(
        "INSERT INTO jobs VALUES ('x','greenhouse','1','Acme','Backend Engineer',"
        "'Bangalore, Karnataka',0,NULL,'Build things.',NULL,NULL,'k','t','t',NULL)"
    )
    raw.commit()
    raw.close()


def test_opening_an_older_database_migrates_instead_of_crashing(tmp_path):
    path = tmp_path / "old.db"
    _old_database(path)
    conn = db.connect(path)  # used to raise "no such column: location_canonical"
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "location_canonical" in columns
    conn.close()


def test_migrated_rows_are_flagged_as_needing_a_rebuild(tmp_path):
    path = tmp_path / "old.db"
    _old_database(path)
    conn = db.connect(path)
    assert db.needs_rebuild(conn) is True
    db.rebuild(conn)
    db.upsert_job(conn, make_job(location="Bangalore, Karnataka"))
    assert db.needs_rebuild(conn) is False
    conn.close()


def test_rebuild_clears_everything(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    db.upsert_job(conn, make_job())
    db.rebuild(conn)
    assert db.stats(conn)["total"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM jobs_fts").fetchone()["c"] == 0
    conn.close()


def test_canonical_location_is_stored_not_just_computed(tmp_path):
    conn = db.connect(tmp_path / "c.db")
    db.upsert_job(conn, make_job(location="Bengaluru-VTP, India"))
    row = conn.execute("SELECT location_canonical FROM jobs").fetchone()
    assert row["location_canonical"] == "bengaluru, in"
    conn.close()
