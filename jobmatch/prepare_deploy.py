"""Build a slim copy of the index for deployment.

The working database is ~117MB, almost all of it full job descriptions. The
app never shows more than the first ~1400 characters of one, so most of that
weight is dead in production. This writes a trimmed copy alongside the
vectors, small enough to ship in a container image.

    python -m jobmatch.prepare_deploy
    python -m jobmatch.prepare_deploy --max-jobs 3000 --engineering-only
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from . import db
from .config import DATA_DIR, DB_PATH
from .embed import DEFAULT_MODEL
from .index import open_store, payload_for
from .vectorstore import NumpyStore

DEPLOY_DIR = DATA_DIR / "deploy"
KEEP_CHARS = 1600   # a little above what the API ever returns

ENGINEERING = """(
    lower(title) LIKE '%engineer%' OR lower(title) LIKE '%developer%'
 OR lower(title) LIKE '%scientist%' OR lower(title) LIKE '%machine learning%'
 OR lower(title) LIKE '%data%'      OR lower(title) LIKE '%software%'
 OR lower(title) LIKE '%research%'  OR lower(title) LIKE '%architect%'
)"""


def build(max_jobs: int | None, engineering_only: bool, keep_chars: int) -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    target = DEPLOY_DIR / "jobs.db"
    if target.exists():
        target.unlink()

    source = db.connect(DB_PATH)
    where = ["expired_at IS NULL"]
    if engineering_only:
        where.append(ENGINEERING)
    sql = f"SELECT id FROM jobs WHERE {' AND '.join(where)} ORDER BY last_seen DESC"
    if max_jobs:
        sql += f" LIMIT {int(max_jobs)}"
    keep = [r["id"] for r in source.execute(sql)]
    if not keep:
        raise SystemExit("Nothing to ship. Run: python -m jobmatch.ingest")

    out = db.connect(target)
    db.rebuild(out)
    kept = 0
    for job_id in keep:
        job = db.get_job(source, job_id)
        if not job:
            continue
        job.description = job.description[:keep_chars]
        db.upsert_job(out, job)
        kept += 1
    total_source = source.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]
    out.commit()
    out.execute("VACUUM")
    out.commit()
    # Fold the write-ahead log back into the database file and leave WAL mode,
    # so what ships is one self-contained file. Copying jobs.db on its own
    # while a 16MB -wal sat beside it would have silently shipped a
    # near-empty index.
    out.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out.execute("PRAGMA journal_mode=DELETE")
    out.commit()
    out.close()

    # Vectors for exactly the jobs that survived, in the same store format.
    store = open_store("numpy", DEFAULT_MODEL)
    by_id = dict(zip(store.ids, store.vectors)) if store.ids else {}
    ids = [i for i in keep if i in by_id]
    slim = NumpyStore(DEPLOY_DIR / f"vectors-{DEFAULT_MODEL.split('/')[-1].lower()}.npz")
    slim.reset(store.vectors.shape[1] if store.ids else 384)
    slim.upsert(ids, np.vstack([by_id[i] for i in ids]), [{} for _ in ids])
    slim.save()

    leftovers = [p for p in DEPLOY_DIR.iterdir() if p.suffix in {".db-wal", ".db-shm"}
                 or p.name.endswith(("-wal", "-shm"))]
    for stale in leftovers:
        stale.unlink(missing_ok=True)

    sizes = {p.name: p.stat().st_size / 1e6 for p in DEPLOY_DIR.iterdir() if p.is_file()}
    before = DB_PATH.stat().st_size / 1e6
    print(f"kept {kept} of {total_source} jobs, {len(ids)} with vectors")
    for name, mb in sorted(sizes.items()):
        print(f"  {name:<40} {mb:7.1f} MB")
    print(f"  (working jobs.db was {before:.1f} MB)")
    # Prove the shipped file stands alone before anyone deploys it.
    check = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    rows = check.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    fts = check.execute("SELECT COUNT(*) FROM jobs_fts").fetchone()[0]
    check.close()
    if rows != kept or fts != kept:
        raise SystemExit(f"shipped file is incomplete: {rows} jobs, {fts} indexed, expected {kept}")
    print(f"  verified: {rows} jobs and {fts} FTS rows readable from the file alone")
    print(f"\nShip {DEPLOY_DIR} as the container's /app/data")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a deployable index.")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--engineering-only", action="store_true",
                    help="drop sales, support and other non-technical postings")
    ap.add_argument("--keep-chars", type=int, default=KEEP_CHARS)
    args = ap.parse_args()
    build(args.max_jobs, args.engineering_only, args.keep_chars)


if __name__ == "__main__":
    main()
