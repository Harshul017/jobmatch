"""Pull every configured source into SQLite.

Safe to run repeatedly: jobs are upserted on a stable id, cross-posted
duplicates are collapsed, and anything a source stops returning is marked
expired. That is what keeps the index 'constantly changing' rather than a
one-off dump.

    python -m jobmatch.ingest              # everything
    python -m jobmatch.ingest --ats-only   # skip the aggregator boards
    python -m jobmatch.ingest --stats      # just show what's stored
"""
import argparse
import json
from collections import Counter

from . import db
from .config import SLUGS_FILE
from .sources import arbeitnow, ashby, greenhouse, lever, remotive
from .sources.base import SourceError

ATS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby}


def _store(conn, jobs, counts, seen):
    for job in jobs:
        counts[db.upsert_job(conn, job)] += 1
        seen.setdefault(job.source, set()).add(job.id)
    counts["fetched"] += len(jobs)


def ingest_ats(conn) -> Counter:
    if not SLUGS_FILE.exists():
        print("No data/slugs.json yet. Run: python -m jobmatch.resolve")
        return Counter()

    entries = json.loads(SLUGS_FILE.read_text(encoding="utf-8"))
    counts, seen = Counter(), {}
    print(f"Fetching {len(entries)} company boards...")

    for i, entry in enumerate(entries, 1):
        provider, slug, name = entry["provider"], entry["slug"], entry["name"]
        try:
            jobs = ATS[provider].fetch(slug, company=name)
            _store(conn, jobs, counts, seen)
            print(f"  [{i}/{len(entries)}] {name:<28} {provider:<11} {len(jobs):>4} jobs")
        except SourceError as exc:
            print(f"  [{i}/{len(entries)}] {name:<28} {provider:<11} failed: {exc}")
        except Exception as exc:
            print(f"  [{i}/{len(entries)}] {name:<28} {provider:<11} error: {exc!r}")
        conn.commit()

    for source, ids in seen.items():
        counts["expired"] += db.expire_missing(conn, source, ids)
    conn.commit()
    return counts


def ingest_boards(conn) -> Counter:
    counts, seen = Counter(), {}
    for module, label, kwargs in [
        (remotive, "remotive", {}),
        (arbeitnow, "arbeitnow", {"pages": 3}),
    ]:
        try:
            jobs = module.fetch(**kwargs)
            _store(conn, jobs, counts, seen)
            print(f"  {label:<12} {len(jobs):>4} jobs   ({module.ATTRIBUTION})")
        except Exception as exc:
            print(f"  {label:<12} failed: {exc!r}")
        conn.commit()

    for source, ids in seen.items():
        counts["expired"] += db.expire_missing(conn, source, ids)
    conn.commit()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch jobs into the local index.")
    ap.add_argument("--ats-only", action="store_true", help="company boards only")
    ap.add_argument("--boards-only", action="store_true", help="aggregators only")
    ap.add_argument("--stats", action="store_true", help="show the index and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop the index first (needed after a schema change)")
    ap.add_argument("--backfill", action="store_true",
                    help="recompute derived columns from stored text, no fetching")
    args = ap.parse_args()

    conn = db.connect()

    if args.stats:
        print(json.dumps(db.stats(conn), indent=2))
        if db.needs_rebuild(conn):
            print("\nSome rows predate the current schema. Run: "
                  "python -m jobmatch.ingest --rebuild")
        return

    if args.backfill:
        changed = db.backfill(conn)
        print(f"recomputed derived fields: {changed['location']} locations, "
              f"{changed['years']} experience requirements, "
              f"{changed['dedupe_key']} dedupe keys changed")
        print(json.dumps(db.stats(conn), indent=2))
        return

    if args.rebuild:
        print("Rebuilding the index from scratch...\n")
        db.rebuild(conn)

    totals = Counter()
    if not args.boards_only:
        totals += ingest_ats(conn)
    if not args.ats_only:
        print("\nFetching aggregator boards...")
        totals += ingest_boards(conn)

    db.record_run(conn, "all", totals)
    conn.commit()

    print(
        f"\nfetched {totals['fetched']}  "
        f"new {totals['inserted']}  updated {totals['updated']}  "
        f"cross-posted duplicates {totals['duplicate']}  expired {totals['expired']}"
    )
    print(json.dumps(db.stats(conn), indent=2))


if __name__ == "__main__":
    main()
