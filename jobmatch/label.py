"""Build the labelled evaluation set by hand.

Judging only what one system returns would rig the comparison in its favour,
so candidates are *pooled*: the union of what BM25, the embeddings and the
fusion each return, shown in a shuffled order with no indication of which
engine found what. You judge the pool; anything outside it counts as
non-relevant, which is the usual and openly-stated limitation of pooled
evaluation.

    python -m jobmatch.label "semantic search and ranking"
    python -m jobmatch.label --seed            # work through data/seed_queries.txt
    python -m jobmatch.label --review          # see what is already labelled
"""
import argparse
import json
import random
from pathlib import Path

from . import db
from .config import DATA_DIR
from .embed import DEFAULT_MODEL, get_embedder
from .index import open_store
from .search import Searcher

EVAL_FILE = DATA_DIR / "eval.json"
SEED_FILE = DATA_DIR / "seed_queries.txt"
POOL_PER_MODE = 12


def load_eval() -> dict:
    if EVAL_FILE.exists():
        return json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return {"queries": []}


def save_eval(data: dict) -> None:
    EVAL_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_selection(raw: str, size: int) -> list[int]:
    """Accepts '1 3 7-9' and returns zero-based indices."""
    picked: set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token[1:]:
            lo, _, hi = token.partition("-")
            try:
                picked.update(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        elif token.isdigit():
            picked.add(int(token))
    return sorted(i - 1 for i in picked if 1 <= i <= size)


def pool_candidates(searcher: Searcher, query: str) -> list[str]:
    seen, pool = set(), []
    for mode in ("bm25", "dense", "hybrid"):
        for job_id, _, _ in searcher.rank_ids(query, POOL_PER_MODE, mode):
            if job_id not in seen:
                seen.add(job_id)
                pool.append(job_id)
    random.shuffle(pool)  # so position carries no hint about which engine found it
    return pool


def label_one(searcher: Searcher, conn, query: str, data: dict) -> bool:
    """Returns False if the user asked to stop."""
    existing = next((q for q in data["queries"] if q["query"] == query), None)
    if existing:
        print(f"\nAlready labelled ({len(existing['relevant'])} relevant). Re-judging.")

    pool = pool_candidates(searcher, query)
    if not pool:
        print("No candidates returned. Is the vector index built?")
        return True

    print(f"\n{'=' * 78}\nQUERY: {query}\n{'=' * 78}")
    jobs = []
    for n, job_id in enumerate(pool, 1):
        job = db.get_job(conn, job_id)
        jobs.append((job_id, job))
        loc = job.location_canonical or "-"
        print(f"{n:3}. {job.title[:58]:60} {job.company[:18]:20} {loc}")

    print("\nWhich are relevant? (e.g. '1 4 7-9'; blank = none; 's' skip; 'q' quit)")
    raw = input("> ").strip().lower()
    if raw == "q":
        return False
    if raw == "s":
        return True

    chosen = parse_selection(raw, len(jobs))
    entry = {
        "query": query,
        "relevant": [jobs[i][0] for i in chosen],
        "judged": pool,
        "relevant_titles": [f"{jobs[i][1].title} @ {jobs[i][1].company}" for i in chosen],
    }
    data["queries"] = [q for q in data["queries"] if q["query"] != query] + [entry]
    save_eval(data)
    print(f"Saved {len(entry['relevant'])} relevant of {len(pool)} judged. "
          f"({len(data['queries'])} queries labelled so far)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label queries for evaluation.")
    ap.add_argument("query", nargs="?", help="a single query to judge")
    ap.add_argument("--seed", action="store_true", help=f"work through {SEED_FILE.name}")
    ap.add_argument("--review", action="store_true", help="list what is labelled")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    data = load_eval()

    if args.review:
        for q in data["queries"]:
            print(f"\n{q['query']}  ({len(q['relevant'])} relevant / {len(q['judged'])} judged)")
            for t in q.get("relevant_titles", []):
                print(f"    - {t}")
        print(f"\n{len(data['queries'])} queries labelled.")
        return

    conn = db.connect()
    searcher = Searcher(conn, open_store("numpy", args.model), get_embedder(args.model))

    if args.seed:
        if not SEED_FILE.exists():
            raise SystemExit(f"No {SEED_FILE}. Put one query per line in it.")
        todo = [l.strip() for l in SEED_FILE.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
        done = {q["query"] for q in data["queries"]}
        remaining = [q for q in todo if q not in done]
        print(f"{len(done)} labelled, {len(remaining)} to go.")
        for query in remaining:
            if not label_one(searcher, conn, query, data):
                break
    elif args.query:
        label_one(searcher, conn, args.query, data)
    else:
        ap.error("give a query, or --seed, or --review")


if __name__ == "__main__":
    main()
