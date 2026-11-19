"""Search the index from the command line.

Useful on its own, and the quickest way to sanity-check retrieval before the
agent is wired up.

    python -m jobmatch.query "backend engineer working on search"
    python -m jobmatch.query "llm agents" --location india --remote
    python -m jobmatch.query "data engineer" --mode bm25 --compare
"""
import argparse

from . import db
from .embed import DEFAULT_MODEL, get_embedder
from .index import open_store
from .search import Filters, Searcher


def show(hits, label: str = "") -> None:
    if label:
        print(f"\n--- {label} ---")
    if not hits:
        print("  (nothing)")
        return
    for h in hits:
        found = "+".join(h.found_by)
        print(f"{h.rank:3}. {h.job.title[:56]:58} {h.job.company[:16]:18} "
              f"{(h.job.location_canonical or '-')[:18]:20} [{found}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the job index.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--compare", action="store_true", help="show all three modes")
    ap.add_argument("--location")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--seniority", nargs="*", default=[])
    ap.add_argument("--company")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--store", choices=["qdrant", "numpy"], default="numpy")
    ap.add_argument("--full", action="store_true", help="print the top result's text")
    ap.add_argument("--diverse", type=int, metavar="N",
                    help="allow at most N results per company")
    args = ap.parse_args()

    conn = db.connect()
    searcher = Searcher(conn, open_store(args.store, args.model), get_embedder(args.model))
    filters = Filters(
        location=args.location,
        remote=True if args.remote else None,
        seniority=args.seniority,
        company=args.company,
    )

    if args.compare:
        for mode in ("bm25", "dense", "hybrid"):
            show(searcher.search(args.query, args.k, mode, filters,
                                 max_per_company=args.diverse), mode)
        return

    hits = searcher.search(args.query, args.k, args.mode, filters,
                           max_per_company=args.diverse)
    show(hits)
    if args.full and hits:
        top = hits[0].job
        print(f"\n{'=' * 70}\n{top.title} @ {top.company}\n{top.url}\n{'=' * 70}")
        print(top.description[:1500])


if __name__ == "__main__":
    main()
