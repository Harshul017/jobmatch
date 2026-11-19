"""Measure retrieval quality instead of assuming it.

Reports Recall@k, MRR and nDCG@10 for keyword-only, embeddings-only and the
fusion, over the hand-labelled queries in data/eval.json.

    python -m jobmatch.evaluate
    python -m jobmatch.evaluate --model sentence-transformers/all-MiniLM-L6-v2
    python -m jobmatch.evaluate --per-query
"""
import argparse
import json
import math
import time

from . import db
from .embed import ALTERNATE_MODEL, DEFAULT_MODEL, get_embedder
from .index import open_store
from .label import EVAL_FILE
from .search import Searcher

MODES = ("bm25", "dense", "hybrid")


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, job_id in enumerate(ranked, start=1):
        if job_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary gains, so DCG rewards putting relevant results near the top."""
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, job_id in enumerate(ranked[:k], start=1)
        if job_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(searcher: Searcher, queries: list[dict], depth: int = 20) -> dict:
    results = {}
    for mode in MODES:
        totals = {"r@5": 0.0, "r@10": 0.0, "p@5": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
        latencies = []
        for item in queries:
            relevant = set(item["relevant"])
            if not relevant:
                continue
            started = time.perf_counter()
            ranked = [j for j, _, _ in searcher.rank_ids(item["query"], depth, mode)]
            latencies.append((time.perf_counter() - started) * 1000)
            totals["r@5"] += recall_at_k(ranked, relevant, 5)
            totals["r@10"] += recall_at_k(ranked, relevant, 10)
            totals["p@5"] += precision_at_k(ranked, relevant, 5)
            totals["mrr"] += reciprocal_rank(ranked, relevant)
            totals["ndcg@10"] += ndcg_at_k(ranked, relevant, 10)
        n = sum(1 for q in queries if q["relevant"]) or 1
        results[mode] = {k: v / n for k, v in totals.items()}
        results[mode]["ms"] = sum(latencies) / len(latencies) if latencies else 0.0
    return results


def print_table(results: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'mode':<9} {'R@5':>7} {'R@10':>7} {'P@5':>7} {'MRR':>7} {'nDCG@10':>8} {'ms':>7}")
    print("-" * 56)
    best = max(results, key=lambda m: results[m]["ndcg@10"])
    for mode, m in results.items():
        star = " *" if mode == best else ""
        print(f"{mode:<9} {m['r@5']:>7.3f} {m['r@10']:>7.3f} {m['p@5']:>7.3f} "
              f"{m['mrr']:>7.3f} {m['ndcg@10']:>8.3f} {m['ms']:>7.1f}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--compare-models", action="store_true",
                    help=f"run both {DEFAULT_MODEL} and {ALTERNATE_MODEL}")
    ap.add_argument("--store", choices=["qdrant", "numpy"], default="numpy")
    ap.add_argument("--per-query", action="store_true", help="show the weakest queries")
    args = ap.parse_args()

    if not EVAL_FILE.exists():
        raise SystemExit("No labelled queries yet. Run: python -m jobmatch.label --seed")
    queries = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["queries"]
    labelled = [q for q in queries if q["relevant"]]
    if not labelled:
        raise SystemExit("eval.json has no relevant judgements yet.")

    conn = db.connect()
    models = [DEFAULT_MODEL, ALTERNATE_MODEL] if args.compare_models else [args.model]

    for model_name in models:
        searcher = Searcher(conn, open_store(args.store, model_name), get_embedder(model_name))
        results = evaluate(searcher, labelled)
        print_table(results, f"{len(labelled)} queries | {model_name} | {args.store}")

        if args.per_query:
            print("\nWeakest queries (hybrid, by nDCG@10):")
            rows = []
            for item in labelled:
                ranked = [j for j, _, _ in searcher.rank_ids(item["query"], 20, "hybrid")]
                rows.append((ndcg_at_k(ranked, set(item["relevant"]), 10), item["query"]))
            for score, query in sorted(rows)[:5]:
                print(f"  {score:.3f}  {query}")


if __name__ == "__main__":
    main()
