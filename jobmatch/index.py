"""Build the vector index over whatever is currently in SQLite.

    python -m jobmatch.index                 # Qdrant, default model
    python -m jobmatch.index --store numpy   # exact brute-force baseline
    python -m jobmatch.index --model sentence-transformers/all-MiniLM-L6-v2
"""
import argparse
import time

import numpy as np

from . import db
from .config import DATA_DIR, QDRANT_PATH
from .embedcache import EmbeddingCache, text_key
from .embed import DEFAULT_MODEL, get_embedder
from .vectorstore import FILTER_FIELDS, NumpyStore, QdrantStore


def open_store(kind: str, model_name: str = DEFAULT_MODEL):
    """Each model gets its own collection, so two models can be compared
    without re-indexing one over the other."""
    tag = model_name.split("/")[-1].lower()
    if kind == "numpy":
        return NumpyStore(DATA_DIR / f"vectors-{tag}.npz")
    store = QdrantStore(QDRANT_PATH)
    store.COLLECTION = f"jobs-{tag}"
    return store


def payload_for(job) -> dict:
    data = {
        "company": job.company,
        "location_canonical": job.location_canonical,
        "remote": "yes" if job.remote else "no",
        "seniority": job.seniority or "",
        "source": job.source,
    }
    return {k: data[k] for k in FILTER_FIELDS}


def build(kind: str = "numpy", model_name: str = DEFAULT_MODEL,
          batch: int = 256, limit: int | None = None,
          threads: int | None = None, chars: int = 1200,
          use_cache: bool = True) -> None:
    conn = db.connect()
    jobs = db.active_jobs(conn, limit)
    if not jobs:
        raise SystemExit("Index is empty. Run: python -m jobmatch.ingest")

    cache = EmbeddingCache(DATA_DIR / "embeddings.db", model_name) if use_cache else None
    texts = {j.id: j.embedding_text(chars) for j in jobs}
    keys = {j.id: text_key(model_name, j.id, texts[j.id]) for j in jobs}
    cached = cache.get_many(list(keys.values())) if cache else {}
    todo = [j for j in jobs if keys[j.id] not in cached]

    print(f"{len(jobs)} jobs | {len(cached)} cached | {len(todo)} to embed "
          f"| model {model_name}")

    embedder = None
    if todo:
        embedder = get_embedder(model_name, threads=threads)
        print(f"  {getattr(embedder, 'threads', 1)} threads, {chars} chars per job")

    started = time.time()
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        vectors = embedder.encode([texts[j.id] for j in chunk])
        if cache:
            cache.put_many([keys[j.id] for j in chunk], vectors)
        for job, vector in zip(chunk, vectors):
            cached[keys[job.id]] = vector
        done = min(start + batch, len(todo))
        rate = done / max(time.time() - started, 1e-6)
        remaining = (len(todo) - done) / max(rate, 1e-6)
        print(f"  embedded {done}/{len(todo)}  ({rate:.1f}/sec, "
              f"~{remaining/60:.0f} min left)   ", end="\r", flush=True)
    if todo:
        print()

    dim = len(next(iter(cached.values()))) if cached else (embedder.dim if embedder else 0)
    store = open_store(kind, model_name)
    store.reset(dim)
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        vectors = np.vstack([cached[keys[j.id]] for j in chunk])
        store.upsert([j.id for j in chunk], vectors, [payload_for(j) for j in chunk])

    if isinstance(store, NumpyStore):
        store.save()
    if cache:
        cache.close()
    print(f"Indexed {store.count()} vectors ({dim}-dim) into {store.name} "
          f"in {time.time() - started:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the vector index.")
    ap.add_argument("--store", choices=["qdrant", "numpy"], default="numpy")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, help="index only the first N jobs")
    ap.add_argument("--threads", type=int, help="ONNX threads (default: all cores)")
    ap.add_argument("--chars", type=int, default=1200,
                    help="characters of description to embed (lower is faster)")
    ap.add_argument("--no-cache", action="store_true", help="re-embed everything")
    args = ap.parse_args()
    build(args.store, args.model, limit=args.limit, threads=args.threads,
          chars=args.chars, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
