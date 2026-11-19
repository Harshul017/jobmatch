"""Find out why embedding is slow, on the machine where it is slow.

Indexing 7k jobs took ~3 hours on an 8GB laptop, which is far off what a 33M
parameter model should do. Rather than guess, this times a small sample under
different settings so the actual cause is visible.

    python -m jobmatch.bench_embed
"""
import argparse
import os
import time

from . import db
from .embed import DEFAULT_MODEL

SAMPLE = 24


def time_config(texts, threads, chars, batch, parallel):
    from fastembed import TextEmbedding

    cut = [t[:chars] for t in texts]
    load0 = time.time()
    model = TextEmbedding(model_name=DEFAULT_MODEL, threads=threads)
    load = time.time() - load0

    list(model.embed(cut[:2], batch_size=batch))  # warm up, exclude from timing
    t0 = time.time()
    vectors = list(model.embed(cut, batch_size=batch, parallel=parallel))
    elapsed = time.time() - t0
    assert len(vectors) == len(cut)
    return load, elapsed, len(cut) / elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark embedding throughput.")
    ap.add_argument("-n", type=int, default=SAMPLE, help="documents per config")
    args = ap.parse_args()

    cores = os.cpu_count() or 4
    conn = db.connect()
    jobs = db.active_jobs(conn, args.n)
    if not jobs:
        raise SystemExit("No jobs indexed. Run: python -m jobmatch.ingest")
    texts = [j.embedding_text(4000) for j in jobs]
    print(f"{len(texts)} real job texts | {cores} cores | "
          f"avg {sum(len(t) for t in texts)//len(texts)} chars\n")

    configs = [
        ("1 thread,  1200 chars", 1, 1200, 32, None),
        (f"{cores} threads, 1200 chars", cores, 1200, 32, None),
        (f"{cores} threads,  600 chars", cores, 600, 32, None),
        (f"{cores} threads,  300 chars", cores, 300, 32, None),
        (f"{cores} threads, 1200 chars, batch 8", cores, 1200, 8, None),
        (f"{cores} threads, 1200 chars, multiproc", cores, 1200, 32, 0),
    ]

    print(f"{'config':<38} {'load':>7} {'embed':>8} {'docs/s':>8} {'est 7k':>9}")
    print("-" * 74)
    for label, threads, chars, batch, parallel in configs:
        try:
            load, elapsed, rate = time_config(texts, threads, chars, batch, parallel)
            est = 7164 / rate
            est_s = f"{est/60:.0f} min" if est > 90 else f"{est:.0f}s"
            print(f"{label:<38} {load:>6.1f}s {elapsed:>7.1f}s {rate:>8.1f} {est_s:>9}")
        except Exception as exc:
            print(f"{label:<38}  failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
