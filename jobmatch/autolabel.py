"""Judge relevance with a model, then measure how far to trust it.

Hand-labelling is the gold standard and this is not that. It is the standard
fallback: a model judges the pooled candidates, and a human spot-checks a
random sample so the agreement rate is known rather than assumed. Retrieval
numbers built on these judgements have to be reported with that caveat, which
is why `--verify` exists and why the agreement is written into the file.

    python -m jobmatch.autolabel            # judge every seed query
    python -m jobmatch.autolabel --verify 40  # spot-check a sample yourself
    python -m jobmatch.autolabel --report
"""
from __future__ import annotations

import argparse
import json
import random

from pydantic import BaseModel, Field, ValidationError

from . import db, normalize
from .embed import DEFAULT_MODEL as EMBED_MODEL
from .embed import get_embedder
from .index import open_store
from .label import EVAL_FILE, SEED_FILE, load_eval, pool_candidates, save_eval
from .llm import DEFAULT_MODEL as LLM_MODEL
from .llm import GroqLLM, RateLimited
from .search import Searcher

PROMPT = """You are judging search results for a job search engine.

Someone searched for: "{query}"

For each listing below, decide whether it is a result they would be glad to
see. Judge the kind of work, not how senior it is and not where it is: a
reasonable match in the wrong city is still relevant to the query.

{candidates}

Reply with JSON only, no prose:
{{"relevant": [list of numbers that are relevant]}}"""


class Judgement(BaseModel):
    relevant: list[int] = Field(default_factory=list)


def judge(llm, query: str, candidates: list[dict]) -> set[int] | None:
    listing = "\n".join(
        f"{n}. {c['title']} — {c['company']} — {c['snippet']}"
        for n, c in enumerate(candidates, 1)
    )
    messages = [{"role": "user",
                 "content": PROMPT.format(query=query, candidates=listing)}]
    for attempt in range(2):
        reply = llm.chat(messages)
        text = reply.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                picked = Judgement(**json.loads(text[start:end + 1])).relevant
                return {n for n in picked if 1 <= n <= len(candidates)}
            except (json.JSONDecodeError, ValidationError):
                pass
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": 'That was not valid JSON. Reply with exactly '
                                    '{"relevant": [1, 2]} and nothing else.'})
    return None


def run(llm, searcher, conn, queries: list[str], data: dict) -> None:
    done = {q["query"] for q in data["queries"]}
    for i, query in enumerate(queries, 1):
        if query in done:
            print(f"  [{i}/{len(queries)}] {query[:48]:50} already labelled")
            continue
        pool = pool_candidates(searcher, query)
        if not pool:
            print(f"  [{i}/{len(queries)}] {query[:48]:50} no candidates")
            continue
        candidates = []
        for job_id in pool:
            job = db.get_job(conn, job_id)
            if job:
                candidates.append({"job_id": job_id, "title": job.title,
                                   "company": job.company,
                                   "snippet": normalize.role_excerpt(job.description)})
        try:
            picked = judge(llm, query, candidates)
        except RateLimited as exc:
            # Everything judged so far is already saved.
            print(f"\n  stopped at query {i}: the model's daily budget is used up.")
            print(f"  {str(exc)[:200]}")
            print("\n  Either wait, or finish with a model that has its own budget:")
            print("    python -m jobmatch.autolabel --model openai/gpt-oss-20b")
            return
        if picked is None:
            print(f"  [{i}/{len(queries)}] {query[:48]:50} judge failed")
            continue
        relevant = [candidates[n - 1]["job_id"] for n in sorted(picked)]
        data["queries"].append({
            "query": query,
            "relevant": relevant,
            "judged": [c["job_id"] for c in candidates],
            "relevant_titles": [f"{candidates[n-1]['title']} @ {candidates[n-1]['company']}"
                                for n in sorted(picked)],
            "judged_by": "llm",
        })
        save_eval(data)
        print(f"  [{i}/{len(queries)}] {query[:48]:50} {len(relevant)}/{len(candidates)} relevant")


def verify(data: dict, conn, sample_size: int) -> None:
    """Spot-check the model's judgements, and record the agreement rate.

    Without this the labels are an unmeasured guess. With it they are a
    measured one, and the measurement goes in the file so any metric computed
    from these labels can be reported honestly.
    """
    pairs = []
    for entry in data["queries"]:
        if entry.get("judged_by") != "llm":
            continue
        relevant = set(entry["relevant"])
        for job_id in entry["judged"]:
            pairs.append((entry["query"], job_id, job_id in relevant))
    if not pairs:
        raise SystemExit("Nothing to verify. Run the labeller first.")

    random.seed(0)                       # same sample every run, so it is reviewable
    sample = random.sample(pairs, min(sample_size, len(pairs)))
    agree = checked = 0
    print(f"Checking {len(sample)} of {len(pairs)} judgements. "
          f"y = relevant, n = not, s = skip, q = stop.\n")

    for n, (query, job_id, model_said) in enumerate(sample, 1):
        job = db.get_job(conn, job_id)
        if not job:
            continue
        print(f"[{n}/{len(sample)}] query: {query}")
        print(f"          {job.title} — {job.company}")
        print(f"          {normalize.role_excerpt(job.description, 320)}...")
        answer = input("  relevant? ").strip().lower()
        if answer == "q":
            break
        if answer not in {"y", "n"}:
            continue
        checked += 1
        agree += (answer == "y") == model_said
        print(f"          model said {'relevant' if model_said else 'not relevant'}"
              f" -> {'agree' if (answer == 'y') == model_said else 'DISAGREE'}\n")

    if checked:
        rate = agree / checked
        data["verification"] = {"checked": checked, "agreed": agree,
                                "agreement": round(rate, 3)}
        save_eval(data)
        print(f"\nAgreement: {agree}/{checked} = {rate:.0%}")
        print("Recorded in data/eval.json. Quote it alongside any retrieval metric.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label with a model, verify by hand.")
    ap.add_argument("--verify", type=int, metavar="N", help="spot-check N judgements")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--redo", action="store_true",
                    help="discard model judgements and judge again (keeps human ones)")
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    args = ap.parse_args()

    data = load_eval()
    conn = db.connect()

    if args.report:
        by = {}
        for q in data["queries"]:
            by[q.get("judged_by", "human")] = by.get(q.get("judged_by", "human"), 0) + 1
        total = sum(len(q["relevant"]) for q in data["queries"])
        print(f"{len(data['queries'])} queries, {total} relevant judgements")
        print(f"source: {by}")
        if "verification" in data:
            v = data["verification"]
            print(f"human agreement: {v['agreed']}/{v['checked']} = {v['agreement']:.0%}")
        else:
            print("not yet verified - run with --verify 40")
        return

    if args.verify:
        verify(data, conn, args.verify)
        return

    if args.redo:
        before = len(data["queries"])
        data["queries"] = [q for q in data["queries"] if q.get("judged_by") != "llm"]
        data.pop("verification", None)
        save_eval(data)
        print(f"discarded {before - len(data['queries'])} model judgements\n")

    if not SEED_FILE.exists():
        raise SystemExit(f"No {SEED_FILE}")
    queries = [l.strip() for l in SEED_FILE.read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#")]
    searcher = Searcher(conn, open_store("numpy", args.embed_model),
                        get_embedder(args.embed_model))
    print(f"Judging {len(queries)} queries with {args.model}\n")
    run(GroqLLM(model=args.model), searcher, conn, queries, data)
    print(f"\n{len(data['queries'])} queries labelled. "
          f"Spot-check them: python -m jobmatch.autolabel --verify 40")


if __name__ == "__main__":
    main()
