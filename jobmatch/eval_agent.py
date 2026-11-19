"""Measure the agent, not just retrieval.

Retrieval metrics say nothing about whether the model called the right tool,
honoured a constraint, or made a job up. These scenarios check behaviour that
has to hold whatever wording the model chooses, so the suite stays meaningful
when the prompt or the model changes.

Unlike the unit tests, this calls the real model and costs API requests.

    python -m jobmatch.eval_agent
    python -m jobmatch.eval_agent --model openai/gpt-oss-20b
    python -m jobmatch.eval_agent --only senior
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable

from . import db
from .agent import Agent, AgentResult
from .embed import DEFAULT_MODEL as EMBED_MODEL
from .embed import get_embedder
from .index import open_store
from .llm import DEFAULT_MODEL as LLM_MODEL
from .llm import GroqLLM
from .models import Job
from .search import Searcher

SENIOR_LEVELS = {"senior", "staff", "principal", "management"}


@dataclass
class Scenario:
    name: str
    message: str
    # Each check gets the result and the jobs it cited, and returns a reason
    # for failing or None for a pass.
    checks: list[Callable[[AgentResult, list[Job]], str | None]] = field(default_factory=list)
    setup: Callable | None = None


# --- reusable checks ------------------------------------------------------

def must_search(result, jobs):
    if not any(s.tool == "search_jobs" for s in result.steps):
        return "never called search_jobs"
    return None


def must_not_fall_back(result, jobs):
    return "fell back to plain search" if result.fell_back else None


def must_be_grounded(result, jobs):
    return "cited a job no tool returned" if not result.grounded else None


def no_invalid_calls(result, jobs):
    if result.invalid_calls:
        bad = [s.error for s in result.steps if s.error]
        return f"{result.invalid_calls} invalid tool calls: {bad[0]}"
    return None


def must_recommend(result, jobs):
    return "recommended nothing" if not jobs else None


def no_senior_roles(result, jobs):
    bad = [j.title for j in jobs if (j.seniority or "") in SENIOR_LEVELS]
    return f"returned senior roles: {bad[:2]}" if bad else None


def all_in_india(result, jobs):
    bad = [f"{j.title} ({j.location_canonical})" for j in jobs
           if not (j.location_canonical or "").endswith("in")]
    return f"returned non-India roles: {bad[:2]}" if bad else None


def all_remote(result, jobs):
    bad = [j.title for j in jobs if not j.remote]
    return f"returned on-site roles: {bad[:2]}" if bad else None


def only_interns(result, jobs):
    bad = [f"{j.title} ({j.seniority})" for j in jobs if j.seniority != "intern"]
    return f"returned non-intern roles: {bad[:2]}" if bad else None


def remembered_something(result, jobs):
    return "saved no preference" if not result.remembered else None


def remembered_nothing(result, jobs):
    if result.remembered:
        return f"saved a preference it was never told: {result.remembered}"
    return None


def at_most_two_per_company(result, jobs):
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.company] = counts.get(job.company, 0) + 1
    over = {c: n for c, n in counts.items() if n > 2}
    return f"one employer dominated: {over}" if over else None


BASE = [must_search, must_not_fall_back, must_be_grounded, no_invalid_calls]

SCENARIOS = [
    Scenario("plain_search", "backend engineering roles in Bengaluru",
             BASE + [must_recommend, all_in_india, at_most_two_per_company]),
    Scenario("senior_excluded", "backend roles in bengaluru, nothing senior",
             BASE + [must_recommend, no_senior_roles, all_in_india]),
    Scenario("intern_only", "software engineering internships in India",
             BASE + [must_recommend, only_interns]),
    Scenario("remote_only", "remote machine learning roles I can do from India",
             BASE + [must_recommend, all_remote]),
    Scenario("semantic", "jobs about making search results more relevant",
             BASE + [must_recommend]),
    Scenario("states_a_preference", "I'm in Bengaluru and only want remote work",
             [must_be_grounded, no_invalid_calls, remembered_something]),
    Scenario("asks_without_stating", "what backend roles are there?",
             [must_search, must_be_grounded, no_invalid_calls, remembered_nothing]),
    Scenario("nothing_matches", "underwater basket weaving roles in Antarctica",
             [must_be_grounded, no_invalid_calls]),
    Scenario("refuses_to_invent",
             "tell me about the Senior Rust Engineer role at Mozilla",
             [must_be_grounded, no_invalid_calls]),
]


def run_one(scenario: Scenario, make_agent) -> dict:
    agent = make_agent()
    if scenario.setup:
        scenario.setup(agent)
    started = time.perf_counter()
    try:
        result = agent.run(scenario.message)
    except Exception as exc:
        return {"name": scenario.name, "failures": [f"raised {type(exc).__name__}: {exc}"],
                "seconds": time.perf_counter() - started, "result": None}
    elapsed = time.perf_counter() - started

    jobs = [j for j in (db.get_job(agent.conn, i) for i in result.cited) if j]
    failures = [reason for check in scenario.checks
                if (reason := check(result, jobs)) is not None]
    return {"name": scenario.name, "failures": failures, "seconds": elapsed,
            "result": result, "jobs": jobs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate agent behaviour.")
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--store", choices=["qdrant", "numpy"], default="numpy")
    ap.add_argument("--only", help="run scenarios whose name contains this")
    ap.add_argument("--verbose", action="store_true", help="show each reply")
    args = ap.parse_args()

    conn = db.connect()
    searcher = Searcher(conn, open_store(args.store, args.embed_model),
                        get_embedder(args.embed_model))
    llm = GroqLLM(model=args.model)

    def make_agent():
        # A fresh user per scenario, so saved preferences cannot leak between
        # them and make a later result depend on an earlier one.
        return Agent(conn, searcher, llm, user_id=f"eval-{time.time_ns()}")

    scenarios = [s for s in SCENARIOS if not args.only or args.only in s.name]
    print(f"{len(scenarios)} scenarios | {args.model}\n")

    rows, totals = [], {"grounded": 0, "fell_back": 0, "invalid": 0, "steps": 0}
    for scenario in scenarios:
        row = run_one(scenario, make_agent)
        rows.append(row)
        status = "pass" if not row["failures"] else "FAIL"
        print(f"  {status}  {row['name']:<22} {row['seconds']:>5.1f}s")
        for reason in row["failures"]:
            print(f"        - {reason}")
        if args.verbose and row["result"]:
            print(f"        {row['result'].reply[:300]}")
        if row["result"]:
            totals["grounded"] += row["result"].grounded
            totals["fell_back"] += row["result"].fell_back
            totals["invalid"] += row["result"].invalid_calls
            totals["steps"] += len(row["result"].steps)

    n = len(rows)
    passed = sum(1 for r in rows if not r["failures"])
    ran = sum(1 for r in rows if r["result"])
    print(f"\n{passed}/{n} scenarios passed")
    if ran:
        print(f"grounded      {totals['grounded']}/{ran}")
        print(f"fell back     {totals['fell_back']}/{ran}")
        print(f"invalid calls {totals['invalid']}")
        print(f"tool calls    {totals['steps'] / ran:.1f} per turn")
    print(f"latency       {sum(r['seconds'] for r in rows) / n:.1f}s mean")


if __name__ == "__main__":
    main()
