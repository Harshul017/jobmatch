"""Talk to the agent from the terminal.

    python -m jobmatch.chat
    python -m jobmatch.chat --once "backend roles in bengaluru, nothing senior"
    python -m jobmatch.chat --model openai/gpt-oss-20b --verbose
"""
import argparse
import json
import time

from . import db
from .agent import Agent
from .embed import DEFAULT_MODEL
from .index import open_store
from .llm import DEFAULT_MODEL as LLM_MODEL
from .llm import GroqLLM
from .models import Preference
from .search import Searcher


def live_step(step, output) -> None:
    """Print each tool call the moment it completes."""
    if step.error:
        print(f"  ... {step.tool} -> {step.error}", flush=True)
        return
    if step.repeated:
        print(f"  ... {step.tool} (repeat, served from cache)", flush=True)
        return
    detail = ""
    if step.tool == "search_jobs":
        n = len(output.get("results", []))
        relaxed = " (filters relaxed)" if output.get("relaxed_filters") else ""
        detail = f"-> {n} results{relaxed}"
    elif step.tool == "get_job":
        detail = f"-> {output.get('title', output.get('error', ''))}"
    elif step.tool == "save_preference":
        detail = f"-> {output.get('changed') or 'no change'}"
    args = {k: v for k, v in step.arguments.items() if v not in (None, [], "")}
    print(f"  ... {step.tool}({args}) {detail}", flush=True)


def build(model: str, embed_model: str, store: str, user: str,
          verbose: bool = False) -> Agent:
    print("loading index and embedding model...", flush=True)
    conn = db.connect()
    from .embed import get_embedder
    searcher = Searcher(conn, open_store(store, embed_model), get_embedder(embed_model))
    return Agent(conn, searcher, GroqLLM(model=model), user_id=user,
                 on_step=live_step if verbose else None)


def report(result, verbose: bool, elapsed: float = 0.0) -> None:
    print(f"\n{result.reply}\n")
    if result.remembered:
        kept = ", ".join(f"{k}={v}" for k, v in result.remembered.items())
        print(f"  (remembered: {kept} - say 'forget that' or run with "
              f"--forget to clear)")
    if verbose:
        turns = " + ".join(f"{s:.1f}" for s in result.llm_seconds)
        other = elapsed - result.llm_total
        print(f"  [{len(result.steps)} calls, {result.invalid_calls} invalid, "
              f"grounded={result.grounded}, fell_back={result.fell_back}]")
        print(f"  [{elapsed:.1f}s total = {result.llm_total:.1f}s model "
              f"({turns}) + {other:.1f}s local]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with the job agent.")
    ap.add_argument("--once", help="send one message and exit")
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--embed-model", default=DEFAULT_MODEL)
    ap.add_argument("--store", choices=["qdrant", "numpy"], default="numpy")
    ap.add_argument("--user", default="default", help="whose preferences to use")
    ap.add_argument("--verbose", action="store_true", help="show tool calls")
    ap.add_argument("--show-prefs", action="store_true", help="print what is remembered")
    ap.add_argument("--forget", action="store_true", help="clear remembered preferences")
    args = ap.parse_args()

    if args.show_prefs or args.forget:
        conn = db.connect()
        if args.forget:
            db.save_preference(conn, Preference(user_id=args.user))
            print(f"Cleared preferences for {args.user!r}.")
        else:
            pref = db.load_preference(conn, args.user)
            known = {k: v for k, v in
                     pref.model_dump(exclude={"user_id", "updated_at"}).items() if v}
            print(json.dumps(known, indent=2) if known else "Nothing remembered yet.")
        return

    agent = build(args.model, args.embed_model, args.store, args.user, args.verbose)

    if args.once:
        started = time.time()
        report(agent.run(args.once), args.verbose, time.time() - started)
        return

    print(f"Job agent ({args.model}). Ctrl-C or 'quit' to exit.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message.lower() in {"quit", "exit", "q"}:
            return
        if message.lower() in {"/prefs", "prefs"}:
            pref = db.load_preference(agent.conn, args.user)
            known = {k: v for k, v in
                     pref.model_dump(exclude={"user_id", "updated_at"}).items() if v}
            print(json.dumps(known, indent=2) if known else "  nothing remembered yet")
            continue
        if message.lower() in {"/forget", "forget"}:
            db.save_preference(agent.conn, Preference(user_id=args.user))
            print("  cleared")
            continue
        if not message:
            continue
        try:
            started = time.time()
            report(agent.run(message), args.verbose, time.time() - started)
        except Exception as exc:
            print(f"\n  failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
