"""The agent loop.

A small open-weights model drives the conversation, but it is kept on a short
leash: it can only act through three validated tools, the loop is capped, and
anything it claims about a job has to cite an id that a tool actually
returned. Ranking stays in the retrieval code, where it can be measured --
the model chooses what to look for and explains the result, it does not decide
what is relevant.
"""
from __future__ import annotations

import json
import re
import time
import sqlite3
from dataclasses import dataclass, field

from . import db, tools
from .llm import LLM, Reply
from .search import Filters, Searcher

# Each recommendation can cost a search plus a get_job to read the listing,
# so a budget of 4 ran out mid-task and fell back on perfectly good runs.
MAX_STEPS = 8
CITATION = re.compile(r"\[([0-9a-f]{12,16})\]")

SYSTEM = """You help someone find a job from a live index of real listings.

Rules you must follow:
- Never describe, recommend or invent a job you have not retrieved with a tool.
  If you have not searched yet, search first.
- search_jobs already returns a snippet of each listing. That is normally
  enough to say why a role fits. Only call get_job when the person asks about
  one role in detail, or the snippet genuinely does not answer the question.
- Answer as soon as you can. One good search then a reply is the usual shape
  of a turn. Repeated searching is how you run out of time and return nothing.
- Cite every job you mention with its id in square brackets, like [a1b2c3d4e5f6].
  Only ids returned by a tool in this conversation are valid.
- Call save_preference only for what the person actually said, and only for
  things that should still hold next time. Send just those fields. Do not
  guess the rest: saving remote_only=false because remote was never mentioned
  silently hides every remote job from them later. Asking for one search with
  a constraint is not a standing preference.
- "nothing senior", "entry level" and similar mean exclude_seniority, not a
  narrow search for intern and junior titles only. Most good matches are
  mid-level, which is seniority "mid".
- Recommend at most 4 roles. Say briefly why each one fits, using only what
  the listing says. If little fits, say so rather than padding the list.
- Be concise and plain. No preamble, no bullet-point boilerplate.
{preferences}"""


@dataclass
class Step:
    tool: str
    arguments: dict
    error: str | None = None
    repeated: bool = False


@dataclass
class AgentResult:
    reply: str
    steps: list[Step] = field(default_factory=list)
    cited: set[str] = field(default_factory=set)
    grounded: bool = True
    fell_back: bool = False
    remembered: dict = field(default_factory=dict)
    llm_seconds: list[float] = field(default_factory=list)

    @property
    def llm_total(self) -> float:
        return sum(self.llm_seconds)

    @property
    def invalid_calls(self) -> int:
        return sum(1 for s in self.steps if s.error)


def _preferences_block(pref) -> str:
    known = {k: v for k, v in pref.model_dump(exclude={"user_id", "updated_at"}).items() if v}
    if not known:
        return ""
    lines = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in known.items())
    return f"\nWhat you already know about this person:\n{lines}"


class Agent:
    def __init__(self, conn: sqlite3.Connection, searcher: Searcher, llm: LLM,
                 user_id: str = "default", max_steps: int = MAX_STEPS,
                 on_step=None):
        self.conn, self.searcher, self.llm = conn, searcher, llm
        self.user_id, self.max_steps = user_id, max_steps
        self.history: list[dict] = []
        # Called as each tool finishes, so a caller can show progress rather
        # than leaving the person watching a blank terminal for half a minute.
        self.on_step = on_step

    def run(self, message: str) -> AgentResult:
        ctx = tools.ToolContext(self.conn, self.searcher, self.user_id)
        system = SYSTEM.format(preferences=_preferences_block(ctx.preference))
        messages = [{"role": "system", "content": system},
                    *self.history,
                    {"role": "user", "content": message}]
        result = AgentResult(reply="")
        # Models re-issue an identical call when they lose track of what they
        # already did. Serving it from memory costs nothing and keeps the
        # step budget for real work.
        done: dict[str, dict] = {}

        for _ in range(self.max_steps):
            _compact(messages)
            started = time.perf_counter()
            reply = self.llm.chat(messages, tools=tools.schemas())
            result.llm_seconds.append(time.perf_counter() - started)
            if not reply.tool_calls:
                result.reply = reply.content
                break
            messages.append(_assistant_turn(reply))
            for call in reply.tool_calls:
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                if key in done:
                    output, error = done[key], None
                    repeated = True
                else:
                    output, error = tools.execute(ctx, call.name, call.arguments)
                    if error is None:
                        done[key] = output
                    repeated = False
                step = Step(call.name, call.arguments, error, repeated=repeated)
                result.steps.append(step)
                if self.on_step:
                    self.on_step(step, output)
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": json.dumps({"error": error} if error else output)[:2500],
                })
        else:
            # Ran out of steps without an answer: do the obvious search ourselves
            # rather than returning nothing.
            result.fell_back = True
            result.reply = self._fallback(ctx, message)

        if not result.reply:
            result.fell_back = True
            result.reply = self._fallback(ctx, message)

        result.cited = set(CITATION.findall(result.reply))
        invented = result.cited - ctx.seen
        result.grounded = not invented
        if invented:
            result.reply = self._strip_invented(result.reply, invented)

        result.remembered = dict(ctx.remembered)
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": result.reply}]
        return result

    def _fallback(self, ctx: tools.ToolContext, message: str) -> str:
        """Plain retrieval, no model. The person still gets results.

        It reuses whatever filters the model had already worked out, because
        dropping them answers a different question: a fallback that ignored
        "nothing senior" cheerfully returned four senior roles.
        """
        filters = ctx.first_filters or _filters_from(ctx.preference)
        query = ctx.best_query or message
        hits = self.searcher.search(query, k=4, filters=filters)
        note = ""
        if not hits and filters and not filters.is_empty():
            hits = self.searcher.search(query, k=4)
            note = " (nothing matched your filters, so these ignore them)"
        if not hits:
            return "I could not find anything matching that in the index."
        ctx.seen.update(h.job_id for h in hits)
        lines = [f"- {h.job.title} at {h.job.company}"
                 f" ({h.job.location_canonical or 'location unknown'}) [{h.job_id}]"
                 for h in hits]
        return f"Here is what the search returned{note}:\n" + "\n".join(lines)

    @staticmethod
    def _strip_invented(text: str, invented: set[str]) -> str:
        """Drop lines citing ids no tool returned, so a hallucinated listing
        never reaches the person."""
        kept = [line for line in text.splitlines()
                if not any(f"[{i}]" in line for i in invented)]
        cleaned = "\n".join(kept).strip()
        return cleaned or "I could not confirm those roles against the index."


KEEP_FULL_RESULTS = 3


def _compact(messages: list[dict]) -> None:
    """Shrink tool results the model has already acted on.

    Every tool result is resent on every later turn, so latency grows with the
    square of how many calls have been made -- measured at 0.9s for the first
    round trip and 24.8s for the eighth. Older results are replaced with a
    stub: the model keeps the thread of what it did without re-reading all of
    it. The most recent few stay intact, since those are what it is reasoning
    about now.
    """
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_indexes[:-KEEP_FULL_RESULTS]:
        content = messages[i].get("content") or ""
        if content.startswith('{"seen_earlier"'):
            continue
        messages[i]["content"] = json.dumps(
            {"seen_earlier": True,
             "note": f"{messages[i].get('name')} result omitted to save context"}
        )


def _filters_from(pref) -> Filters:
    """Fall back to the person's standing preferences when the model never got
    as far as making a search of its own."""
    return Filters(
        location=pref.locations[0] if pref.locations else None,
        remote=pref.remote_only,
        seniority=[pref.seniority] if pref.seniority else [],
        exclude_companies=pref.avoid,
    )


def _assistant_turn(reply: Reply) -> dict:
    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [{"id": c.id, "type": "function",
                        "function": {"name": c.name,
                                     "arguments": json.dumps(c.arguments)}}
                       for c in reply.tool_calls],
    }
