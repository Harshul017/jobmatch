"""LLM access behind one interface.

Groq serves open-weights models (gpt-oss, Qwen, Llama) over an OpenAI-shaped
API. Everything here goes through `LLM`, so the agent never imports a vendor
SDK and the tests never need a network.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import GROQ_API_KEY, GROQ_MODEL

DEFAULT_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "openai/gpt-oss-20b"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply: ...


class RateLimited(RuntimeError):
    """The daily or per-minute budget is gone. Waiting is the only fix, so
    callers should stop cleanly and keep whatever work is already done."""


class GroqLLM:
    """Retries on transient failures. A free-tier rate limit is a 429, which is
    worth waiting out rather than failing the whole conversation."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 60.0):
        from groq import Groq

        key = api_key or GROQ_API_KEY
        if not key:
            raise RuntimeError("No GROQ_API_KEY. Copy .env.example to .env and add one.")
        self.model = model or GROQ_MODEL or DEFAULT_MODEL
        self.temperature = temperature
        self.client = Groq(api_key=key, timeout=timeout)

    def chat(self, messages, tools=None, attempts: int = 3) -> Reply:
        last = None
        for attempt in range(attempts):
            try:
                kwargs = {"model": self.model, "messages": messages,
                          "temperature": self.temperature}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**kwargs)
                return _to_reply(response)
            except Exception as exc:
                last = exc
                text = str(exc)
                # A per-day budget will not come back during this run, so
                # retrying three times just wastes the caller's time.
                if "rate_limit" in text and ("per day" in text or "TPD" in text):
                    raise RateLimited(text) from exc
                if attempt == attempts - 1:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq call failed after {attempts} attempts: {last}")


def _to_reply(response) -> Reply:
    message = response.choices[0].message
    calls = []
    for call in (getattr(message, "tool_calls", None) or []):
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # Keep the raw text: the agent reports it back to the model so it
            # can correct itself, rather than silently dropping the call.
            args = {"__malformed__": call.function.arguments}
        calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))
    return Reply(content=message.content or "", tool_calls=calls)


class ScriptedLLM:
    """Returns a fixed sequence of replies. Lets the tests drive the agent
    through tool loops, malformed arguments and refusals deterministically."""

    model = "scripted"

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None) -> Reply:
        self.calls.append(messages)
        if not self.replies:
            return Reply(content="(no more scripted replies)")
        return self.replies.pop(0)
