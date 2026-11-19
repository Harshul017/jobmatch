"""Shared HTTP plumbing for the public job feeds.

These are other people's free endpoints, so every call gets a timeout, a small
delay, and a bounded retry. A source that fails never takes the whole run down.
"""
import time

import httpx

from ..config import HTTP_TIMEOUT, REQUEST_DELAY

USER_AGENT = "jobmatch/0.1 (student project; contact via repo)"

RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceError(Exception):
    pass


def get_json(url: str, params: dict | None = None, attempts: int = 3) -> dict | list:
    """GET returning parsed JSON, with backoff on transient failures.

    Raises SourceError on a 404 (usually a wrong slug) without retrying, since
    retrying a wrong slug just wastes the source's bandwidth.
    """
    last = None
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 404:
                raise SourceError(f"404 {url}")
            if resp.status_code in RETRY_STATUS:
                last = SourceError(f"{resp.status_code} {url}")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except SourceError:
            raise
        except Exception as exc:  # network error, bad JSON
            last = exc
            time.sleep(2 ** attempt)
    raise SourceError(f"failed after {attempts} attempts: {url} ({last})")


def first(d: dict, *keys, default=None):
    """Return the first key present and truthy. Feeds rename fields over time."""
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default
