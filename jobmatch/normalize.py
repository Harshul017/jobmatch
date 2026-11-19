"""Turning messy source data into consistent fields.

Every source gives us a different shape: Greenhouse sends HTML-escaped HTML,
Lever sends plain text, Ashby sends either. These helpers flatten all of that
into the same plain-text form so retrieval sees one consistent corpus.
"""
import hashlib
import html
import re

from . import locations

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9+#. ]")
# A dot at the end of a word is noise ("inc."); a dot inside one is not ("node.js").
_TRAILING_DOT = re.compile(r"\.(?=\s|$)")

REMOTE_HINTS = ("remote", "work from home", "wfh", "anywhere", "distributed")

# Checked in order; the first match wins. Management is checked before the
# individual-contributor levels so "Senior Engineering Manager" lands on
# management rather than senior, and junior is checked last so a title like
# "Graduate Program Manager" is not mistaken for an entry-level role.
SENIORITY_RULES = [
    ("intern",     r"\b(intern|interns|internship|trainee|apprentice)\b"),
    ("management", r"\b(manager|managers|director|head of|vp|vice president|chief|president)\b"),
    ("principal",  r"\b(principal|distinguished|fellow)\b"),
    ("staff",      r"\b(staff|architect)\b"),
    ("lead",       r"\b(lead|leads|leader|leadership)\b"),
    ("senior",     r"\b(senior|sr)\b"),
    ("junior",     r"\b(junior|jr|entry[- ]level|graduate|new grad|fresher)\b"),
]
_SENIORITY = [(level, re.compile(pat, re.I)) for level, pat in SENIORITY_RULES]


def strip_html(raw: str | None) -> str:
    """HTML (possibly entity-escaped) to readable plain text."""
    if not raw:
        return ""
    text = html.unescape(raw)
    # Unescape twice: Greenhouse escapes HTML that itself contains entities.
    if "&lt;" in text or "&amp;" in text:
        text = html.unescape(text)
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6])[^>]*>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    return collapse(text)


def collapse(text: str | None) -> str:
    """Collapse runs of whitespace but keep paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def norm_key(text: str | None) -> str:
    """Aggressively normalised form used only for comparing and hashing.

    Keeps the punctuation that is part of a technology name (node.js, c++, c#)
    but drops a trailing dot, so "Acme, Inc." and "acme inc" hash alike.
    """
    if not text:
        return ""
    cleaned = _PUNCT.sub(" ", text.lower())
    cleaned = _TRAILING_DOT.sub("", cleaned)
    return _WS.sub(" ", cleaned).strip()


def detect_remote(location: str | None, description: str = "", flag: bool | None = None) -> bool:
    """Sources disagree on how they mark remote, so check the flag then the text."""
    if flag is not None:
        return bool(flag)
    haystack = f"{location or ''} {description[:600]}".lower()
    return any(hint in haystack for hint in REMOTE_HINTS)


def detect_seniority(title: str | None) -> str | None:
    text = title or ""
    for level, pattern in _SENIORITY:
        if pattern.search(text):
            return level
    return None


# "5+ years", "3-5 years", "at least 6 years", "minimum of 4 yrs".
_YEARS = re.compile(r"""
    (?:at\s+least\s+|minimum\s+(?:of\s+)?|min\.?\s*|over\s+|)
    (?P<lo>\d{1,2})
    \s*(?:\+|plus)?
    (?:\s*(?:-|–|—|to)\s*\d{1,2}\s*(?:\+|plus)?)?
    \s*(?:\+|plus)?\s*
    (?:years?|yrs?)\b
""", re.I | re.X)

# Only count a figure that is actually about experience. "in the last 5 years"
# and "founded 10 years ago" are not requirements.
_EXPERIENCE = re.compile(r"experien|背景", re.I)
_LOOKAHEAD = 60


def min_years_required(text: str | None) -> int | None:
    """Lowest years-of-experience figure the posting asks for.

    Title-based seniority misses the common case: a role called "Software
    Engineer" that wants 6-8 years is senior in everything but its name. The
    requirement is in the description, so read it from there.
    """
    if not text:
        return None
    found = []
    for match in _YEARS.finditer(text):
        tail = text[match.end():match.end() + _LOOKAHEAD]
        if not _EXPERIENCE.search(tail):
            continue
        try:
            years = int(match.group("lo"))
        except (TypeError, ValueError):
            continue
        if 0 < years <= 25:       # ignore "1000 years" style noise
            found.append(years)
    return min(found) if found else None


# Paragraphs that are about the employer rather than the work.
_BOILERPLATE = re.compile(
    r"^(about (us|the company|\w+)|who we are|our (mission|story|company)|"
    r"company overview|at [A-Z][\w.& ]{2,30},)", re.I)
# Phrases that mark the start of the actual job.
_ROLE_SIGNAL = re.compile(
    r"(you will|you'll|responsibilit|what you.{0,10}(do|bring)|the role|"
    r"requirements|qualifications|we.{0,5}re looking for|your impact|"
    r"experience with|skills|minimum|preferred)", re.I)


def role_excerpt(text: str | None, limit: int = 700) -> str:
    """The part of a posting that describes the job.

    Most listings open with several paragraphs about the employer. Taking the
    first N characters therefore returns marketing copy, which is useless for
    judging whether a role matches a query -- and is what a relevance judge
    would otherwise be reading.
    """
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[:limit]

    start = 0
    for i, para in enumerate(paragraphs[:4]):
        if _ROLE_SIGNAL.search(para):
            start = i
            break
        if _BOILERPLATE.match(para):
            start = i + 1
    kept = " ".join(paragraphs[start:]) or " ".join(paragraphs)
    return collapse(kept)[:limit]


def job_id(source: str, source_id: str) -> str:
    """Stable primary key, so re-running ingestion updates rather than duplicates."""
    return hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()[:16]


def dedupe_key(company: str, title: str, location: str | None) -> str:
    """Catches the same role listed on two boards (e.g. Ashby and Arbeitnow).

    The location goes through the canonicaliser first. Comparing raw strings
    here meant "Bengaluru, India" and "Bangalore, Karnataka" never matched, so
    cross-posted roles were counted twice.
    """
    return hashlib.sha1(
        "|".join([norm_key(company), norm_key(title), locations.canonical(location)]).encode()
    ).hexdigest()[:16]
