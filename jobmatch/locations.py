"""Canonicalising location strings.

The same office shows up as "bengaluru", "Bangalore, Karnataka",
"Bengaluru, India" and "Bengaluru-VTP, India" depending on which board the
listing came from. Left alone, that breaks two things: filtering by city, and
recognising that two boards are advertising the same role.

This maps the messy string onto a canonical "city, country" pair where it can,
and degrades gracefully to a tidied string where it cannot.
"""
import re

# Older names, anglicisations and metro areas that should collapse together.
CITY_ALIASES = {
    "bangalore": "bengaluru", "bangalore city": "bengaluru", "blr": "bengaluru",
    "bombay": "mumbai", "navi mumbai": "mumbai", "thane": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "poona": "pune",
    "gurgaon": "gurugram", "new delhi": "delhi", "noida": "delhi",
    "ncr": "delhi", "delhi ncr": "delhi", "gaziabad": "delhi",
    "trivandrum": "thiruvananthapuram", "mysore": "mysuru",
    "nyc": "new york", "new york city": "new york", "sf": "san francisco",
    "bay area": "san francisco", "sfo": "san francisco",
    "bengaluru urban": "bengaluru",
}

INDIAN_STATES = {
    "karnataka", "maharashtra", "telangana", "tamil nadu", "delhi", "haryana",
    "uttar pradesh", "west bengal", "gujarat", "kerala", "rajasthan", "punjab",
    "andhra pradesh", "madhya pradesh", "odisha", "bihar", "assam", "goa",
    "chandigarh", "jharkhand", "uttarakhand", "chhattisgarh", "himachal pradesh",
}

# Cities we can attribute to a country without being told.
CITY_COUNTRY = {
    "in": {"bengaluru", "mumbai", "delhi", "gurugram", "hyderabad", "chennai",
           "pune", "kolkata", "ahmedabad", "jaipur", "surat", "indore", "kochi",
           "coimbatore", "nagpur", "chandigarh", "bhubaneswar", "ranchi",
           "thiruvananthapuram", "mysuru", "vadodara", "lucknow"},
    "us": {"new york", "san francisco", "seattle", "austin", "boston", "chicago",
           "los angeles", "denver", "atlanta", "san jose", "palo alto",
           "mountain view", "washington", "miami", "portland", "dallas"},
    "gb": {"london", "manchester", "edinburgh", "cambridge", "bristol"},
    "de": {"berlin", "munich", "münchen", "hamburg", "frankfurt", "cologne", "köln"},
    "sg": {"singapore"},
    "ie": {"dublin"},
    "nl": {"amsterdam"},
    "fr": {"paris"},
    "ca": {"toronto", "vancouver", "montreal"},
    "au": {"sydney", "melbourne"},
    "ae": {"dubai", "abu dhabi"},
    "pl": {"warsaw", "krakow", "kraków"},
    "es": {"madrid", "barcelona"},
    "jp": {"tokyo"},
}
_CITY_TO_COUNTRY = {c: cc for cc, cities in CITY_COUNTRY.items() for c in cities}

COUNTRY_ALIASES = {
    "india": "in", "bharat": "in",
    "united states": "us", "usa": "us", "u.s.": "us", "u.s.a.": "us",
    "united states of america": "us", "america": "us",
    "united kingdom": "gb", "uk": "gb", "england": "gb", "great britain": "gb",
    "germany": "de", "deutschland": "de", "singapore": "sg", "ireland": "ie",
    "netherlands": "nl", "france": "fr", "canada": "ca", "australia": "au",
    "united arab emirates": "ae", "uae": "ae", "poland": "pl", "spain": "es",
    "japan": "jp", "israel": "il", "brazil": "br", "mexico": "mx",
}

REMOTE_WORDS = ("remote", "anywhere", "work from home", "wfh", "distributed",
                "virtual", "home based", "home-based")

_COUNTRY_CODES = set(COUNTRY_ALIASES.values())

# Work arrangements and placeholders that boards put in the location field.
# Without this, "In-Office" reduced to "in" and collided with India's country
# code, so Austin and London roles matched a filter for India.
NON_PLACE = {
    "in office", "office", "hybrid", "onsite", "on site", "remote first",
    "various", "multiple", "multiple locations", "flexible", "tbd", "n a",
    "other", "field", "virtual", "unspecified", "worldwide", "global",
}

# "IN-Bengaluru", "US-Austin": an ISO country code prefixed to the city. The
# old suffix rule cut everything after the hyphen and threw the city away.
_CC_PREFIX = re.compile(r"^([a-z]{2})[-–]\s*(.+)$")

# Internal site codes some employers append: "Bengaluru-VTP", "Mumbai-MSO".
_SITE_CODE = re.compile(r"[-–]\s*[a-z0-9]{2,5}$")
_PAREN = re.compile(r"\(.*?\)")
_NOISE = re.compile(r"\b(office|campus|hq|headquarters|hybrid|onsite|on-site|based)\b")


def _clean(part: str) -> str:
    """Light tidying only. Anything that might be a city has to survive."""
    part = _PAREN.sub(" ", part.strip().lower())
    return re.sub(r"\s+", " ", part).strip(" .,/")


def _is_non_place(part: str) -> bool:
    return part.replace("-", " ").replace("_", " ").strip() in NON_PLACE


def parse(raw: str | None) -> dict:
    """Break a location string into {city, country, remote, canonical}.

    `canonical` is what dedupe and filtering compare on: "bengaluru, in",
    "remote", or a tidied fallback when the string is unrecognisable.
    """
    text = (raw or "").strip()
    if not text:
        return {"city": None, "country": None, "remote": False, "canonical": ""}

    low = text.lower()
    remote = any(w in low for w in REMOTE_WORDS)

    parts = [_clean(p) for p in re.split(r"[,/|]| - ", text)]
    parts = [p for p in parts if p]

    city = country = None
    for part in parts:
        if not part or _is_non_place(part):
            continue

        # A leading ISO country code names the country and leaves the city.
        prefix = _CC_PREFIX.match(part)
        if prefix and prefix.group(1) in _COUNTRY_CODES:
            country = country or prefix.group(1)
            part = prefix.group(2)

        part = _NOISE.sub(" ", _SITE_CODE.sub("", part))
        part = re.sub(r"\s+", " ", part).strip(" .,/-")
        if not part or _is_non_place(part):
            continue

        candidate = CITY_ALIASES.get(part, part)
        # Known cities win over the state list: Delhi, Goa and Chandigarh are
        # both, and "Delhi, Delhi" should resolve to the city, not just India.
        if city is None and candidate in _CITY_TO_COUNTRY:
            city = candidate
            continue
        if part in COUNTRY_ALIASES:
            country = country or COUNTRY_ALIASES[part]
            continue
        if part in INDIAN_STATES:
            country = country or "in"
            continue
        if any(w in part for w in REMOTE_WORDS):
            continue
        # A two-letter token is a country code or an abbreviation, never a
        # city we can trust. Guessing one is how "in" became a place.
        if city is None and len(candidate) >= 3 and candidate not in _COUNTRY_CODES:
            city = candidate

    if city:
        city = CITY_ALIASES.get(city, city)
        country = country or _CITY_TO_COUNTRY.get(city)

    if city and country:
        canonical = f"{city}, {country}"
    elif city:
        canonical = city
    elif remote and country:
        # "Remote - India" is narrower than "Remote"; keep the restriction.
        canonical = f"remote, {country}"
    elif remote:
        canonical = "remote"
    elif country:
        canonical = country
    else:
        # Nothing recognisable. Keep a tidied string so an unknown city still
        # groups with itself, but a work arrangement is not a place at all.
        tidied = _clean(text)
        canonical = "" if _is_non_place(tidied) else tidied

    return {"city": city, "country": country, "remote": remote, "canonical": canonical}


def canonical(raw: str | None) -> str:
    return parse(raw)["canonical"]


def matches(raw: str | None, wanted: str) -> bool:
    """Does this listing's location satisfy a user asking for `wanted`?

    Deliberately loose: someone asking for "India" should get Bengaluru roles,
    and someone asking for "Bangalore" should get "Bengaluru, in".
    """
    want = parse(wanted)
    have = parse(raw)
    if want["remote"]:
        return have["remote"]
    if want["city"]:
        return have["city"] == want["city"]
    if want["country"]:
        return have["country"] == want["country"]
    return want["canonical"] in (have["canonical"] or "")
