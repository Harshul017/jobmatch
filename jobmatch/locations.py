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

# Office suffixes some boards append: "Bengaluru-VTP", "Hyderabad (HYD-2)".
_SUFFIX = re.compile(r"[-–(].*$")
_NOISE = re.compile(r"\b(office|campus|hq|headquarters|hybrid|onsite|on-site|based)\b")


def _clean(part: str) -> str:
    part = part.strip().lower()
    part = _NOISE.sub(" ", part)
    part = _SUFFIX.sub("", part)
    return re.sub(r"\s+", " ", part).strip(" .,/")


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
        if city is None and candidate:
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
        canonical = _clean(text)

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
