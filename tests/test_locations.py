import pytest

from jobmatch import locations


@pytest.mark.parametrize("raw,expected", [
    ("bengaluru", "bengaluru, in"),
    ("Bangalore, Karnataka", "bengaluru, in"),
    ("Bengaluru, India", "bengaluru, in"),
    ("Bengaluru-VTP, India", "bengaluru, in"),   # office suffix stripped
    ("Gurgaon", "gurugram, in"),                 # old name folded to the new one
    ("Delhi, Delhi", "delhi, in"),               # both a city and a state
    ("Mumbai, Maharashtra", "mumbai, in"),
    ("London, UK", "london, gb"),
    ("New York City, NY", "new york, us"),
    ("India", "in"),
    ("", ""),
])
def test_canonical_forms(raw, expected):
    assert locations.canonical(raw) == expected


def test_every_bengaluru_spelling_collapses_to_one():
    spellings = ["bengaluru", "Bangalore", "Bangalore, Karnataka",
                 "Bengaluru, India", "Bengaluru-VTP, India", "BLR"]
    assert len({locations.canonical(s) for s in spellings}) == 1


def test_remote_keeps_its_country_restriction():
    assert locations.canonical("Remote - India") == "remote, in"
    assert locations.canonical("Remote") == "remote"
    assert locations.parse("Remote - India")["remote"] is True


def test_unknown_city_degrades_to_a_tidied_string():
    assert locations.canonical("Vizag") == "vizag"


def test_matching_is_loose_in_the_right_direction():
    # Asking for a country should return its cities...
    assert locations.matches("Bangalore, Karnataka", "India")
    # ...and an old city name should find the new one.
    assert locations.matches("Bengaluru, India", "Bangalore")
    # ...but a different city should not match.
    assert not locations.matches("Mumbai, Maharashtra", "Delhi")


def test_remote_query_only_matches_remote_listings():
    assert locations.matches("Remote - India", "remote")
    assert not locations.matches("Gurugram", "remote")


# --- work arrangements are not places -------------------------------------

@pytest.mark.parametrize("raw", ["In-Office", "Multiple Locations", "Hybrid",
                                 "Various", "TBD", "Flexible"])
def test_a_work_arrangement_is_not_a_location(raw):
    """Regression: "In-Office" reduced to "in", which collided with India's
    country code, so Austin and London roles matched a filter for India."""
    assert locations.canonical(raw) == ""
    assert not locations.matches(raw, "India")


@pytest.mark.parametrize("raw,expected", [
    ("IN-Bengaluru", "bengaluru, in"),
    ("IN-Pune", "pune, in"),
    ("IN-Mumbai-MSO", "mumbai, in"),      # internal site code stripped
    ("IN-Bangalore-MSO", "bengaluru, in"),
    ("US-Austin", "austin, us"),
])
def test_a_country_code_prefix_keeps_the_city(raw, expected):
    """Regression: the suffix rule cut everything after the first hyphen, so
    "IN-Bengaluru" lost Bengaluru and became an unlocated India role."""
    assert locations.canonical(raw) == expected


def test_a_two_letter_token_is_never_treated_as_a_city():
    assert locations.parse("IN")["city"] is None
    assert locations.parse("CA")["city"] is None


def test_multi_city_strings_take_the_first():
    assert locations.canonical("San Francisco, CA • New York, NY") == "san francisco, us"
