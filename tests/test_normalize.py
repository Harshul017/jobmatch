import pytest

from jobmatch import normalize


def test_strip_html_unescapes_twice():
    # Greenhouse escapes HTML that itself contains entities.
    raw = "&lt;p&gt;Build &amp;amp; ship&lt;/p&gt;"
    assert normalize.strip_html(raw) == "Build & ship"


def test_strip_html_turns_breaks_into_newlines():
    out = normalize.strip_html("<p>One</p><p>Two</p>")
    assert "One" in out and "Two" in out
    assert "<" not in out


def test_collapse_keeps_paragraphs_but_squashes_spaces():
    assert normalize.collapse("a   b\n\n\n\nc") == "a b\n\nc"


def test_norm_key_keeps_tech_tokens():
    key = normalize.norm_key("C++, Node.js & Go!")
    assert "c++" in key and "node.js" in key


def test_detect_remote_prefers_explicit_flag_over_text():
    assert normalize.detect_remote("Bengaluru", "remote friendly", flag=False) is False
    assert normalize.detect_remote("Bengaluru", "on site only", flag=True) is True


def test_detect_remote_falls_back_to_text():
    assert normalize.detect_remote("Remote - India", "") is True
    assert normalize.detect_remote("Gurugram", "onsite role") is False


def test_seniority_specific_rule_beats_generic():
    assert normalize.detect_seniority("Senior Backend Engineer") == "senior"
    assert normalize.detect_seniority("Software Engineer Intern") == "intern"
    assert normalize.detect_seniority("Backend Engineer") is None


def test_dedupe_key_ignores_case_and_punctuation():
    a = normalize.dedupe_key("Acme, Inc.", "Backend Engineer", "Gurugram")
    b = normalize.dedupe_key("acme inc", "backend  engineer", "GURUGRAM")
    assert a == b


def test_job_id_is_stable_and_source_scoped():
    assert normalize.job_id("lever", "1") == normalize.job_id("lever", "1")
    assert normalize.job_id("lever", "1") != normalize.job_id("ashby", "1")


def test_management_beats_senior_in_a_hybrid_title():
    assert normalize.detect_seniority("Senior Engineering Manager") == "management"


def test_ic_levels_are_not_lumped_into_lead():
    assert normalize.detect_seniority("VP of Engineering") == "management"
    assert normalize.detect_seniority("Tech Lead") == "lead"
    assert normalize.detect_seniority("Staff Engineer") == "staff"


def test_graduate_programme_manager_is_not_junior():
    assert normalize.detect_seniority("Graduate Program Manager") == "management"


def test_dedupe_matches_across_city_spellings():
    a = normalize.dedupe_key("Acme", "Backend Engineer", "Bangalore, Karnataka")
    b = normalize.dedupe_key("Acme", "Backend Engineer", "Bengaluru, India")
    assert a == b


# --- years of experience --------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("We need 6-8 years of experience in backend systems.", 6),
    ("5+ years of professional experience required.", 5),
    ("At least 3 years experience with Python.", 3),
    ("Minimum of 10 years of relevant experience", 10),
    ("2 to 4 years of industry experience", 2),
    ("You have 1+ year of experience", 1),
    ("8+ yrs experience leading teams", 8),
    ("Requires 4+ years of experience; 6 years preferred.", 4),
])
def test_reads_the_lowest_year_requirement(text, expected):
    assert normalize.min_years_required(text) == expected


@pytest.mark.parametrize("text", [
    "Our company was founded 10 years ago.",
    "Revenue tripled in the last 5 years.",
    "A fast-growing team working on hard problems.",
    "",
    None,
])
def test_ignores_years_that_are_not_a_requirement(text):
    assert normalize.min_years_required(text) is None


def test_ignores_implausible_figures():
    assert normalize.min_years_required("40 years of experience") is None


# --- role excerpt ---------------------------------------------------------

BOILERPLATE_FIRST = """Who we are

About Stripe

Stripe is a financial infrastructure platform for businesses worldwide.

What you'll do

Build payment services in Python and Go, owning them in production."""


def test_excerpt_skips_the_company_blurb():
    """Taking the first N characters returned marketing copy, so both the
    relevance judge and the reader were shown the wrong evidence."""
    out = normalize.role_excerpt(BOILERPLATE_FIRST)
    assert out.startswith("What you'll do")
    assert "financial infrastructure platform" not in out


def test_excerpt_keeps_a_posting_that_starts_with_the_role():
    text = "Responsibilities\n\nBuild and run backend services."
    assert normalize.role_excerpt(text).startswith("Responsibilities")


def test_excerpt_falls_back_rather_than_returning_nothing():
    text = "A short note with no recognisable structure at all."
    assert normalize.role_excerpt(text) == text


def test_excerpt_respects_the_limit():
    assert len(normalize.role_excerpt("word " * 500, limit=120)) <= 120


def test_excerpt_handles_empty_input():
    assert normalize.role_excerpt(None) == ""
    assert normalize.role_excerpt("") == ""
