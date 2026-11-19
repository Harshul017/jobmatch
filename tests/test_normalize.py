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
