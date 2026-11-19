import pytest

from jobmatch import db, search
from jobmatch.embed import HashEmbedder
from jobmatch.models import Job
from jobmatch.search import Filters, Searcher
from jobmatch.vectorstore import NumpyStore

JOBS = [
    ("Search Relevance Engineer", "Shopfind", "Bengaluru, India", "senior",
     "Tune BM25 and vector search for e-commerce. Learning to rank, OpenSearch."),
    ("AI Agent Engineer", "Helpdeskr", "Remote - India", None,
     "Design LLM agents with tool calling, long-term memory and guardrails."),
    ("Frontend Engineer", "Pixelwave", "Mumbai, Maharashtra", None,
     "Build responsive web apps with React and TypeScript, design systems."),
    ("Data Engineer", "Finlytics", "Gurgaon", None,
     "Batch pipelines that clean and load noisy financial data. Airflow, Spark."),
    ("Software Engineer Intern", "Acme", "Bangalore, Karnataka", "intern",
     "Work with our backend team on Python services and SQL."),
]


@pytest.fixture
def searcher(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    jobs = [
        Job(source="greenhouse", source_id=str(i), company=c, title=t,
            location=loc, seniority=sen, description=d,
            remote="remote" in loc.lower())
        for i, (t, c, loc, sen, d) in enumerate(JOBS)
    ]
    for job in jobs:
        db.upsert_job(conn, job)
    conn.commit()

    embedder = HashEmbedder()
    store = NumpyStore()
    store.reset(embedder.dim)
    store.upsert([j.id for j in jobs],
                 embedder.encode([j.embedding_text() for j in jobs]),
                 [{} for _ in jobs])
    yield Searcher(conn, store, embedder), {j.title: j.id for j in jobs}
    conn.close()


# --- FTS5 query building -------------------------------------------------

def test_fts_query_quotes_terms_so_operators_cannot_leak():
    # Bare AND/OR/NOT and quotes are FTS5 syntax; unquoted they raise errors.
    assert search.fts_query('python AND "rust"') == '"python" OR "AND" OR "rust"'


def test_fts_query_keeps_tech_tokens_intact():
    assert '"node.js"' in search.fts_query("node.js and c++")
    assert '"c++"' in search.fts_query("node.js and c++")


def test_fts_query_is_empty_for_junk():
    assert search.fts_query("!!! ?") == ""
    assert search.fts_query("") == ""


def test_punctuation_heavy_query_does_not_raise(searcher):
    s, _ = searcher
    s.search('c++ / "rust" (backend) AND go', k=3, mode="bm25")


# --- Reciprocal Rank Fusion ----------------------------------------------

def test_rrf_promotes_what_both_engines_found():
    fused = search.rrf({"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]})
    assert [job for job, _, _ in fused][:2] == ["a", "b"]


def test_rrf_records_which_engines_found_each_hit():
    fused = dict((job, found) for job, _, found in
                 search.rrf({"bm25": ["a"], "dense": ["a", "z"]}))
    assert sorted(fused["a"]) == ["bm25", "dense"]
    assert fused["z"] == ["dense"]


def test_rrf_keeps_a_hit_only_one_engine_found():
    assert "z" in [job for job, _, _ in search.rrf({"bm25": ["a"], "dense": ["z"]})]


def test_rrf_ranks_by_position_not_by_score_scale():
    # Engine scores are never compared, so wildly different scales cannot skew it.
    fused = search.rrf({"bm25": ["x", "y"], "dense": ["y", "x"]})
    assert {job for job, _, _ in fused} == {"x", "y"}


# --- Retrieval behaviour --------------------------------------------------

def test_keyword_query_finds_the_literal_match(searcher):
    s, ids = searcher
    hits = s.search("OpenSearch learning to rank", k=3, mode="bm25")
    assert hits[0].job_id == ids["Search Relevance Engineer"]


def test_hybrid_returns_results_for_every_mode(searcher):
    s, _ = searcher
    for mode in ("bm25", "dense", "hybrid"):
        assert s.search("engineer", k=3, mode=mode), f"{mode} returned nothing"


def test_hits_report_which_engine_found_them(searcher):
    s, _ = searcher
    hits = s.search("agents with tool calling", k=5, mode="hybrid")
    assert all(h.found_by for h in hits)


# --- Filters --------------------------------------------------------------

def test_location_filter_matches_across_city_spellings(searcher):
    s, ids = searcher
    # Stored as "Bangalore, Karnataka"; the user types Bengaluru.
    hits = s.search("intern", k=5, filters=Filters(location="Bengaluru"))
    assert ids["Software Engineer Intern"] in {h.job_id for h in hits}


def test_country_filter_includes_its_cities(searcher):
    s, _ = searcher
    hits = s.search("engineer", k=10, filters=Filters(location="India"))
    assert hits
    assert all(h.job.location_canonical.endswith("in") for h in hits)


def test_remote_filter_excludes_onsite_roles(searcher):
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(remote=True))
    assert {h.job_id for h in hits} == {ids["AI Agent Engineer"]}


def test_seniority_filter(searcher):
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(seniority=["intern"]))
    assert {h.job_id for h in hits} == {ids["Software Engineer Intern"]}


def test_excluded_company_never_appears(searcher):
    s, _ = searcher
    hits = s.search("engineer", k=10, filters=Filters(exclude_companies=["Pixelwave"]))
    assert "Pixelwave" not in {h.job.company for h in hits}


def test_impossible_filter_returns_nothing_rather_than_ignoring_it(searcher):
    s, _ = searcher
    assert s.search("engineer", k=10, filters=Filters(location="Antarctica")) == []


def test_empty_filters_do_not_restrict(searcher):
    s, _ = searcher
    assert search.allowed_ids(s.conn, Filters()) is None


# --- seniority: "mid" is NULL, and exclusion must not drop it -------------

def test_mid_level_means_no_seniority_marker(searcher):
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(seniority=["mid"]))
    found = {h.job_id for h in hits}
    assert ids["Frontend Engineer"] in found          # untagged title
    assert ids["Software Engineer Intern"] not in found


def test_excluding_senior_keeps_mid_level_roles(searcher):
    """NOT IN never matches NULL, so a naive exclusion silently drops every
    mid-level role -- which is most of them."""
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(exclude_seniority=["senior"]))
    found = {h.job_id for h in hits}
    assert ids["Frontend Engineer"] in found
    assert ids["Data Engineer"] in found
    assert ids["Search Relevance Engineer"] not in found   # the senior one


def test_excluding_mid_keeps_only_tagged_levels(searcher):
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(exclude_seniority=["mid"]))
    assert all(h.job.seniority is not None for h in hits)


def test_seniority_list_may_mix_mid_with_named_levels(searcher):
    s, ids = searcher
    hits = s.search("engineer", k=10, filters=Filters(seniority=["mid", "intern"]))
    found = {h.job_id for h in hits}
    assert ids["Software Engineer Intern"] in found
    assert ids["Frontend Engineer"] in found
    assert ids["Search Relevance Engineer"] not in found


# --- result diversity -----------------------------------------------------

def _dominated_by_one_company(tmp_path):
    """A corpus where one employer genuinely wins on relevance.

    The earlier version of this fixture gave every company equally matching
    text, so the small employers ranked top anyway and there was nothing to
    diversify -- the test passed without exercising the feature.
    """
    conn = db.connect(tmp_path / "d.db")
    jobs = [
        Job(source="greenhouse", source_id=f"big{i}", company="Bigcorp",
            title="Backend Engineer", location="Bengaluru, India",
            description="Backend engineer building Python backend services and "
                        "Python APIs. Backend Python backend engineering.")
        for i in range(6)
    ]
    jobs += [
        Job(source="lever", source_id=f"s{i}", company=name,
            title="Software Engineer", location="Bengaluru, India",
            description="Some backend work, mostly in Python.")
        for i, name in enumerate(["Smallco", "Tinyco"])
    ]
    for j in jobs:
        db.upsert_job(conn, j)
    conn.commit()
    embedder, store = HashEmbedder(), NumpyStore()
    store.reset(embedder.dim)
    store.upsert([j.id for j in jobs],
                 embedder.encode([j.embedding_text() for j in jobs]), [{}] * len(jobs))
    return Searcher(conn, store, embedder)


def test_without_a_cap_one_company_takes_the_whole_page(tmp_path):
    s = _dominated_by_one_company(tmp_path)
    hits = s.search("backend engineer python", k=5)
    assert all(h.job.company == "Bigcorp" for h in hits)


def test_cap_makes_room_for_other_employers(tmp_path):
    s = _dominated_by_one_company(tmp_path)
    hits = s.search("backend engineer python", k=5, max_per_company=2)
    companies = [h.job.company for h in hits]
    assert set(companies) == {"Bigcorp", "Smallco", "Tinyco"}
    assert companies[:2] == ["Bigcorp", "Bigcorp"]   # still relevance-ordered


def test_the_cap_is_soft_so_the_page_still_fills(tmp_path):
    """With only three employers available, returning three results would be
    worse than topping up from the strongest one."""
    s = _dominated_by_one_company(tmp_path)
    hits = s.search("backend engineer python", k=8, max_per_company=2)
    assert len(hits) == 8
    assert hits[2].job.company in {"Smallco", "Tinyco"}  # promoted above overflow


def test_diversification_is_off_by_default(tmp_path):
    s = _dominated_by_one_company(tmp_path)
    plain = [h.job.company for h in s.search("backend engineer python", k=5)]
    capped = [h.job.company for h in s.search("backend engineer python", k=5,
                                              max_per_company=2)]
    assert plain != capped
