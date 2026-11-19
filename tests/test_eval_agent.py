"""The behavioural checks are pure functions, so they can be tested without
calling a model. If a check is wrong, the whole agent evaluation is wrong."""
from jobmatch import eval_agent as ev
from jobmatch.agent import AgentResult, Step
from jobmatch.models import Job


def job(**kw) -> Job:
    base = dict(source="greenhouse", source_id="1", company="Acme",
                title="Backend Engineer", description="Build APIs.",
                location="Bengaluru, India")
    base.update(kw)
    return Job(**base)


def result(**kw) -> AgentResult:
    base = dict(reply="hi", steps=[Step("search_jobs", {})], grounded=True,
                fell_back=False)
    base.update(kw)
    return AgentResult(**base)


def test_must_search_fails_when_no_search_happened():
    assert ev.must_search(result(steps=[]), []) is not None
    assert ev.must_search(result(steps=[Step("get_job", {})]), []) is not None
    assert ev.must_search(result(), []) is None


def test_grounding_and_fallback_checks():
    assert ev.must_be_grounded(result(grounded=False), []) is not None
    assert ev.must_not_fall_back(result(fell_back=True), []) is not None


def test_invalid_call_check_reports_the_first_error():
    bad = result(steps=[Step("search_jobs", {}, error="boom")])
    assert "boom" in ev.no_invalid_calls(bad, [])


def test_no_senior_roles_catches_every_senior_level():
    for level in ("senior", "staff", "principal", "management"):
        assert ev.no_senior_roles(result(), [job(seniority=level)]) is not None
    assert ev.no_senior_roles(result(), [job(seniority="intern")]) is None
    assert ev.no_senior_roles(result(), [job(seniority=None)]) is None


def test_location_check_uses_canonical_form():
    assert ev.all_in_india(result(), [job(location="Bangalore, Karnataka")]) is None
    assert ev.all_in_india(result(), [job(location="London, UK")]) is not None


def test_remote_check():
    assert ev.all_remote(result(), [job(remote=True)]) is None
    assert ev.all_remote(result(), [job(remote=False)]) is not None


def test_intern_check():
    assert ev.only_interns(result(), [job(seniority="intern")]) is None
    assert ev.only_interns(result(), [job(seniority=None)]) is not None


def test_memory_checks_are_opposites():
    saved = result(remembered={"locations": ["Bengaluru"]})
    assert ev.remembered_something(saved, []) is None
    assert ev.remembered_nothing(saved, []) is not None
    assert ev.remembered_something(result(), []) is not None
    assert ev.remembered_nothing(result(), []) is None


def test_company_concentration_check():
    three = [job(source_id=str(i), company="Bigcorp") for i in range(3)]
    assert ev.at_most_two_per_company(result(), three) is not None
    assert ev.at_most_two_per_company(result(), three[:2]) is None


def test_empty_recommendation_fails():
    assert ev.must_recommend(result(), []) is not None
    assert ev.must_recommend(result(), [job()]) is None


def test_every_scenario_is_runnable():
    names = [s.name for s in ev.SCENARIOS]
    assert len(names) == len(set(names))
    for scenario in ev.SCENARIOS:
        assert scenario.message.strip()
        assert scenario.checks


def test_a_scenario_that_raises_is_recorded_not_propagated():
    class Exploding:
        conn = None
        def run(self, message):
            raise RuntimeError("model down")

    row = ev.run_one(ev.SCENARIOS[0], lambda: Exploding())
    assert row["result"] is None
    assert "model down" in row["failures"][0]
