"""Agent behaviour, driven by a scripted model so every path is deterministic.

The interesting cases are the failures: a small open-weights model will send
malformed JSON, invent a field, call a tool that does not exist, and cite a
job it never retrieved. None of those should reach the person.
"""
import json

import pytest

from jobmatch import db, tools
from jobmatch.agent import Agent
from jobmatch.embed import HashEmbedder
from jobmatch.llm import Reply, ScriptedLLM, ToolCall
from jobmatch.models import Job, Preference
from jobmatch.search import Searcher
from jobmatch.vectorstore import NumpyStore

JOBS = [
    ("Backend Engineer, Search", "Shopfind", "Bengaluru, India", None,
     "Tune BM25 and vector search. Python, OpenSearch, ranking."),
    ("AI Agent Engineer", "Helpdeskr", "Remote - India", None,
     "LLM agents with tool calling and long-term memory."),
    ("Senior Platform Engineer", "Cloudnine", "London, UK", "senior",
     "Kubernetes, Terraform, CI/CD and on-call."),
]


def call(name, **args):
    return ToolCall(id=f"c_{name}", name=name, arguments=args)


@pytest.fixture
def setup(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    jobs = [Job(source="greenhouse", source_id=str(i), company=c, title=t,
                location=loc, seniority=sen, description=d,
                remote="remote" in loc.lower())
            for i, (t, c, loc, sen, d) in enumerate(JOBS)]
    for j in jobs:
        db.upsert_job(conn, j)
    conn.commit()
    embedder, store = HashEmbedder(), NumpyStore()
    store.reset(embedder.dim)
    store.upsert([j.id for j in jobs],
                 embedder.encode([j.embedding_text() for j in jobs]), [{}] * len(jobs))
    searcher = Searcher(conn, store, embedder)

    def make(replies=(), user="u1"):
        return Agent(conn, searcher, ScriptedLLM(list(replies)), user_id=user), conn, jobs

    yield make
    conn.close()


# --- the happy path ------------------------------------------------------

def test_search_then_cite_is_grounded(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="search and ranking")]),
        Reply(content=f"Good fit: Backend Engineer, Search [{jobs[0].id}]."),
    ])
    result = agent.run("backend search roles")
    assert result.grounded is True
    assert result.cited == {jobs[0].id}
    assert result.fell_back is False
    assert jobs[0].id in result.reply


# --- model mistakes -------------------------------------------------------

def test_malformed_json_arguments_are_reported_back_not_raised(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[ToolCall("c1", "search_jobs", {"__malformed__": "{query:"})]),
        Reply(tool_calls=[call("search_jobs", query="search")]),
        Reply(content=f"Found one [{jobs[0].id}]."),
    ])
    result = agent.run("find me something")
    assert result.invalid_calls == 1
    assert "not valid JSON" in result.steps[0].error
    assert result.grounded and not result.fell_back


def test_wrong_argument_type_produces_a_correctable_error(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="x", limit=99)]),  # above the cap
        Reply(tool_calls=[call("search_jobs", query="x", limit=3)]),
        Reply(content=f"ok [{jobs[0].id}]"),
    ])
    result = agent.run("anything")
    assert "limit" in result.steps[0].error
    assert result.steps[1].error is None


def test_unknown_tool_name_is_reported_back(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("apply_to_job", job_id="x")]),
        Reply(content="Sorry, I cannot apply on your behalf."),
    ])
    result = agent.run("apply for me")
    assert "No tool named" in result.steps[0].error


def test_get_job_with_a_bad_id_returns_an_error_not_a_crash(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("get_job", job_id="deadbeef")]),
        Reply(content="That listing is not in the index."),
    ])
    result = agent.run("tell me about deadbeef")
    assert result.steps[0].error is None  # handled inside the tool, as data


# --- grounding ------------------------------------------------------------

def test_invented_citation_is_stripped_and_flagged(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="search")]),
        Reply(content=f"Real one [{jobs[0].id}].\nMade-up one [ffffffffffffffff]."),
    ])
    result = agent.run("find roles")
    assert result.grounded is False
    assert "ffffffffffffffff" not in result.reply
    assert jobs[0].id in result.reply


def test_answering_with_no_tool_call_and_a_fake_id_leaves_nothing_behind(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([Reply(content="I recommend Acme [abcdefabcdefabcd].")])
    result = agent.run("recommend something")
    assert result.grounded is False
    assert "abcdefabcdefabcd" not in result.reply


# --- fallbacks ------------------------------------------------------------

def test_endless_tool_calling_falls_back_to_plain_search(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="loop")]) for _ in range(10)
    ])
    result = agent.run("backend engineer")
    assert result.fell_back is True
    assert result.reply.strip()
    assert result.grounded is True  # the fallback only cites what it retrieved


def test_empty_model_reply_falls_back(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([Reply(content="")])
    result = agent.run("search engineer")
    assert result.fell_back is True
    assert result.reply


# --- memory ---------------------------------------------------------------

def test_save_preference_persists_across_agents(setup):
    agent, conn, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("save_preference", locations=["Bengaluru"], remote_only=True)]),
        Reply(content="Noted."),
    ])
    agent.run("I want remote roles in Bengaluru")
    stored = db.load_preference(conn, "u1")
    assert stored.locations == ["Bengaluru"] and stored.remote_only is True


def test_saved_preferences_reach_the_system_prompt(setup):
    agent, conn, _ = setup()
    db.save_preference(conn, Preference(user_id="u1", skills=["python"], seniority="intern"))
    agent.llm = ScriptedLLM([Reply(content="ok")])
    agent.run("hello")
    system = agent.llm.calls[0][0]["content"]
    assert "python" in system and "intern" in system


def test_saved_preferences_filter_the_search(setup):
    agent, conn, jobs = setup()
    db.save_preference(conn, Preference(user_id="u1", remote_only=True))
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="engineer")]),
        Reply(content="ok"),
    ])
    agent.run("find engineering roles")
    tool_message = [m for m in agent.llm.calls[-1] if m.get("role") == "tool"][0]
    returned = json.loads(tool_message["content"])["results"]
    assert returned and all(r["remote"] for r in returned)


def test_preferences_merge_rather_than_overwrite(setup):
    agent, conn, _ = setup()
    db.save_preference(conn, Preference(user_id="u1", skills=["python"]))
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("save_preference", skills=["sql"])]),
        Reply(content="ok"),
    ])
    agent.run("I also know sql")
    assert db.load_preference(conn, "u1").skills == ["python", "sql"]


def test_impossible_filters_are_relaxed_and_flagged(setup):
    agent, conn, _ = setup()
    db.save_preference(conn, Preference(user_id="u1", locations=["Antarctica"]))
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="engineer")]),
        Reply(content="ok"),
    ])
    agent.run("find me roles")
    tool_message = [m for m in agent.llm.calls[-1] if m.get("role") == "tool"][0]
    payload = json.loads(tool_message["content"])
    assert payload["relaxed_filters"] is True
    assert payload["results"]


# --- schemas --------------------------------------------------------------

def test_tool_schemas_are_well_formed():
    for schema in tools.schemas():
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_every_declared_tool_is_executable():
    assert set(tools.BY_NAME) == {t["function"]["name"] for t in tools.schemas()}


# --- the fallback must answer the question that was asked -----------------

def test_fallback_keeps_the_filters_the_model_had_established(setup):
    """Regression: asking for "nothing senior" and running out of steps
    returned four senior roles, because the fallback searched unfiltered."""
    agent, _, jobs = setup()
    agent.max_steps = 2
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="engineer",
                               exclude_seniority=["senior"])]),
        Reply(tool_calls=[call("search_jobs", query="engineer again")]),
    ])
    result = agent.run("engineering roles, nothing senior")
    assert result.fell_back is True
    cited = set(result.cited)
    senior = {j.id for j in jobs if j.seniority == "senior"}
    assert cited and not (cited & senior)


def test_fallback_uses_standing_preferences_when_no_search_happened(setup):
    agent, conn, jobs = setup()
    db.save_preference(conn, Preference(user_id="u1", remote_only=True))
    agent.llm = ScriptedLLM([Reply(content="")])   # model says nothing at all
    result = agent.run("find me something")
    assert result.fell_back is True
    remote_ids = {j.id for j in jobs if j.remote}
    assert result.cited and result.cited <= remote_ids


def test_fallback_says_so_when_it_has_to_drop_the_filters(setup):
    agent, conn, _ = setup()
    db.save_preference(conn, Preference(user_id="u1", locations=["Antarctica"]))
    agent.llm = ScriptedLLM([Reply(content="")])
    result = agent.run("engineer")
    assert "ignore them" in result.reply
    assert result.cited


def test_step_budget_allows_search_then_read_then_answer(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="search")]),
        Reply(tool_calls=[call("get_job", job_id=jobs[0].id)]),
        Reply(tool_calls=[call("get_job", job_id=jobs[1].id)]),
        Reply(content=f"Two options: [{jobs[0].id}] and [{jobs[1].id}]."),
    ])
    result = agent.run("backend roles")
    assert result.fell_back is False
    assert result.cited == {jobs[0].id, jobs[1].id}


# --- memory should only hold what was actually said ----------------------

def test_preference_only_records_fields_that_changed(setup):
    """Regression: the model saved remote_only=False on a question that never
    mentioned remote, which would hide every remote job from then on."""
    agent, conn, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("save_preference", locations=["Bengaluru"],
                               skills=[], avoid=[])]),
        Reply(content="Noted."),
    ])
    result = agent.run("I'm in Bengaluru")
    assert result.remembered == {"locations": ["Bengaluru"]}
    assert db.load_preference(conn, "u1").remote_only is None


def test_empty_values_do_not_count_as_changes(setup):
    agent, conn, _ = setup()
    db.save_preference(conn, Preference(user_id="u1", locations=["Bengaluru"]))
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("save_preference", locations=["Bengaluru"], skills=[])]),
        Reply(content="ok"),
    ])
    result = agent.run("still in bengaluru")
    assert result.remembered == {}


def test_what_was_remembered_is_reported_back(setup):
    agent, _, _ = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("save_preference", remote_only=True, seniority="intern")]),
        Reply(content="ok"),
    ])
    result = agent.run("remote internships only")
    assert result.remembered == {"remote_only": True, "seniority": "intern"}


# --- repeated calls ------------------------------------------------------

def test_identical_tool_call_is_served_from_cache(setup):
    """The model re-issued the same search verbatim, burning a step and a
    round trip for a result it already had."""
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="backend")]),
        Reply(tool_calls=[call("search_jobs", query="backend")]),
        Reply(content=f"ok [{jobs[0].id}]"),
    ])
    result = agent.run("backend roles")
    assert [s.repeated for s in result.steps] == [False, True]
    assert result.grounded


def test_a_different_query_is_not_treated_as_a_repeat(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="backend")]),
        Reply(tool_calls=[call("search_jobs", query="frontend")]),
        Reply(content=f"ok [{jobs[0].id}]"),
    ])
    result = agent.run("anything")
    assert [s.repeated for s in result.steps] == [False, False]


# --- fallback query quality ----------------------------------------------

def test_fallback_searches_the_models_query_not_the_raw_sentence(setup):
    """Regression: the raw message carries the location, and "bengaluru" in
    the query text matches job titles containing the city -- which is how a
    backend search fell back to an Account Executive posting."""
    agent, _, jobs = setup()
    agent.max_steps = 1
    agent.llm = ScriptedLLM([
        Reply(tool_calls=[call("search_jobs", query="backend engineer search")]),
    ])
    result = agent.run("backend roles in London, nothing senior")
    assert result.fell_back is True
    # London appears in a job title in the fixture; the distilled query should
    # rank the backend role above it.
    assert jobs[0].id in result.cited


# --- context compaction ---------------------------------------------------

def test_older_tool_results_are_stubbed_out(setup):
    """Latency grew from 0.9s to 24.8s across eight calls because every tool
    result was resent each turn."""
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        *[Reply(tool_calls=[call("search_jobs", query=f"q{i}")]) for i in range(5)],
        Reply(content=f"done [{jobs[0].id}]"),
    ])
    agent.run("find me roles")
    sent = agent.llm.calls[-1]
    tool_messages = [m for m in sent if m.get("role") == "tool"]
    stubbed = [m for m in tool_messages if '"seen_earlier"' in (m.get("content") or "")]
    assert len(tool_messages) == 5
    assert len(stubbed) == 2           # all but the most recent three
    assert "results" in tool_messages[-1]["content"]


def test_compaction_keeps_the_most_recent_results_intact(setup):
    agent, _, jobs = setup()
    agent.llm = ScriptedLLM([
        *[Reply(tool_calls=[call("search_jobs", query=f"q{i}")]) for i in range(4)],
        Reply(content=f"done [{jobs[0].id}]"),
    ])
    agent.run("roles")
    recent = [m for m in agent.llm.calls[-1] if m.get("role") == "tool"][-3:]
    assert all('"seen_earlier"' not in m["content"] for m in recent)
