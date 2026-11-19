"""API tests against a temporary index and a stub embedder, so no model
download, no API key and no network."""
import pytest
from fastapi.testclient import TestClient

from jobmatch import api, db
from jobmatch.embed import HashEmbedder
from jobmatch.models import Job
from jobmatch.search import Searcher
from jobmatch.vectorstore import NumpyStore

JOBS = [
    ("Backend Engineer, Search", "Shopfind", "Bengaluru, India", None,
     "Tune BM25 and vector search. Python, OpenSearch, ranking."),
    ("Senior Platform Engineer", "Cloudnine", "London, UK", "senior",
     "Kubernetes, Terraform, CI/CD and on-call."),
    ("Software Engineer Intern", "Acme", "Bangalore, Karnataka", "intern",
     "Backend Python services and SQL."),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "api.db")
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

    monkeypatch.setattr(api, "build_state", lambda: {
        "conn": conn, "agents": {}, "searcher": Searcher(conn, store, embedder)})
    with TestClient(api.app) as c:
        c.jobs = jobs
        yield c


def test_health_reports_index_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["jobs"] == 3 and body["vectors"] == 3


def test_health_fails_when_the_index_is_empty(tmp_path, monkeypatch):
    """Up but unable to answer should not look healthy to a load balancer."""
    conn = db.connect(tmp_path / "empty.db")
    embedder, store = HashEmbedder(), NumpyStore()
    store.reset(embedder.dim)
    monkeypatch.setattr(api, "build_state", lambda: {
        "conn": conn, "agents": {}, "searcher": Searcher(conn, store, embedder)})
    with TestClient(api.app, raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 503


def test_search_returns_hits(client):
    body = client.get("/search", params={"q": "search ranking"}).json()
    assert body["count"] >= 1
    assert body["results"][0]["title"] == "Backend Engineer, Search"
    assert body["results"][0]["found_by"]


def test_search_filters_by_location(client):
    body = client.get("/search", params={"q": "engineer", "location": "India"}).json()
    assert body["count"] >= 1
    assert all(r["location"].endswith("in") for r in body["results"])


def test_search_excludes_seniority(client):
    body = client.get("/search",
                      params={"q": "engineer", "exclude_seniority": ["senior"]}).json()
    assert all(r["seniority"] != "senior" for r in body["results"])


def test_search_rejects_an_unknown_mode(client):
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_search_rejects_an_empty_query(client):
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_search_caps_k(client):
    assert client.get("/search", params={"q": "x", "k": 500}).status_code == 422


def test_job_lookup(client):
    job_id = client.jobs[0].id
    body = client.get(f"/jobs/{job_id}").json()
    assert body["title"] == "Backend Engineer, Search"


def test_unknown_job_is_404(client):
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_stats(client):
    assert client.get("/stats").json()["active"] == 3


def test_chat_without_an_api_key_is_503_not_500(client, monkeypatch):
    """A missing key is a configuration problem, and should say so rather than
    surfacing as an unhandled error."""
    monkeypatch.setattr(api, "GroqLLM",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("No GROQ_API_KEY")))
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


def test_chat_rejects_an_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
