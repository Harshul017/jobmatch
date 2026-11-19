"""Both stores must behave identically, including under selective filters."""
import hashlib

import numpy as np
import pytest

from jobmatch.vectorstore import NumpyStore, QdrantStore

N = 600
DIM = 32


def job_id(i: int) -> str:
    """Ids shaped like the real ones: a 16-character hash, not a counter.

    Sequential hex ids differ only in their last characters, which hid a
    truncation bug in the Qdrant point-id mapping behind a test that happened
    not to exercise it."""
    return hashlib.sha1(str(i).encode()).hexdigest()[:16]


def corpus():
    """Points spread along a quarter arc, so cosine similarity to the query is
    strictly decreasing in the index: job 0 is nearest, job N-1 furthest.

    No random noise: an earlier version added jitter larger than the gap
    between neighbours, so "nearest first" was not actually true of the data
    and the test failed for reasons that had nothing to do with the store."""
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0
    angles = np.linspace(0.0, np.pi / 2, N, dtype=np.float32)
    vectors = np.zeros((N, DIM), dtype=np.float32)
    vectors[:, 0] = np.cos(angles)
    vectors[:, 1] = np.sin(angles)
    ids = [job_id(i) for i in range(N)]
    return ids, vectors, query


@pytest.fixture(params=["numpy", "qdrant"])
def store(request, tmp_path):
    ids, vectors, _ = corpus()
    s = NumpyStore() if request.param == "numpy" else QdrantStore(tmp_path / "q")
    s.reset(DIM)
    s.upsert(ids, vectors, [{"job_id": i} for i in ids])
    return s


def test_unfiltered_search_returns_nearest_first(store):
    _, _, query = corpus()
    hits = store.search(query, k=3)
    assert [h[0] for h in hits] == [job_id(i) for i in range(3)]


def test_selective_filter_finds_a_match_buried_deep_in_the_ranking(store):
    """The regression this guards: a fixed over-fetch of 100 would miss this
    entirely and return nothing, even though a match exists."""
    _, _, query = corpus()
    buried = job_id(550)
    hits = store.search(query, k=5, allowed_ids={buried})
    assert [h[0] for h in hits] == [buried]


def test_filter_respects_ordering_among_allowed_results(store):
    _, _, query = corpus()
    allowed = {job_id(i) for i in (400, 10, 580)}
    hits = store.search(query, k=5, allowed_ids=allowed)
    assert [h[0] for h in hits] == [job_id(i) for i in (10, 400, 580)]


def test_filter_matching_nothing_returns_nothing(store):
    _, _, query = corpus()
    assert store.search(query, k=5, allowed_ids={"deadbeefdeadbeef"}) == []


def test_count_reflects_what_was_upserted(store):
    assert store.count() == N


def test_point_ids_do_not_collide():
    """Regression: the mapping used to drop the last hex character."""
    from jobmatch.vectorstore import _point_id
    ids = [job_id(i) for i in range(2000)]
    assert len({_point_id(i) for i in ids}) == len(ids)
