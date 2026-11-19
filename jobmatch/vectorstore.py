"""Where the vectors live.

Two backends behind one interface. Qdrant is the real one: it does the
filtering next to the vectors, which is what you want once the index grows.
The numpy store is an exact brute-force baseline, kept so the Qdrant results
can be checked against ground truth and the two benchmarked against each other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Payload fields we filter on. Kept small and flat so Qdrant can index them.
FILTER_FIELDS = ("company", "location_canonical", "remote", "seniority", "source")


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self, dim: int) -> None: ...


class NumpyStore(VectorStore):
    """Exact search by matrix multiply. At a few thousand rows this is already
    sub-millisecond, which is the honest reason a vector DB is not *needed*
    here -- it is used for filtering and for the path to a larger index."""

    name = "numpy"

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        if self.path and self.path.exists():
            self.load()

    def reset(self, dim: int) -> None:
        self.ids, self.vectors = [], np.zeros((0, dim), dtype=np.float32)

    def upsert(self, ids, vectors, payloads=None) -> None:
        if self.vectors.size == 0:
            self.ids, self.vectors = list(ids), np.asarray(vectors, dtype=np.float32)
            return
        keep = [i for i, j in enumerate(self.ids) if j not in set(ids)]
        self.ids = [self.ids[i] for i in keep] + list(ids)
        self.vectors = np.vstack([self.vectors[keep], np.asarray(vectors, dtype=np.float32)])

    def search(self, vector, k, allowed_ids=None):
        if not self.ids:
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32).ravel()
        order = np.argsort(-scores)
        out = []
        for idx in order:
            job_id = self.ids[idx]
            if allowed_ids is not None and job_id not in allowed_ids:
                continue
            out.append((job_id, float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        return len(self.ids)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, ids=np.array(self.ids, dtype=object), vectors=self.vectors)

    def load(self) -> None:
        data = np.load(self.path, allow_pickle=True)
        self.ids, self.vectors = list(data["ids"]), data["vectors"]


class QdrantStore(VectorStore):
    """Qdrant in local (embedded) mode: a real vector database with payload
    filtering, running in-process with no server and no Docker."""

    name = "qdrant"
    COLLECTION = "jobs"
    declare_payload_indexes = False  # True when pointing at a Qdrant server

    def __init__(self, path: Path | str):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=str(path))

    def reset(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Local (embedded) Qdrant ignores payload indexes -- it warns and
        # scans instead. They are declared anyway so that pointing this at a
        # Qdrant server needs no code change, but filtering here is resolved
        # in SQL and applied on the results, not pushed down to the engine.
        if getattr(self, "declare_payload_indexes", False):
            for field in FILTER_FIELDS:
                try:
                    self.client.create_payload_index(self.COLLECTION, field, "keyword")
                except Exception:
                    pass

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_point_id(i), vector=v.tolist(), payload={**p, "job_id": i})
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.COLLECTION, points=points)

    def search(self, vector, k, allowed_ids=None):
        """Vector search, restricted to `allowed_ids` when given.

        The restriction is resolved in SQL (the authoritative source for
        filters) and applied here. A fixed over-fetch is not enough: a
        selective filter such as "intern roles in India" can easily have no
        match in the top 100 by similarity, which would return nothing while
        matches existed. So we page deeper until we have k hits or run out.
        """
        query = np.asarray(vector).ravel().tolist()
        if allowed_ids is None:
            hits = self.client.query_points(
                collection_name=self.COLLECTION, query=query,
                limit=k, with_payload=True,
            ).points
            return [((h.payload or {}).get("job_id"), float(h.score)) for h in hits]

        total = self.count()
        out, depth = [], max(k * 8, 200)
        while True:
            hits = self.client.query_points(
                collection_name=self.COLLECTION, query=query,
                limit=depth, with_payload=True,
            ).points
            out = [((h.payload or {}).get("job_id"), float(h.score)) for h in hits
                   if (h.payload or {}).get("job_id") in allowed_ids]
            if len(out) >= k or depth >= total or len(hits) < depth:
                break
            depth = min(depth * 4, total)
        return out[:k]

    def count(self) -> int:
        try:
            return self.client.count(self.COLLECTION).count
        except Exception:
            return 0


def _point_id(job_id: str) -> int:
    """Qdrant wants an unsigned int or a UUID; our ids are 16 hex characters,
    which is exactly 64 bits, so reinterpret the whole thing.

    An earlier version truncated to 15 characters and threw away 4 bits for no
    reason, which collided for ids differing only in the last character."""
    return int(job_id, 16)
