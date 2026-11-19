"""Persist embeddings so the expensive part is paid once.

Embedding the corpus is by far the slowest step, and almost all of it is
wasted work on a re-index: the listings have not changed, only the code
around them. Vectors are cached against the hash of the exact text that
produced them, so editing what gets embedded correctly invalidates the entry
while an unrelated change does not.
"""
import hashlib
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key   TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim   INTEGER NOT NULL,
    data  BLOB NOT NULL
);
"""


def text_key(model: str, job_id: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{model}|{job_id}|{digest}"


class EmbeddingCache:
    def __init__(self, path: Path | str, model: str):
        self.model = model
        self.conn = sqlite3.connect(path)
        # WAL is faster but needs shared-memory support, which network drives
        # and some mounted filesystems do not provide. Fall back rather than
        # failing outright on a project folder that lives on one.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("SELECT 1").fetchone()
        except sqlite3.OperationalError:
            self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.executescript(SCHEMA)

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found = {}
        for start in range(0, len(keys), 500):  # SQLite caps variables per query
            chunk = keys[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for key, blob in self.conn.execute(
                f"SELECT key, data FROM vectors WHERE key IN ({marks})", chunk
            ):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, keys: list[str], vectors: np.ndarray) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, model, dim, data) VALUES (?,?,?,?)",
            [(k, self.model, int(v.shape[0]), v.astype(np.float32).tobytes())
             for k, v in zip(keys, vectors)],
        )
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE model = ?", (self.model,)
        ).fetchone()[0]

    def close(self) -> None:
        self.conn.close()
