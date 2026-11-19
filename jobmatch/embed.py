"""Turning job text into vectors.

Two implementations behind one interface: the real ONNX model, and a
deterministic hashing embedder used by tests so the suite never needs to
download a model or reach the network.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
ALTERNATE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class FastEmbedder:
    """fastembed runs these models on ONNX, so there is no PyTorch to install."""

    def __init__(self, name: str = DEFAULT_MODEL, batch_size: int = 64,
                 threads: int | None = None):
        import os

        from fastembed import TextEmbedding

        self.name = name
        self.batch_size = batch_size
        # ONNX Runtime defaults conservatively here and ends up effectively
        # single-threaded, which made a 33M-parameter model crawl.
        self.threads = threads or max(1, (os.cpu_count() or 4))
        self.model = TextEmbedding(model_name=name, threads=self.threads)
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = list(self.model.embed(list(texts), batch_size=self.batch_size))
        arr = np.asarray(vectors, dtype=np.float32)
        return _normalise(arr)


class HashEmbedder:
    """Deterministic stand-in: hashes word unigrams and bigrams into buckets.

    Nowhere near a real model semantically, but it is stable, instant and
    offline, which is what unit tests need in order to exercise the retrieval
    plumbing without asserting anything about embedding quality.
    """

    name = "hash-ngram (test stand-in)"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9+#.]+", (text or "").lower())
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for gram in grams:
                bucket = int.from_bytes(
                    hashlib.md5(gram.encode()).digest()[:4], "little"
                ) % self.dim
                out[row, bucket] += 1.0
        return _normalise(out)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


def get_embedder(name: str | None = None, threads: int | None = None) -> Embedder:
    """Real model by default; the stand-in only when explicitly asked for."""
    if name == "hash":
        return HashEmbedder()
    return FastEmbedder(name or DEFAULT_MODEL, threads=threads)
