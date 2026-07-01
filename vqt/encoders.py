"""Sentence encoders. The encoder is pluggable so the pipeline can be
tested without downloading torch; swap in RealEncoder for real runs."""
from __future__ import annotations
import numpy as np


class Encoder:
    """Interface: embed(list[str]) -> (N, d) float32, L2-normalized rows."""
    dim: int

    def embed(self, texts) -> np.ndarray:
        raise NotImplementedError


class RealEncoder(Encoder):
    """all-MiniLM-L6-v2 (d=384). Requires `sentence-transformers`.

    This is the only place a real neural model is loaded. Training is
    limited to the K-means codebook; no LLM fine-tuning anywhere.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts) -> np.ndarray:
        v = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(v, dtype=np.float32)


class FastEmbedEncoder(Encoder):
    """all-MiniLM-L6-v2 via fastembed (ONNX, no torch). Same weights as
    RealEncoder, lighter runtime. Good for reproducible CPU experiments."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding  # lazy import
        self.model = TextEmbedding(model_name=model_name)
        self.dim = 384

    def embed(self, texts) -> np.ndarray:
        v = np.array(list(self.model.embed(list(texts))), dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v


class FakeEncoder(Encoder):
    """Deterministic hash-based encoder for offline plumbing tests.

    Same string -> same vector. Near-duplicate strings share an anchor so
    K-means clusters and cache collisions actually occur. NOT for real numbers.
    """
    def __init__(self, dim: int = 384, n_anchors: int = 40, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n_anchors, dim)).astype(np.float32)
        self._anchors = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    def embed(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            key = t.lower().strip()
            anchor = self._anchors[abs(hash(key)) % len(self._anchors)]
            jitter = np.random.default_rng(abs(hash(t)) % (2**32)).standard_normal(self.dim)
            v = anchor + 0.15 * jitter.astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out
