"""K-means codebook over sentence embeddings. Nearest-centroid encoding
yields a deterministic integer code per message."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass
class Codebook:
    centroids: np.ndarray          # (K, d) float32, L2-normalized
    version: int = 1
    seed: int = 42

    @property
    def K(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    # -- training / io ----------------------------------------------------- #
    @classmethod
    def train(cls, embeddings: np.ndarray, K: int = 256,
              seed: int = 42, version: int = 1) -> "Codebook":
        km = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        km.fit(embeddings)
        c = km.cluster_centers_.astype(np.float32)
        c /= (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        return cls(centroids=c, version=version, seed=seed)

    def save(self, path: str) -> None:
        np.savez(path, centroids=self.centroids,
                 version=self.version, seed=self.seed)

    @classmethod
    def load(cls, path: str) -> "Codebook":
        z = np.load(path)
        return cls(centroids=z["centroids"].astype(np.float32),
                   version=int(z["version"]), seed=int(z.get("seed", 42)))

    # -- encoding ---------------------------------------------------------- #
    def encode(self, vecs: np.ndarray):
        """(N,d) -> (codes int32, qerr float32, margin float32).

        margin = distance gap between nearest and 2nd-nearest centroid,
        a cheap confidence signal (small margin => ambiguous assignment).
        """
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        # (N, K) squared L2 via ||a||^2 - 2 a.b + ||b||^2
        a2 = (vecs ** 2).sum(1, keepdims=True)
        b2 = (self.centroids ** 2).sum(1)[None, :]
        d2 = a2 - 2.0 * vecs @ self.centroids.T + b2
        np.maximum(d2, 0, out=d2)
        nearest_idx = np.argmin(d2, axis=1)
        codes = nearest_idx.astype(np.int32)
        rows = np.arange(len(vecs))
        nearest = d2[rows, nearest_idx]
        d2[rows, nearest_idx] = np.inf
        second = d2.min(axis=1)
        qerr = np.sqrt(nearest).astype(np.float32)
        margin = (np.sqrt(second) - np.sqrt(nearest)).astype(np.float32)
        return codes, qerr, margin

    # -- diagnostics ------------------------------------------------------- #
    def coverage(self, vecs: np.ndarray) -> dict:
        codes, qerr, _ = self.encode(vecs)
        counts = np.bincount(codes, minlength=self.K)
        active = int((counts > 0).sum())
        # true Gini coefficient of centroid loads (0=even, 1=concentrated)
        c = np.sort(counts.astype(np.float64))
        n = len(c)
        gini = (2 * np.sum((np.arange(1, n + 1)) * c) / (n * c.sum())
                - (n + 1) / n) if c.sum() > 0 else 0.0
        return {
            "active_codes": active,
            "utilization": active / self.K,
            "dead_codes": self.K - active,
            "gini": float(gini),
            "mean_qerr": float(qerr.mean()),
        }
