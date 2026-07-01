"""Exact-match cache keyed on the integer code. This is the property
continuous latent methods (C2C, LatentMAS) cannot offer: a float vector is
not a hashable key, so their payloads are neither dedupable nor cacheable.
A discrete code turns semantic identity into an O(1) hash lookup."""
from __future__ import annotations


class SemanticCache:
    def __init__(self):
        self.store: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, code: int):
        code = int(code)
        if code in self.store:
            self.hits += 1
            return self.store[code]
        self.misses += 1
        return None

    def put(self, code: int, value) -> None:
        self.store[int(code)] = value

    def reset_stats(self) -> None:
        self.hits = self.misses = 0

    @property
    def serve_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
