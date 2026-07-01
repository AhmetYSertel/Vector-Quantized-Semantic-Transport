"""The code->text bridge. Each centroid gets one canonical natural-language
label (the medoid: the corpus sentence closest to that centroid). On a cache
miss, the receiver looks up this label and hands plain text to any text-API
LLM. No model ever consumes a vector."""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from .codebook import Codebook


@dataclass
class Dictionary:
    labels: dict          # {code(int): sentence(str) | None for dead codes}
    version: int = 1

    def get(self, code: int):
        return self.labels.get(int(code))

    @classmethod
    def build(cls, codebook: Codebook, corpus, corpus_vecs) -> "Dictionary":
        """Medoid label per centroid. corpus[i] <-> corpus_vecs[i]."""
        corpus_vecs = np.ascontiguousarray(corpus_vecs, dtype=np.float32)
        codes, _, _ = codebook.encode(corpus_vecs)
        # squared distance of each corpus point to its own centroid
        diff = corpus_vecs - codebook.centroids[codes]
        d2_self = (diff ** 2).sum(1)
        labels = {}
        for k in range(codebook.K):
            members = np.where(codes == k)[0]
            if len(members) == 0:
                labels[k] = None                     # dead code
                continue
            best = members[np.argmin(d2_self[members])]
            labels[k] = corpus[int(best)]
        return cls(labels=labels, version=codebook.version)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": self.version,
                       "labels": {str(k): v for k, v in self.labels.items()}},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Dictionary":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(labels={int(k): v for k, v in d["labels"].items()},
                   version=int(d["version"]))
