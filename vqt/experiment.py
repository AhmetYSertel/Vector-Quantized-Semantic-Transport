"""Experiment driver.

Baseline A : raw message -> LLM -> tool call
VQT path B : embed -> code -> cache? -> (miss) dictionary label -> LLM -> tool call

Metrics (control-plane framing):
    function_name_accuracy : did we route to the right tool?  (the claim)
    full_call_accuracy     : name + args   (expected to lag; that gap IS the
                             control-plane vs data-plane boundary)
    cache_serve_rate       : share of messages answered from cache, LLM skipped
    collision_rate         : share of messages whose code was already seen with
                             a *different* gold function (semantic-corruption risk)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .encoders import Encoder
from .codebook import Codebook
from .dictionary import Dictionary
from .cache import SemanticCache
from .scoring import score_ast
from .llm import stub_llm


@dataclass
class ExperimentResult:
    mode: str
    n: int = 0
    fn_name_correct: int = 0
    full_call_correct: int = 0
    cache_served: int = 0
    collisions: int = 0
    extra: dict = field(default_factory=dict)

    def report(self) -> dict:
        n = max(self.n, 1)
        return {
            "mode": self.mode,
            "n": self.n,
            "function_name_accuracy": round(self.fn_name_correct / n, 4),
            "full_call_accuracy": round(self.full_call_correct / n, 4),
            "cache_serve_rate": round(self.cache_served / n, 4),
            "collision_rate": round(self.collisions / n, 4),
            **self.extra,
        }


def run(examples, encoder: Encoder, *, codebook: Codebook = None,
        dictionary: Dictionary = None, cache: SemanticCache = None,
        llm=stub_llm, scorer=score_ast, use_vqt: bool = True) -> ExperimentResult:
    """Run one condition over `examples`.

    Set use_vqt=False for baseline A (codebook/dictionary/cache ignored).
    """
    mode = "VQT" if use_vqt else "baseline"
    res = ExperimentResult(mode=mode)

    if use_vqt:
        if codebook is None or dictionary is None:
            raise ValueError("VQT mode needs a codebook and dictionary")
        cache = cache or SemanticCache()
        vecs = encoder.embed([ex["message"] for ex in examples])
        codes, _, _ = codebook.encode(vecs)
        seen_code_gold: dict = {}

    for i, ex in enumerate(examples):
        res.n += 1
        gold = ex["gold"]
        tools = ex.get("tools")

        if use_vqt:
            code = int(codes[i])
            gname = gold.get("name")
            if code in seen_code_gold and seen_code_gold[code] != gname:
                res.collisions += 1
            seen_code_gold.setdefault(code, gname)

            cached = cache.get(code)
            if cached is not None:
                pred = cached
                res.cache_served += 1
            else:
                label = dictionary.get(code) or ex["message"]   # dead-code fallback
                pred = llm(label, tools)
                cache.put(code, pred)
        else:
            pred = llm(ex["message"], tools)

        name_ok, full_ok = scorer(pred, gold)
        res.fn_name_correct += int(name_ok)
        res.full_call_correct += int(full_ok)

    if use_vqt:
        res.extra["unique_codes_used"] = len(seen_code_gold)
    return res
