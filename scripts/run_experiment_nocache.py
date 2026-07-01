#!/usr/bin/env python3
"""Exp 9 variant: VQT with the exact-match cache DISABLED.

Isolates whether the function_name drop on BFCL is a stale-cache artifact
(serving the first-seen message's cached answer to a colliding code) or a
codebook-resolution artifact (the dictionary medoid label itself collapses
distinct functions onto one code). With the cache off, every message is routed
via dictionary[code] -> LLM, so any remaining drop is purely code/dictionary
resolution, not cache staleness.

On BFCL simple+multiple (600 msgs, K=256) this recovers function_name_accuracy
to ~0.91 (vs ~0.93 baseline) while cache-on collapses to ~0.52 at
collision_rate 0.43 -- i.e. the drop is the cache meeting a distribution with
no repeated intents, exactly why caching belongs on intent-structured data.

Usage:
    python scripts/run_experiment_nocache.py \
        --bfcl-dir data/bfcl \
        --codebook artifacts/codebook.npz \
        --dictionary artifacts/dictionary.json \
        --backend openai --model gpt-4o-mini \
        --out results/exp9_nocache.json
"""
import argparse
import json
from vqt.encoders import RealEncoder
from vqt.codebook import Codebook
from vqt.dictionary import Dictionary
from vqt.experiment import run
from vqt.data import bfcl
from vqt import llm as llm_mod


class NoCache:
    """Always-miss cache: get() -> None, put() -> no-op."""
    def get(self, code):
        return None

    def put(self, code, value):
        pass


def get_llm(backend, model):
    if backend == "openai":
        return llm_mod.openai_chat_llm(model=model)
    if backend == "stub":
        return llm_mod.stub_llm
    raise ValueError(f"unknown backend {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-dir", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--dictionary", required=True)
    ap.add_argument("--categories", nargs="+", default=list(bfcl.CATEGORIES))
    ap.add_argument("--backend", default="openai", choices=["openai", "stub"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--model-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = bfcl.load(args.bfcl_dir, args.categories)
    enc = RealEncoder(args.model_encoder)
    cb = Codebook.load(args.codebook)
    dic = Dictionary.load(args.dictionary)
    call = get_llm(args.backend, args.model)

    vqt_nc = run(examples, enc, codebook=cb, dictionary=dic,
                 cache=NoCache(), llm=call, use_vqt=True)

    out = {"n_examples": len(examples),
           "categories": args.categories,
           "vqt_nocache": vqt_nc.report()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
