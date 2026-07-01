#!/usr/bin/env python3
"""Run Experiment 9: baseline vs VQT control-plane path on BFCL.

Usage:
    python scripts/run_experiment.py \
        --bfcl-dir data/bfcl \
        --codebook artifacts/codebook.npz \
        --dictionary artifacts/dictionary.json \
        --backend openai --model gpt-4o-mini \
        --out results/exp9.json
"""
import argparse
import json
from vqt.encoders import RealEncoder
from vqt.codebook import Codebook
from vqt.dictionary import Dictionary
from vqt.cache import SemanticCache
from vqt.experiment import run
from vqt.data import bfcl
from vqt import llm as llm_mod


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

    base = run(examples, enc, llm=call, use_vqt=False)
    vqt = run(examples, enc, codebook=cb, dictionary=dic,
              cache=SemanticCache(), llm=call, use_vqt=True)

    out = {"n_examples": len(examples),
           "categories": args.categories,
           "baseline": base.report(),
           "vqt": vqt.report()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
