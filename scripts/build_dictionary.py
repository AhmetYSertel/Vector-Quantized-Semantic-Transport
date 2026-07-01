#!/usr/bin/env python3
"""Build the code->label dictionary (medoid per centroid).

Usage:
    python scripts/build_dictionary.py \
        --corpus data/control_messages.txt \
        --codebook artifacts/codebook.npz \
        --out artifacts/dictionary.json
"""
import argparse
from vqt.encoders import RealEncoder
from vqt.codebook import Codebook
from vqt.dictionary import Dictionary


def read_corpus(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    corpus = read_corpus(args.corpus)
    enc = RealEncoder(args.model)
    vecs = enc.embed(corpus)
    cb = Codebook.load(args.codebook)
    dic = Dictionary.build(cb, corpus, vecs)
    dic.save(args.out)
    dead = sum(v is None for v in dic.labels.values())
    print("saved", args.out, f"| dead codes: {dead}/{cb.K}")


if __name__ == "__main__":
    main()
