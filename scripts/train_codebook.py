#!/usr/bin/env python3
"""Train the K-means codebook from a corpus.

The reframed paper trains on control-plane messages (BFCL queries +
handoff/routing corpora), not generic sentences, so fidelity is measured
on the target distribution.

Usage:
    python scripts/train_codebook.py \
        --corpus data/control_messages.txt \
        --out artifacts/codebook.npz --K 256 --seed 42
"""
import argparse
import numpy as np
from vqt.encoders import RealEncoder
from vqt.codebook import Codebook


def read_corpus(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="one message per line")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    corpus = read_corpus(args.corpus)
    enc = RealEncoder(args.model)
    vecs = enc.embed(corpus)
    cb = Codebook.train(vecs, K=args.K, seed=args.seed, version=args.version)
    cb.save(args.out)
    print("saved", args.out, "| coverage:", cb.coverage(vecs))


if __name__ == "__main__":
    main()
