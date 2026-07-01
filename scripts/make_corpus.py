#!/usr/bin/env python3
"""Regenerate the exp9 codebook corpus (data/control_messages.txt) from BFCL.

The codebook/dictionary for the live-LLM exp9 are trained on the BFCL user
messages themselves (the target distribution *is* control-plane messages).
This script makes that corpus reproducible instead of hand-built: it extracts
the first user message of every BFCL example, in the deterministic load order
of vqt.data.bfcl.load, one message per line (newlines flattened to spaces).

Usage:
    python scripts/make_corpus.py --bfcl-dir data/bfcl \
        --out data/control_messages.txt
"""
import argparse
from vqt.data import bfcl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--categories", nargs="+", default=list(bfcl.CATEGORIES))
    ap.add_argument("--out", default="data/control_messages.txt")
    args = ap.parse_args()

    examples = bfcl.load(args.bfcl_dir, args.categories)
    msgs = [e["message"].replace("\n", " ").strip()
            for e in examples if e["message"].strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(m + "\n")
    print(f"wrote {args.out} | {len(msgs)} messages "
          f"from {len(examples)} BFCL examples ({', '.join(args.categories)})")


if __name__ == "__main__":
    main()
