"""Embed one split with a stratified cap and cache it as an npz
(vecs + texts + labels), so the experiment runner stays fast and uses the
exact same subset. Kept separate because CPU/ONNX embedding is the slow step."""
import argparse, os, sys, json, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vqt.encoders import FastEmbedEncoder
from vqt.data import intent

EMB = os.path.join(os.path.dirname(__file__), "..", "artifacts", "emb")
os.makedirs(EMB, exist_ok=True)


def stratified(pairs, per_label, seed=0):
    by = {}
    for u, l in pairs:
        by.setdefault(l, []).append(u)
    rng = random.Random(seed)
    out = []
    for l, us in by.items():
        rng.shuffle(us)
        for u in us[:per_label]:
            out.append((u, l))
    rng.shuffle(out)
    return out


def load(dataset, root):
    if dataset == "clinc150":
        return intent.load_clinc150(os.path.join(root, "data/intent/clinc150.json"))
    return intent.load_banking77(
        os.path.join(root, "data/intent/banking77_train.csv"),
        os.path.join(root, "data/intent/banking77_test.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--per-label", type=int, required=True)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
    args = ap.parse_args()

    tr, te = load(args.dataset, args.root)
    pairs = tr if args.split == "train" else te
    sub = stratified(pairs, args.per_label, seed=0)
    texts = [u for u, l in sub]
    labels = [l for u, l in sub]

    enc = FastEmbedEncoder()
    print(f"embedding {len(texts)} {args.dataset}/{args.split} ...", flush=True)
    vecs = enc.embed(texts).astype(np.float32)

    out = os.path.join(EMB, f"{args.dataset}_{args.split}.npz")
    np.savez(out, vecs=vecs, texts=np.array(texts, dtype=object),
             labels=np.array(labels, dtype=object))
    print("saved", out, vecs.shape)


if __name__ == "__main__":
    main()
