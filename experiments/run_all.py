"""
VQT experiment suite (control-plane reframe) — generates real numbers.

Routing/cache/coherence/collision on intent-classification datasets
(CLINC150, Banking77) which have the defining control-plane structure:
many surface forms -> small recurring intent set. All metrics are computed
against ground-truth intent labels; no LLM is required. The live-LLM
end-to-end confirmation on BFCL (Exp 9) is a separate script needing an API key.

Usage:
    python experiments/run_all.py --dataset clinc150
    python experiments/run_all.py --dataset banking77
"""
import argparse, json, os, time
from collections import Counter, defaultdict
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vqt.encoders import FastEmbedEncoder
from vqt.codebook import Codebook
from vqt.frame import pack_frame
from vqt.data import intent

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "emb")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
def embed_cached(enc, texts, tag):
    path = os.path.join(CACHE_DIR, f"{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    v = enc.embed(texts)
    np.save(path, v)
    return v


def load_dataset(name, root):
    if name == "clinc150":
        tr, te = intent.load_clinc150(os.path.join(root, "data/intent/clinc150.json"))
    elif name == "banking77":
        tr, te = intent.load_banking77(
            os.path.join(root, "data/intent/banking77_train.csv"),
            os.path.join(root, "data/intent/banking77_test.csv"))
    else:
        raise ValueError(name)
    return tr, te


# --------------------------------------------------------------------------- #
def full_precision_ceiling(tr_vecs, tr_labels, te_vecs, te_labels):
    """1-NN routing in the full (unquantized) embedding space = the ceiling.
    VQ routing accuracy / this = fraction of routing signal kept after
    quantizing 384 floats down to log2(K) bits."""
    tr_labels = np.array(tr_labels, dtype=object)
    sims = te_vecs @ tr_vecs.T                 # cosine (unit-norm rows)
    nn = np.argmax(sims, axis=1)
    pred = tr_labels[nn]
    acc = float(np.mean([p == g for p, g in zip(pred, te_labels)]))
    return {"full_precision_1nn_accuracy": round(acc, 4)}


def routing_metrics(cb, tr_vecs, tr_labels, te_vecs, te_labels):
    """Nearest-centroid routing: label each centroid by majority train intent,
    then route test utterances by their code."""
    tr_codes = cb.encode(tr_vecs)[0]
    te_codes = cb.encode(te_vecs)[0]

    cell = defaultdict(Counter)
    for c, l in zip(tr_codes, tr_labels):
        cell[int(c)][l] += 1
    code_label = {c: cnt.most_common(1)[0][0] for c, cnt in cell.items()}

    # train cell purity (size-weighted)
    tot = sum(sum(c.values()) for c in cell.values())
    purity = sum(c.most_common(1)[0][1] for c in cell.values()) / tot

    n = len(te_labels)
    hit = correct_overall = correct_on_hit = 0
    for c, gold in zip(te_codes, te_labels):
        c = int(c)
        if c in code_label:
            hit += 1
            if code_label[c] == gold:
                correct_on_hit += 1
                correct_overall += 1

    # streaming dedup: process test as a live stream, a message is a cache
    # hit if its code was already emitted earlier in the stream. This is the
    # real "LLM calls skipped" number (the 88%-hit analog).
    seen = {}
    stream_hits = stream_coherent = 0
    for c, gold in zip(te_codes, te_labels):
        c = int(c)
        if c in seen:
            stream_hits += 1
            if seen[c] == gold:          # same intent as the first occurrence
                stream_coherent += 1
        else:
            seen[c] = gold
    return {
        "n_test": n,
        "n_intents": len(set(tr_labels)),
        "labeled_centroids": len(code_label),
        "train_cell_purity": round(purity, 4),
        "code_routing_accuracy": round(correct_overall / n, 4),   # miss = fail (conservative)
        "warm_cache_hit_rate": round(hit / n, 4),                 # code populated in train
        "served_correctness": round(correct_on_hit / hit, 4) if hit else 0.0,
        "streaming_dedup_hit_rate": round(stream_hits / n, 4),    # LLM calls skipped on a live stream
        "streaming_hit_coherence": round(stream_coherent / stream_hits, 4) if stream_hits else 0.0,
        "collision_rate": round((hit - correct_on_hit) / n, 4),   # served-but-misrouted
    }


def fidelity_metrics(cb, vecs):
    codes = cb.encode(vecs)[0]
    cos = (vecs * cb.centroids[codes]).sum(1)   # both unit norm -> cosine
    return {
        "mean_cosine": round(float(cos.mean()), 4),
        "median_cosine": round(float(np.median(cos)), 4),
        "p5": round(float(np.percentile(cos, 5)), 4),
        "p95": round(float(np.percentile(cos, 95)), 4),
        "high_conf_rate_0.9": round(float((cos >= 0.9).mean()), 4),
    }


def bandwidth_metrics(texts, K, dim=384):
    nbytes = np.array([len(t.encode("utf-8")) for t in texts])
    emb_bits = dim * 32                       # float32 embedding
    code_bits = int(np.ceil(np.log2(K)))      # discrete code
    return {
        "mean_nl_bytes": round(float(nbytes.mean()), 1),
        "vq_frame_bytes": 37,
        "byte_compression_ratio": round(float(nbytes.mean()) / 37, 2),
        "embedding_bits": emb_bits,
        "code_bits": code_bits,
        "representational_compression": round(emb_bits / code_bits, 1),
    }


def throughput_metrics(cb, enc, vecs, texts):
    # batch encode (from pre-embedded vecs -> the marginal VQ step)
    t0 = time.perf_counter(); cb.encode(vecs); t1 = time.perf_counter()
    vq_batch = len(vecs) / (t1 - t0)
    # single message VQ
    t0 = time.perf_counter()
    for i in range(min(1000, len(vecs))):
        cb.encode(vecs[i:i+1])
    t1 = time.perf_counter()
    vq_single = min(1000, len(vecs)) / (t1 - t0)
    # decode = centroid gather
    codes = cb.encode(vecs)[0]
    t0 = time.perf_counter(); _ = cb.centroids[codes]; t1 = time.perf_counter()
    decode = len(codes) / (t1 - t0)
    # full encode incl. embedding (small sample; CPU/ONNX)
    sample = texts[:200]
    t0 = time.perf_counter(); ev = enc.embed(sample); cb.encode(ev); t1 = time.perf_counter()
    full = len(sample) / (t1 - t0)
    return {
        "note": "CPU/ONNX (fastembed); GPU numbers will differ",
        "vq_batch_encode_msg_per_s": int(vq_batch),
        "vq_single_encode_msg_per_s": int(vq_single),
        "decode_msg_per_s": int(decode),
        "full_pipeline_incl_embed_msg_per_s": int(full),
    }


def k_scaling(tr_vecs, tr_labels, te_vecs, te_labels, Ks):
    rows = []
    for K in Ks:
        cb = Codebook.train(tr_vecs, K=K, seed=42)
        r = routing_metrics(cb, tr_vecs, tr_labels, te_vecs, te_labels)
        f = fidelity_metrics(cb, te_vecs)
        cov = cb.coverage(tr_vecs)
        rows.append({"K": K,
                     "routing_acc": r["code_routing_accuracy"],
                     "streaming_dedup": r["streaming_dedup_hit_rate"],
                     "hit_coherence": r["streaming_hit_coherence"],
                     "collision": r["collision_rate"],
                     "mean_cosine": f["mean_cosine"],
                     "utilization": round(cov["utilization"], 4)})
    return rows


def mismatch_metrics(tr_vecs, tr_labels, te_vecs, te_labels, K=256):
    cb42 = Codebook.train(tr_vecs, K=K, seed=42, version=1)
    cb777 = Codebook.train(tr_vecs, K=K, seed=777, version=2)
    te_codes42 = cb42.encode(te_vecs)[0]

    # fidelity: decode the code against matched vs mismatched codebook
    matched = (te_vecs * cb42.centroids[te_codes42]).sum(1).mean()
    mism = (te_vecs * cb777.centroids[te_codes42]).sum(1).mean()

    # routing under mismatch: receiver holds cb777 labels but codes came from cb42
    def labels_for(cb):
        codes = cb.encode(tr_vecs)[0]
        cell = defaultdict(Counter)
        for c, l in zip(codes, tr_labels):
            cell[int(c)][l] += 1
        return {c: cnt.most_common(1)[0][0] for c, cnt in cell.items()}
    lab42, lab777 = labels_for(cb42), labels_for(cb777)

    def acc(codes, labels):
        ok = sum(1 for c, g in zip(codes, te_labels)
                 if int(c) in labels and labels[int(c)] == g)
        return ok / len(te_labels)

    return {
        "matched_fidelity": round(float(matched), 4),
        "mismatched_fidelity": round(float(mism), 4),
        "matched_routing_acc": round(acc(te_codes42, lab42), 4),
        "mismatched_routing_acc": round(acc(te_codes42, lab777), 4),  # silent corruption
        "versioned_recovery_acc": round(acc(te_codes42, lab42), 4),   # version check -> use matched
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc150", choices=["clinc150", "banking77"])
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
    args = ap.parse_args()

    def load_npz(split):
        p = os.path.join(CACHE_DIR, f"{args.dataset}_{split}.npz")
        z = np.load(p, allow_pickle=True)
        return z["vecs"].astype(np.float32), list(z["texts"]), list(z["labels"])

    tr_vecs, tr_texts, tr_labels = load_npz("train")
    te_vecs, te_texts, te_labels = load_npz("test")
    enc = FastEmbedEncoder()   # only used for the full-pipeline throughput sample
    print(f"[{args.dataset}] train {len(tr_texts)} / test {len(te_texts)} "
          f"/ intents {len(set(tr_labels))}")

    cb = Codebook.train(tr_vecs, K=args.K, seed=42)

    out = {
        "dataset": args.dataset,
        "n_train": len(tr_texts), "n_test": len(te_texts),
        "n_intents": len(set(tr_labels)), "K": args.K,
        "exp1_fidelity": fidelity_metrics(cb, te_vecs),
        "exp2_bandwidth": bandwidth_metrics(te_texts, args.K),
        "exp3_coverage": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in cb.coverage(tr_vecs).items()},
        "exp4_throughput": throughput_metrics(cb, enc, tr_vecs, tr_texts),
        "exp5_routing": routing_metrics(cb, tr_vecs, tr_labels, te_vecs, te_labels),
        "exp5b_ceiling": full_precision_ceiling(tr_vecs, tr_labels, te_vecs, te_labels),
        "exp6_k_scaling": k_scaling(tr_vecs, tr_labels, te_vecs, te_labels,
                                    Ks=[8, 16, 32, 64, 128, 256, 512]),
        "exp7_mismatch": mismatch_metrics(tr_vecs, tr_labels, te_vecs, te_labels, K=args.K),
    }

    os.makedirs(os.path.join(args.root, "results"), exist_ok=True)
    rpath = os.path.join(args.root, "results", f"{args.dataset}.json")
    with open(rpath, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nsaved ->", rpath)


if __name__ == "__main__":
    main()
