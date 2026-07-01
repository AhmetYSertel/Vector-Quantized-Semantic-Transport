"""Generate the two headline figures from results JSON."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)


def fig_kscaling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ["clinc150", "banking77"]):
        d = json.load(open(os.path.join(ROOT, "results", f"{ds}.json")))
        ks = d["exp6_k_scaling"]
        K = [r["K"] for r in ks]
        ax.plot(K, [r["routing_acc"] for r in ks], "o-", label="routing accuracy")
        ax.plot(K, [r["streaming_dedup"] for r in ks], "s--", label="dedup hit rate")
        ax.plot(K, [r["collision"] for r in ks], "^:", label="collision rate")
        ax.axvline(256, color="gray", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("codebook size K")
        ax.set_title(f"{ds}  ({d['n_intents']} intents)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Routing vs. dedup vs. collision across codebook size (K=256 marked)")
    fig.tight_layout()
    p = os.path.join(FIG, "k_scaling.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_mismatch():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    conds = ["matched", "mismatched\n(no versioning)", "versioned\n+ recovery"]
    width = 0.35
    x = range(len(conds))
    for i, ds in enumerate(["clinc150", "banking77"]):
        d = json.load(open(os.path.join(ROOT, "results", f"{ds}.json")))["exp7_mismatch"]
        vals = [d["matched_routing_acc"], d["mismatched_routing_acc"],
                d["versioned_recovery_acc"]]
        ax.bar([xi + (i - 0.5) * width for xi in x], vals, width,
               label=ds)
    ax.axhspan(0, 0.1, color="red", alpha=0.08)
    ax.set_xticks(list(x)); ax.set_xticklabels(conds)
    ax.set_ylabel("routing accuracy")
    ax.set_title("Silent corruption under codebook mismatch, recovered by versioning")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(FIG, "mismatch.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig_kscaling())
    print(fig_mismatch())
