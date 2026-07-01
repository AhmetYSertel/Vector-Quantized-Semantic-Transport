# VQT — Experimental Results (control-plane reframe)

All numbers are real, produced by `experiments/run_all.py` on
`all-MiniLM-L6-v2` (d=384) embeddings. Throughput is CPU/ONNX (fastembed);
GPU numbers will differ. The live-LLM end-to-end confirmation (Exp 9) is a
separate harness requiring an API key (see below) and is **not** included in
these numbers.

## Design rationale

The reframed claim is about the **control plane**: many surface forms of a
message map to a small recurring set of intents/tools, and a discrete code
should (a) route to the right handler and (b) let repeated intents be served
from cache. BFCL `simple`/`multiple` has no repeated intents (each example is
a distinct function), so it cannot test routing/caching. Intent-classification
benchmarks are purpose-built for exactly this structure, so they are the
primary evaluation; BFCL is retained only for the live-LLM confirmation.

- **Datasets:** CLINC150 (150 intents), Banking77 (77 intents, single-domain).
- **Protocol:** train K-means codebook on train embeddings; label each centroid
  by its majority train intent (nearest-centroid router); evaluate on held-out test.
- **No LLM** is used for any number below — every metric is against ground-truth
  intent labels.

Subsample sizes (CPU budget): CLINC150 6000 train / 3000 test;
Banking77 4575 train / 2310 test (stratified). Full-scale runs on GPU are
expected to move numbers slightly upward.

## Headline (K = 256)

| metric | CLINC150 | Banking77 | reads as |
|---|---|---|---|
| **code routing accuracy** | **0.826** | **0.816** | right handler from an 8-bit code |
| full-precision 1-NN ceiling | 0.859 | 0.898 | unquantized routing upper bound |
| **routing signal retained** | **96%** | **91%** | acc / ceiling |
| representational compression | 1536× | 1536× | 12288 bits (384×fp32) → 8 bits |
| streaming dedup hit rate | 0.918 | 0.891 | live-stream LLM calls skipped |
| streaming hit coherence | 0.755 | 0.745 | of hits, same intent as first occurrence |
| train cell purity | 0.846 | 0.822 | intra-code intent homogeneity |
| collision rate | 0.174 | 0.184 | served-but-misrouted |
| fidelity mean / median cosine | 0.714 / 0.728 | 0.835 / 0.851 | vector-space distortion |
| codebook utilization / dead codes | 1.00 / 0 | 1.00 / 0 | no collapse |
| load Gini | 0.271 | 0.284 | even centroid load |

Banking77's dedup hit rate (0.891) and fidelity (0.835) closely track the
original paper's 0.88 cache-hit and 0.854 fidelity — continuity with the prior
result, now grounded in a labeled benchmark.

## K-scaling (routing accuracy)

| K | CLINC150 routing | CLINC150 dedup | Banking77 routing | Banking77 dedup |
|---|---|---|---|---|
| 8   | 0.053 | ~1.00 | 0.099 | ~1.00 |
| 16  | 0.103 | ~0.99 | 0.192 | ~0.99 |
| 32  | 0.203 | ~0.99 | 0.360 | ~0.98 |
| 64  | 0.390 | ~0.98 | 0.603 | ~0.97 |
| 128 | 0.691 | ~0.96 | 0.724 | ~0.95 |
| **256** | **0.826** | **0.918** | **0.816** | **0.891** |
| 512 | 0.856 | ~0.85 | 0.856 | ~0.80 |

Routing rises steeply then plateaus after K=256 (+3% to K=512) while dedup
falls; K=256 is the operating point where routing is near-saturated and dedup
still high. See `results/figures/k_scaling.png`.

## Codebook mismatch and versioning (K = 256)

| condition | CLINC150 routing | Banking77 routing |
|---|---|---|
| matched codebooks | 0.826 | 0.816 |
| **mismatched, no versioning** | **0.009** | **0.017** |
| versioned + recovery | 0.826 | 0.816 |

Fidelity collapses in parallel (CLINC150 0.714 → 0.108). The failure is silent:
the frame is valid, the code is in range, nothing signals corruption. A version
ID in the header restores the matched baseline. See `results/figures/mismatch.png`.

## Throughput (CPU/ONNX; GPU differs)

| op | CLINC150 |
|---|---|
| VQ batch encode | ~350K msg/s |
| VQ single encode | ~15K msg/s |
| decode (centroid gather) | ~9M msg/s |
| full pipeline incl. embedding | ~23 msg/s (embedding-bound) |

The VQ step is negligible; end-to-end cost is dominated by embedding, which any
retrieval/memory pipeline already pays.

## Interpretation for the paper

- **The headline is retained routing signal, not cosine fidelity.** 91–96% of
  full-precision routing survives quantization to 8 bits — this is the number
  that replaces the old 0.854 cosine as the central result.
- **The `collision_rate` / `hit_coherence` (~75%) is the control-plane / data-plane
  boundary, reported honestly.** ~1 in 4 cache hits mixes a near-neighbor intent;
  the code preserves *which region* of intent space, not fine distinctions — which
  is exactly why VQT is a control-plane, not data-plane, primitive.
- **Mismatch is the safety result and it is dramatic on routing** (→ <2%),
  making the versioning requirement non-negotiable, as in the original paper.

## Exp 9 — live-LLM end-to-end on BFCL (needs API key)

`scripts/run_experiment.py` runs baseline (raw message → LLM → tool call) vs the
VQT path (embed → code → cache → dictionary label → LLM → tool call) on BFCL
`simple`+`multiple`, scoring `function_name_accuracy`, `full_call_accuracy`,
`cache_serve_rate`, `collision_rate` with the AST scorer. Set an OpenAI-compatible
backend (`--backend openai`, or point `OPENAI_BASE_URL` at a local vLLM/Ollama
server) to produce these numbers. This confirms the routing result end-to-end
with a real model in the loop.
