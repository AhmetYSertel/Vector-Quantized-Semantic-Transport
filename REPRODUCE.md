# Reproducing the VQT results

This repo is split into two evaluation paths with **different reproducibility
guarantees**. Read this before trusting or regenerating any number.

## Environment

Committed `results/` and `artifacts/` were produced with:

| component | version |
|---|---|
| Python | 3.12.10 (Windows 11) |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| sentence-transformers | 5.3.0 |
| fastembed | 0.8.0 |
| openai | 2.44.0 |
| pyyaml | 6.0.3 |

Exact-environment install:

```bash
pip install -r requirements-lock.txt      # pinned == versions above
# or a flexible install:
pip install -r requirements.txt
```

## Encoders — read this, it affects the numbers

Two encoders wrap the **same** `all-MiniLM-L6-v2` weights via different runtimes:

- `RealEncoder` (sentence-transformers / **torch**) — used by `scripts/` (the
  exp9 codebook, dictionary, and live-LLM run).
- `FastEmbedEncoder` (**ONNX** / fastembed) — used by `experiments/` (the intent
  path: CLINC150 / Banking77).

Same weights, but torch and ONNX inference can differ at ~1e-3, which changes
K-means centroids and therefore codes. **Do not mix artifacts across encoders.**
The committed `artifacts/codebook.npz` and `artifacts/dictionary.json` are
`RealEncoder` artifacts (exp9).

## Determinism

| step | deterministic? | why |
|---|---|---|
| corpus (`data/control_messages.txt`) | ✅ byte-identical | `scripts/make_corpus.py`, fixed BFCL load order |
| codebook (`artifacts/codebook.npz`) | ✅ bit-identical | KMeans `random_state=42`, `n_init="auto"` — verified `max abs diff 0.0` on this env |
| intent subset (embed) | ✅ | stratified `seed=0` in `experiments/embed.py` |
| intent metrics (`results/clinc150.json`, `results/banking77.json`) | ✅ given pinned versions | no LLM; all vs ground-truth labels |
| **live-LLM exp9 (`results/exp9*.json`)** | ❌ **not bit-reproducible** | depends on the OpenAI model snapshot; `temperature=0` is not a determinism guarantee |

Codebook/intent determinism holds **for the pinned versions above** — a
different scikit-learn or sentence-transformers/fastembed build can shift codes.

## Path A — intent evaluation (offline, no API key)

```bash
bash scripts/download_data.sh                 # or use the committed data/
python experiments/embed.py --dataset clinc150 --split train --per-label 40
python experiments/embed.py --dataset clinc150 --split test  --per-label 20
python experiments/run_all.py --dataset clinc150 --K 256      # -> results/clinc150.json
# same three for --dataset banking77
python experiments/make_figures.py                            # -> results/figures/*.png
```

## Path B — live-LLM exp9 on BFCL (needs an OpenAI key)

```bash
export OPENAI_API_KEY=sk-...                                  # never commit this
python scripts/make_corpus.py --bfcl-dir data/bfcl \
    --out data/control_messages.txt                          # regenerate the corpus
python scripts/train_codebook.py --corpus data/control_messages.txt \
    --out artifacts/codebook.npz --K 256 --seed 42
python scripts/build_dictionary.py --corpus data/control_messages.txt \
    --codebook artifacts/codebook.npz --out artifacts/dictionary.json
python scripts/run_experiment.py --bfcl-dir data/bfcl \
    --codebook artifacts/codebook.npz --dictionary artifacts/dictionary.json \
    --backend openai --model gpt-4o-mini-2024-07-18 --out results/exp9.json
python scripts/run_experiment_nocache.py --bfcl-dir data/bfcl \
    --codebook artifacts/codebook.npz --dictionary artifacts/dictionary.json \
    --backend openai --model gpt-4o-mini-2024-07-18 --out results/exp9_nocache.json
```

### Provenance of the committed exp9 numbers

`results/exp9.json` and `results/exp9_nocache.json` were produced on
**2026-07-01** with the `gpt-4o-mini` alias (OpenAI's then-current snapshot),
`temperature=0`, over all 600 BFCL `simple`+`multiple` examples, on the
bit-identical `RealEncoder` codebook above. The `--model` default has since been
pinned to `gpt-4o-mini-2024-07-18`; re-running will produce numbers **close but
not identical** to the committed ones, because the served model can change and
sampling is not guaranteed deterministic even at `temperature=0`. Treat the
live-LLM result as a **confirmation of the offline routing signal**, not a
bit-exact artifact — the exact, bit-reproducible evidence is Path A.
