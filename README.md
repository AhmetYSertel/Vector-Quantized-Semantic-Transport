# VQT — Vector-Quantized Semantic Transport

**A discrete, text-API-compatible control-plane channel for LLM agents.**

Most inter-agent traffic in multi-agent LLM systems is natural language: routing
labels, handoffs, status updates, tool selections. Generating and re-parsing text
for these is wasteful. Continuous latent-communication methods (C2C, LatentMAS,
Interlat) remove that cost but break compatibility with the text APIs current
models actually expose — you cannot feed a raw hidden state to a text-API model.

VQT keeps the message **discrete**: an integer code pointing to the nearest
centroid in a learned codebook. Discreteness buys three things continuous methods
cannot offer — exact-match caching, deduplication, and version tracking — while a
small canonical **dictionary** (one NL label per code) keeps the channel
compatible with *any* text-API model on a cache miss. No LLM fine-tuning; the only
training is the K-means codebook.

```
Sender:   message --embed--> vector --VQ--> integer code  (37-byte frame)
Receiver: code --cache?--> HIT : cached call         (LLM skipped)
                        --MISS: dictionary[code] -> NL label -> text-API LLM
```

## Install

```bash
pip install -e ".[encoder,llm,dev]"     # or: pip install -r requirements.txt
```

`sentence-transformers` is only needed for the real encoder; the test suite runs
offline with a deterministic `FakeEncoder`.

## Pipeline

```bash
# 1. train the codebook on control-plane messages (not generic sentences)
python scripts/train_codebook.py --corpus data/control_messages.txt \
    --out artifacts/codebook.npz --K 256 --seed 42

# 2. build the code -> NL-label dictionary (medoid per centroid)
python scripts/build_dictionary.py --corpus data/control_messages.txt \
    --codebook artifacts/codebook.npz --out artifacts/dictionary.json

# 3. run Experiment 9: baseline vs VQT on BFCL (simple + multiple)
python scripts/run_experiment.py --bfcl-dir data/bfcl \
    --codebook artifacts/codebook.npz --dictionary artifacts/dictionary.json \
    --backend openai --model gpt-4o-mini --out results/exp9.json
```

BFCL data (place under `--bfcl-dir`):
`BFCL_v3_<cat>.json` and `possible_answer/BFCL_v3_<cat>.json` from the
Gorilla repo. The codebook corpus can be the BFCL messages themselves, since
the target distribution *is* control-plane messages.

## Metrics (Experiment 9)

| metric | meaning |
|---|---|
| `function_name_accuracy` | routed to the right tool — the central claim |
| `full_call_accuracy` | name + args; expected to lag, marking the control-plane vs data-plane boundary |
| `cache_serve_rate` | fraction answered from cache with the LLM skipped |
| `collision_rate` | fraction whose code was already seen with a *different* gold function (semantic-corruption risk) |

The `function_name` vs `full_call` gap is the story: the discrete channel
preserves *which tool*, but a fixed dictionary does not preserve precise
arguments — which is exactly why VQT is a control-plane, not data-plane, primitive.

## Reproduce the numbers

```bash
bash scripts/download_data.sh                     # BFCL + CLINC150 + Banking77
python experiments/embed.py --dataset clinc150 --split train --per-label 40
python experiments/embed.py --dataset clinc150 --split test  --per-label 20
python experiments/run_all.py --dataset clinc150 --K 256      # -> results/clinc150.json
python experiments/make_figures.py                            # -> results/figures/*.png
```

Real numbers and paper-ready tables are in [`RESULTS.md`](RESULTS.md).
Headline: an 8-bit code retains **91–96%** of full-precision routing accuracy
at **1536×** representational compression, with silent codebook-mismatch
corruption (routing → <2%) fully recovered by versioning.

## Layout

```
vqt/            encoders · codebook · dictionary · frame · cache · scoring · experiment
vqt/data/       bfcl loader
scripts/        train_codebook · build_dictionary · run_experiment
tests/          offline plumbing tests (FakeEncoder + synthetic BFCL fixtures)
configs/        default.yaml
```

## Tests

```bash
pytest -q
```

## Swapping in the official BFCL evaluator

`vqt/scoring.py` implements an AST check mirroring BFCL's. For leaderboard
parity, replace `score_ast` with BFCL's own evaluator and pass it via the
`scorer=` argument to `vqt.experiment.run`.
