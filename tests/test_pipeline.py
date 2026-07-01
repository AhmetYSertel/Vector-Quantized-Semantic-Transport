"""End-to-end plumbing tests using the offline FakeEncoder + synthetic BFCL.
Proves the pipeline runs; real numbers require RealEncoder + real BFCL data."""
import os
import numpy as np
import pytest

from vqt.encoders import FakeEncoder
from vqt.codebook import Codebook
from vqt.dictionary import Dictionary
from vqt.cache import SemanticCache
from vqt.frame import pack_frame, unpack_frame, FRAME_BYTES
from vqt.scoring import parse_call, score_ast
from vqt.experiment import run
from vqt.data import bfcl

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def corpus():
    templates = [
        "get the current weather in Paris",
        "what is the weather in Paris right now",
        "process a payment of 50 dollars",
        "charge the customer 50 usd",
        "book a table for four at eight pm",
        "reserve a table party of four 8pm",
    ]
    return templates * 30


def test_frame_roundtrip():
    f = pack_frame(code=123, qerr=0.4, margin=0.1, version=2, source_id=7, ts=99)
    assert len(f) == FRAME_BYTES == 37
    d = unpack_frame(f)
    assert d["code"] == 123 and d["version"] == 2 and d["source_id"] == 7


def test_codebook_encode_and_coverage(corpus):
    enc = FakeEncoder(seed=1)
    v = enc.embed(corpus)
    cb = Codebook.train(v, K=16, seed=42)
    codes, qerr, margin = cb.encode(v)
    assert codes.shape[0] == len(corpus)
    assert (qerr >= 0).all() and (margin >= -1e-5).all()
    cov = cb.coverage(v)
    assert 0 < cov["utilization"] <= 1.0


def test_codebook_save_load(tmp_path, corpus):
    enc = FakeEncoder(seed=1)
    cb = Codebook.train(enc.embed(corpus), K=16, seed=42, version=3)
    p = tmp_path / "cb.npz"
    cb.save(str(p))
    cb2 = Codebook.load(str(p))
    assert cb2.K == cb.K and cb2.version == 3
    np.testing.assert_allclose(cb.centroids, cb2.centroids)


def test_dictionary_medoid(tmp_path, corpus):
    enc = FakeEncoder(seed=1)
    v = enc.embed(corpus)
    cb = Codebook.train(v, K=16, seed=42)
    dic = Dictionary.build(cb, corpus, v)
    # every active code maps to an actual corpus sentence
    for code, label in dic.labels.items():
        if label is not None:
            assert label in corpus
    p = tmp_path / "dic.json"
    dic.save(str(p))
    assert Dictionary.load(str(p)).labels == dic.labels


def test_cache_exact_match():
    c = SemanticCache()
    assert c.get(5) is None
    c.put(5, "call()")
    assert c.get(5) == "call()"
    assert c.serve_rate == 0.5   # one miss, one hit


def test_scorer_ast():
    gold = {"name": "get_weather", "args": {"city": ["Paris"]}}
    assert score_ast('get_weather(city="Paris")', gold) == (True, True)
    assert score_ast('get_weather(city="London")', gold) == (True, False)
    assert score_ast('process_payment(amount=50)', gold) == (False, False)
    name, kw = parse_call("Sure! get_weather(city='Paris')")
    assert name == "get_weather" and kw["city"] == "Paris"


def test_bfcl_loader():
    ex = bfcl.load(FIX, categories=["simple"])
    assert len(ex) == 4
    assert ex[0]["gold"]["name"] == "get_weather"
    assert "city" in ex[0]["gold"]["args"]
    assert ex[0]["message"].startswith("get the current weather")


def test_full_pipeline_vqt_vs_baseline():
    ex = bfcl.load(FIX, categories=["simple"])
    # replicate so cache hits happen
    ex = ex * 10
    enc = FakeEncoder(seed=2)
    cb = Codebook.train(enc.embed([e["message"] for e in ex]), K=8, seed=42)
    dic = Dictionary.build(cb, [e["message"] for e in ex],
                           enc.embed([e["message"] for e in ex]))

    def fake_llm(prompt, tools=None):
        p = prompt.lower()
        if "weather" in p:
            return 'get_weather(city="Paris")'
        if "payment" in p or "charge" in p:
            return "process_payment(amount=50)"
        return "unknown()"

    base = run(ex, enc, llm=fake_llm, use_vqt=False)
    vqt = run(ex, enc, codebook=cb, dictionary=dic,
              cache=SemanticCache(), llm=fake_llm, use_vqt=True)

    rb, rv = base.report(), vqt.report()
    # routing preserved
    assert rv["function_name_accuracy"] == rb["function_name_accuracy"]
    # cache actually skips LLM calls after first sighting of each code
    assert rv["cache_serve_rate"] > 0.5
    # paraphrases share codes without cross-intent collision here
    assert rv["collision_rate"] == 0.0
