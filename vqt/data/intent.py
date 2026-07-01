"""Loaders for intent-classification datasets used as control-plane
distributions: many surface forms mapping to a small recurring intent set.
Each returns (train, test) as lists of (utterance, intent_label)."""
from __future__ import annotations
import json
import csv


def load_clinc150(path: str, include_oos: bool = False):
    with open(path) as f:
        d = json.load(f)
    train = [(u, i) for u, i in d["train"]]
    test = [(u, i) for u, i in d["test"]]
    if include_oos:
        train += [(u, "oos") for u, i in d.get("oos_train", [])]
        test += [(u, "oos") for u, i in d.get("oos_test", [])]
    return train, test


def load_banking77(train_csv: str, test_csv: str):
    def _read(p):
        rows = []
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append((row["text"], row["category"]))
        return rows
    return _read(train_csv), _read(test_csv)
