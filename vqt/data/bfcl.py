"""Load BFCL v3 (simple / multiple) into the control-plane example schema.

BFCL native layout:
    data_dir/BFCL_v3_<cat>.json                 (questions, JSON-lines)
    data_dir/possible_answer/BFCL_v3_<cat>.json (ground truth, JSON-lines)

Questions line:  {"id", "question": [[{role, content}, ...]], "function": [ {name,...} ]}
Answer line:     {"id", "ground_truth": [ {fn_name: {arg: [acceptable, ...]}} ]}

We treat each user query as an inter-agent control message and its
ground-truth call as the routing/handoff target.
"""
from __future__ import annotations
import json
import os

CATEGORIES = ("simple", "multiple")


def _read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_user_content(question) -> str:
    """question is [[{role, content}, ...], ...]; take first user turn."""
    try:
        turns = question[0]
        for t in turns:
            if t.get("role") == "user":
                return t["content"]
        return turns[0]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _gold_from_ground_truth(gt) -> dict:
    """gt = [{fn_name: {arg: [values]}}]; take the first call."""
    if not gt:
        return {"name": None, "args": {}}
    call = gt[0]
    name = next(iter(call))
    return {"name": name, "args": call[name]}


def load_category(data_dir: str, category: str) -> list:
    q_path = os.path.join(data_dir, f"BFCL_v3_{category}.json")
    a_path = os.path.join(data_dir, "possible_answer", f"BFCL_v3_{category}.json")
    questions = {r["id"]: r for r in _read_jsonl(q_path)}
    answers = {r["id"]: r for r in _read_jsonl(a_path)}
    examples = []
    for _id, q in questions.items():
        if _id not in answers:
            continue
        gold = _gold_from_ground_truth(answers[_id].get("ground_truth", []))
        examples.append({
            "id": _id,
            "message": _first_user_content(q.get("question", [])),
            "tools": q.get("function", []),
            "gold": gold,
        })
    return examples


def load(data_dir: str, categories=CATEGORIES) -> list:
    out = []
    for cat in categories:
        out.extend(load_category(data_dir, cat))
    return out
