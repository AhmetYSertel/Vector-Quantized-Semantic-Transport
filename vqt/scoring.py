"""Scoring. Mirrors BFCL's AST check at its core: parse the predicted
function call into (name, kwargs), then verify the name and that each
ground-truth argument's predicted value is among the acceptable values.

For the official leaderboard, swap `score_ast` for BFCL's own evaluator.
This implementation is self-contained so the repo runs without it.
"""
from __future__ import annotations
import ast
import re
from typing import Optional

_CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\(")


def parse_call(text: str):
    """Extract the first function call from model output -> (name, kwargs).

    Tolerates prose/markdown around the call. Positional args are captured
    under integer keys (0, 1, ...) as a fallback.
    """
    if not text:
        return None, {}
    m = _CALL_RE.search(text)
    if not m:
        return None, {}
    start = m.end() - 1                      # index of "("
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None, {}
    expr = text[m.start():end + 1]
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None, {}
    if not isinstance(node, ast.Call):
        return None, {}
    name = _dotted_name(node.func)
    kwargs = {}
    for j, arg in enumerate(node.args):
        kwargs[j] = _safe_literal(arg)
    for kw in node.keywords:
        if kw.arg is not None:
            kwargs[kw.arg] = _safe_literal(kw.value)
    return name, kwargs


def _dotted_name(func) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _dotted_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return None


def _safe_literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _values_match(pred, acceptable) -> bool:
    """acceptable is a list of allowed values (BFCL ground_truth arg list)."""
    if not isinstance(acceptable, (list, tuple)):
        acceptable = [acceptable]
    for a in acceptable:
        if pred == a:
            return True
        if str(pred).strip().lower() == str(a).strip().lower():
            return True
    return False


def score_ast(pred_text: str, gold: dict):
    """gold = {"name": str, "args": {arg: [acceptable, ...]}}.
    Returns (fn_name_correct: bool, full_call_correct: bool)."""
    name, kwargs = parse_call(pred_text)
    if name is None:
        return False, False
    name_ok = (name == gold["name"]) or name.endswith("." + gold["name"])
    if not name_ok:
        return False, False
    for arg, acceptable in gold.get("args", {}).items():
        if arg not in kwargs or not _values_match(kwargs[arg], acceptable):
            return True, False       # right tool, wrong/missing args
    return True, True
