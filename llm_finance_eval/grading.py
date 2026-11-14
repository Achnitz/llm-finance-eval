"""Deterministic grading: pull the final number out of a model response and
compare it with the reference answer within a relative tolerance."""
from __future__ import annotations

import re

_FINAL = re.compile(r"FINAL ANSWER\s*[:=]\s*\$?\s*(-?[\d.,]+)\s*%?", re.IGNORECASE)


def extract_final(text: str) -> float | None:
    """Return the number after the last 'FINAL ANSWER:' marker, or None."""
    matches = _FINAL.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].rstrip(".")
    # Accept "1,234.5" and "1.234,5": the last separator is the decimal one.
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = raw.replace(",", "") if len(tail) == 3 and head else raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def grade(task: dict, response: str) -> dict:
    """Grade one response. Returns a dict with the parsed value and verdict."""
    value = extract_final(response)
    answer = float(task["answer"])
    if value is None:
        return {"id": task["id"], "parsed": None, "correct": False, "error": "no_final_answer"}
    denom = max(abs(answer), 1e-9)
    rel_err = abs(value - answer) / denom
    # Zero answers (e.g. "no impairment") need an absolute check.
    correct = abs(value) < 1e-6 if answer == 0 else rel_err <= task["rel_tolerance"]
    return {"id": task["id"], "parsed": value, "correct": correct, "rel_error": rel_err}
