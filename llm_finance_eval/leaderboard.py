"""Build results/LEADERBOARD.md from the raw run files in results/*.jsonl.

For each model, only the most recent full run (all tasks of the current set)
is used. Mock runs are ignored.

Run:  python -m llm_finance_eval.leaderboard
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_STAMP = re.compile(r"^(?P<label>.+)-(?P<stamp>\d{8}-\d{6})\.jsonl$")


def _latest_runs(results_dir: Path, task_ids: set[str]) -> dict[str, list[dict]]:
    latest: dict[str, tuple[str, list[dict]]] = {}
    for path in results_dir.glob("*.jsonl"):
        m = _STAMP.match(path.name)
        if not m or m["label"].startswith("mock"):
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if {r["id"] for r in rows} != task_ids:
            continue  # partial run (e.g. one category) or an older task set
        if m["label"] not in latest or m["stamp"] > latest[m["label"]][0]:
            latest[m["label"]] = (m["stamp"], rows)
    return {label: rows for label, (_, rows) in latest.items()}


def build_leaderboard(results_dir: Path, tasks: list[dict]) -> str:
    meta = {t["id"]: t for t in tasks}
    cats = sorted({t["category"] for t in tasks})
    runs = _latest_runs(results_dir, set(meta))
    if not runs:
        return ""

    table = []
    for label, rows in runs.items():
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            by_cat[meta[r["id"]]["category"]].append(r["correct"])
        total = sum(r["correct"] for r in rows)
        model = label.split("_", 1)[-1]
        table.append((total, model, {c: (sum(by_cat[c]), len(by_cat[c])) for c in cats}))
    table.sort(key=lambda x: (-x[0], x[1]))

    head = "| Rank | Model | Overall | " + " | ".join(cats) + " |"
    sep = "|---|---|---|" + "---|" * len(cats)
    lines = [f"# Leaderboard ({len(meta)} tasks)", "", head, sep]
    for i, (total, model, by_cat) in enumerate(table, 1):
        cells = " | ".join(f"{a}/{b}" for a, b in by_cat.values())
        lines.append(f"| {i} | {model} | **{total}/{len(meta)}** | {cells} |")
    board = "\n".join(lines) + "\n"
    (results_dir / "LEADERBOARD.md").write_text(board, encoding="utf-8")
    return board


def main() -> None:
    from .run_eval import load_tasks
    tasks = load_tasks(ROOT / "data" / "tasks.jsonl", None)
    print(build_leaderboard(ROOT / "results", tasks) or "no full runs found in results/")


if __name__ == "__main__":
    main()
