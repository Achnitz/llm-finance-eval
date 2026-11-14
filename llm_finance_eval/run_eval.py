"""Run the benchmark against a model and write a scorecard.

Examples:
  python -m llm_finance_eval.run_eval --provider mock
  python -m llm_finance_eval.run_eval --provider anthropic --model claude-sonnet-5-5
  python -m llm_finance_eval.run_eval --provider openai --model gpt-5 --category deal_mechanics
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .grading import grade
from .providers import get_provider

ROOT = Path(__file__).resolve().parent.parent


def load_tasks(path: Path, category: str | None) -> list[dict]:
    if not path.exists():
        from .build_tasks import main as build_main
        build_main()
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [t for t in tasks if category is None or t["category"] == category]


def scorecard(rows: list[dict], tasks: list[dict], label: str) -> str:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    meta = {t["id"]: t for t in tasks}
    for r in rows:
        by_cat[meta[r["id"]]["category"]].append(r["correct"])
    total = sum(r["correct"] for r in rows)
    lines = [f"# Scorecard: {label}", "",
             f"**Overall: {total}/{len(rows)} ({total / len(rows):.0%})**", "",
             "| Category | Correct | Accuracy |", "|---|---|---|"]
    for cat, res in sorted(by_cat.items()):
        lines.append(f"| {cat} | {sum(res)}/{len(res)} | {sum(res) / len(res):.0%} |")
    misses = [r for r in rows if not r["correct"]]
    if misses:
        lines += ["", "## Misses", "", "| Task | Expected | Got |", "|---|---|---|"]
        for r in misses:
            lines.append(f"| {r['id']} | {meta[r['id']]['answer']:,.4f} | {r['parsed']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks), args.category)
    call = get_provider(args.provider, args.model, tasks)
    label = f"{args.provider}{':' + args.model if args.model else ''}"

    rows = []
    for t in tasks:
        response = call(t)
        result = grade(t, response)
        result["response"] = response
        rows.append(result)
        print(f"{t['id']:<8} {'OK ' if result['correct'] else 'X  '} parsed={result['parsed']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    safe = label.replace(":", "_").replace("/", "_")
    (out_dir / f"{safe}-{stamp}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    card = scorecard(rows, tasks, label)
    (out_dir / f"{safe}-{stamp}.md").write_text(card, encoding="utf-8")
    print("\n" + card)


if __name__ == "__main__":
    main()
