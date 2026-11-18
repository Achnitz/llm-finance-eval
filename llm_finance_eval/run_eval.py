"""Run the benchmark against one or more models and write scorecards.

Examples:
  python -m llm_finance_eval.run_eval --provider mock
  python -m llm_finance_eval.run_eval --provider anthropic --model claude-sonnet-5-5
  python -m llm_finance_eval.run_eval --provider openrouter --model openai/gpt-5,google/gemini-2.5-pro
  python -m llm_finance_eval.run_eval --provider openai --model gpt-5 --category deal_mechanics

After each run the leaderboard (results/LEADERBOARD.md) is rebuilt from every
scorecard in results/.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .grading import grade
from .leaderboard import build_leaderboard
from .providers import get_provider

ROOT = Path(__file__).resolve().parent.parent


def load_tasks(path: Path, category: str | None) -> list[dict]:
    from .build_tasks import OUT, main as build_main
    # Always rebuild the default task file, so a stale data/tasks.jsonl from an
    # older checkout never hides new tasks. Custom --tasks files are used as-is.
    if not path.exists() or path.resolve() == OUT.resolve():
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


def call_with_retry(call, task: dict, retries: int = 2) -> str:
    """Retry transient API errors; a persistent failure is scored as a miss."""
    for attempt in range(retries + 1):
        try:
            return call(task)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            if attempt == retries:
                return f"API_ERROR: {type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return ""


def run_model(provider: str, model: str | None, tasks: list[dict]) -> str:
    call = get_provider(provider, model, tasks)
    label = f"{provider}{':' + model if model else ''}"
    print(f"\n=== {label} ===")

    rows = []
    for t in tasks:
        response = call_with_retry(call, t)
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
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "anthropic", "openai", "openrouter"])
    ap.add_argument("--model", default=None,
                    help="one model or a comma-separated list, e.g. openai/gpt-5,google/gemini-2.5-pro")
    ap.add_argument("--category", default=None)
    ap.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks), args.category)
    models = [m.strip() for m in args.model.split(",")] if args.model else [None]
    for model in models:
        run_model(args.provider, model, tasks)

    if args.category is None:
        board = build_leaderboard(ROOT / "results", tasks)
        print(board)


if __name__ == "__main__":
    main()
