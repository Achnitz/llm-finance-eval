from llm_finance_eval.build_tasks import build, build_v2
from llm_finance_eval.grading import grade


def test_task_schema_and_unique_ids():
    tasks = build()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for t in tasks:
        assert {"id", "category", "difficulty", "prompt", "answer", "unit",
                "rel_tolerance", "solution"} <= set(t)
        assert "FINAL ANSWER" in t["prompt"]


def test_v2_tasks_are_hard_and_explained():
    for t in build_v2():
        assert t["difficulty"] == "hard"
        assert len(t["solution"]) > 80  # must say where the trap is


def _ms01_reference() -> float:
    """Independent recomputation of ms-01, written year by year."""
    revs = [1080.0]
    for g in (0.07, 0.06, 0.05, 0.04):
        revs.append(revs[-1] * (1 + g))
    margins = (0.22, 0.23, 0.24, 0.25, 0.25)
    nwc_prev, pv = 150.0, 0.0
    for year, (r, m) in enumerate(zip(revs, margins), start=1):
        nopat = (m - 0.04) * r * 0.75
        fcff = nopat + 0.04 * r - 0.06 * r - (0.12 * r - nwc_prev)
        nwc_prev = 0.12 * r
        pv += fcff / 1.095 ** (year - 0.5)
    pv += fcff * 1.03 / (0.095 - 0.03) / 1.095 ** 5
    net_debt = 900 - 210 + 120
    return (pv - net_debt - 80 + 50) / 100


def test_ms01_answer_and_traps():
    t = next(x for x in build_v2() if x["id"] == "ms-01")
    ref = _ms01_reference()
    assert abs(t["answer"] - ref) < 1e-4
    assert grade(t, f"FINAL ANSWER: {ref:.2f}")["correct"]
    # Each classic slip must land outside the tolerance.
    for wrong in (15.36, 16.48, 16.05, 16.85, 15.37):
        assert not grade(t, f"FINAL ANSWER: {wrong}")["correct"]
