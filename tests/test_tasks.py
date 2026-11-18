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


def _trap01_price(debt_items=None, cash=100.0, nwc=239.0, peg=205.0, band=12.0) -> float:
    """Independent recomputation of trap-01; arguments reproduce each classic slip."""
    debt = sum(debt_items if debt_items is not None else (420, 9, 24, 16, 33, 50))
    gap = nwc - peg
    adj = 0.0 if abs(gap) <= band else gap - band if gap > 0 else gap + band
    return 1500 - debt + cash + adj


def test_trap01_answer_and_traps():
    t = next(x for x in build_v2() if x["id"] == "trap-01")
    assert t["category"] == "deal_traps"
    assert abs(t["answer"] - _trap01_price()) < 1e-6
    assert abs(t["answer"] - 1070) < 1e-6
    assert grade(t, "FINAL ANSWER: 1070")["correct"]
    slips = {
        "debt-like items left in NWC": _trap01_price(nwc=190.0),
        "band treated as a tipping basket": _trap01_price(band=0.0),
        "December used as the peg": _trap01_price(peg=176.0),
        "June used as the peg": _trap01_price(peg=221.0),
        "provision gross of tax": _trap01_price(debt_items=(420, 9, 24, 16, 50, 50)),
        "bonus missed": _trap01_price(debt_items=(420, 9, 24, 16, 33)),
        "payroll charges missed": _trap01_price(debt_items=(420, 9, 24, 16, 33, 40)),
        "pledged cash counted": _trap01_price(cash=130.0),
        "leases deducted": _trap01_price(debt_items=(420, 9, 24, 16, 33, 50, 85)),
    }
    for name, wrong in slips.items():
        assert not grade(t, f"FINAL ANSWER: {wrong}")["correct"], name
