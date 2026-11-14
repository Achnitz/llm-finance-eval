from llm_finance_eval.grading import extract_final, grade


def test_extract_formats():
    assert extract_final("... FINAL ANSWER: 1234.5") == 1234.5
    assert extract_final("FINAL ANSWER: 1,234.5") == 1234.5
    assert extract_final("FINAL ANSWER: 1.234,5") == 1234.5
    assert extract_final("FINAL ANSWER: 12,5%") == 12.5
    assert extract_final("final answer = -3.2") == -3.2
    assert extract_final("no marker here 42") is None


def test_uses_last_marker():
    assert extract_final("FINAL ANSWER: 1\nwait, correction\nFINAL ANSWER: 2") == 2


def test_grade_tolerance_and_zero():
    t = {"id": "x", "answer": 100.0, "rel_tolerance": 0.005}
    assert grade(t, "FINAL ANSWER: 100.4")["correct"]
    assert not grade(t, "FINAL ANSWER: 101")["correct"]
    z = {"id": "z", "answer": 0.0, "rel_tolerance": 0.005}
    assert grade(z, "FINAL ANSWER: 0")["correct"]
    assert not grade(z, "FINAL ANSWER: 5")["correct"]
