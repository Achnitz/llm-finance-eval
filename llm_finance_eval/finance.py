"""Small, dependency-free finance helpers used to compute the answer key."""
from __future__ import annotations


def npv(rate: float, cashflows: list[float]) -> float:
    """NPV with the first cash flow at t=0."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def pv_of_flows(flows: list[float], rate: float) -> float:
    """PV of end-of-year flows starting at t=1."""
    return sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(flows))


def irr(cashflows: list[float], lo: float = -0.99, hi: float = 10.0, tol: float = 1e-10) -> float:
    """IRR by bisection. Assumes a single sign change (conventional project)."""
    f_lo = npv(lo, cashflows)
    for _ in range(500):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def annuity_payment(pv: float, rate: float, n: int) -> float:
    """Level end-of-period payment that amortizes pv over n periods."""
    return pv * rate / (1 - (1 + rate) ** -n)
