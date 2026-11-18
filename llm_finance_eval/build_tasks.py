"""Build the benchmark task set.

Every task is generated from explicit inputs and its reference answer is
computed here, in code. That keeps the answer key auditable: if a number
looks wrong, the formula that produced it is a few lines below.

Run:  python -m llm_finance_eval.build_tasks  ->  data/tasks.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from .finance import annuity_payment, irr, npv, pv_of_flows

OUT = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"

FINAL_INSTRUCTION = (
    "Show your reasoning, then end with a single line in the format "
    "'FINAL ANSWER: <number>' (no units, no thousands separators)."
)


def task(tid, category, difficulty, prompt, answer, unit, tol=0.005, solution=""):
    return {
        "id": tid,
        "category": category,
        "difficulty": difficulty,
        "prompt": f"{prompt}\n\n{FINAL_INSTRUCTION}",
        "answer": round(answer, 6),
        "unit": unit,
        "rel_tolerance": tol,
        "solution": solution,
    }


def build() -> list[dict]:
    t: list[dict] = []

    # ---------------- Cost of capital ----------------
    ke = 0.045 + 1.1 * 0.055 + 0.025
    t.append(task(
        "coc-01", "cost_of_capital", "easy",
        "Estimate the cost of equity for an emerging-market utility using CAPM with a "
        "country risk premium. Risk-free rate 4.5%, levered beta 1.1, equity risk premium "
        "5.5%, country risk premium 2.5% (added to the CAPM result). Answer in percent.",
        ke * 100, "%", solution="ke = rf + beta*ERP + CRP",
    ))

    E, D, k_e, k_d, tax = 600, 400, 0.14, 0.10, 0.34
    wacc = E / (E + D) * k_e + D / (E + D) * k_d * (1 - tax)
    t.append(task(
        "coc-02", "cost_of_capital", "easy",
        f"A company has market value of equity of {E} and debt of {D} (millions). Cost of "
        f"equity is 14%, pre-tax cost of debt is 10% and the marginal tax rate is 34%. "
        "Compute the WACC in percent.",
        wacc * 100, "%", solution="WACC = E/V*ke + D/V*kd*(1-t)",
    ))

    bl, de, tx, de_new = 1.2, 0.5, 0.25, 1.0
    bu = bl / (1 + (1 - tx) * de)
    bl_new = bu * (1 + (1 - tx) * de_new)
    t.append(task(
        "coc-03", "cost_of_capital", "medium",
        "A peer has a levered beta of 1.20, a debt-to-equity ratio of 0.50 and a 25% tax rate. "
        "Unlever the beta with the Hamada formula and relever it for a target capital "
        "structure with debt-to-equity of 1.00 (same tax rate). Report the relevered beta.",
        bl_new, "x", solution="bu = bl/(1+(1-t)D/E); bl' = bu*(1+(1-t)D'/E')",
    ))

    real, infl = 0.06, 0.04
    nominal = (1 + real) * (1 + infl) - 1
    t.append(task(
        "coc-04", "cost_of_capital", "easy",
        "A regulator sets a real WACC of 6.0%. Expected inflation is 4.0%. Using the exact "
        "Fisher relation, what is the nominal WACC in percent?",
        nominal * 100, "%", solution="(1+r)(1+i)-1",
    ))

    # ---------------- Valuation ----------------
    fcf1, w, g = 120, 0.10, 0.03
    t.append(task(
        "val-01", "valuation", "easy",
        "Next year's free cash flow to the firm is 120. WACC is 10% and the perpetual "
        "growth rate is 3%. What is the value of the firm today (Gordon growth)?",
        fcf1 / (w - g), "currency", solution="V = FCF1/(WACC-g)",
    ))

    flows, w, g = [100, 110, 120], 0.09, 0.02
    tv = flows[-1] * (1 + g) / (w - g)
    ev = pv_of_flows(flows, w) + tv / (1 + w) ** 3
    t.append(task(
        "val-02", "valuation", "medium",
        "Explicit-period FCFF forecasts are 100, 110 and 120 for years 1 to 3 (end-of-year). "
        "After year 3, cash flows grow at 2% forever. WACC is 9%. Compute enterprise value today.",
        ev, "currency", solution="PV(explicit) + PV(TV at year 3 = FCF3*(1+g)/(WACC-g))",
    ))

    ev_, nd, mi, inv, sh = 2500, 800, 120, 60, 100
    eq_ps = (ev_ - nd - mi + inv) / sh
    t.append(task(
        "val-03", "valuation", "easy",
        "Enterprise value is 2,500 (millions). Net debt is 800, non-controlling interests "
        "are 120 and non-consolidated investments are worth 60. There are 100 million shares. "
        "What is the equity value per share?",
        eq_ps, "currency/share", solution="(EV - net debt - NCI + investments)/shares",
    ))

    peers, ebitda, nd2 = [7.5, 8.2, 9.0, 6.8, 8.6], 210, 600
    med = sorted(peers)[len(peers) // 2]
    t.append(task(
        "val-04", "valuation", "easy",
        f"Peer EV/EBITDA multiples are {peers}. Apply the median to a target with EBITDA of "
        f"{ebitda} and net debt of {nd2}. What is the implied equity value?",
        med * ebitda - nd2, "currency", solution="median multiple * EBITDA - net debt",
    ))

    syn, gs, ws, cost = 30, 0.02, 0.10, 50
    t.append(task(
        "val-05", "valuation", "medium",
        "Run-rate cost synergies are 30 per year starting next year, growing at 2% a year "
        "forever. The discount rate is 10%. One-off integration costs of 50 are paid today. "
        "What is the net present value of the synergies?",
        syn / (ws - gs) - cost, "currency", solution="30/(0.10-0.02) - 50",
    ))

    debt, tx2 = 500, 0.34
    t.append(task(
        "val-06", "valuation", "easy",
        "A firm keeps permanent debt of 500 at a fixed amount. The tax rate is 34% and the "
        "tax shield has the same risk as the debt. What is the present value of the interest "
        "tax shield?",
        debt * tx2, "currency", solution="PV(TS) = t*D for perpetual fixed debt",
    ))

    # ---------------- Capital budgeting ----------------
    cf = [-1000] + [300] * 5
    t.append(task(
        "cb-01", "capital_budgeting", "easy",
        "A project costs 1,000 today and generates 300 at the end of each of the next five "
        "years. The discount rate is 8%. What is the NPV?",
        npv(0.08, cf), "currency", solution="sum CF_t/(1.08)^t",
    ))

    cf2 = [-1000, 250, 300, 350, 400, 450]
    t.append(task(
        "cb-02", "capital_budgeting", "medium",
        "Cash flows are -1,000 today, then 250, 300, 350, 400 and 450 at the end of years 1 "
        "to 5. What is the IRR in percent?",
        irr(cf2) * 100, "%", solution="rate where NPV = 0",
    ))

    t.append(task(
        "cb-03", "capital_budgeting", "medium",
        "A regulated asset requires capex of 1,000 and the allowed revenue is set as a level "
        "annual payment that recovers the investment over 30 years at an 8% allowed return "
        "(end-of-year payments, ignore O&M and taxes). What is the annual allowed revenue?",
        annuity_payment(1000, 0.08, 30), "currency", solution="capital recovery factor",
    ))

    # ---------------- Deal mechanics ----------------
    hp, nd3, nwc_t, nwc_a = 1000, 210, 150, 132
    t.append(task(
        "deal-01", "deal_mechanics", "medium",
        "An SPA uses completion accounts on a cash-free/debt-free basis. Headline enterprise "
        "value is 1,000. Net debt at closing is 210. The agreed normalized working capital "
        "target is 150 and actual closing working capital is 132 (dollar-for-dollar "
        "adjustment, no collar). What equity purchase price does the buyer pay?",
        hp - nd3 + (nwc_a - nwc_t), "currency",
        solution="EV - net debt + (actual NWC - target NWC)",
    ))

    t.append(task(
        "deal-02", "deal_mechanics", "easy",
        "Under a locked-box mechanism the agreed equity price is 900. Between the locked-box "
        "date and closing the seller received a dividend of 10 and management fees of 5, both "
        "classified as leakage. No permitted leakage applies. What should the buyer pay?",
        900 - 15, "currency", solution="price - leakage",
    ))

    probs, pays, years, rate = [0.3, 0.5, 0.2], [0, 100, 250], 2, 0.12
    ev_earn = sum(p * x for p, x in zip(probs, pays)) / (1 + rate) ** years
    t.append(task(
        "deal-03", "deal_mechanics", "medium",
        "An earn-out pays 0 (probability 30%), 100 (probability 50%) or 250 (probability "
        "20%) at the end of year 2. Discounting at 12%, what is its present value?",
        ev_earn, "currency", solution="E[payout]/(1.12)^2",
    ))

    ni_a, sh_a, ni_t, price, px, syn_at = 500, 250, 80, 1200, 20, 20
    new_sh = sh_a + price / px
    eps0, eps1 = ni_a / sh_a, (ni_a + ni_t + syn_at) / new_sh
    t.append(task(
        "deal-04", "deal_mechanics", "hard",
        "Acquirer net income 500, 250 million shares. Target net income 80. The acquirer pays "
        "1,200 entirely in new shares issued at 20 per share. After-tax synergies are 20 in "
        "year one. What is the year-one EPS accretion (+) or dilution (-) in percent?",
        (eps1 / eps0 - 1) * 100, "%", tol=0.01,
        solution="EPS_pro forma = (500+80+20)/(250+60) vs 2.00",
    ))

    eq_in, eq_out, n = 400, 1000, 5
    t.append(task(
        "deal-05", "deal_mechanics", "easy",
        "A sponsor invests 400 of equity and exits after 5 years receiving 1,000, with no "
        "interim distributions. What is the IRR in percent?",
        ((eq_out / eq_in) ** (1 / n) - 1) * 100, "%", solution="MOIC^(1/n)-1",
    ))

    # ---------------- Accounting / impairment ----------------
    cfs, r_imp, carrying = [180, 190, 200, 210, 220], 0.09, 950
    viu = pv_of_flows(cfs, r_imp)
    t.append(task(
        "acc-01", "accounting", "medium",
        "Under IAS 36, a cash-generating unit has a carrying amount of 950. Its projected "
        "pre-tax cash flows are 180, 190, 200, 210 and 220 for years 1-5, after which the "
        "asset is retired with no residual value. The pre-tax discount rate is 9%. Fair value "
        "less costs of disposal is 700. What impairment loss should be recognised? "
        "(Answer 0 if none.)",
        max(0.0, carrying - max(viu, 700)), "currency",
        solution="recoverable = max(VIU, FVLCD); loss = carrying - recoverable",
    ))

    # ---------------- Project finance / credit ----------------
    cfads, dscr, rpf, yrs = 120, 1.30, 0.09, 10
    t.append(task(
        "pf-01", "project_finance", "hard",
        "A project generates CFADS of 120 per year for 10 years (end-of-year). Lenders require "
        "a minimum DSCR of 1.30 in every year and the interest rate is 9%. With debt service "
        "sculpted to exactly CFADS/1.30 each year, what is the maximum debt size?",
        pv_of_flows([cfads / dscr] * yrs, rpf), "currency",
        solution="Debt = PV(CFADS/DSCR) at the debt rate",
    ))

    t.append(task(
        "pf-02", "project_finance", "easy",
        "CFADS for the year is 150 and scheduled debt service (interest + principal) is 110. "
        "What is the DSCR?",
        150 / 110, "x", solution="CFADS / debt service",
    ))

    eb, cov, nd4 = 300, 3.5, 800
    t.append(task(
        "pf-03", "project_finance", "easy",
        "A borrower has EBITDA of 300 and net debt of 800. The covenant caps net debt/EBITDA "
        "at 3.5x. How much additional net debt can it raise before breaching the covenant?",
        eb * cov - nd4, "currency", solution="3.5*300 - 800",
    ))

    t += build_v2()
    return t


def build_v2() -> list[dict]:
    """v2: hard tasks. Multi-step chains, hidden deal traps and inputs where the
    right move is to flag the problem. v1 (above) is saturated by frontier models."""
    t: list[dict] = []

    # ---------------- Multi-step chains ----------------
    # ms-01: revenue -> margins -> FCFF -> mid-year DCF -> EV-to-equity bridge -> per share
    rev0, growth = 1000.0, [0.08, 0.07, 0.06, 0.05, 0.04]
    margins = [0.22, 0.23, 0.24, 0.25, 0.25]
    da_pct, capex_pct, nwc_pct, tax_r, nwc0 = 0.04, 0.06, 0.12, 0.25, 150.0
    wacc_ms, g_ms = 0.095, 0.03
    revs, rev = [], rev0
    for gr in growth:
        rev *= 1 + gr
        revs.append(rev)
    fcff, prev_nwc = [], nwc0
    for r_, m_ in zip(revs, margins):
        ebit = r_ * m_ - r_ * da_pct
        d_nwc = nwc_pct * r_ - prev_nwc
        fcff.append(ebit * (1 - tax_r) + r_ * da_pct - r_ * capex_pct - d_nwc)
        prev_nwc = nwc_pct * r_
    pv_explicit = sum(f / (1 + wacc_ms) ** (i + 0.5) for i, f in enumerate(fcff))
    tv_ms = fcff[-1] * (1 + g_ms) / (wacc_ms - g_ms)
    ev_ms = pv_explicit + tv_ms / (1 + wacc_ms) ** 5
    debt_ms, cash_ms, trapped, lease_ms, nci_ms, assoc_ms, sh_ms = 900, 250, 40, 120, 80, 50, 100
    net_debt_ms = debt_ms - (cash_ms - trapped) + lease_ms
    eq_ms = ev_ms - net_debt_ms - nci_ms + assoc_ms
    t.append(task(
        "ms-01", "multi_step", "hard",
        "You are valuing a mid-cap industrial company as of 31 December 2026 with a five-year "
        "DCF (2027-2031).\n"
        "- 2026 revenue was 1,000 (millions). Revenue grows 8%, 7%, 6%, 5% and 4% in 2027 to 2031.\n"
        "- EBITDA margin is 22%, 23%, 24%, 25% and 25% in 2027 to 2031. EBITDA is reported "
        "under IFRS 16, so lease payments are not in operating costs.\n"
        "- D&A is 4% of revenue and capex is 6% of revenue every year (capex excludes "
        "right-of-use asset additions).\n"
        "- Net working capital was 150 at end-2026 (inflated by a one-off customer payment "
        "delay). From 2027 it is held at 12% of same-year revenue.\n"
        "- Tax rate on EBIT is 25%.\n"
        "- WACC is 9.5%. Discount the explicit-year FCFF with the mid-year convention (year 1 "
        "at 0.5 years, year 2 at 1.5 years, and so on).\n"
        "- Terminal value: Gordon growth at end-2031, using 2031 FCFF grown by 3.0% nominal. "
        "Discount the terminal value 5.0 years.\n"
        "Balance sheet at 31 December 2026: gross financial debt 900; cash 250, of which 40 is "
        "restricted cash in a foreign subsidiary that cannot be upstreamed (only freely "
        "available cash reduces net debt); IFRS 16 lease liabilities 120; non-controlling "
        "interests at fair value 80; a 30% stake in an associate worth 50; 100 million shares "
        "outstanding, no dilutive instruments.\n"
        "What is the equity value per share?",
        eq_ms / sh_ms, "currency/share",
        solution=(
            "FCFF = EBIT*(1-t) + D&A - capex - dNWC. Year-1 dNWC = 12%*1,080 - 150 = -20.4 "
            "(a cash release, because the opening NWC is the actual 150, not 12% of 2026 "
            "revenue); later years = 12% of the revenue change. "
            "Explicit flows at t-0.5, TV = FCFF2031*1.03/(0.095-0.03) at 5.0. Bridge: "
            "net debt = 900 - (250-40) + 120 leases (EBITDA is post-IFRS 16, so leases are "
            "debt-like); equity = EV - net debt - NCI + associate. Traps: missing the year-1 "
            "NWC release (using 120 as opening NWC), mid-year on the TV, counting the restricted "
            "cash, dropping the leases."
        ),
    ))

    # ---------------- Hidden deal traps ----------------
    # trap-01: SPA definitions decide what is debt, what is NWC and how the peg works
    ev_spa, loan, accrued_int, div_payable, tax_payable = 1500.0, 420.0, 9.0, 24.0, 16.0
    cash_total, cash_pledged, env_prov, env_tax = 130.0, 30.0, 50.0, 0.34
    bonus, bonus_charges = 40.0, 0.25
    reported_nwc, collar = 190.0, 12.0
    peg_months = [188, 196, 214, 228, 236, 221, 203, 194, 187, 192, 225, 176]
    indebtedness = (loan + accrued_int + div_payable + tax_payable
                    + env_prov * (1 - env_tax) + bonus * (1 + bonus_charges))
    free_cash = cash_total - cash_pledged
    closing_nwc = reported_nwc + accrued_int + div_payable + tax_payable
    peg = sum(peg_months) / len(peg_months)
    gap = closing_nwc - peg
    nwc_adj = max(abs(gap) - collar, 0.0) * (1 if gap > 0 else -1)
    t.append(task(
        "trap-01", "deal_traps", "hard",
        "You are on the buy-side deal team checking the completion statement for the "
        "acquisition of 100% of a company. Completion date is 30 June 2026. The agreed "
        "enterprise value is 1,500 (millions), set on a pre-IFRS 16 EBITDA multiple.\n"
        "SPA definitions:\n"
        "- Purchase price = enterprise value - Indebtedness + Free Cash + NWC Adjustment.\n"
        "- Indebtedness means bank debt including accrued interest; dividends declared but "
        "unpaid; income tax payable; environmental remediation provisions, net of the tax "
        "shield at 34%; and transaction bonuses triggered by the deal, including employer "
        "payroll charges, with no tax effect. IFRS 16 lease liabilities are excluded.\n"
        "- Free Cash means cash at bank excluding any cash that is pledged or restricted.\n"
        "- Closing NWC excludes every item that counts as Indebtedness or cash.\n"
        "- Target NWC is the simple average of the twelve month-end NWC balances for 2025, "
        "already stated on the SPA definition.\n"
        "- NWC Adjustment: no adjustment if Closing NWC is within 12 of Target NWC. Outside "
        "that band, the price moves only by the amount beyond the band, in the direction of "
        "the difference.\n"
        "Month-end NWC for 2025, January to December: 188, 196, 214, 228, 236, 221, 203, 194, "
        "187, 192, 225, 176.\n"
        "Seller's completion balance sheet at 30 June 2026:\n"
        "- Bank loan principal 420.\n"
        "- Cash at bank 130, including a 30 deposit pledged against a bank guarantee.\n"
        "- Reported net working capital 190. Its other current liabilities line includes "
        "accrued interest on the bank loan of 9, a dividend declared in May and not yet paid "
        "of 24, and income tax payable of 16.\n"
        "- Non-current environmental remediation provision 50 (deductible when paid).\n"
        "- IFRS 16 lease liabilities 85, shown separately from NWC.\n"
        "Disclosure letter, item 14: retention bonuses of 40 become payable to management on "
        "completion. They are not accrued in the balance sheet. Employer payroll charges on "
        "them are 25%.\n"
        "What is the purchase price payable for the shares?",
        ev_spa - indebtedness + free_cash + nwc_adj, "currency",
        solution=(
            "Indebtedness = 420 + 9 + 24 + 16 + 50*(1-34%) + 40*1.25 = 552. Free Cash = 130 - 30 "
            "= 100. Closing NWC = 190 + 9 + 24 + 16 = 239, because the three debt-like items "
            "must leave reported NWC. Target = average of the 12 months = 205. Gap = +34, and "
            "the band works as a deductible, so the adjustment is +22. Price = 1,500 - 552 + "
            "100 + 22 = 1,070. Traps: leaving the reclassified items in NWC, paying the full 34 "
            "instead of the excess over the band, using December or June as the peg, taking the "
            "provision gross of tax, missing the unaccrued bonus or its payroll charges, "
            "counting the pledged cash, deducting the leases."
        ),
    ))

    return t


def main() -> None:
    tasks = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for item in tasks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {OUT}")


if __name__ == "__main__":
    main()
