# llm-finance-eval

**A small, auditable benchmark for testing how well LLMs handle corporate finance and M&A work: cost of capital, valuation, deal mechanics, impairment and project finance.**

Most finance benchmarks for LLMs test textbook definitions or stock-market trivia. Practitioners care about something else. Does the model get the EV-to-equity bridge right? Does it apply a completion-accounts working capital adjustment in the right direction? Does it pick the right recoverable amount under IAS 36? Can it size sculpted project finance debt off a DSCR? This repo turns that kind of everyday deal-team work into graded tasks.

Built by an M&A practitioner, covering both sell-side and buy-side transactions in energy and infrastructure.

## What's inside

| Category | Tasks | Examples |
|---|---|---|
| Cost of capital | 4 | CAPM with country risk premium, WACC, Hamada unlever/relever, Fisher real→nominal |
| Valuation | 6 | Gordon growth, 3-year DCF + terminal value, EV-to-equity bridge, peer multiples, synergy NPV, tax shield |
| Capital budgeting | 3 | NPV, IRR, capital recovery (regulated revenue) |
| Deal mechanics | 5 | Completion accounts, locked-box leakage, earn-out PV, EPS accretion/dilution, sponsor IRR |
| Accounting | 1 | IAS 36 impairment (VIU vs FVLCD) |
| Project finance | 3 | Sculpted debt sizing, DSCR, leverage covenant headroom |
| Multi-step chains (v2, hard) | 1 | Revenue → margins → FCFF (capex, D&A, NWC) → mid-year DCF → EV-to-equity bridge → value per share |
| Deal traps (v2, hard) | 1 | Completion statement under SPA definitions: debt-like items, NWC peg and collar |
| LBO (v2, hard) | 1 | Sponsor IRR with cash sweep, year-3 dividend recap, exit costs and a management incentive plan |

**25 tasks** in total, each with a deterministic numeric answer: the 22-task v1 set plus 3 hard v2 tasks (v2 is in progress, see below).

## Design principles

1. **The answer key is code, not a spreadsheet.** Every reference answer is computed in [`build_tasks.py`](llm_finance_eval/build_tasks.py) from the task inputs, so anyone can audit it. No hand-typed numbers.
2. **Deterministic grading.** The model must end its answer with `FINAL ANSWER: <number>`. The grader parses it (it handles `1,234.5`, `1.234,5` and `%`) and checks it against a relative tolerance, 0.5% by default.
3. **Practitioner framing.** The prompts read like the requests a deal team gets, not exam questions. Several contain a trap: the sign of the NWC adjustment, NCI in the bridge, or a fair value that is below VIU.
4. **Zero dependencies to run the core.** Model SDKs are only needed to call a model.

## Results (v1, November 2026)

These scores are on the **22-task v1 set**. The hard v2 tasks added since then have not been run yet.

| Model | Overall | Cost of capital | Valuation | Capital budgeting | Deal mechanics | Accounting | Project finance |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | **22/22** | 4/4 | 6/6 | 3/3 | 5/5 | 1/1 | 3/3 |
| Claude Sonnet 5.5 | **22/22** | 4/4 | 6/6 | 3/3 | 5/5 | 1/1 | 3/3 |

Raw scorecards are in [`results/`](results/).

**What this tells us:** frontier models no longer miss single-step corporate finance calculations, traps included. The largest relative error across both runs was 0.27%, inside tolerance. The v1 set is saturated, which is a finding in itself. The next version targets where models still fail in real deal work:

- **Multi-step chains:** projections → FCFF → DCF → EV-to-equity bridge in one prompt, where one early slip carries through to the answer
- **Hidden deal traps:** debt-like items buried in a footnote, NWC peg definitions, capped earn-outs, real vs nominal growth mismatches
- **Incomplete or contradictory inputs:** the right answer is to flag the problem, not to compute
- **Model review:** spotting planted errors in a valuation model (see the upcoming `dcf-red-team`)

## v2: hard tasks (in progress)

v2 tasks live in `build_v2()` in [`build_tasks.py`](llm_finance_eval/build_tasks.py), all with `difficulty: "hard"` and a `solution` that names the trap. Each trap is checked in the tests to move the answer outside the tolerance, so a model cannot pass by accident.

| ID | Category | What it tests | Where it breaks |
|---|---|---|---|
| ms-01 | multi_step | Five-year FCFF build, mid-year DCF, Gordon TV, EV-to-equity bridge, per share | Opening NWC above target (year-1 cash release), TV discounted at 5.0 not 4.5, restricted cash, IFRS 16 leases as debt |
| trap-01 | deal_traps | Purchase price from a completion statement, applying the SPA definitions of Indebtedness, Free Cash and NWC | Debt-like items sitting inside reported NWC, collar that works as a deductible, 12-month average peg, provision net of tax, unaccrued bonus plus payroll charges in the disclosure letter, pledged cash, leases excluded |
| lbo-01 | lbo | Sponsor IRR through a five-year hold: interest on opening debt, tax after interest, 100% cash sweep, dividend recap in year 3, exit at a lower multiple | Fees left out of the equity cheque, recap arrangement fee, dividend timed at exit instead of year 3, exit costs, 10% MIP on exit equity only, tax before interest |

## Quickstart

```bash
git clone https://github.com/Achnitz/llm-finance-eval.git
cd llm-finance-eval

# build the task file from source (also done automatically on the first run)
python -m llm_finance_eval.build_tasks

# sanity check the pipeline offline
python -m llm_finance_eval.run_eval --provider mock

# run against a real model
pip install anthropic            # or: pip install openai
export ANTHROPIC_API_KEY=...
python -m llm_finance_eval.run_eval --provider anthropic --model claude-sonnet-5-5

# many vendors with one key (OpenRouter), several models in one go
pip install openai
export OPENROUTER_API_KEY=...
python -m llm_finance_eval.run_eval --provider openrouter --model openai/gpt-5,google/gemini-2.5-pro

# one category only
python -m llm_finance_eval.run_eval --provider openai --model gpt-5 --category deal_mechanics
```

Each run writes the raw responses (`results/*.jsonl`) and a scorecard (`results/*.md`) with accuracy by category and a list of misses. Full runs also rebuild [`results/LEADERBOARD.md`](results/LEADERBOARD.md), which ranks the latest run of every model. Model names for OpenRouter are listed at openrouter.ai/models.

## Task format

```json
{
  "id": "deal-01",
  "category": "deal_mechanics",
  "difficulty": "medium",
  "prompt": "An SPA uses completion accounts on a cash-free/debt-free basis...",
  "answer": 772,
  "unit": "currency",
  "rel_tolerance": 0.005,
  "solution": "EV - net debt + (actual NWC - target NWC)"
}
```

## Repo structure

```
llm_finance_eval/
  build_tasks.py   # task definitions + answer key computed in code
  finance.py       # NPV, IRR, annuity helpers (no dependencies)
  grading.py       # FINAL ANSWER parser + tolerance check
  providers.py     # mock / Anthropic / OpenAI callers
  run_eval.py      # runner + scorecard
data/tasks.jsonl   # generated task set (built from build_tasks.py, git-ignored)
tests/             # grader and task-set tests, incl. independent recomputation of v2 answers (pytest)
```

## Roadmap

- [ ] Rubric-based grading of the reasoning, not only the final number (LLM-as-judge with a practitioner rubric)
- [ ] Multi-step tasks: full three-statement → DCF → football field
- [ ] Purchase price allocation (IFRS 3), IFRS 16 lease adjustments to EV/EBITDA
- [ ] Portuguese (pt-BR) version of the task set
- [ ] Public leaderboard across frontier models

## Contributing

New tasks are welcome. Add them to `build_tasks.py` with the answer computed in code and a one-line `solution`.

## Author

**Felipe Achnitz**: M&A and corporate development in energy & infrastructure (valuation, project finance, impairment testing, due diligence). Economist (Mackenzie), postgraduate in Finance & Economics (FGV EESP).
[LinkedIn](https://www.linkedin.com/in/felipe-achnitz)

## License

MIT
