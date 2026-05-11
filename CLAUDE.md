# CLAUDE.md — Project Context for Claude Code

## What this repo is

Freddie Mac 30-year fixed-rate mortgage data pipeline and survival analysis codebase.
Assignment: IR&C Homework #3 (Baruch College, Prof. Lesniewski, due May 13 2026).
Scope: Parts A–C are core; D–E are advanced extensions.

## Key files

| File | Purpose |
|---|---|
| `src/credit_data.py` | All data loading — always use these loaders, never read parquet directly |
| `src/schemas.py` | Column names, dtypes, sentinel values |
| `DATA.md` | Per-column reference for origination and monthly tables |
| `PLAN.md` | Full pipeline spec (schema, event derivation rules, disk constraints) |
| `scripts/partB_cox.py` | Part B: Cox PH model — our primary analysis script |
| `notebooks/partA_survival.ipynb` | Part A: KM, hazard rates, stratified curves (complete) |
| `notebooks/partA_calendar.ipynb` | Part A supplement: calendar-time prepayment vs. rate |

## Data layout (after `python scripts/download_data.py`)

```
data/processed/
├── origination/    vintage_year=YYYY/part-QN.parquet   — one row per loan
├── monthly/        vintage_year=YYYY/part-QN.parquet   — one row per loan-month
├── outcomes/       vintage_year=YYYY/part-QN.parquet   — derived events (small, ~70 MB)
└── macro/          fred_monthly.parquet                 — FRED monthly series
```

## Loading data

```python
from src.credit_data import load_loans, load_origination, load_monthly, load_outcomes, load_macro

# Most common: origination features + event outcome joined
df = load_loans(years=[2006, 2007], columns=["loan_seq_num", "fico", "ltv",
                                              "event_type", "event_time_months"])

# Macro (small, always load fully)
macro = load_macro()   # columns: month, MORTGAGE30US, GS10, UNRATE, CPIAUCSL, CSUSHPISA

# Monthly panel — use lazy=True for wide vintage ranges
lf = load_monthly(years=list(range(2006, 2023)), lazy=True)
```

## Event definitions (from outcomes table)

- `prepaid` ← `zb_code == "01"` (full prepayment — the target event for Parts A–C)
- `defaulted` ← `zb_code ∈ {03,06,09,15}` or ever 180+ days delinquent
- `other_termination` ← other terminal zb_code
- `censored` ← no terminal state reached

Assignment rule: treat only full repayment as prepayment; ignore curtailments.

## Train / test convention

- **In-sample:** vintages 2006–2022
- **Held out:** 2024–2025 (2023 Q3–Q4 are absent from upstream data)
- `YEARS = list(range(2006, 2023))` is the standard in-sample range used in all scripts

## Key columns (processed names)

**Origination:** `loan_seq_num`, `fico`, `ltv`, `cltv`, `dti`, `orig_rate`, `orig_upb`,
`loan_purpose` (P/C/N), `channel` (R/B/T/C), `n_borrowers`, `state`, `vintage_year`, `vintage_quarter`, `first_payment_date`

**Monthly:** `loan_seq_num`, `month`, `loan_age`, `actual_upb`, `current_rate`, `eltv`, `dq_status`, `zb_code`

**Outcomes:** `loan_seq_num`, `event_type`, `event_time_months`, `first_obs_date`, `last_obs_date`

**Macro:** `month`, `MORTGAGE30US`, `GS10`, `UNRATE`, `CPIAUCSL`, `CSUSHPISA`

## Figures

All plots go to `figures/`. Naming convention:
- `partA_*` — Part A exploratory analysis
- `partB_*` — Part B Cox model

## Environment

Devcontainer (Linux). Install packages with `pip install --break-system-packages`.
No venv needed — the container is the isolation layer.
All required packages are in `requirements.txt`.

## Part B Cox script — design notes

`scripts/partB_cox.py` runs two Cox PH models:

1. **Static:** FICO, LTV, DTI, orig_rate, loan_purpose, channel, n_borrowers, vintage_year
2. **Macro:** Static + rate_incentive_orig (orig_rate − MORTGAGE30US), mort_treasury_spread
   (MORTGAGE30US − GS10), UNRATE, log_hpi, log_cpi — all joined at `first_payment_date`

Key parameters (edit at top of file or pass as CLI flags):
- `DEFAULT_SAMPLE = 2_000_000` — loans sampled for Cox fitting
- `SCHOENFELD_MAX = 200_000` — subsample for PH test (sufficient; avoids hours of runtime)
- `penalizer = 0.01` — L2 regularization (small at n=2M; likelihood dominates)
- `TRAIN_YEARS = list(range(2006, 2023))`

The macro covariates are joined **at origination** (static). Time-varying rate incentive
(current market rate vs. orig_rate each month) is the correct treatment but requires
the monthly panel — that belongs in Part E (advanced extensions).

## Reference reading

`../Sources/Chen2024_sections2-3_notes.md` — clean notes on the survival analysis
theory used in Part B (right-censored framework, hazard function, Cox partial
likelihood, Breslow estimator, hazard ratio interpretation, PH assumption check).
Read this instead of re-running pdfminer on the source PDF.
