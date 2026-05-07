# Data reference

Schema, join keys, and per-column documentation for `data/processed/`. The
underlying data is Freddie Mac's single-family loan-level dataset, filtered
at conversion time to **30-year fixed-rate mortgages** (`term_months == 360`
AND `amort_type == 'FRM'`).

For the original field definitions consult `data/user_guide.pdf` and
`data/file_layout.xlsx` (Freddie Mac docs, kept locally — not redistributed
in this repo).

## Quick map

| Table | Grain | Rows | Size | Partition |
|---|---|---|---|---|
| `origination/` | one per loan | ~14M | ~295 MB | `vintage_year=YYYY/part-QN.parquet` |
| `monthly/` | one per loan-month | ~600M | ~9.4 GB | `vintage_year=YYYY/part-QN.parquet` |
| `outcomes/` | one per loan | ~14M | ~70 MB | `vintage_year=YYYY/part-QN.parquet` |
| `macro/fred_monthly.parquet` | one per month | 436 | ~12 KB | single file |

**Vintages covered:** 2006 Q1 through 2025 Q3 (2023 Q3 + Q4 missing upstream).

> **Important partition convention.** `vintage_year` everywhere refers to
> the *loan's origination year*, not the reporting period. A loan that
> originated in 2006 stays in `monthly/vintage_year=2006/` even when
> we observe it making payments in 2020. This means every loan's full
> monthly history is co-located in one partition — ideal for vintage-
> stratified analysis and per-loan groupbys.

## Join keys at a glance

| Joining... | Key | Type | Notes |
|---|---|---|---|
| `origination` ↔ `outcomes` | `loan_seq_num` | 1-to-1 inner | both tables have one row per loan; counts match per partition |
| `origination` ↔ `monthly` | `loan_seq_num` | 1-to-many | the typical loan-history attach |
| `monthly` ↔ `outcomes` | `loan_seq_num` | many-to-1 | use to attach the eventual outcome to a loan-month |
| `monthly` ↔ `macro` | `month` (Date) | many-to-1 | `month` is the month-start `Date` in both |
| `origination` ↔ `macro` | `first_payment_date` (Date) | many-to-1 | for "macro at origination" features |

`loan_seq_num` is a 12-character string of the form `F<YY>Q<N><7-digit ID>`,
e.g. `F06Q10000001` = 2006 Q1, loan 0000001. It is unique across the entire
dataset.

## Loaders (`src/credit_data.py`)

```python
from src.credit_data import (
    load_origination, load_monthly, load_outcomes, load_macro, load_loans,
)

orig    = load_origination(years=[2006, 2007], columns=[...])     # eager
out     = load_outcomes(years=None)                                # all vintages
macro   = load_macro()
monthly = load_monthly(years=[2010], columns=[...], lazy=True)     # always prefer lazy

# Convenience: origination + outcomes joined on loan_seq_num.
# One row per loan with both features (X) and event (y).
loans   = load_loans(years=[2006, 2007],
                     columns=["loan_seq_num", "fico", "ltv", "dti",
                              "orig_rate", "event_type", "event_time_months"])
```

All loaders use parquet partition pruning so unrequested years are not read.

> **Why isn't this two tables merged on disk?** `origination/` is static
> (raw → typed → filtered, never re-derived); `outcomes/` is a derived
> table that gets rewritten whenever the event-classification rules
> change. Keeping them separate localizes the blast radius of those
> changes, keeps features and labels in distinct files (helpful for
> avoiding label leakage), and lets pure-X or pure-y queries scan less
> data. `load_loans()` papers over the join when you actually want both.

---

## 1. `origination/` — one row per loan

Loan-level snapshot at origination. Used as the source of static features
(FICO, LTV, DTI, rate, location) for any modeling task.

### Identity and partitioning

| Column | Dtype | Description |
|---|---|---|
| `loan_seq_num` | String | **Primary key.** Unique loan ID, format `F<YY>Q<N><7-digit ID>`. Join key against `monthly`, `outcomes`. |
| `vintage_year` | UInt16 | Loan origination year (= partition key). Derived from `first_payment_date`. |
| `vintage_quarter` | UInt8 | 1-4. Origination quarter, derived from `first_payment_date`. |

### Origination dates

| Column | Dtype | Description |
|---|---|---|
| `first_payment_date` | Date | First scheduled monthly payment. Stored as the first day of that month. Defines the vintage. |
| `maturity_date` | Date | Scheduled final payment date (origination + term). For the 30y FRM filter, always 360 months after `first_payment_date`. |

### Borrower and underwriting features

| Column | Dtype | Sentinels → null | Description |
|---|---|---|---|
| `fico` | UInt16 | 9999 | Credit score at origination, range typically 300-850. Industry-standard underwriting feature. |
| `first_time_homebuyer` | String | 9 | `Y` / `N`. First-time-homebuyer flag. |
| `n_borrowers` | UInt8 | 99 | Number of borrowers on the note (1-10). |
| `dti` | UInt8 | 999 | Debt-to-income ratio (whole percentage points). |

### Loan-amount and rate features

| Column | Dtype | Sentinels → null | Description |
|---|---|---|---|
| `orig_upb` | Float32 | — | Original unpaid principal balance (USD). |
| `orig_rate` | Float32 | — | Note interest rate at origination (annual %, e.g. `6.5`). |
| `ltv` | UInt8 | 999 | Original loan-to-value ratio (whole percentage points). |
| `cltv` | UInt8 | 999 | Combined LTV including subordinate liens. |
| `mi_pct` | UInt8 | 999 | Mortgage insurance coverage % at origination. 0 if not applicable. |
| `term_months` | UInt16 | — | Original loan term in months. **Always 360** here (the FRM filter). |
| `amort_type` | String | — | Always `'FRM'` here. Dropped at modeling time; useful as a sanity-check column. |

### Property features

| Column | Dtype | Sentinels → null | Description |
|---|---|---|---|
| `n_units` | UInt8 | 99 | Number of units in the property (1-4). |
| `occupancy` | String | 9 | `P` (primary), `I` (investor), `S` (second home). |
| `prop_type` | String | 99 | `SF` (single-family), `PU` (PUD), `CO` (condo), `CP` (co-op), `MH` (manufactured), `LH` (leasehold). |
| `state` | String | — | 2-letter US state code. |
| `msa` | UInt32 | — | Metropolitan Statistical Area code (CBSA). Null for non-MSA properties. |
| `zip3` | UInt32 | — | Postal code. Anonymized by Freddie to first 3 digits + `00`. |
| `prop_valuation_method` | UInt8 | 9 | 1=ACE, 2=full appraisal, 3=external-only, 4=desktop. |

### Origination economics

| Column | Dtype | Sentinels → null | Description |
|---|---|---|---|
| `channel` | String | 9 | `R` (retail), `B` (broker), `T` (correspondent), `C` (TPO not specified). |
| `loan_purpose` | String | 9 | `P` (purchase), `C` (cash-out refi), `N` (no-cash-out refi). |
| `ppm_flag` | String | — | `Y` / `N`. Prepayment-penalty mortgage flag. |
| `seller_name` | String | — | Originator/seller of the loan to Freddie. Free text. |
| `servicer_name` | String | — | Current servicer at the time the data extract was cut. Free text. |

### Program / structural flags

| Column | Dtype | Description |
|---|---|---|
| `super_conforming` | String | `Y` / blank. Flag for loans above the conforming limit but below the high-balance limit. |
| `preharp_loan_seq_num` | String | If this is a HARP refi, the original (pre-HARP) `loan_seq_num`. Mostly null. |
| `program` | String | `H` (HARP), `F` (HFA Advantage), `R` (Refi Plus), or 9. |
| `harp` | String | `Y` / `N`. HARP indicator. |
| `io_indicator` | String | `Y` / `N`. Interest-only flag. |
| `mi_cancel` | String | MI cancellation indicator. |

---

## 2. `monthly/` — one row per loan-month

The performance panel. One row per (loan, reporting month). This is the
table you join macro features onto and aggregate to build outcome / event
columns.

### Identity and time

| Column | Dtype | Description |
|---|---|---|
| `loan_seq_num` | String | Loan ID. Join key to `origination`, `outcomes`. |
| `month` | Date | Monthly reporting period, stored as first-of-month. **Join key to `macro`.** |
| `loan_age` | UInt16 | Months since origination, 0-based. `loan_age == event_time_months` at the loan's last observation. |
| `rem_months` | UInt16 | Remaining months to legal maturity. |

### Balance and rate (time-varying)

| Column | Dtype | Description |
|---|---|---|
| `actual_upb` | Float32 | Current unpaid principal balance at end of period. Hits 0 when the loan terminates. |
| `current_rate` | Float32 | Note rate in effect for this period. Constant for FRMs unless modified. |
| `deferred_upb` | Float32 | Portion of UPB that's been deferred (non-interest-bearing) — only non-zero for modified loans. |
| `interest_bearing_upb` | Float32 | UPB excluding the deferred portion. |

### Delinquency and servicing

| Column | Dtype | Description |
|---|---|---|
| `dq_status` | String | Months delinquent as a string: `"0"`, `"1"`, ..., `"30"`, plus special codes `"RA"` (Reactive — recently re-performing) and `"XX"` (status unknown). Cast to int with `pl.col("dq_status").cast(pl.Int16, strict=False)` — special codes become null. |
| `ddlpi` | Date | Due-date of last paid installment. |
| `mod_flag` | String | `Y` if the loan has ever been modified, else null/`N`. |
| `step_mod_flag` | String | Step-modification flag. |
| `deferred_payment_plan` | String | Deferral plan code (e.g. forbearance treatments). |
| `disaster_dq` | String | `Y` if delinquency is attributed to a federally-declared disaster. |
| `borrower_assistance` | String | Borrower assistance code (forbearance / repayment plan / trial / permanent mod). |

### Termination

`zb_code` (zero-balance code) is the field that drives event classification
in `outcomes/`. Values used in the data:

| `zb_code` | Meaning |
|---|---|
| `01` | Prepaid / matured / refinanced (full payoff). **PLAN treats this as the only true prepayment event.** |
| `02` | Third-party sale (similar economics to prepay). |
| `03` | Short sale or charge-off. |
| `06` | Repurchased by seller (typically for rep-and-warrant breach). |
| `09` | REO disposition (foreclosure-completion sale). |
| `15` | Note sale. |
| `96` | Removal — modified loan removed from the dataset after re-performing. |

Other termination columns:

| Column | Dtype | Description |
|---|---|---|
| `zb_date` | Date | Effective date of the zero-balance event. |
| `zb_removal_upb` | Float32 | UPB at the time of removal — useful for sizing the loss. |
| `defect_date` | Date | If the loan was identified as having an underwriting defect, this is the settlement date. Mostly null. |

### Loss / recovery (only populated on default-resolution rows)

These are zero or null for performing and prepaid loans. Useful for the
optional default-loss extension (assignment §E).

| Column | Dtype | Description |
|---|---|---|
| `mi_recoveries` | Float32 | Mortgage insurance proceeds received. |
| `non_mi_recoveries` | Float32 | Other recovery proceeds (e.g. sale of property). |
| `net_sales_proceeds` | String | Sale proceeds. **Sometimes alphabetic** (`"U"` = unknown, `"C"` = covered by recoveries) — kept as String. Cast to Float32 with `cast(..., strict=False)` after filtering out non-numeric values. |
| `expenses` | Float32 | Total expenses associated with the default. |
| `legal_costs` | Float32 | Legal costs subset of `expenses`. |
| `maint_costs` | Float32 | Property preservation / maintenance subset. |
| `taxes_insurance` | Float32 | Taxes and insurance advances. |
| `misc_expenses` | Float32 | Other expenses. |
| `actual_loss` | Float32 | Realized loss after recoveries. **Negative values mean a gain (rare).** |
| `mod_cost` | Float32 | Cumulative modification cost. |
| `current_mod_cost` | Float32 | Modification cost incurred this period. |
| `dq_accrued_interest` | Float32 | Accrued interest on delinquent loans. |

### Estimated mark-to-market

| Column | Dtype | Sentinels → null | Description |
|---|---|---|---|
| `eltv` | UInt16 | 999 | Estimated current LTV using Freddie's mark-to-market home price index. Updated periodically. Useful as a time-varying covariate; null for many recent loans. |

### Sort order

Within each partition file, rows are written in the order pyarrow received
them from the streaming CSV reader, which is the same order as the source
text — already sorted by `(loan_seq_num, month)`. Pulling one loan's full
history is a contiguous read.

---

## 3. `outcomes/` — derived per-loan event table

Built from `monthly/` by aggregating per `loan_seq_num` and applying the
event-classification rules in PLAN.md §2.3. Small (~70 MB total), so safe
to load every vintage at once with `load_outcomes()`.

### Columns

| Column | Dtype | Description |
|---|---|---|
| `loan_seq_num` | String | Loan ID. Join key. |
| `vintage_year` | UInt16 | From `origination` (partition key). |
| `vintage_quarter` | UInt8 | From `origination`. |
| `first_obs_date` | Date | Earliest `month` in `monthly`. Null if the loan has no monthly observations yet (rare; happens for very new originations). |
| `last_obs_date` | Date | Most recent `month` in `monthly`. |
| `event_time_months` | UInt16 | `loan_age.max()` — months from origination to event-or-censoring. **The duration variable for survival models.** Null if no monthly history yet. |
| `event_type` | String | Categorical: `prepaid` / `defaulted` / `other_termination` / `censored`. See rules below. |
| `final_zero_balance_code` | String | Last non-null `zb_code` in `monthly` for this loan. Null if the loan never reached a terminal state. |
| `max_dq_status` | Int16 | Maximum integer `dq_status` ever observed. `RA` / `XX` ignored. Null if all dq codes were special. |
| `final_delinquency_status` | String | The last raw `dq_status` value (preserves `RA` / `XX`). |

### Event classification rules

In priority order:

1. **`prepaid`** — `final_zero_balance_code == "01"`.
2. **`defaulted`** — `final_zero_balance_code` ∈ `{"03", "06", "09", "15"}` **OR** `max_dq_status >= 6` (i.e. ever 180+ days delinquent).
3. **`other_termination`** — any other non-null `final_zero_balance_code` (e.g. `"02"` third-party sale, `"96"` modified-and-removed).
4. **`censored`** — no terminal `zb_code` reached by the last observed month, and never hit D180+.

The rule order matters: a loan that prepaid at the end is `prepaid` even
if it had a transient delinquency earlier in life. A loan that hit D180+
but later cured and was still active at extract is `defaulted`, because
serious delinquency is treated as the modeled event.

### Counts by event_type (sanity benchmarks)

For the 2006 vintage on first build:
- prepaid: 860,706 (85%)
- defaulted: 115,660 (11%)
- censored: 24,318 (2%)
- other_termination: 6,979 (1%)

Recent vintages (2020+) are dominated by `censored` since they haven't
had time to terminate — this is what makes them suitable as held-out
test data.

---

## 4. `macro/fred_monthly.parquet` — FRED monthly series

Single-file table. One row per month. Series fetched via
`pandas-datareader` from FRED.

| Column | Dtype | FRED ID | Description |
|---|---|---|---|
| `month` | Datetime[ns] | — | Month-start. Cast to Date with `.cast(pl.Date)` before joining to `monthly`. |
| `MORTGAGE30US` | Float64 | [MORTGAGE30US](https://fred.stlouisfed.org/series/MORTGAGE30US) | 30-year fixed mortgage rate, US average. Originally weekly Freddie PMMS — resampled to monthly mean. |
| `GS10` | Float64 | [GS10](https://fred.stlouisfed.org/series/GS10) | 10-year US Treasury constant-maturity yield. Useful as a refi-incentive denominator. |
| `UNRATE` | Float64 | [UNRATE](https://fred.stlouisfed.org/series/UNRATE) | Civilian unemployment rate (%). |
| `CPIAUCSL` | Float64 | [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI All Urban Consumers, seasonally adjusted (1982-1984=100). |
| `CSUSHPISA` | Float64 | [CSUSHPISA](https://fred.stlouisfed.org/series/CSUSHPISA) | S&P/Case-Shiller US National Home Price Index, seasonally adjusted. |

The most recent month or two may have nulls for series that report on a
lag (e.g. `CSUSHPISA`, `UNRATE`).

### Joining macro to monthly

Both tables store `month` as month-start. After casting macro's column to
`Date`, a plain `left` join on `month` works:

```python
macro = load_macro().with_columns(pl.col("month").cast(pl.Date))
panel = (
    load_monthly(years=[2010], lazy=True)
    .join(macro.lazy(), on="month", how="left")
    .collect()
)
```

For "macro at origination" features, join on `first_payment_date` instead:

```python
orig = load_origination(years=[2010])
features = orig.join(macro, left_on="first_payment_date", right_on="month", how="left")
```

---

## Sentinel and missing-value handling

Freddie encodes "not available" with width-specific 9s (`9`, `99`, `999`,
`9999`). The conversion pipeline strips these to null **before** dtype
casting. The complete sentinel map lives in `src/schemas.py`:

- **Origination** : `fico=9999`, `first_time_homebuyer=9`, `mi_pct=999`, `n_units=99`, `occupancy=9`, `cltv=999`, `dti=999`, `ltv=999`, `channel=9`, `prop_type=99`, `loan_purpose=9`, `n_borrowers=99`, `prop_valuation_method=9`.
- **Monthly** : `eltv=999`. Empty strings on all fields → null universally.

If you see `9999` or `999` in a numeric column, the file you're reading is
upstream of the sentinel pass — check that you're loading from
`data/processed/`, not from a raw `.txt`.

## Data-quality notes

- **2023 H2 missing.** The upstream Freddie archive did not include
  Q3/Q4 2023. Cross-vintage analyses that use 2023 must end at Q2.
- **2024-2025 should be held out.** Because 2023 is partial, train/eval
  splits that include 2024-2025 in training would be biased. The
  recommended pattern is train on 2006-2022 and evaluate on 2024-2025.
- **`net_sales_proceeds` mixed type.** Mostly numeric but sometimes the
  letter `"U"` (unknown). Stored as String; cast to float with
  `strict=False` and treat the null as missing.
- **`max_dq_status` null.** Possible if every `dq_status` value the loan
  ever had was a special code (`RA`/`XX`). Null does not necessarily mean
  the loan was performing — check `final_zero_balance_code`.
- **Recent vintages have low default counts.** A 2024 Q3 loan has had
  at most 7-8 monthly observations as of the data cut, so reaching 180+
  days delinquent is essentially impossible. The right interpretation is
  "censored," not "low default risk."
