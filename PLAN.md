# Mortgage Default / Prepayment Project — Data Processing Plan

This plan describes how to convert the Freddie Mac single-family loan-level
zipped text data in `data/` into a compact, fast, multi-user-friendly format
suitable for the survival-analysis and ML work specified in
`IR&C_Assignment3_2026.pdf`.

The plan is detailed enough that a fresh agent can implement it from a short
prompt such as *"implement PLAN.md"*.

---

## 1. Goals and constraints

| Goal | Why it matters |
|------|----------------|
| Minimize disk footprint | Only ~17 GB free on this machine; raw zips already take 21 GB. Full unzip (~150 GB+) is not feasible. |
| Fast filter / scan / column subset | KM / Cox / XGBoost workflows repeatedly slice by vintage, FICO, LTV, term — columnar storage with predicate pushdown is essential. |
| Low memory at research time | Monthly panel has ~10⁹ rows across all years. Must stream / chunk; never load whole panel into RAM. |
| Lightweight, simple for many users | Single `pip install`, single `python prepare_data.py` step. After that, a one-line `load_*` function returns a Polars/Pandas DataFrame. |
| Reproducible | Deterministic output for given input zips; schema and dtypes specified once. |

Hard rules that fall out of these constraints:

- **Stream zip → Parquet directly**. Never write unzipped `.txt` to disk.
- **Filter to 30-year FRM at conversion time** (assignment requires this). Drops 30-50% of rows.
- **Process one quarter at a time**, free memory between quarters.
- **Keep the raw `.zip` files** as the cold archive (they are the source of truth). Parquet is the hot working format.

---

## 2. Output layout

All converted data lives in `data/processed/`. Three tables, all Parquet, all
ZSTD-compressed, partitioned by `vintage_year`.

```
data/processed/
├── origination/                        ~150-300 MB total
│   ├── vintage_year=2006/part-Q1.parquet
│   ├── vintage_year=2006/part-Q2.parquet
│   └── ...
├── monthly/                            ~5-8 GB total (the big one)
│   ├── vintage_year=2006/part-Q1.parquet
│   └── ...
├── outcomes/                           ~30-80 MB total (derived)
│   └── vintage_year=2006/part-Q1.parquet
└── macro/
    └── fred_monthly.parquet            ~50 KB (separate fetch from FRED)
```

Estimated total output: **~6-10 GB**. Comfortably fits in remaining disk
alongside the 21 GB of zips.

### 2.1 `origination/` — one row per loan

Filtered to **30-year fixed rate** loans (`Original Loan Term == 360` AND
`Amortization Type == 'FRM'`). All 32 columns from the data dictionary,
typed and renamed to snake_case. Partition key `vintage_year` is derived
from `first_payment_date`.

Add a derived `vintage_quarter` (1-4) column for convenience.

### 2.2 `monthly/` — one row per loan-month

Same 30-year-FRM filter, applied via `loan_seq_num` membership in the
filtered origination set. Columns: keep all 32 but downcast aggressively
(see §3). Partition key `vintage_year` (from the loan's vintage, not the
reporting period — keeps monthly rows for one loan in one partition, which
is what we want for grouping).

This is the largest table. Within each partition, sort by
`(loan_seq_num, monthly_reporting_period)` so reading one loan's full
history is a contiguous read.

### 2.3 `outcomes/` — one row per loan (derived)

Derived from `monthly/` in a second pass. Columns:

- `loan_seq_num`
- `vintage_year`, `vintage_quarter`
- `first_obs_date`, `last_obs_date`
- `event_time_months` — months from origination to event or censoring
- `event_type` — categorical: `prepaid` | `defaulted` | `other_termination` | `censored`
- `final_zero_balance_code`
- `final_delinquency_status`

Event derivation rules (see assignment §1 and Freddie data dictionary):

- **`prepaid`** ← `Zero Balance Code == 01` (full prepayment). Per the
  assignment, this is the only event treated as prepayment; partial
  curtailments are ignored.
- **`defaulted`** ← `Zero Balance Code ∈ {03, 06, 09, 15}` (short sale, repurchase,
  REO disposition, note sale) **or** delinquency status ≥ `D180`.
- **`other_termination`** ← any other terminal `Zero Balance Code` (e.g. 02, 96).
- **`censored`** ← no terminal status reached by last observed month.

This table is what most modeling code will load — small, indexed, immediate.

### 2.4 `macro/fred_monthly.parquet`

Fetched separately by `scripts/fetch_macro.py` (FRED API or `pandas-datareader`).
Series:

- `MORTGAGE30US` — 30-year fixed mortgage rate (weekly → resampled monthly)
- `GS10` — 10-year Treasury yield
- `UNRATE` — civilian unemployment rate
- `CPIAUCSL` — CPI all urban consumers
- `CSUSHPISA` — Case-Shiller national HPI

Index: monthly date. Joined to `monthly/` on `monthly_reporting_period`
at modeling time, not at conversion time (keeps `monthly/` independent of
macro refreshes).

---

## 3. Schema and dtype map

Aggressive downcasting is the single biggest disk-space lever. Approximate
expected savings vs naive int64/float64/string: 4-8x.

### 3.1 Origination

| Field (data dictionary) | Output column | Dtype | Notes / sentinel-null mapping |
|---|---|---|---|
| Credit Score | `fico` | `UInt16` (nullable) | 9999 → null |
| First Payment Date | `first_payment_date` | `Date` | YYYYMM → first of month |
| First Time Homebuyer Flag | `first_time_homebuyer` | `Categorical` | 9 → null |
| Maturity Date | `maturity_date` | `Date` | YYYYMM |
| MSA | `msa` | `UInt32` (nullable) | empty → null |
| MI % | `mi_pct` | `UInt8` | 999 → null |
| Number of Units | `n_units` | `UInt8` | 99 → null |
| Occupancy Status | `occupancy` | `Categorical` | 9 → null |
| CLTV | `cltv` | `UInt8` | 999 → null |
| DTI | `dti` | `UInt8` | 999 → null |
| Original UPB | `orig_upb` | `Float32` | |
| Original LTV | `ltv` | `UInt8` | 999 → null |
| Original Interest Rate | `orig_rate` | `Float32` | |
| Channel | `channel` | `Categorical` | R/B/T/C, 9 → null |
| PPM Flag | `ppm_flag` | `Categorical` | |
| Amortization Type | `amort_type` | `Categorical` | drop after FRM filter |
| Property State | `state` | `Categorical` | |
| Property Type | `prop_type` | `Categorical` | |
| Postal Code | `zip3` | `UInt32` | already 5-digit anonymized to 3-digit + "00" by Freddie |
| Loan Sequence Number | `loan_seq_num` | `String` (dictionary) | join key |
| Loan Purpose | `loan_purpose` | `Categorical` | P/C/N |
| Original Loan Term | `term_months` | `UInt16` | drop after term==360 filter |
| Number of Borrowers | `n_borrowers` | `UInt8` | 99 → null |
| Seller Name | `seller_name` | `Categorical` | |
| Servicer Name | `servicer_name` | `Categorical` | |
| Super Conforming Flag | `super_conforming` | `Categorical` | |
| Pre-HARP Loan Sequence Number | `preharp_loan_seq_num` | `String` (dictionary) | mostly null |
| Program Indicator | `program` | `Categorical` | |
| HARP Indicator | `harp` | `Categorical` | |
| Property Valuation Method | `prop_valuation_method` | `UInt8` | |
| I/O Indicator | `io_indicator` | `Categorical` | |
| MI Cancellation Indicator | `mi_cancel` | `Categorical` | |

### 3.2 Monthly performance

| Field | Output column | Dtype | Notes |
|---|---|---|---|
| Loan Sequence Number | `loan_seq_num` | `String` (dict) | |
| Monthly Reporting Period | `month` | `Date` | first of month |
| Current Actual UPB | `actual_upb` | `Float32` | |
| Current Loan Delinquency Status | `dq_status` | `Categorical` | "0", "1", ..., "RA", "XX" |
| Loan Age | `loan_age` | `UInt16` | |
| Remaining Months to Legal Maturity | `rem_months` | `UInt16` | |
| Defect Settlement Date | `defect_date` | `Date` | mostly null |
| Modification Flag | `mod_flag` | `Categorical` | |
| Zero Balance Code | `zb_code` | `Categorical` | |
| Zero Balance Effective Date | `zb_date` | `Date` | |
| Current Interest Rate | `current_rate` | `Float32` | |
| Current Deferred UPB | `deferred_upb` | `Float32` | |
| DDLPI | `ddlpi` | `Date` | |
| MI Recoveries | `mi_recoveries` | `Float32` | |
| Net Sales Proceeds | `net_sales_proceeds` | `String` (dict) | sometimes alpha codes |
| Non MI Recoveries | `non_mi_recoveries` | `Float32` | |
| Expenses | `expenses` | `Float32` | |
| Legal Costs | `legal_costs` | `Float32` | |
| Maintenance | `maint_costs` | `Float32` | |
| Taxes & Insurance | `taxes_insurance` | `Float32` | |
| Misc Expenses | `misc_expenses` | `Float32` | |
| Actual Loss | `actual_loss` | `Float32` | |
| Modification Cost | `mod_cost` | `Float32` | |
| Step Modification Flag | `step_mod_flag` | `Categorical` | |
| Deferred Payment Plan | `deferred_payment_plan` | `Categorical` | |
| Estimated LTV | `eltv` | `UInt16` | 999 → null |
| Zero Balance Removal UPB | `zb_removal_upb` | `Float32` | |
| Delinquent Accrued Interest | `dq_accrued_interest` | `Float32` | |
| Delinquency Due to Disaster | `disaster_dq` | `Categorical` | |
| Borrower Assistance Status Code | `borrower_assistance` | `Categorical` | |
| Current Month Modification Cost | `current_mod_cost` | `Float32` | |
| Interest Bearing UPB | `interest_bearing_upb` | `Float32` | |

### 3.3 Sentinel handling

Freddie uses 9, 99, 999, 9999 as "not available" depending on field width.
A small `MISSING_SENTINELS` dict in the conversion code maps each numeric
column to its sentinel(s) and replaces them with null **before** dtype
casting. Document the mapping in code comments referencing
`data/user_guide.pdf` and `data/file_layout.xlsx`.

---

## 4. Conversion pipeline

### 4.1 Tooling

- **Polars** for I/O and transforms (lazy, multithreaded, low memory).
- **PyArrow** for Parquet writing (fine-grained control over compression and
  dictionary encoding).
- Stdlib **`zipfile`** for streaming `.zip` members without extracting.

Why Polars and not Pandas for ETL: Pandas materializes the full frame and
its string columns are expensive. Polars `scan_csv` over a streaming reader
plus `sink_parquet` keeps peak RAM in the low GBs even for the biggest
quarters.

Users at modeling time can still use Pandas (`pd.read_parquet(...)`) — the
output format is interchange-friendly.

### 4.2 Per-quarter pipeline (`scripts/convert_quarter.py`)

```
for each year_zip in data/historical_data_YYYY.zip:
    open zip in streaming mode (no extraction)
    for each quarter (Q1..Q4) inside:
        # 1. Read origination
        with zip.open("historical_data_YYYYQn.txt") as f:
            df_orig = polars.read_csv(f, separator="|", has_header=False,
                                      new_columns=ORIG_COLS, schema=ORIG_SCHEMA)
            df_orig = (df_orig
                       .pipe(apply_sentinels)
                       .filter((pl.col("term_months") == 360) &
                               (pl.col("amort_type") == "FRM"))
                       .with_columns(vintage_year=YYYY, vintage_quarter=n))
            df_orig.write_parquet(
                f"data/processed/origination/vintage_year={YYYY}/part-Q{n}.parquet",
                compression="zstd", compression_level=9,
                use_pyarrow=True)
            kept_loan_ids = set(df_orig["loan_seq_num"])

        # 2. Read monthly in CHUNKS (1-2 GB raw text, ~20M rows)
        with zip.open("historical_data_time_YYYYQn.txt") as f:
            writer = pyarrow.parquet.ParquetWriter(
                f"data/processed/monthly/vintage_year={YYYY}/part-Q{n}.parquet",
                schema=MONTHLY_ARROW_SCHEMA,
                compression="zstd", compression_level=9,
                use_dictionary=True)
            for chunk in polars.read_csv_batched(f, separator="|", batch_size=2_000_000,
                                                  has_header=False,
                                                  new_columns=MONTHLY_COLS,
                                                  schema=MONTHLY_SCHEMA):
                chunk = (chunk
                         .pipe(apply_sentinels)
                         .filter(pl.col("loan_seq_num").is_in(kept_loan_ids)))
                writer.write_table(chunk.to_arrow())
            writer.close()

        # 3. Free memory before next quarter
        del df_orig, kept_loan_ids
```

`polars.read_csv_batched` (or `pl.scan_csv(...).collect(streaming=True)`) is
the key to keeping peak RAM bounded — a single 2006Q1 monthly file has
20M rows and would be ~5-8 GB in Polars memory if loaded whole.

### 4.3 Outcomes derivation (`scripts/derive_outcomes.py`)

After all `monthly/` partitions are written, run a separate pass:

```
for each monthly partition file:
    df = pl.scan_parquet(file).group_by("loan_seq_num").agg([
        pl.col("month").min().alias("first_obs_date"),
        pl.col("month").max().alias("last_obs_date"),
        pl.col("loan_age").max().alias("event_time_months"),
        pl.col("zb_code").drop_nulls().last().alias("final_zero_balance_code"),
        pl.col("dq_status").max().alias("max_dq_status"),
        pl.col("dq_status").last().alias("final_delinquency_status"),
    ]).with_columns(event_type=classify_event(...))
    df.sink_parquet(corresponding outcomes path)
```

`classify_event` implements §2.3 rules. Done in pure Polars expressions —
no Python loop.

### 4.4 Macro (`scripts/fetch_macro.py`)

Standalone, network-dependent, rarely re-run. Uses
`pandas-datareader` against FRED. Output:
`data/processed/macro/fred_monthly.parquet`, indexed by month.

### 4.5 Master script (`scripts/prepare_data.py`)

Orchestrates §4.2 → §4.3 → §4.4 in order, with progress bars (`tqdm`).
Idempotent: if a partition already exists and is non-empty, skip. CLI
flags: `--years 2006 2007 ...`, `--force`, `--no-macro`.

---

## 5. Data loader (user-facing API)

A thin module `src/credit_data.py` is what every modeling notebook imports.
Three functions, no surprises:

```python
def load_origination(years: list[int] | None = None,
                     columns: list[str] | None = None) -> pl.DataFrame: ...

def load_monthly(years: list[int] | None = None,
                 columns: list[str] | None = None,
                 loan_ids: list[str] | None = None,
                 lazy: bool = False) -> pl.DataFrame | pl.LazyFrame: ...

def load_outcomes(years: list[int] | None = None) -> pl.DataFrame: ...

def load_macro() -> pl.DataFrame: ...
```

Each function builds a glob like `data/processed/monthly/vintage_year={2006,2007,...}/*.parquet`
and uses `pl.scan_parquet(...)` so partitions outside the year set are
never touched (predicate pushdown). `load_monthly(lazy=True)` returns a
`LazyFrame` for users who want to chain filters before materializing —
critical for memory when working across many vintages.

A 5-line "how to use" example block at the top of the file shows the
common patterns: KM by vintage, Cox feature build, XGBoost training set.

---

## 6. Project layout

```
Project2/
├── data/                         (input — leave as-is)
│   ├── historical_data_*.zip
│   ├── file_layout.xlsx
│   ├── user_guide.pdf
│   ├── faq.pdf
│   └── processed/                (generated; gitignored)
├── scripts/
│   ├── prepare_data.py           # orchestrator
│   ├── convert_quarter.py        # zip → parquet
│   ├── derive_outcomes.py        # monthly → outcomes
│   └── fetch_macro.py            # FRED → parquet
├── src/
│   ├── credit_data.py            # user-facing loaders
│   └── schemas.py                # ORIG_COLS, MONTHLY_COLS, dtype maps, sentinels
├── notebooks/                    # research notebooks (later)
├── PLAN.md                       # this file
├── README.md                     # quickstart
├── requirements.txt
└── .gitignore                    # excludes data/, notebooks/.ipynb_checkpoints/
```

`requirements.txt` (minimal — keep install fast for new users):

```
polars>=1.0
pyarrow>=15
pandas>=2.0
numpy
tqdm
pandas-datareader      # macro only
lifelines              # modeling only
scikit-learn
xgboost
matplotlib
```

`README.md` quickstart (one-liner each):

```
# 1. install
pip install -r requirements.txt

# 2. one-time data prep (1-2 hours, ~8 GB output)
python scripts/prepare_data.py

# 3. use in any notebook
from src.credit_data import load_origination, load_monthly, load_outcomes
orig = load_origination(years=[2006, 2007, 2008])
```

---

## 7. Disk-space watchdog

Order of operations is chosen so peak disk is always safe:

1. **Before any work**, delete the already-unzipped
   `data/historical_data_2006/` directory (1.7 GB). User has authorized
   this; re-extraction from `historical_data_2006.zip` is always
   available if needed.
2. Convert quarter-by-quarter; each quarter writes ~100-300 MB of parquet
   and then frees its in-RAM frame.
3. After every full year, log current disk free; abort with a clear
   message if free space drops below 3 GB.
4. Never extract `.txt` files to disk — always stream from inside the
   `.zip` via `zipfile.ZipFile.open()`.

After full conversion, expected layout:

| Component | Disk | Notes |
|---|---|---|
| Source zips | 21 GB | unchanged, archival |
| `data/processed/origination/` | 0.2 GB | |
| `data/processed/monthly/` | 6-8 GB | dominant |
| `data/processed/outcomes/` | < 0.1 GB | |
| `data/processed/macro/` | < 1 MB | |
| **Total** | **~28-30 GB** | leaves ~7 GB free |

If disk pressure is still a concern, an opt-in `--drop-zips-after` flag
can delete each year's zip after its parquet partitions are written and
verified. Default off.

---

## 8. Validation / smoke test

After conversion, `scripts/validate.py` (small, fast) checks:

- Row counts in `origination/` match `wc -l`-equivalent counts of source
  txt for known quarters (within FRM filter — recompute filter against raw).
- Each `monthly/` partition's `loan_seq_num` set is a subset of the
  matching `origination/` partition's set.
- `outcomes/` row count == `origination/` row count per partition.
- Spot check: 2006Q1 origination row count after FRM filter matches the
  count of `term==360 & amort=='FRM'` in raw text.
- No nulls in primary keys.

Validation runs automatically at the end of `prepare_data.py` and prints
a one-line PASS/FAIL summary.

---

## 9. Resolved decisions (do not re-ask)

1. **2013**: `historical_data_2013.zip` is being re-downloaded by the
   user and will be present in `data/` before implementation runs.
   Treat 2006-2023 as a contiguous range; no special-case for 2013.
2. **Unzipped `data/historical_data_2006/` (1.7 GB)**: delete it as
   step 1 of the pipeline. Authorized.
3. **30-year FRM filter at conversion time**: yes. Apply
   `term_months == 360 AND amort_type == 'FRM'` while writing
   `origination/`, then propagate the kept `loan_seq_num` set to filter
   `monthly/`. Non-30y-FRM loans are not stored.
4. **Monthly partition key**: `vintage_year` (loan origination year).
   Vintage is the model input, so partitioning by it gives free
   predicate pushdown for vintage-stratified analyses.
5. **Loss / recovery columns** (monthly fields 14-23): keep. Required
   for the optional default-modeling extension (E). They compress to
   near-zero with ZSTD when null-dominated, so the cost is small.

---

## 10. Distribution to teammates

Code lives in a public GitHub repo. Prepared parquet is hosted as **GitHub
Release assets** attached to a tagged release. Teammates clone the repo and
run one download script — no Dropbox, no LFS, no per-user cloud credentials.

### 10.1 Bundle layout per release

GitHub caps a single release asset at 2 GB. Per-vintage tarballs come in
well under that (~300-700 MB each, since the inner parquet is already
ZSTD-compressed) and let users grab partial data.

```
release v1.0/
├── processed-2006.tar       # origination/, monthly/, outcomes/ for vintage 2006
├── processed-2007.tar
├── ...
├── processed-2023.tar       # ~17 tarballs total, 2013 included
├── macro.parquet            # global, ~50 KB
└── manifest.sha256          # one line per asset: "<sha256>  <filename>"
```

Each tarball, when extracted at the repo root, lays files into
`data/processed/origination/vintage_year=YYYY/`,
`data/processed/monthly/vintage_year=YYYY/`, and
`data/processed/outcomes/vintage_year=YYYY/`. So extracting all of them
reconstructs the full `data/processed/` tree.

Tarballs use `tar` (no `gzip`/`zstd` outer compression — parquet is
already compressed, so a second pass costs CPU and saves nothing).

### 10.2 Upload script (`scripts/publish_release.py`)

Run by the data maintainer (you), not by teammates. Steps:

1. Build per-vintage tarballs from `data/processed/` into a `dist/` dir.
2. Compute `manifest.sha256`.
3. Call `gh release create vX.Y --title ... --notes ... dist/*` to upload
   all assets in one shot. Requires `gh` CLI and `gh auth login` once.

### 10.3 Download script (`scripts/download_data.py`)

Run by every teammate as the second command of the quickstart. Behaviour:

- Reads the latest (or pinned) release tag from a `RELEASE_TAG` constant
  defined at the top of the file, so that pinning to a specific data
  version is a one-line change.
- Fetches `manifest.sha256` first, then downloads each tarball with
  `requests` + `tqdm` progress bar. GitHub asset URLs are anonymous for
  public repos — no token needed.
- Verifies SHA256 of each asset against the manifest before extraction.
- Idempotent: skips assets whose extracted `vintage_year=YYYY/` directory
  already exists and matches the recorded checksum.
- CLI flags: `--years 2006 2007 ...` to fetch a subset, `--force` to
  re-download.

### 10.4 README quickstart (final form)

```bash
# 1. install
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt

# 2. one-time data download (~8 GB, ~15-30 min)
python scripts/download_data.py

# 3. start modeling
jupyter lab notebooks/
```

The optional re-preparation path (download raw zips from the assignment's
Dropbox link, run `scripts/prepare_data.py`) is documented as an
"advanced / contributor" section of the README — not the default path.

### 10.5 Repo size hygiene

`.gitignore` excludes `data/`, `dist/`, `notebooks/.ipynb_checkpoints/`,
`__pycache__/`, `*.parquet`, `*.tar`. The git repo itself stays under
~5 MB (code + plan + readme + small static assets only). All bulk data
moves through Releases.

### 10.6 Updating the dataset later

If the schema or filter changes, bump the release tag (`v1.1`, `v2.0`),
re-run `publish_release.py`, and update `RELEASE_TAG` in
`download_data.py`. Teammates pull the latest commit and re-run
`download_data.py --force`. Old releases stay available so anyone needing
to reproduce a prior run can pin to an older tag.

---

## 11. Implementation checklist for the next agent

A short-prompt brief like *"implement PLAN.md"* should be enough. The order:

1. [ ] Create `src/schemas.py` with column lists, dtype maps, sentinel maps.
2. [ ] Implement `scripts/convert_quarter.py` (one quarter end-to-end, test on 2006Q1 first — already unzipped, easy to spot-check).
3. [ ] Add zip-streaming wrapper so the same code path works on the still-zipped years.
4. [ ] Implement `scripts/derive_outcomes.py`.
5. [ ] Implement `scripts/fetch_macro.py` (FRED).
6. [ ] Implement `scripts/prepare_data.py` orchestrator with `tqdm` and disk-space watchdog.
7. [ ] Implement `src/credit_data.py` loaders.
8. [ ] Implement `scripts/validate.py`.
9. [ ] Run end-to-end on 2006 only; verify outputs and disk usage; then full run.
10. [ ] Implement `scripts/publish_release.py` (tarball builder + `gh release create`).
11. [ ] Implement `scripts/download_data.py` (asset fetch + SHA256 verify + extract).
12. [ ] Write `README.md` with both the teammate quickstart and the maintainer re-prepare path.
13. [ ] Initialize git repo, push to a public GitHub repo, cut release `v1.0`, run `publish_release.py`, and verify a clean download from a fresh checkout.
