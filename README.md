# Mortgage Default / Prepayment Project

Survival-analysis and ML workflows over the Freddie Mac single-family
loan-level dataset, restricted to 30-year fixed-rate mortgages.

## Vintage coverage

The processed dataset spans **2006 Q1 → 2025 Q3**, with two gaps:

- **2023 Q3 and 2023 Q4 are absent** from the upstream Freddie Mac
  archive available to us. Any cross-vintage analysis that would have
  used 2023 H2 must end on 2023 Q2.
- **2025 ends at Q3.**

Because 2023 is partial, **2024 and 2025 are best treated as a held-out
test set** (or as evaluation data for models trained on 2006-2022)
rather than mixed into in-sample estimation.

This repo ships:

- `scripts/` – data-pipeline entry points (zip → Parquet, outcomes, FRED) **and analysis scripts**.
- `src/credit_data.py` – one-line loaders for the processed tables.
- `src/schemas.py` – column / dtype / sentinel definitions.
- [`DATA.md`](DATA.md) – per-column reference, join keys, sentinel map.
- [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) – 5-section tour of the loaders with plots.

## Analysis scripts (IR&C Assignment 3)

### Part A — Exploratory Survival Analysis (`notebooks/partA_survival.ipynb`)

Kaplan-Meier survival estimates, implied hazard rates, and stratified curves across
FICO, LTV, DTI, orig_rate, loan purpose, channel, and vintage. Also includes:
- Aalen-Johansen competing-risks CIF (prepayment vs. default vs. other)
- Bin-free univariate Cox ranking vs. log-rank test comparison
- Age × calendar-year heatmap of monthly prepayment rates

Output figures are written to `figures/partA_*.png`.

### Part B — Classical Cox Modeling (`scripts/partB_cox.py`)

Cox proportional hazards model for prepayment. Runs two models:

1. **Static model** — origination features only (FICO, LTV, DTI, orig_rate, loan purpose, channel, n_borrowers, vintage_year)
2. **Macro model** — adds FRED covariates joined at origination month: rate incentive (orig_rate − MORTGAGE30US), mortgage/Treasury spread, unemployment, log HPI, log CPI

Also produces:
- Forest plots of hazard ratios with 95% CI
- Breslow baseline hazard and survival curves
- Schoenfeld residuals test (PH assumption)
- Log-log survival plots for FICO, LTV, orig_rate

```bash
# Full run — 2006-2022 in-sample, 2M loan sample
python scripts/partB_cox.py

# Quick test on a small subset
python scripts/partB_cox.py --years 2006 2007 --sample 50000

# Skip macro model
python scripts/partB_cox.py --skip-macro
```

Output figures are written to `figures/partB_*.png` and `figures/partB_*.csv`.

**Train/test split:** 2006–2022 in-sample; 2024–2025 held out (2023 H2 absent upstream).

The pipeline output is ~6–10 GB of ZSTD-compressed Parquet, partitioned by
`vintage_year`. Loaders use predicate pushdown so vintage-stratified work
only touches the partitions it needs.

## Quickstart (teammates)

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt

# Pulls per-vintage tarballs (~8 GB total, 15-30 min) and the macro file.
python scripts/download_data.py

# Optional: subset
python scripts/download_data.py --years 2006 2007 2008
```

Then in any notebook / script:

```python
from src.credit_data import load_origination, load_monthly, load_outcomes, load_macro

orig    = load_origination(years=[2006, 2007, 2008])
out     = load_outcomes(years=[2006, 2007, 2008])
macro   = load_macro()
monthly = load_monthly(years=[2010], columns=["loan_seq_num", "month",
                                              "current_rate", "actual_upb",
                                              "dq_status", "loan_age"])
```

For wide vintages, prefer `load_monthly(..., lazy=True)` so polars can
push your filters into the parquet reader.

## Re-preparing from raw zips (maintainer / contributor)

You only need this if you're rebuilding the dataset (changing the schema
or the FRM filter). The output of `prepare_data.py` is what gets uploaded
as a release; teammates do not need to run this.

```bash
# 1. Drop the raw zips into data/ — see assignment Dropbox link.
ls data/historical_data_*.zip

# 2. Run the full pipeline (1-2 hours, ~8 GB output).
python scripts/prepare_data.py

# 3. Validate.
python scripts/validate.py
```

Useful flags:

- `--years 2006 2007` – process a subset.
- `--force` – overwrite existing parquet partitions.
- `--no-macro` – skip the FRED fetch.
- `--skip-validate` – skip the post-run checks.
- `--keep-legacy-2006` – keep the pre-extracted `data/historical_data_2006/`
  directory in place (default behaviour: delete it to free 1.7 GB).

Logs are tee'd to `logs/prepare_data_<unix-ts>.log` and a per-quarter
JSON summary is written to `logs/prepare_data_<unix-ts>.summary.json`.

## Cutting a release (maintainer)

```bash
python scripts/publish_release.py --tag v1.0 --upload   # build + gh release create
python scripts/publish_release.py --tag v1.0            # build only
```

The script writes per-vintage tarballs to `dist/`, computes
`dist/manifest.sha256`, and (with `--upload`) calls
`gh release create v1.0 --title v1.0 --notes ... dist/*`. Requires the
`gh` CLI to be installed and `gh auth login` to have been run once.

After uploading, bump `RELEASE_TAG` and `REPO_SLUG` in
`scripts/download_data.py` so teammates pick up the new tag automatically.

## Output layout

```
data/processed/
├── origination/        # one row per loan
│   └── vintage_year=2006/part-Q1.parquet
├── monthly/            # one row per loan-month (the big one)
│   └── vintage_year=2006/part-Q1.parquet
├── outcomes/           # one row per loan, derived (event_type, event_time_months, ...)
│   └── vintage_year=2006/part-Q1.parquet
└── macro/
    └── fred_monthly.parquet
```

See `PLAN.md` for the full spec (schema, sentinels, event derivation,
disk-space watchdog).
