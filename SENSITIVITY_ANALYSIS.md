# Cox Model — Sensitivity Analysis Guide

All commands run from the repo root: `/workspaces/Mortgage Project/data-collection-repo-clone/`

---

## Quick reference: new flags

| Flag | Script | Effect |
|---|---|---|
| `--save-model PATH` | both | Pickle the fitted model to PATH after fitting |
| `--load-model PATH` | both | Load a cached model; skip data load + fit entirely |
| `--skip-tests` | partB_cox.py only | Skip Schoenfeld + log-log PH tests (faster) |
| `--skip-macro` | partB_cox.py only | Skip the macro-covariate model |
| `--no-plots` | partB_cox_team.py only | Skip diagnostic plots |
| `--penalizer FLOAT` | both | L2 regularization strength (default: 0.01) |
| `--sample INT` | partB_cox.py only | Max loans used for fitting (default: 2,000,000) |
| `--years INT ...` | partB_cox.py only | Vintage years to train on (default: 2006–2022) |

---

## Typical sensitivity analysis workflow

### Step 1 — Fit the baseline model once and cache it

```bash
# Full run with all diagnostics. Saves model to disk.
python scripts/partB_cox.py \
    --save-model figures/cox_static_baseline.pkl

# Team model version (needed for prediction CSVs)
python scripts/partB_cox_team.py \
    --save-model figures/team_model_baseline.pkl
```

Outputs written to `figures/`:
- `partB_coef_table.csv` — coefficients
- `partB_hazard_ratios.png` — forest plot
- `partB_baseline_hazard.png` — Breslow h0(t) and S0(t)
- `partB_schoenfeld.png` / `partB_schoenfeld_results.csv` — PH test
- `partB_loglog_fico/ltv/orig_rate.png` — log-log PH check plots

---

### Step 2 — Re-run plots from a cached model (no refitting)

```bash
# Regenerates coef table, forest plot, baseline hazard instantly
python scripts/partB_cox.py \
    --load-model figures/cox_static_baseline.pkl

# PH tests are automatically skipped when loading from cache
# (they require the training data in memory)
```

---

### Step 3 — Penalizer sensitivity sweep

Vary the L2 regularization strength. Use `--skip-tests` to avoid the slow
Schoenfeld computation on each variant. Compare the resulting coef tables.

```bash
python scripts/partB_cox.py --penalizer 0.001 --skip-tests \
    --save-model figures/cox_p0001.pkl

python scripts/partB_cox.py --penalizer 0.01  --skip-tests \
    --save-model figures/cox_p001.pkl

python scripts/partB_cox.py --penalizer 0.05  --skip-tests \
    --save-model figures/cox_p005.pkl

python scripts/partB_cox.py --penalizer 0.1   --skip-tests \
    --save-model figures/cox_p01.pkl
```

Compare coefficient tables across runs:
```
figures/partB_coef_table.csv   (overwritten each run — rename or copy between runs)
```

Tip: rename after each run to avoid overwriting, e.g.:
```bash
python scripts/partB_cox.py --penalizer 0.001 --skip-tests && \
    cp figures/partB_coef_table.csv figures/partB_coef_p0001.csv
```

---

### Step 4 — Sample size sensitivity

Check that results are stable at 2M loans vs. smaller subsamples.

```bash
python scripts/partB_cox.py --sample 200000  --skip-tests \
    --save-model figures/cox_n200k.pkl

python scripts/partB_cox.py --sample 500000  --skip-tests \
    --save-model figures/cox_n500k.pkl

python scripts/partB_cox.py --sample 2000000 --skip-tests \
    --save-model figures/cox_n2m.pkl
```

---

### Step 5 — Vintage year sensitivity

Check model stability across different training windows.

```bash
# Train on 2006-2018 only (pre-2019)
python scripts/partB_cox.py \
    --years $(seq 2006 2018) --skip-tests \
    --save-model figures/cox_2006_2018.pkl

# Train on 2010-2022 (post-crisis)
python scripts/partB_cox.py \
    --years $(seq 2010 2022) --skip-tests \
    --save-model figures/cox_2010_2022.pkl
```

---

### Step 6 — Static vs. macro model comparison

```bash
# Static only (fast — skips macro model fit)
python scripts/partB_cox.py --skip-macro --skip-tests

# Both models (default)
python scripts/partB_cox.py --skip-tests
```

---

### Step 7 — Team model: re-score from cached model

Once `team_model_baseline.pkl` exists, re-generating all prediction CSVs
takes ~10 minutes (scoring only) instead of ~30 minutes (fit + score).

```bash
python scripts/partB_cox_team.py \
    --load-model figures/team_model_baseline.pkl \
    --no-plots
```

To test a different penalizer and re-score:
```bash
python scripts/partB_cox_team.py --penalizer 0.05 \
    --save-model figures/team_model_p005.pkl

python scripts/partB_cox_team.py \
    --load-model figures/team_model_p005.pkl \
    --no-plots
```

---

## Notes

- Cached `.pkl` files are standard Python pickle — load with `pickle.load()` to inspect the `CoxPHFitter` object directly if needed.
- `--load-model` always skips PH tests (Schoenfeld, log-log) since those require the training DataFrame in memory. Run PH tests once on the baseline model and consider them fixed for the sensitivity study.
- The macro model (`partB_macro_*` outputs) is not cached — it re-fits every run. Use `--skip-macro` if you only need the static model results.
