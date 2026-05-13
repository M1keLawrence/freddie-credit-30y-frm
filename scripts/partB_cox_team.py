"""
Part B — Cox PH model fitted on team-aligned parquet files

Trains on team_train.parquet (vintages 2006-2013, ~1.95M loans) using the full
feature set agreed with the team.  Produces probability-of-prepayment predictions
for three evaluation files at their respective horizons:

  team_test.parquet       → horizon 60 months  → cox_test_predictions.csv
  team_backtest_a.parquet → horizon 24 months  → cox_backtest_a_predictions.csv
  team_backtest_b.parquet → horizon 12 months  → cox_backtest_b_predictions.csv

All outputs go to figures/.

Usage:
    python scripts/partB_cox_team.py
    python scripts/partB_cox_team.py --no-plots
    python scripts/partB_cox_team.py --penalizer 0.05
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR    = Path("/workspaces/Mortgage Project/teammate-data")

sys.path.insert(0, str(REPO_ROOT))
FIGURES_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# ── Configuration ──────────────────────────────────────────────────────────────

RANDOM_SEED      = 42
PENALIZER        = 0.01
SCORE_BATCH_SIZE = 100_000   # rows per prediction batch — avoids OOM on large eval files
DURATION_COL     = "event_time_months"
EVENT_COL        = "prepay_observed"

# Categorical columns that will be one-hot encoded
CAT_COLS = [
    "loan_purpose",        # C/N/P
    "channel",             # B/C/R/T
    "state",               # 54 values incl. territories
    "occupancy",           # I/P/S
    "first_time_homebuyer",# Y/N
    "prop_type",           # CO/CP/MH/PU/SF
]

# All feature columns in the team parquet files (excludes duration/event/label/id)
FEATURE_COLS = [
    "fico", "ltv", "dti", "orig_rate",
    "loan_purpose", "channel", "n_borrowers", "vintage_year",
    "state", "occupancy", "orig_upb",
    "first_time_homebuyer", "prop_type", "mi_pct", "n_units",
    "GS10", "UNRATE",
    "rate_incentive_orig", "mort_treasury_spread",
    "log_hpi", "log_cpi",
]

# Evaluation files, their prediction horizons, and output CSV names
EVAL_FILES = [
    ("team_test.parquet",       60, "cox_test_predictions.csv"),
    ("team_backtest_a.parquet", 24, "cox_backtest_a_predictions.csv"),
    ("team_backtest_b.parquet", 12, "cox_backtest_b_predictions.csv"),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_parquet(fname: str) -> pd.DataFrame:
    path = DATA_DIR / fname
    print(f"  reading {path.name} ...", end=" ", flush=True)
    df = pd.read_parquet(path)
    print(f"{len(df):,} rows, {df.shape[1]} cols")
    return df


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_train(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """
    Prepare the training DataFrame for CoxPHFitter.

    Returns
    -------
    processed_df        : pandas DataFrame ready for cph.fit()
    impute_map          : {col: median} for numeric nulls — apply to eval data
    train_feature_cols  : ordered list of dummy-expanded feature columns
                          (everything except DURATION_COL and EVENT_COL)
    """
    keep = [DURATION_COL, EVENT_COL] + FEATURE_COLS
    df = df[[c for c in keep if c in df.columns]].copy()

    # Coerce all numeric columns to float64 (some arrive as UInt8)
    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS and c in df.columns]
    for col in num_cols:
        df[col] = df[col].astype("float64")

    # Impute numeric nulls with column median; record fills for eval data
    impute_map: dict[str, float] = {}
    for col in num_cols:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            median_val = float(df[col].median())
            impute_map[col] = median_val
            df[col] = df[col].fillna(median_val)
            print(f"    imputing {col}: {n_null:,} nulls → median {median_val:.4f}")

    # Drop rows with missing duration or event (should be zero after the above)
    before = len(df)
    df = df.dropna(subset=[DURATION_COL, EVENT_COL] + num_cols)
    df = df[df[DURATION_COL] > 0]
    if len(df) < before:
        print(f"    dropped {before - len(df):,} rows (null duration/event)")

    # Ensure categoricals are strings so get_dummies produces consistent names
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode; drop_first=True removes the reference category
    df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns],
                        drop_first=True, dtype=float)

    train_feature_cols = [c for c in df.columns if c not in [DURATION_COL, EVENT_COL]]

    print(f"  train: {len(df):,} rows, {len(train_feature_cols)} features, "
          f"{int(df[EVENT_COL].sum()):,} prepayments ({df[EVENT_COL].mean():.2%})")
    return df, impute_map, train_feature_cols


def preprocess_eval(
    df: pd.DataFrame,
    impute_map: dict[str, float],
    train_feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare an evaluation file for prediction.

    Returns
    -------
    feat_df   : DataFrame aligned to train_feature_cols, ready for predict_survival_function
    loan_ids  : numpy array of loan_seq_num in the same row order
    """
    loan_ids = df["loan_seq_num"].to_numpy()

    feat_df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()

    # Coerce numerics to float64
    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS and c in feat_df.columns]
    for col in num_cols:
        feat_df[col] = feat_df[col].astype("float64")
        n_null = int(feat_df[col].isna().sum())
        if n_null > 0:
            fill = impute_map.get(col, float(feat_df[col].median()))
            feat_df[col] = feat_df[col].fillna(fill)

    for col in CAT_COLS:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].astype(str)

    # Expand dummies — use drop_first=False here so no category is silently lost
    # before reindex; the reindex to train_feature_cols handles reference-category
    # alignment correctly (missing columns → 0, extra columns → dropped)
    feat_df = pd.get_dummies(
        feat_df,
        columns=[c for c in CAT_COLS if c in feat_df.columns],
        drop_first=False,
        dtype=float,
    )
    feat_df = feat_df.reindex(columns=train_feature_cols, fill_value=0.0)

    return feat_df, loan_ids


# ── Cox fitting ────────────────────────────────────────────────────────────────

def fit_cox(df: pd.DataFrame, penalizer: float = PENALIZER) -> CoxPHFitter:
    print(f"\n{'='*60}")
    print(f"  Fitting CoxPHFitter  n={len(df):,}  events={int(df[EVENT_COL].sum()):,}")
    print(f"  features={df.shape[1] - 2}  penalizer={penalizer}")
    print(f"{'='*60}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=DURATION_COL, event_col=EVENT_COL)
    cph.print_summary(decimals=4, style="ascii")
    return cph


# ── Prediction ─────────────────────────────────────────────────────────────────

def score_eval_file(
    cph: CoxPHFitter,
    fname: str,
    horizon: int,
    impute_map: dict[str, float],
    train_feature_cols: list[str],
    output_name: str,
) -> None:
    """Score one eval file and write CSV."""
    print(f"\n  Scoring {fname}  (horizon={horizon}m) ...")
    raw = load_parquet(fname)
    feat_df, loan_ids = preprocess_eval(raw, impute_map, train_feature_cols)
    del raw
    gc.collect()

    # Score in batches to avoid OOM (large eval files × internal matrix alloc)
    probs = []
    n = len(feat_df)
    for start in range(0, n, SCORE_BATCH_SIZE):
        batch = feat_df.iloc[start : start + SCORE_BATCH_SIZE]
        sf_batch = cph.predict_survival_function(batch, times=[horizon])
        probs.append(1.0 - sf_batch.loc[horizon].values)
        if (start // SCORE_BATCH_SIZE) % 10 == 0:
            print(f"    scored {min(start + SCORE_BATCH_SIZE, n):,} / {n:,}", flush=True)
    prob_prepay = np.concatenate(probs)

    col_name = f"cox_prob_prepay_{horizon}mo"
    out = pd.DataFrame({"loan_seq_num": loan_ids, col_name: prob_prepay})

    out_path = FIGURES_DIR / output_name
    out.to_csv(out_path, index=False)

    print(f"    written: {out_path}  ({len(out):,} rows)")
    print(f"    {col_name}: mean={prob_prepay.mean():.4f}"
          f"  p10={np.percentile(prob_prepay, 10):.4f}"
          f"  p50={np.percentile(prob_prepay, 50):.4f}"
          f"  p90={np.percentile(prob_prepay, 90):.4f}")


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def plot_hazard_ratios(cph: CoxPHFitter) -> None:
    s = cph.summary.copy().sort_values("exp(coef)", ascending=True)
    hr = s["exp(coef)"].values
    lo = s["exp(coef) lower 95%"].values
    hi = s["exp(coef) upper 95%"].values
    names = s.index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(6, len(s) * 0.38)))
    y = np.arange(len(s))
    colors = ["tab:red" if h > 1 else "tab:blue" for h in hr]
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr],
                fmt="none", ecolor="grey", capsize=3, lw=0.8, zorder=1)
    ax.scatter(hr, y, c=colors, zorder=2, s=32)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    for xi, yi, h in zip(hr, y, hr):
        ax.text(max(hi) * 1.01, yi, f"{h:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Hazard Ratio  exp(coef)  with 95% CI")
    ax.set_title("Cox PH — Team Model Hazard Ratios\n"
                 "Train: 2006-2013, all team features, 95% CI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_path = FIGURES_DIR / "partB_team_hazard_ratios.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path.name}")


def plot_baseline_hazard(cph: CoxPHFitter) -> None:
    bh = cph.baseline_hazard_.copy()
    bs = cph.baseline_survival_.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(bh.index, bh["baseline hazard"].values, lw=1.2, color="tab:blue")
    axes[0].set_xlabel("Months from origination")
    axes[0].set_ylabel("Baseline hazard  h₀(t)")
    axes[0].set_title("Breslow baseline hazard")
    axes[0].grid(alpha=0.3)

    axes[1].plot(bs.index, bs["baseline survival"].values, lw=1.2, color="tab:orange")
    axes[1].set_xlabel("Months from origination")
    axes[1].set_ylabel("S₀(t)  =  exp(−H₀(t))")
    axes[1].set_title("Baseline survival function")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Cox PH (team model) — Breslow Baseline Hazard", y=1.02)
    fig.tight_layout()
    save_path = FIGURES_DIR / "partB_team_baseline_hazard.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part B Cox — team-aligned parquet files")
    p.add_argument("--penalizer", type=float, default=PENALIZER,
                   help=f"L2 penalizer for CoxPHFitter (default: {PENALIZER})")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip diagnostic plots (faster scoring-only run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── [1/3] Fit Cox on training data ────────────────────────────────────────
    print("\n[1/3] Loading and preprocessing team_train.parquet ...")
    train_raw = load_parquet("team_train.parquet")
    train_df, impute_map, train_feature_cols = preprocess_train(train_raw)
    del train_raw
    gc.collect()

    cph = fit_cox(train_df, penalizer=args.penalizer)

    coef_path = FIGURES_DIR / "partB_team_coef_table.csv"
    cph.summary.to_csv(coef_path)
    print(f"\n  coefficient table: {coef_path}")

    if not args.no_plots:
        plot_hazard_ratios(cph)
        plot_baseline_hazard(cph)

    del train_df
    gc.collect()

    # ── [2/3] Score all evaluation files ──────────────────────────────────────
    print("\n[2/3] Scoring evaluation files ...")
    for fname, horizon, output_name in EVAL_FILES:
        score_eval_file(cph, fname, horizon, impute_map, train_feature_cols, output_name)

    # ── [3/3] Summary ─────────────────────────────────────────────────────────
    print("\n[3/3] Done. Prediction CSVs:")
    for _, _, output_name in EVAL_FILES:
        p = FIGURES_DIR / output_name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
