"""
Part B — Classical Cox Proportional Hazards Model for Prepayment

(i)   Estimate a Cox PH model on static origination features.
(ii)  Interpret coefficients and hazard ratios (forest plot + table).
(iii) Test the proportional hazards assumption (Schoenfeld residuals +
      log-log survival plots for key covariates).
(iv)  Refit with macroeconomic covariates joined at origination month:
      mortgage rate, Treasury yield, unemployment, CPI, HPI.

All outputs are written to figures/partB_*.

Usage:
    python scripts/partB_cox.py
    python scripts/partB_cox.py --years 2006 2007 2008 --sample 200000
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from src.credit_data import load_loans, load_macro

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_YEARS    = list(range(2006, 2023))   # 2006-2022 in-sample
EVAL_YEARS     = [2024, 2025]              # held-out (2023 H2 absent upstream)
DEFAULT_SAMPLE = 2_000_000                   # max loans used for Cox fitting
RANDOM_SEED    = 42

DURATION_COL = "event_time_months"
EVENT_COL    = "prepay_observed"

# Static origination features for model (i)
STATIC_FEATURES = [
    "fico",                  # credit score         — higher → easier to refi → higher prepay
    "ltv",                   # loan-to-value        — higher → less equity → harder to refi
    "dti",                   # debt-to-income       — higher → harder to qualify for new loan
    "orig_rate",             # original interest rate — higher → bigger refi incentive later
    "orig_upb",              # original loan balance — larger loans have stronger refi incentive
    "loan_purpose",          # P=purchase, C=cash-out, N=no-cash refi
    "channel",               # R=retail, B=broker, T=correspondent, C=other
    "n_borrowers",           # 1 or 2 borrowers
    "n_units",               # 1–4 units
    "mi_pct",                # mortgage insurance % — affects refi feasibility
    "occupancy",             # O=owner, I=investor, S=second home
    "prop_type",             # SF/CO/PU/MH/CP
    "first_time_homebuyer",  # Y/N
    "state",                 # 50 states + territories — regional rate environment
    "vintage_year",          # origination year — absorbs rate-cycle era
]

# Macro series available in fred_monthly.parquet
MACRO_COLS = ["MORTGAGE30US", "GS10", "UNRATE", "CPIAUCSL", "CSUSHPISA"]

# Log-log PH check: covariates and their bin specs
LOGLOG_SPECS = {
    "fico":      {"breaks": [660, 700, 740, 780],    "labels": ["<660", "660-700", "700-740", "740-780", "780+"]},
    "ltv":       {"breaks": [60, 80, 90, 95],         "labels": ["<=60", "60-80", "80-90", "90-95", "95+"]},
    "orig_rate": {"breaks": [3.5, 4.5, 5.5, 6.5],    "labels": ["<3.5%", "3.5-4.5%", "4.5-5.5%", "5.5-6.5%", "6.5%+"]},
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_static(years: list[int], sample: int) -> pd.DataFrame:
    """Load origination + outcome columns; return sampled pandas DataFrame."""
    cols = (
        ["loan_seq_num", DURATION_COL, "event_type", "first_payment_date"]
        + STATIC_FEATURES
    )
    df = (
        load_loans(years=years, columns=cols)
        .with_columns(
            (pl.col("event_type") == "prepaid").cast(pl.Int8).alias(EVENT_COL)
        )
        .filter(
            pl.col(DURATION_COL).is_not_null()
            & (pl.col(DURATION_COL) > 0)
        )
        .drop(["event_type", "loan_seq_num", "first_payment_date"])
    )
    if df.height > sample:
        df = df.sample(n=sample, seed=RANDOM_SEED)
    print(f"  loaded {df.height:,} loans ({df[EVENT_COL].sum():,} prepayments)")
    return df.to_pandas()


def load_with_macro(years: list[int], sample: int) -> pd.DataFrame:
    """Load origination + outcome + macro at origination month."""
    cols = (
        ["loan_seq_num", DURATION_COL, "event_type", "first_payment_date"]
        + STATIC_FEATURES
    )
    loans = (
        load_loans(years=years, columns=cols)
        .with_columns(
            (pl.col("event_type") == "prepaid").cast(pl.Int8).alias(EVENT_COL)
        )
        .filter(
            pl.col(DURATION_COL).is_not_null()
            & (pl.col(DURATION_COL) > 0)
        )
        .drop("event_type")
    )

    mac = load_macro().with_columns(pl.col("month").cast(pl.Date))

    loans = (
        loans
        .join(mac.select(["month"] + MACRO_COLS),
              left_on="first_payment_date", right_on="month",
              how="left")
        .with_columns([
            # Rate incentive at origination: loan rate minus prevailing market rate.
            # Positive = loan rate above market = latent refi incentive.
            (pl.col("orig_rate") - pl.col("MORTGAGE30US")).alias("rate_incentive_orig"),
            # Rate spread: mortgage rate minus 10-year Treasury (risk premium proxy).
            (pl.col("MORTGAGE30US") - pl.col("GS10")).alias("mort_treasury_spread"),
            # HPI and CPI are levels; take log so coefficients scale naturally.
            pl.col("CSUSHPISA").log().alias("log_hpi"),
            pl.col("CPIAUCSL").log().alias("log_cpi"),
        ])
        .drop(["loan_seq_num", "first_payment_date",
               # Drop raw macro levels we replaced with derived features
               "MORTGAGE30US", "CSUSHPISA", "CPIAUCSL"])
    )

    if loans.height > sample:
        loans = loans.sample(n=sample, seed=RANDOM_SEED)

    print(f"  loaded {loans.height:,} loans with macro ({loans[EVENT_COL].sum():,} prepayments)")
    return loans.to_pandas()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and drop rows with nulls in any feature."""
    cat_cols = [c for c in ["loan_purpose", "channel", "occupancy", "prop_type",
                            "first_time_homebuyer", "state"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    feature_cols = [c for c in df.columns if c not in [DURATION_COL, EVENT_COL]]
    before = len(df)
    df = df.dropna(subset=feature_cols + [DURATION_COL])
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped:,} rows with nulls ({dropped/before:.1%})")
    return df


# ── Cox fitting ────────────────────────────────────────────────────────────────

def fit_cox(df: pd.DataFrame, label: str, penalizer: float = 0.01) -> CoxPHFitter:
    """Fit CoxPHFitter and print summary."""
    print(f"\n{'='*60}")
    print(f"  Cox PH — {label}")
    print(f"  n={len(df):,}   events={int(df[EVENT_COL].sum()):,}   "
          f"event_rate={df[EVENT_COL].mean():.3f}")
    print(f"{'='*60}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=DURATION_COL, event_col=EVENT_COL)
    cph.print_summary(model=label, decimals=4, style="ascii")
    return cph


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_hazard_ratios(cph: CoxPHFitter, title: str, save_path: Path) -> None:
    """Horizontal forest plot of hazard ratios with 95% CI."""
    s = cph.summary.copy().sort_values("exp(coef)", ascending=True)

    hr = s["exp(coef)"].values
    lo = s["exp(coef) lower 95%"].values
    hi = s["exp(coef) upper 95%"].values
    names = s.index.tolist()

    fig, ax = plt.subplots(figsize=(9, max(5, len(s) * 0.45)))
    y = np.arange(len(s))

    colors = ["tab:red" if h > 1 else "tab:blue" for h in hr]
    ax.errorbar(hr, y,
                xerr=[hr - lo, hi - hr],
                fmt="none", ecolor="grey", capsize=3, lw=1.0, zorder=1)
    ax.scatter(hr, y, c=colors, zorder=2, s=40)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")

    # Annotate HR values
    for xi, yi, h in zip(hr, y, hr):
        ax.text(max(hi) * 1.01, yi, f"{h:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Hazard Ratio  (exp(coef))  with 95% CI")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path.name}")


def plot_schoenfeld(cph: CoxPHFitter, df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    Schoenfeld residuals test for the PH assumption.
    Bar chart of -log10(p); red = violation at 5% level.
    Returns the test summary DataFrame.
    """
    results = proportional_hazard_test(cph, df, time_transform="rank")
    tdf = results.summary.copy()
    tdf["-log10(p)"] = -np.log10(tdf["p"].clip(lower=1e-300))
    tdf = tdf.sort_values("-log10(p)", ascending=True)

    colors = ["tab:red" if p < 0.05 else "steelblue" for p in tdf["p"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(tdf) * 0.45)))
    ax.barh(tdf.index, tdf["-log10(p)"], color=colors, alpha=0.82)
    ax.axvline(-np.log10(0.05), color="black", lw=1.0, ls="--",
               label=f"p = 0.05  (−log₁₀ = {-np.log10(0.05):.2f})")
    ax.set_xlabel("−log₁₀(p)   [red bars: PH assumption violated at 5%]")
    ax.set_title("Schoenfeld Residuals — Proportional Hazards Test\n"
                 "H₀: hazard ratio is constant over time")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print("\n  Schoenfeld residuals test:")
    print(tdf[["test_statistic", "p"]].to_string())
    print(f"  saved: {save_path.name}")
    return tdf


def plot_loglog(df: pd.DataFrame, covariate: str,
                breaks: list[float], labels: list[str],
                save_path: Path, sample_per_group: int = 100_000) -> None:
    """
    Log-log survival plot: log(-log(S(t))) vs log(t).
    Under PH, curves for different groups are parallel vertical shifts.
    Crossing or converging curves indicate PH violation for this covariate.
    """
    df = df[[DURATION_COL, EVENT_COL, covariate]].dropna().copy()
    df["__grp"] = pd.cut(
        df[covariate],
        bins=[-np.inf] + breaks + [np.inf],
        labels=labels,
    )
    df = df.dropna(subset=["__grp"])

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, len(labels)))

    for color, label in zip(cmap, labels):
        sub = df[df["__grp"] == label]
        if len(sub) > sample_per_group:
            sub = sub.sample(sample_per_group, random_state=RANDOM_SEED)
        if len(sub) < 200:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[DURATION_COL], sub[EVENT_COL], label=label)
        sf = kmf.survival_function_
        # Avoid log(0): keep S > 0 and t > 0
        sf = sf[(sf.index > 0) & (sf.iloc[:, 0] > 0)]
        loglog = np.log(-np.log(sf.iloc[:, 0].values))
        logt   = np.log(sf.index.values)
        ax.plot(logt, loglog, color=color, lw=1.4, label=f"{label}  (n={len(sub):,})")

    ax.set_xlabel("log(t)   [log of months from origination]")
    ax.set_ylabel("log(−log(S(t)))")
    ax.set_title(
        f"Log-Log Survival Plot — {covariate}\n"
        "Parallel lines → PH holds   |   Crossing/converging → PH violated"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path.name}")


def plot_baseline_hazard(cph: CoxPHFitter, title: str, save_path: Path) -> None:
    """Plot the Breslow-estimated baseline hazard and survival functions."""
    bh = cph.baseline_hazard_.copy()
    bs = cph.baseline_survival_.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(bh.index, bh["baseline hazard"].values, lw=1.2, color="tab:blue")
    axes[0].set_xlabel("Months from origination")
    axes[0].set_ylabel("Baseline hazard h₀(t)")
    axes[0].set_title("Breslow baseline hazard")
    axes[0].grid(alpha=0.3)

    axes[1].plot(bs.index, bs["baseline survival"].values, lw=1.2, color="tab:orange")
    axes[1].set_xlabel("Months from origination")
    axes[1].set_ylabel("S₀(t)  =  exp(−H₀(t))")
    axes[1].set_title("Baseline survival function")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Model cache helpers ─────────────────────────────────────────────────────────

def save_model(cph: CoxPHFitter, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cph, f)
    print(f"  model cached: {path}")


def load_model(path: str) -> CoxPHFitter:
    print(f"  loading cached model: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part B: Cox PH model for prepayment")
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="Vintage years to use (default: 2006-2022)")
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                   help="Max loans to sample for Cox fitting (default: 500,000)")
    p.add_argument("--penalizer", type=float, default=0.01,
                   help="L2 penalizer for CoxPHFitter (default: 0.1)")
    p.add_argument("--skip-macro", action="store_true",
                   help="Skip the macro-covariate model (step iv)")
    p.add_argument("--save-model", type=str, default=None, metavar="PATH",
                   help="Pickle the fitted static Cox model to PATH after fitting")
    p.add_argument("--load-model", type=str, default=None, metavar="PATH",
                   help="Load a pickled static Cox model; skips data load, fit, and PH tests")
    p.add_argument("--skip-tests", action="store_true",
                   help="Skip Schoenfeld + log-log PH tests (faster for sensitivity runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    years  = args.years or TRAIN_YEARS
    sample = args.sample

    # ── (i) & (ii): Static origination model ──────────────────────────────────
    if args.load_model:
        print("\n[1/4] Loading cached static model (skipping data load + fit) ...")
        cph_static = load_model(args.load_model)
        raw = None
        df  = None
    else:
        print("\n[1/4] Loading static origination data ...")
        raw = load_static(years, sample)
        df  = preprocess(raw.copy())

        cph_static = fit_cox(df, label="Static origination features",
                             penalizer=args.penalizer)

        if args.save_model:
            save_model(cph_static, args.save_model)

    # Save coefficient table
    coef_path = FIGURES_DIR / "partB_coef_table.csv"
    cph_static.summary.to_csv(coef_path)
    print(f"  saved: {coef_path.name}")

    # Forest plot — hazard ratios
    plot_hazard_ratios(
        cph_static,
        title="Cox PH — Prepayment Hazard Ratios\nStatic origination features, 95% CI",
        save_path=FIGURES_DIR / "partB_hazard_ratios.png",
    )

    # Baseline hazard (Breslow estimator)
    plot_baseline_hazard(
        cph_static,
        title="Cox PH — Breslow Baseline Hazard and Survival",
        save_path=FIGURES_DIR / "partB_baseline_hazard.png",
    )

    # ── (iii): Test PH assumption ──────────────────────────────────────────────
    run_tests = not (args.skip_tests or args.load_model)
    if run_tests:
        print("\n[2/4] Testing proportional hazards assumption ...")

        schoenfeld_df = plot_schoenfeld(
            cph_static, df,
            save_path=FIGURES_DIR / "partB_schoenfeld.png",
        )
        schoenfeld_df.to_csv(FIGURES_DIR / "partB_schoenfeld_results.csv")

        # Log-log survival plots for key numeric covariates
        print("\n  Log-log survival plots ...")
        for cov, spec in LOGLOG_SPECS.items():
            if cov not in raw.columns:
                continue
            plot_loglog(
                raw,
                covariate=cov,
                breaks=spec["breaks"],
                labels=spec["labels"],
                save_path=FIGURES_DIR / f"partB_loglog_{cov}.png",
            )
    else:
        print("\n[2/4] Skipping PH tests (--skip-tests or --load-model).")

    if df is not None:
        del df
    gc.collect()

    # ── (iv): Macro-covariate model ────────────────────────────────────────────
    if not args.skip_macro:
        print("\n[3/4] Loading data with macro covariates ...")
        raw_macro = load_with_macro(years, sample)
        df_macro  = preprocess(raw_macro.copy())

        cph_macro = fit_cox(df_macro, label="Static + macro at origination",
                            penalizer=args.penalizer)

        coef_macro_path = FIGURES_DIR / "partB_macro_coef_table.csv"
        cph_macro.summary.to_csv(coef_macro_path)
        print(f"  saved: {coef_macro_path.name}")

        plot_hazard_ratios(
            cph_macro,
            title=(
                "Cox PH — Prepayment Hazard Ratios\n"
                "Static origination + macro at origination, 95% CI"
            ),
            save_path=FIGURES_DIR / "partB_macro_hazard_ratios.png",
        )

        plot_baseline_hazard(
            cph_macro,
            title="Cox PH (macro model) — Breslow Baseline Hazard and Survival",
            save_path=FIGURES_DIR / "partB_macro_baseline_hazard.png",
        )

        # PH test on macro model too
        plot_schoenfeld(
            cph_macro, df_macro,
            save_path=FIGURES_DIR / "partB_macro_schoenfeld.png",
        )

        del df_macro, raw_macro
        gc.collect()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n[4/4] Done. Figures written to:", FIGURES_DIR)
    print("\n  Artifacts:")
    for f in sorted(FIGURES_DIR.glob("partB_*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
