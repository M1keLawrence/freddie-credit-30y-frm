"""
User-facing loaders for the processed Freddie Mac dataset.

All four functions return Polars DataFrames (or LazyFrames where useful)
and rely on parquet partition pruning so that asking for a subset of
years only touches that subset of partitions on disk.

Quick examples:

    from src.credit_data import load_origination, load_monthly, load_outcomes, load_macro

    # 1. KM curves by vintage
    out = load_outcomes(years=[2006, 2007, 2008])

    # 2. Cox feature build
    orig = load_origination(years=[2006, 2007],
                            columns=["loan_seq_num", "fico", "ltv", "dti", "orig_rate"])

    # 3. XGBoost training set with macro joined in
    monthly = load_monthly(years=[2010], columns=["loan_seq_num", "month",
                                                   "current_rate", "actual_upb",
                                                   "dq_status", "loan_age"])
    macro = load_macro()
    train = monthly.join(macro, on="month", how="left")

    # 4. Streaming for very wide vintages
    lf = load_monthly(years=list(range(2006, 2024)), lazy=True)
    big = lf.filter(pl.col("dq_status") != "0").collect(streaming=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _vintage_globs(table: str, years: Sequence[int] | None) -> list[str]:
    """Return parquet path globs for the requested vintages.

    `pl.scan_parquet` will accept a list of paths and stitch them into one
    LazyFrame, so this gives us cheap predicate pushdown.
    """
    root = PROCESSED_DIR / table
    if years is None:
        return [str(p) for p in sorted(root.glob("vintage_year=*/*.parquet"))]
    return [
        str(p)
        for y in sorted(set(years))
        for p in sorted((root / f"vintage_year={y}").glob("*.parquet"))
    ]


def _load(
    table: str,
    years: Iterable[int] | None,
    columns: Sequence[str] | None,
    lazy: bool,
    extra_filter: pl.Expr | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    paths = _vintage_globs(table, list(years) if years is not None else None)
    if not paths:
        raise FileNotFoundError(
            f"No '{table}' partitions found under {PROCESSED_DIR / table}. "
            "Did you run scripts/prepare_data.py?"
        )
    lf = pl.scan_parquet(paths)
    if extra_filter is not None:
        lf = lf.filter(extra_filter)
    if columns is not None:
        lf = lf.select(columns)
    return lf if lazy else lf.collect()


def load_origination(
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Load the origination table (one row per loan)."""
    return _load("origination", years, columns, lazy)


def load_monthly(
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
    loan_ids: Sequence[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Load the monthly performance panel.

    For wide vintages or full-history runs, prefer lazy=True and chain
    your own filters so polars can push them into the parquet reader.
    """
    extra = None
    if loan_ids is not None:
        # In-memory loan_id filter — fine for ad-hoc 1k-1M id slices.
        ids_df = pl.DataFrame({"loan_seq_num": list(loan_ids)})

        paths = _vintage_globs("monthly", list(years) if years is not None else None)
        lf = pl.scan_parquet(paths)
        lf = lf.join(ids_df.lazy(), on="loan_seq_num", how="semi")
        if columns is not None:
            lf = lf.select(columns)
        return lf if lazy else lf.collect()

    return _load("monthly", years, columns, lazy, extra_filter=extra)


def load_outcomes(
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Load the per-loan outcome / event table."""
    return _load("outcomes", years, columns, lazy)


def load_macro(lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Load the FRED monthly macro table. Single-file, small."""
    path = PROCESSED_DIR / "macro" / "fred_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing macro file at {path}. Run scripts/fetch_macro.py."
        )
    lf = pl.scan_parquet(path)
    return lf if lazy else lf.collect()
