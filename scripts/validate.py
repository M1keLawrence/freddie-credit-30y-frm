"""
Sanity-check the converted Parquet partitions.

Checks:
  1. Every monthly partition's loan_seq_num set is a subset of the
     matching origination partition's set.
  2. Every outcomes partition's row count equals its origination partition's.
  3. Spot check: 2006Q1 origination 30y FRM count is recomputed from the
     raw zip and compared against the parquet row count.
  4. No nulls in primary keys (origination.loan_seq_num,
     monthly.loan_seq_num, monthly.month).

Outputs a single PASS/FAIL line and exits non-zero on FAIL.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import ORIG_COLS, transform_origination  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
logger = logging.getLogger(__name__)


def _list_partitions(table: str) -> list[tuple[int, int, Path]]:
    root = PROCESSED_DIR / table
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("vintage_year="):
            continue
        year = int(d.name.split("=")[1])
        for p in sorted(d.glob("part-Q*.parquet")):
            q = int(p.stem.split("Q")[-1])
            out.append((year, q, p))
    return out


def _check_subset(year: int, quarter: int) -> tuple[bool, str]:
    orig_p = PROCESSED_DIR / "origination" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    monthly_p = PROCESSED_DIR / "monthly" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"

    orig_ids = pl.read_parquet(orig_p, columns=["loan_seq_num"])
    monthly_ids = pl.read_parquet(monthly_p, columns=["loan_seq_num"]).unique()
    diff = monthly_ids.join(orig_ids, on="loan_seq_num", how="anti").height
    if diff > 0:
        return False, f"{year}Q{quarter}: {diff} loan_seq_num in monthly NOT in origination"
    return True, f"{year}Q{quarter}: subset OK ({orig_ids.height:,} loans / {monthly_ids.height:,} active)"


def _check_outcomes_count(year: int, quarter: int) -> tuple[bool, str]:
    orig_p = PROCESSED_DIR / "origination" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    out_p = PROCESSED_DIR / "outcomes" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    if not out_p.exists():
        return True, f"{year}Q{quarter}: outcomes not yet derived (skipping)"
    n_orig = pl.read_parquet(orig_p, columns=["loan_seq_num"]).height
    n_out = pl.read_parquet(out_p, columns=["loan_seq_num"]).height
    if n_orig != n_out:
        return False, f"{year}Q{quarter}: outcomes={n_out:,} != origination={n_orig:,}"
    return True, f"{year}Q{quarter}: outcomes count OK ({n_out:,})"


def _check_pk_nulls(year: int, quarter: int) -> tuple[bool, str]:
    orig_p = PROCESSED_DIR / "origination" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    monthly_p = PROCESSED_DIR / "monthly" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    n_o = pl.read_parquet(orig_p, columns=["loan_seq_num"]).null_count().row(0)[0]
    df_m = pl.read_parquet(monthly_p, columns=["loan_seq_num", "month"])
    n_m1 = df_m.select(pl.col("loan_seq_num").is_null().sum()).item()
    n_m2 = df_m.select(pl.col("month").is_null().sum()).item()
    if n_o or n_m1 or n_m2:
        return False, (
            f"{year}Q{quarter}: nulls in PK — orig.loan_seq_num={n_o} "
            f"monthly.loan_seq_num={n_m1} monthly.month={n_m2}"
        )
    return True, f"{year}Q{quarter}: PK null-free"


def _spot_check_2006q1() -> tuple[bool, str]:
    """Recompute the 30y-FRM origination row count from the raw zip and compare."""
    outer = DATA_DIR / "historical_data_2006.zip"
    inner_name = "historical_data_2006Q1.zip"
    if not outer.exists():
        return True, "spot check: 2006 zip absent — skipping"
    with zipfile.ZipFile(outer) as zo:
        with zo.open(inner_name) as fh:
            inner_bytes = fh.read()
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zi:
        with zi.open("historical_data_2006Q1.txt") as fh:
            raw = fh.read()
    df = pl.read_csv(
        io.BytesIO(raw),
        separator="|",
        has_header=False,
        new_columns=ORIG_COLS,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    df = transform_origination(df, 2006, 1).filter(
        (pl.col("term_months") == 360) & (pl.col("amort_type") == "FRM")
    )
    expected = df.height

    parquet_p = PROCESSED_DIR / "origination" / "vintage_year=2006" / "part-Q1.parquet"
    actual = pl.read_parquet(parquet_p, columns=["loan_seq_num"]).height
    if expected != actual:
        return False, f"spot check 2006Q1: parquet={actual:,} vs raw 30yFRM={expected:,}"
    return True, f"spot check 2006Q1: matched ({actual:,})"


def validate_all(years_filter: list[int] | None = None) -> bool:
    parts = [p for p in _list_partitions("monthly") if years_filter is None or p[0] in set(years_filter)]
    if not parts:
        logger.warning("No monthly partitions to validate.")
        return False

    all_ok = True
    for year, q, _ in parts:
        for ok, msg in (_check_subset(year, q), _check_outcomes_count(year, q), _check_pk_nulls(year, q)):
            (logger.info if ok else logger.error)(msg)
            all_ok = all_ok and ok

    ok, msg = _spot_check_2006q1()
    (logger.info if ok else logger.error)(msg)
    all_ok = all_ok and ok

    logger.info("VALIDATION %s", "PASS" if all_ok else "FAIL")
    return all_ok


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"validate_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="*", default=None)
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    ok = validate_all(args.years)
    sys.exit(0 if ok else 2)
