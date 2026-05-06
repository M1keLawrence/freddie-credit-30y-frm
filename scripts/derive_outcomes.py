"""
Derive a per-loan outcome table from the monthly performance Parquet.

For each (vintage_year, quarter) partition the script:
  1. Loads the matching monthly partition lazily.
  2. Aggregates per loan_seq_num.
  3. Joins on origination to attach vintage_year/quarter.
  4. Classifies a survival-style event_type column.
  5. Writes one Parquet partition per vintage to data/processed/outcomes/.

Event classification (PLAN.md §2.3):

  - 'prepaid'         : final zero_balance_code == "01"
  - 'defaulted'       : ZBC in {03,06,09,15}  OR  max_dq_status >= 6 (D180+)
  - 'other_termination': any other terminal ZBC (e.g. "02","96")
  - 'censored'        : no terminal ZBC reached

Idempotent: skips a partition whose output exists and is non-empty unless
force=True.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

logger = logging.getLogger(__name__)


@dataclass
class OutcomesStats:
    year: int
    quarter: int
    n_loans: int
    n_prepaid: int
    n_defaulted: int
    n_other: int
    n_censored: int
    elapsed_sec: float
    skipped: bool = False


DEFAULT_ZBC = ["03", "06", "09", "15"]
OTHER_TERMINAL_ZBC = ["02", "96"]


def _classify_event() -> pl.Expr:
    return (
        pl.when(pl.col("final_zero_balance_code") == "01")
        .then(pl.lit("prepaid"))
        .when(pl.col("final_zero_balance_code").is_in(DEFAULT_ZBC))
        .then(pl.lit("defaulted"))
        .when(pl.col("max_dq_status") >= 6)
        .then(pl.lit("defaulted"))
        .when(pl.col("final_zero_balance_code").is_not_null())
        .then(pl.lit("other_termination"))
        .otherwise(pl.lit("censored"))
        .alias("event_type")
    )


def derive_outcomes_for_quarter(
    year: int,
    quarter: int,
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> OutcomesStats:
    t0 = time.time()
    monthly_path = processed_dir / "monthly" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    orig_path = processed_dir / "origination" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    out_path = processed_dir / "outcomes" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"

    if not monthly_path.exists() or not orig_path.exists():
        raise FileNotFoundError(
            f"Missing inputs for {year}Q{quarter}: monthly={monthly_path.exists()} "
            f"origination={orig_path.exists()}"
        )

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        logger.info("[%dQ%d] outcomes already present; skipping.", year, quarter)
        return OutcomesStats(year, quarter, 0, 0, 0, 0, 0, 0.0, skipped=True)

    logger.info("[%dQ%d] deriving outcomes from %s", year, quarter, monthly_path.name)

    monthly = pl.scan_parquet(monthly_path)

    # max_dq_status: cast dq_status (Utf8) to Int; "RA","XX",null all become null,
    # which max() then ignores.
    agg = (
        monthly.group_by("loan_seq_num")
        .agg(
            [
                pl.col("month").min().alias("first_obs_date"),
                pl.col("month").max().alias("last_obs_date"),
                pl.col("loan_age").max().alias("event_time_months"),
                pl.col("zb_code").drop_nulls().last().alias("final_zero_balance_code"),
                pl.col("dq_status").cast(pl.Int16, strict=False).max().alias("max_dq_status"),
                pl.col("dq_status").last().alias("final_delinquency_status"),
            ]
        )
    )

    # Anchor on origination so EVERY originated loan gets an outcome row,
    # even ones that originated too late in the period to have a monthly
    # observation yet (treated as 'censored' with null event_time_months).
    orig = pl.scan_parquet(orig_path).select(["loan_seq_num", "vintage_year", "vintage_quarter"])
    out = (
        orig.join(agg, on="loan_seq_num", how="left")
        .with_columns(_classify_event())
        .select(
            [
                "loan_seq_num",
                "vintage_year",
                "vintage_quarter",
                "first_obs_date",
                "last_obs_date",
                "event_time_months",
                "event_type",
                "final_zero_balance_code",
                "max_dq_status",
                "final_delinquency_status",
            ]
        )
        .collect()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd", compression_level=9, statistics=True)

    counts = out.group_by("event_type").agg(pl.len().alias("n")).to_dict(as_series=False)
    by_event = dict(zip(counts["event_type"], counts["n"]))
    n_prepaid = by_event.get("prepaid", 0)
    n_defaulted = by_event.get("defaulted", 0)
    n_other = by_event.get("other_termination", 0)
    n_censored = by_event.get("censored", 0)

    elapsed = time.time() - t0
    logger.info(
        "[%dQ%d] outcomes: n=%s prepaid=%s defaulted=%s other=%s censored=%s (%.1fs)",
        year,
        quarter,
        f"{out.height:,}",
        f"{n_prepaid:,}",
        f"{n_defaulted:,}",
        f"{n_other:,}",
        f"{n_censored:,}",
        elapsed,
    )
    return OutcomesStats(
        year, quarter, out.height, n_prepaid, n_defaulted, n_other, n_censored, elapsed
    )


def derive_all_outcomes(
    years: list[int] | None = None,
    quarters: tuple[int, ...] = (1, 2, 3, 4),
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> list[OutcomesStats]:
    """Derive outcomes for every existing (year, quarter) monthly partition."""
    monthly_root = processed_dir / "monthly"
    if not monthly_root.exists():
        logger.warning("No monthly directory at %s — nothing to derive.", monthly_root)
        return []
    if years is None:
        years = sorted(
            int(p.name.split("=")[1]) for p in monthly_root.iterdir() if p.is_dir()
        )

    stats: list[OutcomesStats] = []
    for y in years:
        for q in quarters:
            mp = processed_dir / "monthly" / f"vintage_year={y}" / f"part-Q{q}.parquet"
            if not mp.exists():
                continue
            stats.append(derive_outcomes_for_quarter(y, q, processed_dir, force))
    return stats


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"derive_outcomes_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Derive per-loan outcomes from monthly partitions.")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--quarter", type=int, nargs="*", default=[1, 2, 3, 4])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    s = derive_all_outcomes(args.years, tuple(args.quarter), force=args.force)
    logger.info("derived %d partitions", len(s))
