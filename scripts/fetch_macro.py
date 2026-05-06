"""
Fetch macro time series from FRED and save them as a single monthly Parquet.

Output: data/processed/macro/fred_monthly.parquet

Series (PLAN.md §2.4):
  MORTGAGE30US - 30-year fixed mortgage rate (weekly -> resampled monthly mean)
  GS10         - 10-year Treasury constant-maturity yield
  UNRATE       - civilian unemployment rate
  CPIAUCSL     - CPI all urban consumers
  CSUSHPISA    - Case-Shiller national HPI

The fetch is online and rarely re-run. By default we fetch from the
earliest available date through today. Failures on a single series are
logged but do not abort the run — partial coverage beats no macro file.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MACRO_OUT = PROCESSED_DIR / "macro" / "fred_monthly.parquet"

logger = logging.getLogger(__name__)

SERIES = ["MORTGAGE30US", "GS10", "UNRATE", "CPIAUCSL", "CSUSHPISA"]


def fetch_macro(start: str = "1990-01-01", out_path: Path = MACRO_OUT, force: bool = False) -> Path:
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        logger.info("[macro] file already exists; skipping fetch (%s)", out_path)
        return out_path

    from pandas_datareader import data as pdr  # imported lazily so offline runs don't fail

    end = pd.Timestamp.today().normalize()
    frames: dict[str, pd.Series] = {}
    for s in SERIES:
        try:
            logger.info("[macro] fetching %s ...", s)
            df = pdr.DataReader(s, "fred", start=start, end=end)
            frames[s] = df[s]
        except Exception as e:  # network / rate-limit / etc.
            logger.exception("[macro] failed to fetch %s: %s", s, e)

    if not frames:
        raise RuntimeError("All FRED fetches failed; no macro data written.")

    # Resample everything to month-start (mean within month). MORTGAGE30US
    # is weekly; GS10/UNRATE/CPI/HPI are already monthly but resample is a no-op.
    monthly = pd.concat(frames, axis=1).resample("MS").mean()
    monthly.index.name = "month"
    monthly = monthly.reset_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(monthly).write_parquet(out_path, compression="zstd", compression_level=9)
    logger.info(
        "[macro] wrote %s rows %d series %d -> %s",
        f"{len(monthly):,}",
        monthly.shape[1] - 1,
        len(frames),
        out_path,
    )
    return out_path


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"fetch_macro_{int(time.time())}.log"
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
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    fetch_macro(args.start, force=args.force)
