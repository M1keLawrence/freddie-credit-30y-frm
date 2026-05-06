"""
Convert one quarter of Freddie Mac single-family loan-level data
(zip-streamed) into ZSTD-compressed Parquet partitions.

Public entry point: convert_quarter(year, quarter, ...)

Pipeline:
  1. Open outer year zip; read inner quarter zip into memory.
  2. From the inner zip:
       - Read origination .txt (small), filter to 30y FRM, write Parquet.
       - Stream monthly performance .txt batch-by-batch via pyarrow,
         filter to kept loan_seq_num set, write Parquet.

Idempotent: if both quarter parquet partitions already exist (and are
non-empty) the function is a no-op unless force=True.

Logging is delegated to the caller's configured root logger.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Allow `python scripts/convert_quarter.py` style invocation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import (  # noqa: E402
    MONTHLY_COLS,
    MONTHLY_OUTPUT_COLS_ORDER,
    ORIG_COLS,
    transform_monthly,
    transform_origination,
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

logger = logging.getLogger(__name__)


@dataclass
class QuarterStats:
    year: int
    quarter: int
    orig_raw_rows: int
    orig_kept_rows: int
    monthly_raw_rows: int
    monthly_kept_rows: int
    elapsed_sec: float
    skipped: bool = False


# ---- ZIP HANDLING -----------------------------------------------------------


def _outer_zip_path(year: int) -> Path:
    return DATA_DIR / f"historical_data_{year}.zip"


def _open_inner_zip(year: int, quarter: int) -> zipfile.ZipFile:
    """Read the inner quarter zip from the outer year zip into memory.

    Returns a ZipFile handle backed by an in-memory BytesIO. We read the
    whole inner zip (compressed, ~200-700 MB) into RAM rather than
    extracting to disk; this keeps the pipeline bounded in disk usage,
    and the inner zip's compressed footprint is small enough to fit.
    """
    outer = _outer_zip_path(year)
    inner_name = f"historical_data_{year}Q{quarter}.zip"
    if not outer.exists():
        raise FileNotFoundError(f"Missing outer zip: {outer}")
    with zipfile.ZipFile(outer) as zo:
        if inner_name not in zo.namelist():
            raise FileNotFoundError(
                f"Inner zip {inner_name} not found inside {outer.name}; "
                f"available: {zo.namelist()[:5]}"
            )
        with zo.open(inner_name) as fh:
            data = fh.read()
    return zipfile.ZipFile(io.BytesIO(data))


# ---- ORIGINATION ------------------------------------------------------------


def _process_origination(
    inner_zip: zipfile.ZipFile, year: int, quarter: int, out_path: Path
) -> tuple[int, int, list[str]]:
    """Read origination, type-cast, filter to 30y FRM, write parquet.

    Returns (raw_rows, kept_rows, kept_loan_ids).
    """
    name = f"historical_data_{year}Q{quarter}.txt"
    logger.info("[%dQ%d] reading origination %s", year, quarter, name)
    with inner_zip.open(name) as fh:
        raw = fh.read()

    df = pl.read_csv(
        io.BytesIO(raw),
        separator="|",
        has_header=False,
        new_columns=ORIG_COLS,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    raw_rows = df.height

    df = transform_origination(df, year, quarter)
    df = df.filter((pl.col("term_months") == 360) & (pl.col("amort_type") == "FRM"))
    kept_rows = df.height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        out_path,
        compression="zstd",
        compression_level=9,
        statistics=True,
    )

    kept_ids = df["loan_seq_num"].to_list()
    logger.info(
        "[%dQ%d] origination: raw=%s kept(30yFRM)=%s -> %s",
        year,
        quarter,
        f"{raw_rows:,}",
        f"{kept_rows:,}",
        out_path.name,
    )
    return raw_rows, kept_rows, kept_ids


# ---- MONTHLY ----------------------------------------------------------------


def _build_monthly_arrow_schema() -> pa.Schema:
    """Locked schema for the monthly parquet — must match transform_monthly output."""
    fields = {
        "loan_seq_num": pa.string(),
        "month": pa.date32(),
        "actual_upb": pa.float32(),
        "dq_status": pa.string(),
        "loan_age": pa.uint16(),
        "rem_months": pa.uint16(),
        "defect_date": pa.date32(),
        "mod_flag": pa.string(),
        "zb_code": pa.string(),
        "zb_date": pa.date32(),
        "current_rate": pa.float32(),
        "deferred_upb": pa.float32(),
        "ddlpi": pa.date32(),
        "mi_recoveries": pa.float32(),
        "net_sales_proceeds": pa.string(),
        "non_mi_recoveries": pa.float32(),
        "expenses": pa.float32(),
        "legal_costs": pa.float32(),
        "maint_costs": pa.float32(),
        "taxes_insurance": pa.float32(),
        "misc_expenses": pa.float32(),
        "actual_loss": pa.float32(),
        "mod_cost": pa.float32(),
        "step_mod_flag": pa.string(),
        "deferred_payment_plan": pa.string(),
        "eltv": pa.uint16(),
        "zb_removal_upb": pa.float32(),
        "dq_accrued_interest": pa.float32(),
        "disaster_dq": pa.string(),
        "borrower_assistance": pa.string(),
        "current_mod_cost": pa.float32(),
        "interest_bearing_upb": pa.float32(),
    }
    return pa.schema([(name, dtype) for name, dtype in fields.items()])


MONTHLY_ARROW_SCHEMA = _build_monthly_arrow_schema()


def _process_monthly(
    inner_zip: zipfile.ZipFile,
    year: int,
    quarter: int,
    kept_loan_ids: list[str],
    out_path: Path,
) -> tuple[int, int]:
    """Stream-process monthly performance file → ParquetWriter.

    Returns (raw_rows, kept_rows).
    """
    name = f"historical_data_time_{year}Q{quarter}.txt"
    logger.info("[%dQ%d] streaming monthly %s", year, quarter, name)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 64 MB CSV blocks; ~20-50 batches per quarter. Keeps per-batch RAM
    # well under 1 GB while amortizing per-batch overhead.
    read_opts = pacsv.ReadOptions(
        column_names=MONTHLY_COLS,
        block_size=64 * 1024 * 1024,
    )
    parse_opts = pacsv.ParseOptions(delimiter="|")
    convert_opts = pacsv.ConvertOptions(
        column_types={c: pa.string() for c in MONTHLY_COLS},
        strings_can_be_null=True,
        null_values=[""],
    )

    # Filter via semi-join: faster than is_in for large id sets and avoids
    # polars' is_in deprecation when both sides are the same dtype.
    kept_df = pl.DataFrame({"loan_seq_num": pl.Series(kept_loan_ids, dtype=pl.Utf8)})

    raw_rows = 0
    kept_rows = 0
    writer: pq.ParquetWriter | None = None
    fh = inner_zip.open(name)
    try:
        reader = pacsv.open_csv(fh, read_opts, parse_opts, convert_opts)
        try:
            batch_idx = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                if batch.num_rows == 0:
                    continue
                raw_rows += batch.num_rows
                chunk = pl.from_arrow(pa.Table.from_batches([batch]))
                chunk = transform_monthly(chunk)
                chunk = chunk.join(kept_df, on="loan_seq_num", how="semi")
                if chunk.height == 0:
                    batch_idx += 1
                    continue
                kept_rows += chunk.height
                # Lock column order and arrow schema before writing.
                chunk = chunk.select(MONTHLY_OUTPUT_COLS_ORDER)
                tbl = chunk.to_arrow().cast(MONTHLY_ARROW_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path,
                        MONTHLY_ARROW_SCHEMA,
                        compression="zstd",
                        compression_level=9,
                        use_dictionary=True,
                    )
                writer.write_table(tbl)
                if batch_idx % 5 == 0:
                    logger.info(
                        "[%dQ%d]   batch %d: raw_total=%s kept_total=%s",
                        year,
                        quarter,
                        batch_idx,
                        f"{raw_rows:,}",
                        f"{kept_rows:,}",
                    )
                batch_idx += 1
        finally:
            try:
                reader.close()
            except Exception:
                pass
    finally:
        fh.close()
        if writer is not None:
            writer.close()
        else:
            # No batches produced output rows. Write an empty parquet with
            # the locked schema so downstream consumers can still glob.
            empty = pa.table(
                {f.name: pa.array([], type=f.type) for f in MONTHLY_ARROW_SCHEMA},
                schema=MONTHLY_ARROW_SCHEMA,
            )
            pq.write_table(empty, out_path, compression="zstd", compression_level=9)

    logger.info(
        "[%dQ%d] monthly: raw=%s kept=%s -> %s",
        year,
        quarter,
        f"{raw_rows:,}",
        f"{kept_rows:,}",
        out_path.name,
    )
    return raw_rows, kept_rows


# ---- PUBLIC API -------------------------------------------------------------


def convert_quarter(
    year: int,
    quarter: int,
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> QuarterStats:
    """Convert one (year, quarter) of raw Freddie data to Parquet partitions."""
    t0 = time.time()
    orig_path = processed_dir / "origination" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"
    monthly_path = processed_dir / "monthly" / f"vintage_year={year}" / f"part-Q{quarter}.parquet"

    if not force and orig_path.exists() and monthly_path.exists():
        if orig_path.stat().st_size > 0 and monthly_path.stat().st_size > 0:
            logger.info("[%dQ%d] already converted; skipping.", year, quarter)
            return QuarterStats(year, quarter, 0, 0, 0, 0, 0.0, skipped=True)

    inner = _open_inner_zip(year, quarter)
    try:
        orig_raw, orig_kept, kept_ids = _process_origination(inner, year, quarter, orig_path)
        m_raw, m_kept = _process_monthly(inner, year, quarter, kept_ids, monthly_path)
    finally:
        inner.close()

    elapsed = time.time() - t0
    logger.info(
        "[%dQ%d] DONE in %.1fs  orig=%s monthly=%s",
        year,
        quarter,
        elapsed,
        f"{orig_kept:,}",
        f"{m_kept:,}",
    )
    return QuarterStats(year, quarter, orig_raw, orig_kept, m_raw, m_kept, elapsed)


# ---- CLI --------------------------------------------------------------------


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"convert_quarter_{int(time.time())}.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert one quarter of Freddie data to Parquet.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet partitions.")
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    stats = convert_quarter(args.year, args.quarter, force=args.force)
    logger.info("stats: %s", stats)
