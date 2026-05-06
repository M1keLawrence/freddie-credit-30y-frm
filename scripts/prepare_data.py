"""
Master orchestrator: zip -> Parquet -> outcomes -> macro -> validate.

Usage (most common):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --years 2006 2007
    python scripts/prepare_data.py --no-macro --skip-validate
    python scripts/prepare_data.py --force      # overwrite existing partitions

Logs are tee'd to stdout AND a per-run file at logs/prepare_data_<ts>.log.
A short progress summary is also written to logs/prepare_data_<ts>.summary.json.

Disk-space watchdog: aborts cleanly if free space drops below
DISK_FREE_FLOOR_GB (default 3 GB) between quarters.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_quarter import (  # noqa: E402
    DATA_DIR,
    PROCESSED_DIR,
    QuarterStats,
    convert_quarter,
)
from scripts.derive_outcomes import derive_outcomes_for_quarter  # noqa: E402
from scripts.fetch_macro import fetch_macro  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # tqdm always present in quant env, but guard anyway
    def tqdm(it, **_):
        return it


logger = logging.getLogger("prepare_data")

DEFAULT_YEARS = list(range(2006, 2026))  # 2006..2025 inclusive (2023 ends at Q2)
ALL_QUARTERS = (1, 2, 3, 4)
DISK_FREE_FLOOR_GB = 3.0
LEGACY_2006_DIR = DATA_DIR / "historical_data_2006"


# ---- progress + state -------------------------------------------------------


@dataclass
class RunSummary:
    started_at: str
    finished_at: str | None = None
    years_requested: list[int] = field(default_factory=list)
    quarters: list[QuarterStats] = field(default_factory=list)
    outcomes_derived: int = 0
    macro_written: bool = False
    aborted_reason: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quarters"] = [asdict(q) for q in self.quarters]
        return d


# ---- helpers ----------------------------------------------------------------


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _setup_logging(log_dir: Path) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_file = log_dir / f"prepare_data_{ts}.log"
    summary_file = log_dir / f"prepare_data_{ts}.summary.json"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file, summary_file


def _delete_legacy_2006_dir(log: logging.Logger) -> None:
    """Per PLAN §7 step 1: delete the already-extracted 2006 directory.

    The 2006 outer zip remains as the source of truth, so this is reversible.
    """
    if not LEGACY_2006_DIR.exists():
        return
    size_mb = sum(p.stat().st_size for p in LEGACY_2006_DIR.rglob("*") if p.is_file()) / 1e6
    log.info("Deleting pre-extracted %s (~%.0f MB) per PLAN §7", LEGACY_2006_DIR, size_mb)
    shutil.rmtree(LEGACY_2006_DIR, ignore_errors=True)
    log.info("Deleted. Free disk now: %.2f GB", _free_gb(DATA_DIR))


def _quarter_zip_present(year: int, quarter: int) -> bool:
    """Check that the inner quarter zip is present inside the outer year zip."""
    import zipfile

    outer = DATA_DIR / f"historical_data_{year}.zip"
    if not outer.exists():
        return False
    try:
        with zipfile.ZipFile(outer) as z:
            return f"historical_data_{year}Q{quarter}.zip" in z.namelist()
    except zipfile.BadZipFile:
        return False


# ---- run --------------------------------------------------------------------


def run(
    years: list[int],
    quarters: tuple[int, ...] = ALL_QUARTERS,
    force: bool = False,
    skip_macro: bool = False,
    skip_validate: bool = False,
    keep_legacy_2006: bool = False,
    summary_path: Path | None = None,
) -> RunSummary:
    summary = RunSummary(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        years_requested=list(years),
    )

    if not keep_legacy_2006:
        _delete_legacy_2006_dir(logger)

    free0 = _free_gb(DATA_DIR)
    logger.info("Free disk at start: %.2f GB", free0)

    quarter_jobs = [
        (y, q) for y in years for q in quarters if _quarter_zip_present(y, q)
    ]
    missing = [(y, q) for y in years for q in quarters if not _quarter_zip_present(y, q)]
    if missing:
        logger.warning("Skipping %d quarters with missing zips: %s", len(missing), missing[:8])

    logger.info(
        "Conversion plan: %d quarter jobs across years %s",
        len(quarter_jobs),
        [str(y) for y in years],
    )

    # ----- conversion --------------------------------------------------------
    for y, q in tqdm(quarter_jobs, desc="quarters", unit="qtr"):
        try:
            stats = convert_quarter(y, q, processed_dir=PROCESSED_DIR, force=force)
        except Exception as e:
            logger.exception("[%dQ%d] FAILED: %s", y, q, e)
            summary.aborted_reason = f"convert_quarter failed at {y}Q{q}: {e}"
            break
        summary.quarters.append(stats)

        free = _free_gb(DATA_DIR)
        if free < DISK_FREE_FLOOR_GB:
            msg = f"Disk free dropped to {free:.2f} GB (< {DISK_FREE_FLOOR_GB} GB floor); aborting."
            logger.error(msg)
            summary.aborted_reason = msg
            break

        # Write a partial summary after every quarter for resilience.
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

    if summary.aborted_reason is None:
        # ----- outcomes ------------------------------------------------------
        for y, q in tqdm(quarter_jobs, desc="outcomes", unit="qtr"):
            try:
                derive_outcomes_for_quarter(y, q, processed_dir=PROCESSED_DIR, force=force)
                summary.outcomes_derived += 1
            except Exception as e:
                logger.exception("[%dQ%d] outcomes FAILED: %s", y, q, e)
                summary.aborted_reason = f"derive_outcomes failed at {y}Q{q}: {e}"
                break

    # ----- macro -------------------------------------------------------------
    if summary.aborted_reason is None and not skip_macro:
        try:
            fetch_macro(force=force)
            summary.macro_written = True
        except Exception as e:
            logger.exception("[macro] failed: %s", e)
            # Macro is non-blocking; continue.

    # ----- validate ----------------------------------------------------------
    if summary.aborted_reason is None and not skip_validate:
        try:
            from scripts.validate import validate_all

            ok = validate_all(years_filter=years)
            logger.info("Validation %s", "PASS" if ok else "FAIL")
        except Exception as e:
            logger.exception("Validation crashed: %s", e)

    summary.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    if summary_path is not None:
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
    free1 = _free_gb(DATA_DIR)
    logger.info("Free disk at end: %.2f GB (delta %+.2f GB)", free1, free1 - free0)
    return summary


# ---- CLI --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Freddie Mac data into Parquet.")
    p.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=DEFAULT_YEARS,
        help="Vintage years to process (default: 2006..2023).",
    )
    p.add_argument(
        "--quarter",
        type=int,
        nargs="*",
        default=list(ALL_QUARTERS),
        choices=[1, 2, 3, 4],
        help="Quarters to process within each year.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet partitions.")
    p.add_argument("--no-macro", action="store_true", help="Skip the FRED macro fetch.")
    p.add_argument("--skip-validate", action="store_true", help="Skip the validation pass.")
    p.add_argument(
        "--keep-legacy-2006",
        action="store_true",
        help="Do NOT delete data/historical_data_2006/ (default: delete).",
    )
    args = p.parse_args()

    log_path, summary_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("Run start. log=%s summary=%s", log_path, summary_path)

    summary = run(
        years=args.years,
        quarters=tuple(args.quarter),
        force=args.force,
        skip_macro=args.no_macro,
        skip_validate=args.skip_validate,
        keep_legacy_2006=args.keep_legacy_2006,
        summary_path=summary_path,
    )

    if summary.aborted_reason:
        logger.error("Run aborted: %s", summary.aborted_reason)
        sys.exit(2)
    logger.info("Run complete. Summary at %s", summary_path)


if __name__ == "__main__":
    main()
