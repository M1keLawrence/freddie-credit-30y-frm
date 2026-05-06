"""
One-shot helper: wait until the currently-running prepare_data job
releases its parquet writers, then convert 2020 + 2024 + 2025.

Used because the user reuploaded 2020 and added 2024, 2025 mid-run.
Running a second prepare_data in parallel would compete for CPU and disk;
this script just polls the existing log file for "Run complete." then
launches the extension run.

Not part of the user-facing API — delete after one use.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
PYTHON = r"C:\Users\a\anaconda3\envs\quant\python.exe"

logger = logging.getLogger("queue_extension")


def _wait_for_main(timeout_min: int = 180, marker: str = "Run complete.") -> Path | None:
    """Poll the most-recent prepare_data_*.log until the marker shows up."""
    deadline = time.time() + timeout_min * 60
    last_size = -1
    while time.time() < deadline:
        candidates = sorted(LOG_DIR.glob("prepare_data_*.log"))
        if not candidates:
            time.sleep(15)
            continue
        log = candidates[-1]
        size = log.stat().st_size
        text = log.read_text(encoding="utf-8", errors="ignore")
        if marker in text or "Run aborted" in text:
            logger.info("Detected main run finished in %s", log.name)
            return log
        if size != last_size:
            last_size = size
            logger.info("main job still running (log %d bytes)", size)
        time.sleep(30)
    logger.error("Timed out waiting for main run after %d min", timeout_min)
    return None


def _setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"queue_extension_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    log_path = _setup_logging()
    logger.info("logging to %s", log_path)
    logger.info("waiting for main prepare_data run to finish...")
    finished_log = _wait_for_main()
    if finished_log is None:
        sys.exit(2)

    logger.info("launching extension run for 2020 2024 2025")
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "scripts" / "prepare_data.py"),
        "--years", "2020", "2024", "2025",
        "--no-macro",
        "--skip-validate",
        "--keep-legacy-2006",  # already deleted; avoid the 'doesn't exist' nudge
    ]
    logger.info("cmd: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    logger.info("extension run exit code: %d", rc)
    sys.exit(rc)
