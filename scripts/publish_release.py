"""
Build per-vintage tarballs of data/processed/ and publish them as a
GitHub Release. Maintainer-only.

Pipeline:
  1. For each vintage year present under data/processed/origination/,
     build dist/processed-YYYY.tar containing:
         data/processed/origination/vintage_year=YYYY/
         data/processed/monthly/vintage_year=YYYY/
         data/processed/outcomes/vintage_year=YYYY/
  2. Copy data/processed/macro/fred_monthly.parquet to dist/macro.parquet.
  3. Compute SHA256 of every dist/* file -> dist/manifest.sha256.
  4. (Optional) `gh release create <tag> dist/*`.

Tarballs use plain `tar` — Parquet inside is already ZSTD-compressed, so
gzip/zstd outer compression buys nothing.

CLI:
  python scripts/publish_release.py --tag v1.0 --upload
  python scripts/publish_release.py --tag v1.0           # build only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DIST_DIR = PROJECT_ROOT / "dist"
logger = logging.getLogger(__name__)


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _vintage_years() -> list[int]:
    root = PROCESSED_DIR / "origination"
    if not root.exists():
        return []
    return sorted(int(p.name.split("=")[1]) for p in root.iterdir() if p.is_dir())


def _add_partition(tar: tarfile.TarFile, table: str, year: int) -> int:
    """Add data/processed/<table>/vintage_year=YYYY/ to the tar with full path."""
    src = PROCESSED_DIR / table / f"vintage_year={year}"
    if not src.exists():
        return 0
    n = 0
    for f in sorted(src.glob("*.parquet")):
        # Archive with the path layout that download_data.py expects when extracting
        # at the repo root.
        arcname = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
        tar.add(f, arcname=arcname)
        n += 1
    return n


def build_tarballs(dist_dir: Path = DIST_DIR) -> list[Path]:
    dist_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for year in _vintage_years():
        target = dist_dir / f"processed-{year}.tar"
        logger.info("[%d] building %s", year, target.name)
        with tarfile.open(target, "w") as tar:
            n_o = _add_partition(tar, "origination", year)
            n_m = _add_partition(tar, "monthly", year)
            n_t = _add_partition(tar, "outcomes", year)
        size_mb = target.stat().st_size / 1e6
        logger.info("[%d]   wrote %.1f MB (orig=%d monthly=%d outcomes=%d)", year, size_mb, n_o, n_m, n_t)
        out.append(target)

    macro_src = PROCESSED_DIR / "macro" / "fred_monthly.parquet"
    if macro_src.exists():
        macro_dst = dist_dir / "macro.parquet"
        shutil.copy2(macro_src, macro_dst)
        logger.info("[macro] copied %s (%.1f KB)", macro_dst.name, macro_dst.stat().st_size / 1024)
        out.append(macro_dst)
    else:
        logger.warning("[macro] %s missing — skipping.", macro_src)

    return out


def write_manifest(files: list[Path], dist_dir: Path = DIST_DIR) -> Path:
    manifest = dist_dir / "manifest.sha256"
    lines = []
    for f in files:
        digest = _sha256(f)
        lines.append(f"{digest}  {f.name}")
        logger.info("sha256 %s  %s", digest, f.name)
    manifest.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d entries)", manifest, len(lines))
    return manifest


def gh_release_create(tag: str, dist_dir: Path = DIST_DIR, notes: str = "Processed Freddie Mac data") -> None:
    assets = sorted(dist_dir.glob("*.tar")) + sorted(dist_dir.glob("*.parquet"))
    manifest = dist_dir / "manifest.sha256"
    if manifest.exists():
        assets.append(manifest)
    cmd = ["gh", "release", "create", tag, "--title", tag, "--notes", notes, *map(str, assets)]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"publish_release_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="GitHub release tag (e.g. v1.0).")
    p.add_argument("--upload", action="store_true", help="Run `gh release create` after building.")
    p.add_argument("--notes", default="Processed Freddie Mac single-family loan data.")
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    files = build_tarballs()
    write_manifest(files)
    if args.upload:
        gh_release_create(args.tag, notes=args.notes)
    else:
        logger.info("Build done. Use --upload to publish via gh CLI.")
