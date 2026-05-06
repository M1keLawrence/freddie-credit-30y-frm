"""
Teammate entry point: fetch processed parquet from a GitHub Release.

Usage:
  python scripts/download_data.py                    # all vintages, latest tag
  python scripts/download_data.py --years 2006 2007  # subset
  python scripts/download_data.py --force            # re-download

Behaviour:
  1. Reads RELEASE_TAG and REPO_SLUG below to locate assets.
  2. Fetches manifest.sha256, downloads each requested tarball to dist/,
     verifies SHA256, then extracts at the project root.
  3. Idempotent: skips a vintage when the matching origination/, monthly/,
     and outcomes/ partitions are already on disk.
  4. macro.parquet is fetched in addition (always).

If the repo is private, set GITHUB_TOKEN in the env (the script forwards
it as a Bearer token); for public repos no auth is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DIST_DIR = PROJECT_ROOT / "dist"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- one-line pin point — bump on each release --------------------------------
RELEASE_TAG = "v1.0"
REPO_SLUG = "sdecoster/freddie-credit-30y-frm"  # set to "<user>/<repo>" before release
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _asset_url(name: str) -> str:
    return f"https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}/{name}"


def _http_get(url: str, dest: Path, token: str | None = None) -> None:
    headers = {"User-Agent": "credit-data-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        chunk = 1024 * 1024
        n = 0
        last = time.time()
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            n += len(b)
            now = time.time()
            if now - last > 5:
                pct = f" ({100*n/total:.1f}%)" if total else ""
                logger.info("  %s  %.1f MB%s", dest.name, n / 1e6, pct)
                last = now
    tmp.replace(dest)


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_manifest(token: str | None) -> dict[str, str]:
    url = _asset_url("manifest.sha256")
    dest = DIST_DIR / "manifest.sha256"
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("fetching %s", url)
    _http_get(url, dest, token=token)
    table: dict[str, str] = {}
    for line in dest.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        table[name.strip()] = digest.strip()
    return table


def _vintage_already_present(year: int) -> bool:
    """True iff origination, monthly and outcomes are all on disk for this vintage."""
    for table in ("origination", "monthly", "outcomes"):
        d = PROCESSED_DIR / table / f"vintage_year={year}"
        if not d.exists() or not any(d.glob("*.parquet")):
            return False
    return True


def _extract_tar(tar_path: Path) -> None:
    logger.info("  extracting %s", tar_path.name)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(PROJECT_ROOT)


def _filter_assets(manifest: dict[str, str], years: list[int] | None) -> list[str]:
    out: list[str] = []
    for name in manifest:
        if name.startswith("processed-") and name.endswith(".tar"):
            try:
                yr = int(name.removeprefix("processed-").removesuffix(".tar"))
            except ValueError:
                continue
            if years is None or yr in set(years):
                out.append(name)
        elif name == "macro.parquet":
            out.append(name)
    return sorted(out)


def download_all(years: list[int] | None = None, force: bool = False) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    manifest = _read_manifest(token)
    assets = _filter_assets(manifest, years)
    logger.info("plan: %d assets (%s)", len(assets), assets[:6])

    for name in assets:
        # Skip whole vintages already present.
        if name.startswith("processed-") and not force:
            try:
                yr = int(name.removeprefix("processed-").removesuffix(".tar"))
                if _vintage_already_present(yr):
                    logger.info("  %s: already present, skipping", name)
                    continue
            except ValueError:
                pass

        local = DIST_DIR / name
        if local.exists() and not force:
            actual = _sha256(local)
            if actual == manifest[name]:
                logger.info("  %s: cached + checksum OK", name)
            else:
                logger.warning("  %s: cached but checksum mismatch — redownloading", name)
                local.unlink()
        if not local.exists() or force:
            logger.info("downloading %s", name)
            try:
                _http_get(_asset_url(name), local, token=token)
            except urllib.error.HTTPError as e:
                logger.error("  HTTP %s for %s — skipping", e.code, name)
                continue
            actual = _sha256(local)
            if actual != manifest[name]:
                logger.error("  %s: checksum FAIL (got %s)", name, actual)
                local.unlink(missing_ok=True)
                continue

        if name.endswith(".tar"):
            _extract_tar(local)
        elif name == "macro.parquet":
            dest = PROCESSED_DIR / "macro" / "fred_monthly.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local.read_bytes())
            logger.info("  installed -> %s", dest)


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"download_data_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    log_path = _setup_logging(PROJECT_ROOT / "logs")
    logger.info("logging to %s", log_path)
    download_all(args.years, force=args.force)
