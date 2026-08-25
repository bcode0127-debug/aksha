#!/usr/bin/env python3
"""Fetch ESA-ADB's Mission2 raw dataset into data/, verified against the
pinned Zenodo record's published checksum.

Mirrors scripts/fetch_data.py's structure. Unlike OPSSAT-AD's two small CSVs,
this is one 3.8 GB zip (ESA-Mission2.zip) that unpacks to per-channel pickles,
telecommand pickles, and label CSVs -- see docs/mission2-adapter-notes.md for
what aksha_core/data/mission2.py expects inside it.

Pinned record: 10.5281/zenodo.15237121 (ADR-015). Unpacks into
data/esa-adb/mission2/ESA-Mission2/ -- inside the repo's gitignored data/, not
a path outside the repo, so a fresh clone can reproduce the build without
editing any code.

    python3 scripts/fetch_mission2.py
    python3 scripts/fetch_mission2.py --skip-unzip   # verify/download only
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests

RECORD_ID = "15237121"
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_NAME = "ESA-Mission2.zip"
EXPECTED_MD5 = "bfc72012691427d9327eb41f726ce45e"
EXPECTED_SIZE = 4_098_539_932

# Where aksha_core/data/mission2.py's DEFAULT_DATA_ROOT points -- keep these
# in sync; AKSHA_MISSION2_DIR overrides either at runtime.
UNPACK_DIR = DATA_DIR / "esa-adb" / "mission2" / "ESA-Mission2"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_zip() -> Path:
    dest = DATA_DIR / ZIP_NAME
    if dest.exists() and dest.stat().st_size == EXPECTED_SIZE and md5sum(dest) == EXPECTED_MD5:
        print(f"{ZIP_NAME}: already present and valid, skipping download")
        return dest

    url = f"{BASE_URL}/{ZIP_NAME}/content"
    print(f"{ZIP_NAME}: downloading from {url} ({EXPECTED_SIZE / 1e9:.2f} GB)")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    tmp = dest.with_name(dest.name + ".part")
    total_bytes = 0
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            total_bytes += len(chunk)

    actual_md5 = md5sum(tmp)
    if actual_md5 != EXPECTED_MD5:
        tmp.unlink()
        raise RuntimeError(
            f"{ZIP_NAME}: checksum mismatch -- expected {EXPECTED_MD5}, got {actual_md5}. "
            "Download discarded."
        )

    tmp.rename(dest)
    print(f"{ZIP_NAME}: downloaded {total_bytes:,} bytes, checksum verified")
    return dest


def unpack(zip_path: Path) -> None:
    """Extract the zip directly into UNPACK_DIR.

    Verified against ESA-Mission1.zip (same publisher, same ESA-AD pipeline,
    found already extracted on this machine): its entries are flat --
    channels/, telecommands/, labels.csv, channels.csv, telecommands.csv,
    anomaly_types.csv sit at the top level, with no internal "ESA-Mission1/"
    prefix. The containing folder name is imposed by whoever extracts it, not
    part of the archive. An earlier version of this function extracted to
    UNPACK_DIR.parent on the opposite assumption (that the zip itself
    provided the nesting) -- that was wrong; data_root()'s expectation in
    aksha_core/data/mission2.py was right. Still not verified against the
    actual Mission2 archive's bytes: the only copy found locally
    (~/aksha-datasets/esa-adb/ESA-Mission2.zip) is 3,422,552,064 bytes against
    Zenodo's published 4,098,539,932 for record 15237121, and fails to open
    as a zip at all -- it belongs to an unrelated project's data cache
    (see that directory's PROVENANCE.md) and was not usable for this check.
    """
    if UNPACK_DIR.exists() and any(UNPACK_DIR.iterdir()):
        print(f"{UNPACK_DIR}: already populated, skipping unzip")
        return

    free_bytes = shutil.disk_usage(DATA_DIR).free
    # The zip is compressed pickles; unpacked size is not published by Zenodo,
    # so this is a conservative floor, not a guarantee -- re-check if it fails.
    if free_bytes < EXPECTED_SIZE * 2:
        raise RuntimeError(
            f"only {free_bytes / 1e9:.1f} GB free; unpacking a {EXPECTED_SIZE / 1e9:.1f} GB "
            "zip needs headroom beyond the zip itself. Free up space and re-run "
            "(the verified zip is kept, so this resumes at unzip)."
        )

    UNPACK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"unzipping to {UNPACK_DIR} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(UNPACK_DIR)
    print(f"unpacked: {UNPACK_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-unzip", action="store_true", help="download and verify only, do not unpack"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = fetch_zip()
    if not args.skip_unzip:
        unpack(zip_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (requests.RequestException, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"fetch_mission2.py: FAILED -- {exc}", file=sys.stderr)
        sys.exit(1)
