#!/usr/bin/env python3
"""Fetch OPSSAT-AD dataset.csv and segments.csv into data/, verified against
the pinned Zenodo record's published checksums.

Pinned record: 10.5281/zenodo.15108715 (latest version as of the fetch script
being written; the concept DOI 10.5281/zenodo.12588358 may point elsewhere
by the time you read this — re-verify before repinning).
"""
import hashlib
import sys
from pathlib import Path

import requests

RECORD_ID = "15108715"
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    "dataset.csv": "5246fdc5e4630a4cecbf7fb6bc8b795e",
    "segments.csv": "72f109630abb933a386106897a631188",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(filename: str, expected_md5: str) -> None:
    dest = DATA_DIR / filename
    if dest.exists() and md5sum(dest) == expected_md5:
        print(f"{filename}: already present and valid, skipping")
        return

    url = f"{BASE_URL}/{filename}/content"
    print(f"{filename}: downloading from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    tmp = dest.with_name(dest.name + ".part")
    total_bytes = 0
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            total_bytes += len(chunk)

    actual_md5 = md5sum(tmp)
    if actual_md5 != expected_md5:
        tmp.unlink()
        raise RuntimeError(
            f"{filename}: checksum mismatch — expected {expected_md5}, got {actual_md5}. "
            "Download discarded."
        )

    tmp.rename(dest)
    print(f"{filename}: downloaded {total_bytes} bytes, checksum verified")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in FILES.items():
        fetch(filename, expected_md5)


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, RuntimeError) as exc:
        print(f"fetch_data.py: FAILED — {exc}", file=sys.stderr)
        sys.exit(1)
