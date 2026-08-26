#!/usr/bin/env python3
"""Precompute per-incident signal windows for the dashboard's Pane 1.

`aksha_core.data.mission2.load_channel()` reads one channel's ENTIRE raw
series (years of data) into memory -- there is no windowed/partial read.
Pane 1 needs 11 channels' worth of context (all of LIGHTWEIGHT_CHANNELS) for
every incident, and re-running that per incident, per dashboard interaction,
would mean loading each multi-year channel repeatedly for a few hours of
data each time.

So this loads each of the 11 lightweight channels exactly ONCE, slices out
the padded window (window duration on each side, so "shade the window
region" has something to shade against) for every incident currently in
Firestore, and writes the small result to a long-format parquet the
dashboard reads directly -- no raw-data dependency at dashboard runtime.

    python3 scripts/precompute_signal_cache.py

Re-run whenever new incidents land in Firestore that aren't in the cache yet
(the dashboard falls back to a clear "not cached" state for those, rather
than triggering a slow live load).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_core.data.mission2 import LIGHTWEIGHT_CHANNELS, load_channel  # noqa: E402

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
OUT_PATH = REPO_ROOT / "data" / "processed" / "dashboard_signal_cache.parquet"

# Padding on each side of the incident's own window, so the plot shows
# before/window/after and "shade the window region" means something.
PAD_FRACTION = 1.0  # one window-width of padding each side


def fetch_incidents() -> list[dict]:
    from google.cloud import firestore

    db = firestore.Client(project=PROJECT_ID)
    docs = db.collection("incidents").stream()
    out = []
    for d in docs:
        v = d.to_dict()
        if v.get("t_start") and v.get("t_end") and v.get("channel_id"):
            out.append({"incident_id": d.id, "channel_id": v["channel_id"],
                        "t_start": v["t_start"], "t_end": v["t_end"]})
    return out


def main() -> int:
    started = time.perf_counter()
    incidents = fetch_incidents()
    if not incidents:
        print("no incidents with channel_id/t_start/t_end found in Firestore", file=sys.stderr)
        return 1
    print(f"{len(incidents)} incidents with a usable window")

    print(f"loading {len(LIGHTWEIGHT_CHANNELS)} channels (each read once)...")
    channel_series: dict[str, pd.Series] = {}
    for name in LIGHTWEIGHT_CHANNELS:
        t0 = time.perf_counter()
        channel_series[name] = load_channel(name)
        print(f"  {name}: {len(channel_series[name]):,} points in {time.perf_counter() - t0:.1f}s")
    load_elapsed = time.perf_counter() - started
    print(f"channel load total: {load_elapsed:.1f}s")

    rows = []
    for inc in incidents:
        t_start = pd.Timestamp(inc["t_start"]).tz_localize(None)
        t_end = pd.Timestamp(inc["t_end"]).tz_localize(None)
        pad = (t_end - t_start) * PAD_FRACTION
        lo, hi = t_start - pad, t_end + pad

        for channel_name, series in channel_series.items():
            window = series[(series.index >= lo) & (series.index <= hi)]
            if window.empty:
                continue
            rows.append(pd.DataFrame({
                "incident_id": inc["incident_id"],
                "channel_id": channel_name,
                "is_flagged": channel_name == inc["channel_id"],
                "window_start": t_start,
                "window_end": t_end,
                "timestamp": window.index,
                "value": window.to_numpy(),
            }))

    if not rows:
        print("no overlapping data found for any incident window -- check the archive covers this period", file=sys.stderr)
        return 1

    table = pd.concat(rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)

    total_elapsed = time.perf_counter() - started
    print(f"\nwritten: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB, {len(table):,} rows)")
    print(f"channel load: {load_elapsed:.1f}s | total: {total_elapsed:.1f}s "
          f"for {len(incidents)} incidents x {len(LIGHTWEIGHT_CHANNELS)} channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
