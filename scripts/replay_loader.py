#!/usr/bin/env python3
"""Replay loader: publish real Mission2 test-split windows to telemetry-in
at a fixed rate, so the live pipeline (and the dashboard watching it) can be
exercised continuously rather than one batch at a time.

Referenced but never built in docs/PRD.md and docs/TRD.md ("replay loader
(publishes in sequence)") -- this is the first actual implementation, not an
extension of a prior script.

Reads data/processed/mission2_features.parquet directly (the same file
scripts/build_funnel_stats.py reads), iterates the test split in
window_start order, and publishes one real row every --rate seconds. Ctrl+C
stops it; --count and --duration are optional caps for a bounded run.

    python3 scripts/replay_loader.py                    # every 5s, until stopped
    python3 scripts/replay_loader.py --rate 10           # every 10s
    python3 scripts/replay_loader.py --count 20          # stop after 20 windows
    python3 scripts/replay_loader.py --duration 300      # stop after 5 minutes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google.cloud import pubsub_v1

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_core.detectors.artifact import DetectorArtifact  # noqa: E402

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
TOPIC = "telemetry-in"
FEATURES_PARQUET = REPO_ROOT / "data" / "processed" / "mission2_features.parquet"


def load_test_windows() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        print(f"missing {FEATURES_PARQUET} -- run python3 -m aksha_core.data.mission2 first", file=sys.stderr)
        sys.exit(1)
    frame = pd.read_parquet(FEATURES_PARQUET)
    test = frame[frame["split"] == "test"].copy()
    return test.sort_values("window_start").reset_index(drop=True)


def build_payload(row: pd.Series, feature_cols: list[str], run_tag: int, i: int) -> dict:
    window_start = pd.Timestamp(row["window_start"])
    window_end = window_start + pd.Timedelta(hours=1)
    return {
        "fragment_id": f"frag-replay-{run_tag}-{i:05d}",
        "channel_id": row["channel"],
        "t_start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "t_end": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": {c: float(row[c]) for c in feature_cols},
        "context": {"source": "scripts/replay_loader.py", "label": row["label"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rate", type=float, default=5.0, help="seconds between publishes")
    parser.add_argument("--count", type=int, default=None, help="stop after this many windows")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--start-at", type=int, default=0, help="skip this many rows before starting")
    args = parser.parse_args()

    test = load_test_windows()
    artifact = DetectorArtifact.load()
    feature_cols = artifact.feature_columns
    print(f"loaded {len(test):,} test-split windows from {FEATURES_PARQUET}")

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, args.topic)

    run_tag = int(time.time())
    started = time.perf_counter()
    published = 0
    print(f"replay started: rate={args.rate}s topic={args.topic} run_tag={run_tag}")
    print("stop with Ctrl+C")

    try:
        i = args.start_at
        while i < len(test):
            if args.count is not None and published >= args.count:
                break
            if args.duration is not None and (time.perf_counter() - started) >= args.duration:
                break

            row = test.iloc[i]
            payload = build_payload(row, feature_cols, run_tag, i)
            message_id = publisher.publish(topic_path, json.dumps(payload).encode("utf-8")).result()
            published += 1
            print(f"[{published}] published {payload['fragment_id']} "
                  f"({row['label']}, {row['channel']}) message_id={message_id}")

            i += 1
            time.sleep(args.rate)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl+C")

    elapsed = time.perf_counter() - started
    print(f"\nreplay stopped: {published} windows published in {elapsed:.1f}s "
          f"(run_tag={run_tag}, fragment_ids frag-replay-{run_tag}-*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
