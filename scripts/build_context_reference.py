#!/usr/bin/env python3
"""Build the committed context reference the triage graph's prepare_context uses.

Two things, both derived from the TRAINING split only:

  * per-channel nominal mean/std for every feature, so a window can be placed
    against its own channel's history as z-scores;
  * a stratified subsample of nominal training windows, so prepare_context can
    retrieve the k nearest ones as feature summaries.

Train-only is deliberate. Retrieving neighbours from the test split would put
test rows into the reasoning context of an evaluation run on that same split —
leakage through the back door, even though nothing is fitted on them. The
training split is also nominal by construction (the adapter purges anomalies
and rare events), so every neighbour is a legitimate "here is what normal looks
like" reference rather than an answer key.

    python3 scripts/build_context_reference.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_core.detectors.iforest import feature_columns  # noqa: E402

DEFAULT_PARQUET = "data/processed/mission2_features.parquet"
DEFAULT_OUT = "aksha_core/artifacts/mission2_context_reference.json"
WINDOWS_PER_CHANNEL = 40
SEED = 20260818


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--per-channel", type=int, default=WINDOWS_PER_CHANNEL)
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        print(f"feature table not found: {path}", file=sys.stderr)
        return 1

    frame = pd.read_parquet(path)
    train = frame[frame["split"] == "train"]
    if set(train["label"].unique()) != {"nominal"}:
        print(
            f"training split is not purely nominal ({sorted(train['label'].unique())}); "
            "refusing to build a reference that could leak labels",
            file=sys.stderr,
        )
        return 1

    columns = feature_columns(frame)
    rng = np.random.default_rng(SEED)

    channel_stats: dict[str, dict] = {}
    reference: list[dict] = []

    for channel, rows in train.groupby("channel", sort=True):
        values = rows[columns]
        channel_stats[channel] = {
            "reference_windows": int(len(rows)),
            "mean": {c: float(values[c].mean()) for c in columns},
            "std": {c: float(values[c].std(ddof=0)) for c in columns},
        }
        take = min(args.per_channel, len(rows))
        picked = rows.iloc[rng.choice(len(rows), size=take, replace=False)]
        for _, row in picked.iterrows():
            reference.append(
                {
                    "channel_id": channel,
                    "window_start": str(row["window_start"]),
                    "label": str(row["label"]),
                    "features": {c: round(float(row[c]), 9) for c in columns},
                }
            )

    payload = {
        "source_parquet": str(path),
        "built_from_split": "train",
        "feature_columns": columns,
        "channels": sorted(channel_stats),
        "channel_stats": channel_stats,
        "reference_windows": reference,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    size_kb = out.stat().st_size / 1024

    print(f"channels          : {len(channel_stats)}")
    print(f"reference windows : {len(reference)} ({args.per_channel} per channel)")
    print(f"features          : {len(columns)}")
    print(f"written           : {out}  ({size_kb:,.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
