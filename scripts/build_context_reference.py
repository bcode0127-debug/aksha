#!/usr/bin/env python3
"""Build the labeled context reference the triage graph retrieves against.

Two things, both restricted to the TRAIN PERIOD (window_start < 2001-07-01):

  * per-channel NOMINAL mean/std for every feature, so a window can be placed
    against its own channel's normal envelope as z-scores;
  * labeled exemplars of every category — nominal, rare_event, anomaly — so the
    verifier has a reference for each class it is asked to discriminate between.

WHY THE EXEMPLARS INCLUDE WINDOWS THE ADAPTER PURGED

The adapter purges anomaly- and rare-event-labelled windows from the training
split, and that is correct: an unsupervised detector fitted on examples of the
thing it is meant to find is not fitted on nominal behaviour any more. But the
purge governs *what the model is fitted on*. Retrieval material is a different
question, and conflating the two left the verifier with a reference set that was
100% nominal — it had never seen an example of the class it was supposed to
reject, so it could not discriminate. This script therefore reads the UNPURGED
feature table and keeps the labels.

WHAT IS STILL FORBIDDEN

Nothing at or after 2001-07-01 may enter the reference. That is the train cut
(ESA-ADB's `validation_splits["21_months"]`), and both the calibration block and
the test split lie beyond it. Retrieving a neighbour from either would put
evaluation rows into the reasoning context of a run being evaluated on them —
leakage through the back door, even though nothing is fitted on them. The
constraint is asserted in code below, not merely intended.

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

from aksha_core.data.mission2 import TRAIN_END, build_feature_table  # noqa: E402
from aksha_core.detectors.iforest import feature_columns  # noqa: E402

DEFAULT_UNPURGED = "data/processed/mission2_features_unpurged.parquet"
DEFAULT_OUT = "aksha_core/artifacts/mission2_context_reference.json"

# Ours. Per (channel, category), capped so the file stays small enough to commit
# while every channel still carries its own reference for each class.
PER_CHANNEL_PER_CATEGORY = 30
CATEGORIES = ("nominal", "rare_event", "anomaly")
SEED = 20260818


def load_unpurged(path: Path) -> pd.DataFrame:
    """The feature table WITH its non-nominal windows intact.

    Cached: rebuilding costs a couple of minutes and the raw 3.8 GB dataset.
    """
    if path.exists():
        return pd.read_parquet(path)

    print(f"unpurged table not found at {path}; building it (a few minutes)...")
    table, stats = build_feature_table(purge_non_nominal=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    print(f"  built {len(table):,} rows, drop rate {stats['drop_rate']:.6%}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unpurged", default=DEFAULT_UNPURGED)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--per-channel", type=int, default=PER_CHANNEL_PER_CATEGORY)
    args = parser.parse_args()

    frame = load_unpurged(Path(args.unpurged))
    columns = feature_columns(frame)

    # Train period only, by timestamp rather than by the `split` column, so the
    # constraint is on the thing that actually matters.
    train_period = frame[frame["window_start"] < TRAIN_END]
    if train_period.empty:
        print("no windows before the train cut; nothing to build", file=sys.stderr)
        return 1

    present = sorted(train_period["label"].unique())
    print(f"train period (< {TRAIN_END.date()}): {len(train_period):,} windows")
    print(f"  labels present: {present}")
    for category in CATEGORIES:
        n = int((train_period["label"] == category).sum())
        print(f"    {category:<11} {n:>8,}")
    missing = [c for c in CATEGORIES if c not in present]
    if missing:
        print(
            f"\nERROR: no exemplars for {missing} in the train period. The verifier "
            "cannot discriminate between classes it has never seen; refusing to "
            "build a reference that would put it back in that position.",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(SEED)

    # The nominal envelope for z-scores stays NOMINAL-only: it describes what
    # normal looks like, so anomalous windows must not widen it.
    nominal = train_period[train_period["label"] == "nominal"]
    channel_stats: dict[str, dict] = {}
    for channel, rows in nominal.groupby("channel", sort=True):
        values = rows[columns]
        channel_stats[channel] = {
            "reference_windows": int(len(rows)),
            "mean": {c: float(values[c].mean()) for c in columns},
            "std": {c: float(values[c].std(ddof=0)) for c in columns},
        }

    exemplars: list[dict] = []
    per_category_counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    coverage: dict[str, list[str]] = {c: [] for c in CATEGORIES}

    for (channel, label), rows in train_period.groupby(["channel", "label"], sort=True):
        if label not in CATEGORIES:
            continue
        take = min(args.per_channel, len(rows))
        picked = rows.iloc[rng.choice(len(rows), size=take, replace=False)]
        for _, row in picked.iterrows():
            exemplars.append(
                {
                    "channel_id": channel,
                    "window_start": str(row["window_start"]),
                    "label": label,
                    "features": {c: round(float(row[c]), 9) for c in columns},
                }
            )
        per_category_counts[label] += take
        coverage[label].append(channel)

    # The constraint, asserted rather than assumed.
    cutoff = pd.Timestamp(TRAIN_END)
    latest = max(pd.Timestamp(e["window_start"]) for e in exemplars)
    assert latest < cutoff, (
        f"exemplar at {latest} is at or after the train cut {cutoff}; "
        "this would leak calibration/test data into retrieval"
    )
    assert set(per_category_counts) == set(CATEGORIES)
    assert all(n > 0 for n in per_category_counts.values()), per_category_counts

    payload = {
        "source_parquet": str(args.unpurged),
        "built_from": "train period only (window_start < 2001-07-01), labels retained",
        "train_cut": str(cutoff),
        "latest_exemplar": str(latest),
        "feature_columns": columns,
        "categories": list(CATEGORIES),
        "channels": sorted(channel_stats),
        "channel_stats": channel_stats,
        "reference_windows": exemplars,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

    print(f"\nexemplars written: {len(exemplars):,}")
    for category in CATEGORIES:
        print(
            f"  {category:<11} {per_category_counts[category]:>5}  "
            f"across {len(coverage[category]):>2}/11 channels"
        )
    print(f"latest exemplar : {latest}  (cut {cutoff})")
    print(f"written         : {out}  ({out.stat().st_size / 1024:,.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
