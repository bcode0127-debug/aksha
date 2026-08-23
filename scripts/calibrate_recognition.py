#!/usr/bin/env python3
"""Calibrate what "close to a known rare event" means, from the training data.

The verifier's decision rests on one question: is this window a RECOGNISED
expected pattern? That is only answerable if proximity to a known rare event
actually separates the categories — and if it does, only if we know what
distance counts as close. Both are empirical questions, and this script answers
them before any prompt is written.

For every window in the train period it computes the distance to the nearest
rare_event exemplar, and reports the distribution split by true category.

TWO THINGS THAT WOULD MAKE THE ANSWER A LIE, both handled:

  * Self-match. 330 of the train windows ARE the rare_event exemplars. Left in,
    each would score distance 0 against itself and manufacture the very
    separation we are testing for. Exact (channel, window_start) matches are
    excluded.

  * A different metric from production. The calibration is worthless if the
    verifier sees distances computed another way, so this replicates
    `ReferenceContextProvider._distance` exactly: standardise by the channel's
    NOMINAL std, sum squared deltas over features with non-zero std, divide by
    the count, take the square root.

    python3 scripts/calibrate_recognition.py
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

from aksha_core.data.mission2 import TRAIN_END  # noqa: E402

DEFAULT_UNPURGED = "data/processed/mission2_features_unpurged.parquet"
DEFAULT_REFERENCE = "aksha_core/artifacts/mission2_context_reference.json"
DEFAULT_OUT = "aksha_core/artifacts/mission2_recognition_calibration.json"
CATEGORIES = ("nominal", "rare_event", "anomaly")
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def nearest_rare_distances(
    windows: pd.DataFrame,
    exemplars: list[dict],
    stats: dict,
    columns: list[str],
) -> np.ndarray:
    """Distance from each window to its nearest rare_event exemplar.

    Vectorised, and self-matches are excluded by exact timestamp rather than by
    a distance threshold: a genuinely identical-looking neighbour is a real
    signal and must not be discarded along with the self.
    """
    std = np.array([stats["std"].get(c, 0.0) for c in columns], dtype="float64")
    valid = std > 0.0
    if not valid.any() or not exemplars:
        return np.full(len(windows), np.nan)

    scale = std[valid]
    left = windows[columns].to_numpy(dtype="float64")[:, valid] / scale
    right = np.array(
        [[e["features"][c] for c in columns] for e in exemplars], dtype="float64"
    )[:, valid] / scale

    # squared euclidean via the expansion, then RMS over the counted features
    sq = (
        (left**2).sum(axis=1)[:, None]
        - 2.0 * left @ right.T
        + (right**2).sum(axis=1)[None, :]
    )
    np.maximum(sq, 0.0, out=sq)
    distances = np.sqrt(sq / int(valid.sum()))

    exemplar_keys = [str(e["window_start"]) for e in exemplars]
    window_keys = windows["window_start"].astype(str).to_numpy()
    for j, key in enumerate(exemplar_keys):
        distances[window_keys == key, j] = np.inf  # self-match

    nearest = distances.min(axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unpurged", default=DEFAULT_UNPURGED)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    unpurged = Path(args.unpurged)
    if not unpurged.exists():
        print(
            f"unpurged table not found: {unpurged}\n"
            "build it:  python3 scripts/build_context_reference.py",
            file=sys.stderr,
        )
        return 1

    reference = json.loads(Path(args.reference).read_text())
    columns: list[str] = reference["feature_columns"]
    frame = pd.read_parquet(unpurged)
    train = frame[frame["window_start"] < TRAIN_END]

    rare_by_channel: dict[str, list[dict]] = {}
    for window in reference["reference_windows"]:
        if window["label"] == "rare_event":
            rare_by_channel.setdefault(window["channel_id"], []).append(window)

    parts: list[pd.DataFrame] = []
    for channel, rows in train.groupby("channel", sort=True):
        stats = reference["channel_stats"].get(channel)
        exemplars = rare_by_channel.get(channel, [])
        if not stats or not exemplars:
            continue
        distances = nearest_rare_distances(rows, exemplars, stats, columns)
        parts.append(pd.DataFrame({"label": rows["label"].to_numpy(), "d": distances}))

    scored = pd.concat(parts, ignore_index=True).dropna(subset=["d"])
    print(f"train-period windows scored: {len(scored):,}")
    print(f"rare_event exemplars used  : {sum(len(v) for v in rare_by_channel.values())}")
    print()

    print("DISTANCE TO NEAREST rare_event EXEMPLAR, by true category")
    print("-" * 78)
    header = f"{'category':<12}{'n':>8}" + "".join(f"{f'p{p}':>8}" for p in PERCENTILES)
    print(header)
    summary: dict[str, dict] = {}
    for category in CATEGORIES:
        values = scored.loc[scored["label"] == category, "d"]
        if values.empty:
            continue
        qs = np.percentile(values, PERCENTILES)
        summary[category] = {
            "n": int(values.size),
            **{f"p{p}": float(q) for p, q in zip(PERCENTILES, qs)},
        }
        print(
            f"{category:<12}{values.size:>8,}"
            + "".join(f"{q:>8.2f}" for q in qs)
        )

    # Does proximity separate at all? Compare the two classes the verifier must
    # tell apart. Overlap here means the design does not work.
    print()
    print("SEPARATION")
    print("-" * 78)
    verdict_ok = True
    if "rare_event" in summary and "anomaly" in summary:
        rare_p90 = summary["rare_event"]["p90"]
        anom_p10 = summary["anomaly"]["p10"]
        anom_p50 = summary["anomaly"]["p50"]
        rare_p50 = summary["rare_event"]["p50"]
        print(f"  rare_event median distance : {rare_p50:.2f}")
        print(f"  anomaly    median distance : {anom_p50:.2f}")
        print(f"  ratio (anomaly / rare)     : {anom_p50 / rare_p50:.1f}x")
        print(f"  rare_event p90={rare_p90:.2f}  vs  anomaly p10={anom_p10:.2f}")
        overlap = anom_p10 < rare_p90
        print(
            f"  90% of rare events are within {rare_p90:.2f}; "
            f"10% of anomalies are within {anom_p10:.2f}"
        )
        if overlap:
            print("  -> OVERLAP: the two categories are not cleanly separated here.")
            verdict_ok = anom_p50 > rare_p50 * 2
        else:
            print("  -> SEPARATED at these percentiles.")

    if "nominal" in summary:
        print(f"  nominal    median distance : {summary['nominal']['p50']:.2f}")

    if not verdict_ok:
        print()
        print(
            "VERDICT: rare-event proximity does NOT separate the categories. "
            "The recognition design cannot work on this signal.",
            file=sys.stderr,
        )
        return 2

    # A fine grid over the rare_event distances, so a new window's distance can
    # be placed as a percentile by interpolation rather than bucketed into the
    # nearest of nine coarse points.
    rare_values = scored.loc[scored["label"] == "rare_event", "d"].to_numpy()
    grid = list(range(0, 101))
    rare_curve = [float(v) for v in np.percentile(rare_values, grid)]

    payload = {
        "built_from": "train period only (window_start < 2001-07-01)",
        "train_cut": str(TRAIN_END),
        "metric": "RMS standardised distance to nearest rare_event exemplar, self excluded",
        "percentiles": list(PERCENTILES),
        "by_category": summary,
        # The reference the verifier is given: where a window's distance sits
        # against the rare_event population it is being compared to.
        "rare_event_reference": summary.get("rare_event", {}),
        "rare_event_percentile_grid": grid,
        "rare_event_percentile_values": rare_curve,
        "separability_auc": {
            "note": "AUC of this distance, anomaly scored as the positive class",
            "anomaly_vs_nominal_and_rare": None,  # filled by scripts/eval_verifier.py runs
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
