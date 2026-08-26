#!/usr/bin/env python3
"""Pin a fixed set of windows with expected verdicts, for comparable evaluation.

Every measurement so far drew a fresh random sample, which makes two attempts
incomparable: a change in the numbers could be the change in the design or the
change in the sample, and there is no way to tell which. The golden set removes
that ambiguity -- it is committed, it never resamples, and a future attempt is
measured against exactly the windows this one was.

WHAT GOES IN, and why each group

  clear_fault    : true anomalies well above the gate's distance threshold.
                   Missing one of these is the expensive failure.
  clear_expected : true rare events well below it. Confirming one of these
                   wakes an operator for routine behaviour.
  clear_nominal  : detector-flagged nominal windows -- the false alarms the
                   verifier exists to absorb.
  ambiguous      : windows inside the gate's calibrated ambiguous band. The
                   gate returns `disputed` for these BY CONSTRUCTION, so they
                   carry NO expected verdict -- scoring them would be scoring
                   the band's definition against itself. They are here so that
                   how much decision the band absorbs stays visible.

Windows are drawn from the TEST split wherever possible, so the golden set
measures the system on data it has never been fitted, calibrated or given as
retrieval material.

FAULTS ARE THE EXCEPTION, and the reason is worth stating plainly. The test
split contains exactly one distinct anomaly event, and both of its
detector-flagged windows sit BELOW the gate's distance threshold -- the gate
rejects them. So the test split cannot supply a single "clear fault", and a
golden set built only from it would measure everything except the failure that
matters most. Clear faults are therefore drawn from the held-out train-period
anomalies (absent from the exemplar reference), and every entry carries a
`source` field so train-period rows are never mistaken for test evidence.

    python3 scripts/build_golden_set.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_agent.graph.context import ReferenceContextProvider  # noqa: E402
from aksha_agent.graph.schemas import DetectionSummary  # noqa: E402
from aksha_core.detectors.artifact import DetectorArtifact  # noqa: E402
from aksha_core.detectors.iforest import feature_columns  # noqa: E402

DEFAULT_PARQUET = "data/processed/mission2_features.parquet"
DEFAULT_OUT = "tests/fixtures/golden_set.json"
DEFAULT_HOLDOUT = "data/processed/mission2_anomaly_holdout.parquet"
SEED = 20260822

# Ours. Sizes chosen so the set stays reviewable by hand while each group has
# enough members that one flipped verdict does not swing the reading.
GROUP_SIZES = {"clear_fault": 4, "clear_expected": 7, "clear_nominal": 7, "ambiguous": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet)
    columns = feature_columns(frame)
    artifact = DetectorArtifact.load()
    provider = ReferenceContextProvider()
    threshold = provider.operating_threshold
    band = provider.ambiguous_band
    if threshold is None or band is None:
        print(
            "no operating threshold / ambiguous band in the calibration artifact; "
            "run scripts/calibrate_recognition.py first",
            file=sys.stderr,
        )
        return 1

    test = frame[frame["split"] == "test"].copy()
    scored = artifact.score_frame(test)
    test["score"] = scored["score"].to_numpy()
    test["conformal_p"] = scored["conformal_p"].to_numpy()
    test["threshold"] = scored["threshold"].to_numpy()
    # Only detector-flagged windows reach triage in production.
    test = test[test["score"] >= test["threshold"]].reset_index(drop=True)

    distances = []
    for _, row in test.iterrows():
        summary = DetectionSummary(
            fragment_id="g",
            channel_id=row["channel"],
            t_start=str(row["window_start"]),
            t_end=str(row["window_start"] + pd.Timedelta(hours=1)),
            score=float(row["score"]),
            threshold=float(row["threshold"]),
            conformal_p=float(row["conformal_p"]),
            detector_version="iforest-conformal-0.1.0",
            features={c: float(row[c]) for c in columns},
        )
        distances.append(provider.recognition(summary).distance_to_nearest_expected)
    test["distance"] = distances

    # The SAME bounds the gate uses at runtime, read from the same artifact.
    # Deriving them here independently would let the golden set's notion of
    # "ambiguous" drift away from the gate's without anything failing.
    lo, hi = band
    rng = __import__("numpy").random.default_rng(SEED)

    def take(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        n = min(n, len(pool))
        return pool.iloc[rng.choice(len(pool), size=n, replace=False)] if n else pool.head(0)

    # Clear faults from the held-out train anomalies: see the module docstring.
    holdout_path = Path(args.holdout)
    fault_pool = test[(test["label"] == "anomaly") & (test["distance"] >= hi)]
    fault_source = "test"
    if fault_pool.empty and holdout_path.exists():
        held = pd.read_parquet(holdout_path)
        hscored = artifact.score_frame(held)
        held = held.assign(
            score=hscored["score"].to_numpy(),
            conformal_p=hscored["conformal_p"].to_numpy(),
            threshold=hscored["threshold"].to_numpy(),
        )
        held = held[held["score"] >= held["threshold"]].reset_index(drop=True)
        hd = []
        for _, row in held.iterrows():
            hd.append(
                provider.recognition(
                    DetectionSummary(
                        fragment_id="g", channel_id=row["channel"],
                        t_start=str(row["window_start"]), t_end=str(row["window_start"]),
                        score=float(row["score"]), threshold=float(row["threshold"]),
                        conformal_p=float(row["conformal_p"]),
                        detector_version="iforest-conformal-0.1.0",
                        features={c: float(row[c]) for c in columns},
                    )
                ).distance_to_nearest_expected
            )
        held["distance"] = hd
        fault_pool = held[held["distance"] >= hi]
        fault_source = "train_holdout"
        print(f"test split has no clear fault above {hi:.2f}; "
              f"drawing {GROUP_SIZES['clear_fault']} from the train holdout")

    groups = {
        "clear_fault": take(fault_pool, GROUP_SIZES["clear_fault"]),
        "clear_expected": take(
            test[(test["label"] == "rare_event") & (test["distance"] <= lo)],
            GROUP_SIZES["clear_expected"],
        ),
        "clear_nominal": take(
            test[(test["label"] == "nominal") & (test["distance"] <= lo)],
            GROUP_SIZES["clear_nominal"],
        ),
        # ambiguous: inside the band, whatever the true label
        "ambiguous": take(
            test[(test["distance"] > lo) & (test["distance"] < hi)],
            GROUP_SIZES["ambiguous"],
        ),
    }

    expected = {
        "clear_fault": "confirm",
        "clear_expected": "reject",
        "clear_nominal": "reject",
        "ambiguous": None,  # no expected verdict, by design
    }

    entries = []
    for group, rows in groups.items():
        for _, row in rows.iterrows():
            entries.append(
                {
                    "id": f"{row['channel']}@{pd.Timestamp(row['window_start']).isoformat()}",
                    "group": group,
                    "source": fault_source if group == "clear_fault" else "test",
                    "expected_verdict": expected[group],
                    "true_label": row["label"],
                    "channel_id": row["channel"],
                    "window_start": str(row["window_start"]),
                    "score": round(float(row["score"]), 6),
                    "conformal_p": float(row["conformal_p"]),
                    "detector_threshold": round(float(row["threshold"]), 6),
                    "gate_distance": round(float(row["distance"]), 4),
                    "features": {c: float(row[c]) for c in columns},
                }
            )

    payload = {
        "built_from": "test split, detector-flagged windows only",
        "gate_threshold_at_build": threshold,
        "ambiguous_band": [round(lo, 4), round(hi, 4)],
        "band_source": "aksha_core/artifacts/mission2_recognition_calibration.json",
        "note": (
            "Fixed set. Do not resample -- its purpose is that two evaluation runs "
            "measure the same windows. `ambiguous` entries have no expected verdict."
        ),
        "windows": entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"gate threshold      : {threshold}")
    print(f"ambiguous band      : [{lo:.3f}, {hi:.3f}]")
    print(f"golden set          : {len(entries)} windows -> {out}")
    for group in GROUP_SIZES:
        rows = [e for e in entries if e["group"] == group]
        want = expected[group] or "(none)"
        print(f"  {group:<16} {len(rows):>2}  expect={want}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
