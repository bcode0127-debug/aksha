#!/usr/bin/env python3
"""Train the Mission2 detector: parquet in, committed artifact out.

This is the reproducibility path. From a clone with the feature parquet
present:

    python3 scripts/train_detector.py

Deterministic: the Isolation Forest is seeded, and nothing about the fit
depends on row order. Re-running reproduces the artifact's scores exactly.

Fits on the training split only, calibrates on the calibration split only, and
reports the test split's score distribution without computing any metric — the
split has 4 anomaly windows from a single event (docs/mission2-adapter-notes.md),
which is not enough to support an anomaly-class metric, and this script is not
the place to pretend otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_core.conformal.split import (  # noqa: E402
    DEFAULT_EPSILON,
    QUANTILE_GRID,
    SplitConformalCalibrator,
)
from aksha_core.detectors.artifact import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_ARTIFACT_NAME,
    DetectorArtifact,
)
from aksha_core.detectors.iforest import fit_detector  # noqa: E402

DEFAULT_PARQUET = "data/processed/mission2_features.parquet"
DETECTOR_VERSION = "iforest-conformal-0.1.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR / DEFAULT_ARTIFACT_NAME))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--version", default=DETECTOR_VERSION)
    args = parser.parse_args()

    parquet = Path(args.parquet)
    if not parquet.exists():
        print(
            f"feature table not found at {parquet}\n"
            "build it first:  python3 -m aksha_core.data.mission2",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_parquet(parquet)
    train = frame[frame["split"] == "train"]
    calibration = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "test"]

    print(f"feature table : {parquet}  ({len(frame):,} rows)")
    print(f"  train       : {len(train):,}  (labels: {sorted(train['label'].unique())})")
    print(f"  calibration : {len(calibration):,}  (labels: {sorted(calibration['label'].unique())})")
    print(f"  test        : {len(test):,}  (labels: {sorted(test['label'].unique())})")

    # Refuse to fit on contaminated splits rather than silently producing a
    # detector calibrated against anomalies it was supposed to never see.
    for name, split in (("train", train), ("calibration", calibration)):
        contaminated = set(split["label"].unique()) - {"nominal"}
        if contaminated:
            print(
                f"\nERROR: {name} split contains non-nominal labels {sorted(contaminated)}.\n"
                "The adapter is supposed to purge these; refusing to fit.",
                file=sys.stderr,
            )
            return 1

    started = time.perf_counter()
    detector = fit_detector(train)
    fit_seconds = time.perf_counter() - started
    print(f"\nfit           : {fit_seconds:.1f}s on {detector.n_train_rows:,} rows, "
          f"{len(detector.feature_columns)} features")
    print(f"  params      : {detector.params}")
    if detector.impute_values:
        print(f"  imputed     : {detector.impute_values}")

    calib_started = time.perf_counter()
    calibration_scores = detector.raw_scores(calibration)
    calibrator = SplitConformalCalibrator(calibration_scores, epsilon=args.epsilon)
    calib_seconds = time.perf_counter() - calib_started
    print(f"calibrate     : {calib_seconds:.1f}s on {calibrator.n:,} nominal windows")
    print(f"  threshold @ eps={args.epsilon}: {calibrator.threshold_at():.6f}")
    print(f"  realised alarm rate on calibration: "
          f"{calibrator.realised_alarm_rate(calibration_scores):.5f} (bound: {args.epsilon})")

    artifact = DetectorArtifact(
        detector_version=args.version,
        detector=detector,
        calibrator=calibrator,
        metadata={
            "source_parquet": str(parquet),
            "fit_seconds": round(fit_seconds, 2),
            "train_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
        },
    )
    out_path = artifact.save(args.out)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nartifact      : {out_path}  ({size_kb:,.1f} KB)")
    print(f"  sidecar     : {out_path.with_suffix('.json').name}")

    # --- test split score distribution, no metrics ---------------------------
    scored = artifact.score_frame(test, epsilon=args.epsilon)
    scored = scored.assign(label=test["label"].to_numpy())
    threshold = artifact.threshold(args.epsilon)

    print(f"\ntest split score distribution (threshold {threshold:.6f} @ eps={args.epsilon})")
    print(f"{'label':<12} {'n':>8} {'mean':>10} {'std':>9} {'p50':>10} {'p95':>10} "
          f"{'max':>10} {'above thr':>10}")
    for label in ("nominal", "rare_event", "anomaly"):
        rows = scored[scored["label"] == label]
        if rows.empty:
            continue
        s = rows["score"]
        above = float((s >= threshold).mean())
        print(f"{label:<12} {len(rows):>8,} {s.mean():>10.4f} {s.std():>9.4f} "
              f"{s.median():>10.4f} {s.quantile(0.95):>10.4f} {s.max():>10.4f} "
              f"{above:>9.1%}")

    print("\nfraction above threshold by significance level")
    header = f"{'eps':<10}" + "".join(f"{l:>14}" for l in ("nominal", "rare_event", "anomaly"))
    print(header)
    for eps in QUANTILE_GRID:
        thr = artifact.threshold(eps)
        cells = ""
        for label in ("nominal", "rare_event", "anomaly"):
            rows = scored[scored["label"] == label]
            frac = float((rows["score"] >= thr).mean()) if len(rows) else float("nan")
            cells += f"{frac:>13.2%} "
        print(f"{eps:<10}{cells}")

    print("\nconformal_p direction: LOW = anomalous. No metric is reported here; "
          "the test split has 4 anomaly windows from one event.")

    stats_path = out_path.with_name(out_path.stem + "_train_report.json")
    stats_path.write_text(
        json.dumps(
            {
                "detector_version": args.version,
                "fit_seconds": round(fit_seconds, 2),
                "artifact_kb": round(size_kb, 1),
                "threshold_table": {str(k): v for k, v in calibrator.threshold_table().items()},
                "test_score_summary": {
                    label: {
                        "n": int((scored["label"] == label).sum()),
                        "mean": float(scored.loc[scored["label"] == label, "score"].mean()),
                        "above_threshold": float(
                            (scored.loc[scored["label"] == label, "score"] >= threshold).mean()
                        ),
                    }
                    for label in ("nominal", "rare_event", "anomaly")
                    if (scored["label"] == label).any()
                },
            },
            indent=2,
        )
    )
    print(f"report        : {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
