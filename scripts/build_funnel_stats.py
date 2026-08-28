#!/usr/bin/env python3
"""Precompute the dashboard funnel strip's static numbers, from real artifacts.

Three of the four funnel numbers describe the offline build (the Mission2
adapter's output and the committed detector scoring it) and never change
between dashboard page loads on the same commit -- computing them fresh on
every load would mean shipping the 27 MB feature parquet into the Cloud Run
image and rescoring 168k rows per request, for a number that is always the
same. So this script computes them once, from the real files, and writes a
small committed artifact the dashboard reads instead -- same pattern as
scripts/eval_final.py writing eval/outputs/final_report.json, except this one
is small and static enough to commit (aksha_core/artifacts/ already commits
trained artifacts on purpose, see .gitignore).

The fourth number (incidents triaged) is NOT here: it changes every time the
pipeline runs, so the dashboard reads it live from Firestore on every load.

    python3 scripts/build_funnel_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aksha_core.detectors.artifact import DetectorArtifact  # noqa: E402

FEATURES_PARQUET = REPO_ROOT / "data" / "processed" / "mission2_features.parquet"
OUT_PATH = REPO_ROOT / "aksha_core" / "artifacts" / "mission2_funnel_stats.json"


def main() -> int:
    if not FEATURES_PARQUET.exists():
        print(f"missing {FEATURES_PARQUET} -- run python3 -m aksha_core.data.mission2 first", file=sys.stderr)
        return 1

    frame = pd.read_parquet(FEATURES_PARQUET)
    windows_built = int(len(frame))

    test = frame[frame["split"] == "test"].copy()
    windows_test = int(len(test))

    artifact = DetectorArtifact.load()
    scored = artifact.score_frame(test)
    flagged = int((scored["score"] >= scored["threshold"]).sum())

    payload = {
        "source": "scripts/build_funnel_stats.py, from data/processed/mission2_features.parquet "
                   "and aksha_core/artifacts/mission2_iforest.joblib",
        "detector_version": artifact.detector_version,
        "windows_built": windows_built,
        "windows_built_note": "total rows in the Mission2 adapter's committed feature table "
                               "(train + calibration + test, purged split)",
        "windows_test": windows_test,
        "windows_test_note": "test-split rows, scored live by the committed detector",
        "windows_flagged": flagged,
        "windows_flagged_note": f"test-split rows with score >= the epsilon={artifact.calibrator.epsilon} "
                                 "conformal threshold",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"windows_built={windows_built:,} windows_test={windows_test:,} windows_flagged={flagged:,}")
    print(f"written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
