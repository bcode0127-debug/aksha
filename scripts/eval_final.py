#!/usr/bin/env python3
"""Assemble the final consolidated eval report for the demo and writeup.

No new modeling: everything here re-runs or re-reads outputs of scripts that
already exist (scripts/train_detector.py's committed artifact,
scripts/calibrate_recognition.py's distance metric, scripts/eval_triage.py's
golden-set and holdout passes). This script's only job is to assemble and
cross-check those into one report, plus compute the two AUC numbers
calibrate_recognition.py already reserves a field for
(`separability_auc.anomaly_vs_nominal_and_rare`) but never fills in.

Numbers claimed elsewhere (code comments, a prior task message) are recorded
under `prior_claims` for comparison, never substituted for what this script
actually measures. See docs/EVALUATION.md for what drifted and why.

Requires, run first (real LLM calls, not mocked):

    python3 scripts/eval_triage.py --golden \
        --out eval/outputs/golden_results.json
    python3 scripts/eval_triage.py \
        --holdout data/processed/mission2_anomaly_holdout.parquet \
        --out eval/outputs/holdout_results.json

Then:

    python3 scripts/eval_final.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_set.json"
GOLDEN_RESULTS = REPO_ROOT / "eval" / "outputs" / "golden_results.json"
HOLDOUT_RESULTS = REPO_ROOT / "eval" / "outputs" / "holdout_results.json"
TEST_PARQUET = REPO_ROOT / "data" / "processed" / "mission2_features.parquet"
UNPURGED_PARQUET = REPO_ROOT / "data" / "processed" / "mission2_features_unpurged.parquet"
REFERENCE_PATH = REPO_ROOT / "aksha_core" / "artifacts" / "mission2_context_reference.json"
CALIBRATION_PATH = REPO_ROOT / "aksha_core" / "artifacts" / "mission2_recognition_calibration.json"
TRAIN_REPORT_PATH = REPO_ROOT / "aksha_core" / "artifacts" / "mission2_iforest_train_report.json"
OUT_PATH = REPO_ROOT / "eval" / "outputs" / "final_report.json"

# Numbers claimed elsewhere, kept only for comparison (CLAUDE.md: no
# unverified numbers reported as fact). Never substituted for a measured
# value below.
PRIOR_CLAIMS = {
    "gate_confirm_rate_workflow_comment": {
        "value": 0.93, "n": 100,
        "source": "aksha_agent/graph/workflow.py:170-176 comment",
    },
    "llm_confirm_rate_workflow_comment": {
        "value": 0.70, "n": 100,
        "source": "aksha_agent/graph/workflow.py:170-176 comment",
    },
    "auc_anomaly_vs_rare_event_workflow_comment": {
        "value": 0.593,
        "source": "aksha_agent/graph/workflow.py:174 comment",
    },
    "gate_confirm_rate_task_message": {
        "value": 0.88,
        "source": "user task message (2026-08-23); no corresponding artifact or report found in-repo",
    },
    "llm_confirm_rate_task_message": {
        "value": 0.52,
        "source": "user task message (2026-08-23); no corresponding artifact or report found in-repo",
    },
    "auc_fault_vs_expected_calibrate_comment": {
        "value": 0.788,
        "source": "scripts/calibrate_recognition.py:58 comment",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def detector_section() -> dict:
    """Section (a): 7 benchmark-style metrics + score separation table.

    Positive class is rare_event UNION anomaly, not anomaly alone: the test
    split carries 4 anomaly positives from a single labelled event
    (docs/mission2-adapter-notes.md), too few to support a headline metric.
    ESA-ADB's own paper pools the two classes for the same reason (arXiv
    2406.17826, Table 2: "detection of all events excluding communication
    gaps"). Anomaly-only counts are still reported, unscored, for
    transparency.
    """
    from aksha_core.detectors.artifact import DetectorArtifact

    artifact = DetectorArtifact.load()
    frame = pd.read_parquet(TEST_PARQUET)
    test = frame[frame["split"] == "test"].copy()
    scored = artifact.score_frame(test)
    test["score"] = scored["score"].to_numpy()
    test["threshold"] = scored["threshold"].to_numpy()
    test["conformal_p"] = scored["conformal_p"].to_numpy()
    test["flagged"] = test["score"] >= test["threshold"]

    y_true = test["label"].isin(["rare_event", "anomaly"]).to_numpy()
    y_pred = test["flagged"].to_numpy()
    y_score = test["score"].to_numpy()

    metrics = {
        "positive_class": "rare_event ∪ anomaly (ESA-ADB pooling convention; see note)",
        "n_positive": int(y_true.sum()),
        "n_negative": int((~y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "point_adjusted_f1": "not computed; CLAUDE.md forbids reporting it standalone and it adds nothing here",
    }

    anomaly_only = {
        "n": int((test["label"] == "anomaly").sum()),
        "flagged": int(test.loc[test["label"] == "anomaly", "flagged"].sum()),
        "note": (
            "4 positives from 1 labelled event (2001-12-14, channels 18-20); "
            "not a meaningful precision/recall/F1/MCC -- "
            "docs/mission2-adapter-notes.md 'The number that constrains what can be evaluated'"
        ),
    }

    separation = {}
    for category in ("nominal", "rare_event", "anomaly"):
        rows = test[test["label"] == category]
        if rows.empty:
            continue
        separation[category] = {
            "n": int(len(rows)),
            "mean_score": float(rows["score"].mean()),
            "above_threshold_fraction": float(rows["flagged"].mean()),
        }

    train_report = json.loads(TRAIN_REPORT_PATH.read_text())
    return {
        "source": (
            "aksha_core/artifacts/mission2_iforest.joblib (iforest-conformal-0.1.0), "
            "re-scored live this session against data/processed/mission2_features.parquet test split"
        ),
        "epsilon": 0.01,
        "metrics_rare_event_union_anomaly": metrics,
        "anomaly_only_raw_counts": anomaly_only,
        "score_separation_by_category": separation,
        "cross_check_against_committed_train_report": train_report.get("test_score_summary"),
    }


def separability_auc_section() -> dict:
    """AUC of the recognition distance -- fills the field
    calibrate_recognition.py reserves and never writes
    (mission2_recognition_calibration.json: 'separability_auc').

    Not new modeling: re-runs the exact distance computation
    (`nearest_rare_distances`) calibrate_recognition.py already defines, on
    the same train-period data, then scores it with sklearn.roc_auc_score two
    ways, matching the two different AUC figures cited in code comments so
    both can be checked against a real, current number.
    """
    calib_mod = load_module("calibrate_recognition", REPO_ROOT / "scripts" / "calibrate_recognition.py")

    reference = json.loads(REFERENCE_PATH.read_text())
    columns = reference["feature_columns"]
    frame = pd.read_parquet(UNPURGED_PARQUET)
    train = frame[frame["window_start"] < calib_mod.TRAIN_END]

    rare_by_channel: dict[str, list[dict]] = {}
    for window in reference["reference_windows"]:
        if window["label"] == "rare_event":
            rare_by_channel.setdefault(window["channel_id"], []).append(window)

    parts = []
    for channel, rows in train.groupby("channel", sort=True):
        stats = reference["channel_stats"].get(channel)
        exemplars = rare_by_channel.get(channel, [])
        if not stats or not exemplars:
            continue
        distances = calib_mod.nearest_rare_distances(rows, exemplars, stats, columns)
        parts.append(pd.DataFrame({"label": rows["label"].to_numpy(), "d": distances}))

    scored = pd.concat(parts, ignore_index=True).dropna(subset=["d"])

    fault_vs_expected = scored[scored["label"].isin(["nominal", "rare_event", "anomaly"])]
    y1 = (fault_vs_expected["label"] == "anomaly").to_numpy()
    auc_fault_vs_expected = float(roc_auc_score(y1, fault_vs_expected["d"].to_numpy()))

    rare_or_anomaly = scored[scored["label"].isin(["rare_event", "anomaly"])]
    y2 = (rare_or_anomaly["label"] == "anomaly").to_numpy()
    auc_anomaly_vs_rare = float(roc_auc_score(y2, rare_or_anomaly["d"].to_numpy()))

    return {
        "source": (
            "recomputed live this session via scripts/calibrate_recognition.py:nearest_rare_distances "
            "on the train period -- the same distance the gate uses"
        ),
        "auc_anomaly_vs_nominal_and_rare_event": auc_fault_vs_expected,
        "auc_anomaly_vs_rare_event_only": auc_anomaly_vs_rare,
        "n_scored": int(len(scored)),
    }


def gate_vs_llm_section() -> dict:
    """Section (b): 200 held-out fault confirm rate, gate vs LLM vs naive baseline.

    Freshly re-run this session (STEP 5) against the current code (post-#16)
    and current calibration artifact -- not read from any prior report.
    """
    results = json.loads(HOLDOUT_RESULTS.read_text())
    ok = [r for r in results if not r.get("error")]
    n = len(ok)
    gate_confirm = sum(1 for r in ok if r["gate_verdict"] == "confirm")
    llm_confirm = sum(1 for r in ok if r["llm_verdict"] == "confirm")

    return {
        "source": (
            "scripts/eval_triage.py --holdout data/processed/mission2_anomaly_holdout.parquet, "
            "live-run this session against the triage graph at HEAD (post-#16)"
        ),
        "n_windows": n,
        "n_errored": len(results) - n,
        "gate_confirm_rate": gate_confirm / n if n else None,
        "gate_confirm_count": gate_confirm,
        "gate_verdict_distribution": dict(Counter(r["gate_verdict"] for r in ok)),
        "llm_confirm_rate": llm_confirm / n if n else None,
        "llm_confirm_count": llm_confirm,
        "llm_verdict_distribution": dict(Counter(r["llm_verdict"] for r in ok)),
        "naive_constant_baseline": {
            "rule": "always predict confirm",
            "confirm_rate": 1.0,
            "note": (
                "trivially 100% on this set, because the holdout is anomaly-only by "
                "construction (STEP 6 / HOLDOUT_ANOMALIES design in build_context_reference.py). "
                "It is not a competing detector, it is the ceiling this one metric cannot rule "
                "out on its own -- the gate's real contribution is its specificity on "
                "nominal/rare_event windows, measured separately in the golden set and the "
                "calibration's ambiguous-band table."
            ),
        },
        "prior_claims": {
            "workflow_comment": {
                "gate": PRIOR_CLAIMS["gate_confirm_rate_workflow_comment"]["value"],
                "llm": PRIOR_CLAIMS["llm_confirm_rate_workflow_comment"]["value"],
                "n": PRIOR_CLAIMS["gate_confirm_rate_workflow_comment"]["n"],
                "source": PRIOR_CLAIMS["gate_confirm_rate_workflow_comment"]["source"],
            },
            "task_message": {
                "gate": PRIOR_CLAIMS["gate_confirm_rate_task_message"]["value"],
                "llm": PRIOR_CLAIMS["llm_confirm_rate_task_message"]["value"],
                "source": PRIOR_CLAIMS["gate_confirm_rate_task_message"]["source"],
            },
        },
    }


def golden_section(calibration: dict) -> dict:
    """Section (c): 22-window golden set, group / expected / matched / verdict mix."""
    golden = json.loads(GOLDEN_PATH.read_text())
    results = json.loads(GOLDEN_RESULTS.read_text())
    windows = golden["windows"]
    if len(windows) != len(results):
        raise ValueError("golden set and results out of sync; re-run --golden")

    rows = []
    hits = total = 0
    by_group: dict[str, dict] = {}
    for w, r in zip(windows, results):
        group = w["group"]
        expected = w["expected_verdict"]
        verdict = r["final_verdict"]
        rows.append({
            "channel_id": w["channel_id"],
            "window_start": w["window_start"],
            "group": group,
            "true_label": w["true_label"],
            "expected_verdict": expected,
            "final_verdict": verdict,
        })
        bucket = by_group.setdefault(group, {"n": 0, "expected": expected, "verdicts": Counter()})
        bucket["n"] += 1
        bucket["verdicts"][verdict] += 1
        if expected is not None:
            total += 1
            hits += int(verdict == expected)

    for bucket in by_group.values():
        bucket["verdicts"] = dict(bucket["verdicts"])

    built_band = [golden["ambiguous_band"][0], golden["ambiguous_band"][1]]
    current_band = [calibration["ambiguous_band"]["low"], calibration["ambiguous_band"]["high"]]

    return {
        "source": (
            "tests/fixtures/golden_set.json (scripts/build_golden_set.py), scored live "
            "this session via scripts/eval_triage.py --golden"
        ),
        "n_windows": len(windows),
        "by_group": by_group,
        "scored_agreement": f"{hits}/{total}",
        "scored_agreement_hits": hits,
        "scored_agreement_total": total,
        "scored_agreement_fraction": hits / total if total else None,
        "windows": rows,
        "built_at_gate_threshold": golden["gate_threshold_at_build"],
        "built_at_ambiguous_band": built_band,
        "current_gate_threshold": calibration["operating_threshold"]["distance"],
        "current_ambiguous_band": current_band,
        "calibration_unchanged_since_build": (
            golden["gate_threshold_at_build"] == calibration["operating_threshold"]["distance"]
            and built_band == current_band
        ),
    }


def live_confirmation_section() -> dict:
    """Section (d): today's real end-to-end run through the redeployed service.

    Captured interactively this session: published via
    `scripts/publish_stub.py --window anomaly` to `telemetry-in`, scored by
    detector-service, triaged by triage-service-00006-gnw (the numpy-fix
    revision from PR #16), read back from Firestore
    `incidents/frag-anomaly-1787527553`, delivery confirmed via
    triage_service logs (`SlackNotifier.post()` returned `delivered`, which
    requires a 2xx from the real webhook -- aksha_agent/infra/slack.py:178-185).
    """
    return {
        "incident_id": "frag-anomaly-1787527553",
        "revision": "triage-service-00006-gnw",
        "channel_id": "channel_20",
        "gate_distance": 1.4131,
        "gate_threshold": 2.0658,
        "ambiguous_band": [1.3428, 2.7888],
        "gate_verdict": "disputed",
        "llm_verdict": "confirm",
        "final_verdict": "disputed",
        "routing_destination": "log",
        "routing_outcome": "delivered",
        "severity": "Advisory",
        "conformal_p": 0.0006257561219807267,
        "anomaly_score": 0.10340437637906841,
        "note": (
            "the audit column doing its job: the LLM read this window as a genuine fault "
            "(confirm), the deterministic gate placed the calibrated distance inside the "
            "ambiguous band and returned disputed, and disputed is what routed -- the model's "
            "read never touched the outcome (aksha_agent/graph/workflow.py:168, "
            "'the gate decides, alone')."
        ),
    }


def cost_section(golden_results: list[dict], holdout_results: list[dict]) -> dict:
    def agg(results: list[dict]) -> dict:
        ok = [r for r in results if not r.get("error")]
        total_tokens = sum(r["tokens"] for r in ok)
        total_thought = sum(r["thought_tokens"] for r in ok)
        total_seconds = sum(r["seconds"] for r in ok)
        return {
            "n": len(ok),
            "total_tokens": total_tokens,
            "thought_tokens": total_thought,
            "tokens_per_incident": total_tokens // len(ok) if ok else None,
            "mean_seconds_per_incident": round(total_seconds / len(ok), 2) if ok else None,
        }

    return {
        "source": (
            "in-process scripts/eval_triage.py runs this session (golden set, holdout) "
            "plus the one live Cloud Run incident in section (d)"
        ),
        "golden_set_22": agg(golden_results),
        "holdout_200": agg(holdout_results),
        "live_cloud_run_incident": {
            "incident_id": "frag-anomaly-1787527553",
            "tokens": 6834,
            "triage_seconds": 7.0,
            "note": "includes Pub/Sub + Firestore + Slack delivery overhead the in-process runs above do not pay",
        },
    }


def main() -> int:
    if not HOLDOUT_RESULTS.exists() or not GOLDEN_RESULTS.exists():
        print(
            "run the eval_triage.py golden and holdout passes first (see module docstring)",
            file=sys.stderr,
        )
        return 1

    golden_results = json.loads(GOLDEN_RESULTS.read_text())
    holdout_results = json.loads(HOLDOUT_RESULTS.read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        commit = None

    report = {
        "generated_by": "scripts/eval_final.py",
        "commit": commit,
        "dataset_disclosure": (
            "All numbers below are computed on ESA-ADB's Mission2 benchmark "
            "(aksha_core/data/mission2.py), not on OPSSAT-AD's published dataset.csv/"
            "segments.csv (data/dataset.csv). This is the build dataset per ADR-015, "
            "which supersedes ADR-006's original OPSSAT-AD choice; PRD and README are "
            "updated to match. See docs/EVALUATION.md for the full disclosure."
        ),
        "detector": detector_section(),
        "separability_auc": separability_auc_section(),
        "gate_vs_llm": gate_vs_llm_section(),
        "golden_set": golden_section(calibration),
        "live_confirmation": live_confirmation_section(),
        "cost": cost_section(golden_results, holdout_results),
        "prior_claims": PRIOR_CLAIMS,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
