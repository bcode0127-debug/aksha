"""Tests for the one-sided recognition evidence and its calibration.

Three guarantees, each of which fails silently if broken: the verifier must not
be able to see an anomaly exemplar, the held-out anomaly windows must not appear
in the reference it matches against, and the calibration must be computed on
train-period data only.

The middle one is the subtle one. If a held-out window is also an exemplar, the
fault-sensitivity measurement compares a window against itself, scores distance
zero, and reports a capability the system does not have.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aksha_agent.graph import workflow as wf
from aksha_agent.graph.context import ReferenceContextProvider
from aksha_agent.graph.schemas import DetectionSummary, RecognitionEvidence, ExplainerInput
from aksha_core.data.mission2 import TRAIN_END

ARTIFACTS = Path(__file__).resolve().parent.parent / "aksha_core" / "artifacts"
REFERENCE_PATH = ARTIFACTS / "mission2_context_reference.json"
CALIBRATION_PATH = ARTIFACTS / "mission2_recognition_calibration.json"
HOLDOUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "mission2_anomaly_holdout.parquet"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE_PATH.exists() or not CALIBRATION_PATH.exists(),
    reason="context reference or calibration not built",
)


@pytest.fixture(scope="module")
def provider() -> ReferenceContextProvider:
    return ReferenceContextProvider(REFERENCE_PATH, CALIBRATION_PATH)


@pytest.fixture(scope="module")
def calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text())


def _detection(provider: ReferenceContextProvider, channel: str = "channel_20"):
    stats = provider.channel_stats[channel]
    return DetectionSummary(
        fragment_id="t",
        channel_id=channel,
        t_start="2001-12-14T19:00:00Z",
        t_end="2001-12-14T20:00:00Z",
        score=0.1,
        threshold=-0.0118,
        conformal_p=0.0006,
        detector_version="test",
        features={c: stats["mean"][c] for c in provider.feature_columns},
    )


# --- the anomaly comparison cannot reach the verifier -------------------------


def test_verifier_input_has_no_anomaly_exemplar_field():
    """One-sided by construction. The anomaly comparison measured near-chance
    (AUC 0.593 anomaly vs rare_event) and dragged the verdict toward reject, so
    it is removed from the decision inputs rather than merely de-emphasised.
    """
    fields = set(ExplainerInput.model_fields)
    assert "nearest_labeled" not in fields
    assert "recognition" in fields


def test_recognition_evidence_exposes_only_the_expected_side():
    fields = set(RecognitionEvidence.model_fields)
    assert "distance_to_nearest_expected" in fields
    assert "percentile_among_rare_events" in fields
    for forbidden in ("distance_to_nearest_anomaly", "anomaly_exemplar", "nearest_anomaly"):
        assert forbidden not in fields, f"{forbidden} reintroduces the two-sided comparison"


def test_matched_exemplar_is_always_an_expected_pattern(provider):
    for channel in ("channel_18", "channel_20", "channel_23"):
        evidence = provider.recognition(_detection(provider, channel))
        if evidence.matched_exemplar is not None:
            assert evidence.matched_exemplar.label == "rare_event", (
                "the verifier was matched against something other than a known "
                "expected pattern"
            )


def test_verifier_instruction_frames_the_number_as_outlierness():
    """Three framings have now been ruled out by measurement, and none may
    return: hypothesis-support (always yes for a deviant window), two-sided
    comparison (AUC 0.593), and recognition/resemblance (nominal windows sit
    CLOSER to rare-event exemplars than rare events do to each other, so
    "recognised pattern" describes something false).
    """
    text = wf.EXPLAINER_INSTRUCTION.lower()
    assert "outlierness" in text
    assert "percentile" in text
    assert "does not mean" in text  # the explicit warning against resemblance
    assert "does the evidence support" not in text
    assert "closest to the anomaly exemplar" not in text
    assert "recognised expected pattern" not in text
    assert "recognized expected pattern" not in text


# --- held-out windows are absent from the reference ---------------------------


@pytest.mark.skipif(not HOLDOUT_PATH.exists(), reason="holdout not built")
def test_holdout_windows_are_not_in_the_reference():
    """Otherwise the fault-sensitivity number is measured against the windows
    themselves and means nothing.
    """
    reference = json.loads(REFERENCE_PATH.read_text())
    exemplars = {
        (w["channel_id"], str(w["window_start"])) for w in reference["reference_windows"]
    }
    holdout = pd.read_parquet(HOLDOUT_PATH)
    held = {(str(r["channel"]), str(r["window_start"])) for _, r in holdout.iterrows()}

    overlap = exemplars & held
    assert not overlap, f"{len(overlap)} held-out windows are also exemplars: {list(overlap)[:3]}"


@pytest.mark.skipif(not HOLDOUT_PATH.exists(), reason="holdout not built")
def test_holdout_is_anomalies_from_the_train_period_only():
    holdout = pd.read_parquet(HOLDOUT_PATH)
    assert set(holdout["label"].unique()) == {"anomaly"}
    assert (holdout["window_start"] < TRAIN_END).all(), (
        "a held-out window falls at or after the train cut"
    )


# --- the calibration is train-only --------------------------------------------


def test_calibration_declares_and_respects_the_train_cut(calibration):
    assert pd.Timestamp(calibration["train_cut"]) == pd.Timestamp(TRAIN_END)
    assert "train period only" in calibration["built_from"]


def test_calibration_covers_every_category_with_real_support(calibration):
    for category in ("nominal", "rare_event", "anomaly"):
        stats = calibration["by_category"][category]
        assert stats["n"] > 1000, f"{category} calibrated on only {stats['n']} windows"


def test_calibration_grid_is_monotonic_and_complete(calibration):
    values = calibration["rare_event_percentile_values"]
    grid = calibration["rare_event_percentile_grid"]
    assert len(values) == len(grid) == 101
    assert all(b >= a for a, b in zip(values, values[1:])), "percentile curve is not monotonic"


def test_percentile_placement_is_ordered(provider):
    """A larger distance must never map to a smaller percentile -- the number the
    verifier reasons over has to mean what the prompt says it means.
    """
    values = [provider._percentile_of(d) for d in (0.1, 1.0, 5.0, 20.0, 500.0, 1e9)]
    assert all(b >= a for a, b in zip(values, values[1:])), values
    assert 0.0 <= values[0] <= 100.0
    # The rare-event distribution has a long tail, so even a distance of 500
    # is not the maximum; only something far beyond the tail saturates.
    assert values[-1] == 100.0
    assert values[-2] < 100.0


def test_recognition_is_computed_without_the_calibration_if_absent(tmp_path):
    """A missing calibration must degrade to 'unrecognised', not crash the graph
    or silently report percentile 0 (which would read as 'recognised').
    """
    provider = ReferenceContextProvider(REFERENCE_PATH, tmp_path / "missing.json")
    evidence = provider.recognition(_detection(provider))
    assert evidence.percentile_among_rare_events == 100.0


# --- determinism ---------------------------------------------------------------


def test_verify_node_is_configured_deterministically():
    """A verdict that varied 7:1 across identical inputs was a defect in itself.
    Verified configurable against google-adk 2.3.0.
    """
    graph = wf.build_workflow(
        detection={
            "fragment_id": "f",
            "channel_id": "channel_20",
            "t_start": "2001-12-14T19:00:00Z",
            "t_end": "2001-12-14T20:00:00Z",
            "score": 0.1,
            "threshold": -0.0118,
            "conformal_p": 0.0006,
            "detector_version": "t",
            "features": {},
        },
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        explain_model="gemini-3.5-flash-lite",
    )
    explain = next(
        node
        for edge in graph.graph.edges
        for node in (edge.from_node, edge.to_node)
        if node.name == "explain"
    )
    assert explain.generate_content_config is not None
    assert explain.generate_content_config.temperature == 0.0
