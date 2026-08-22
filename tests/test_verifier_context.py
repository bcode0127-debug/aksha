"""Tests for the labeled context reference and the reframed verifier.

These guard the fix itself: that the reference cannot leak past the train cut,
that it actually contains an exemplar of every class the verifier must
discriminate between, that retrieval returns one of each, and that the verifier
is asked to adjudicate rather than to agree.

The last one matters more than it looks. Asking "does the evidence support the
hypothesis" is a question a rare-but-expected event answers YES to, because it
genuinely is statistically deviant — so that phrasing makes the verifier
structurally incapable of rejecting anything, no matter how good the model is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aksha_agent.graph import workflow as wf
from aksha_agent.graph.context import ALWAYS_REPORTED, ReferenceContextProvider
from aksha_agent.graph.schemas import DetectionSummary, VerifierStatus
from aksha_core.data.mission2 import TRAIN_END

REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "aksha_core"
    / "artifacts"
    / "mission2_context_reference.json"
)
CATEGORIES = ("nominal", "rare_event", "anomaly")

pytestmark = pytest.mark.skipif(
    not REFERENCE_PATH.exists(), reason="context reference not built"
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


@pytest.fixture(scope="module")
def provider() -> ReferenceContextProvider:
    return ReferenceContextProvider(REFERENCE_PATH)


# --- nothing at or after the train cut may enter ------------------------------


def test_no_exemplar_at_or_after_the_train_cut(reference):
    """Retrieval material from the calibration or test period would put
    evaluation rows into the reasoning context of a run being evaluated on them.
    """
    cutoff = pd.Timestamp(TRAIN_END)
    offenders = [
        w["window_start"]
        for w in reference["reference_windows"]
        if pd.Timestamp(w["window_start"]) >= cutoff
    ]
    assert not offenders, f"{len(offenders)} exemplars at/after {cutoff}: {offenders[:3]}"


def test_reference_declares_the_cut_it_was_built_under(reference):
    assert pd.Timestamp(reference["train_cut"]) == pd.Timestamp(TRAIN_END)
    assert pd.Timestamp(reference["latest_exemplar"]) < pd.Timestamp(TRAIN_END)


# --- every category the verifier must judge is represented --------------------


def test_every_category_has_exemplars(reference):
    """The bug this replaces: a reference built from the purged training table
    was 100% nominal, so the verifier had no example of the class it was meant
    to reject and could not discriminate.
    """
    present = {w["label"] for w in reference["reference_windows"]}
    for category in CATEGORIES:
        assert category in present, f"no {category} exemplars — verifier cannot reject"


def test_categories_are_not_trivially_imbalanced(reference):
    counts = {
        c: sum(1 for w in reference["reference_windows"] if w["label"] == c)
        for c in CATEGORIES
    }
    assert all(n >= 50 for n in counts.values()), counts


def test_every_channel_covers_every_category(reference):
    """Retrieval is per channel, so a channel missing a category would silently
    give the verifier a one-sided comparison for windows on that channel.
    """
    seen: dict[str, set[str]] = {}
    for window in reference["reference_windows"]:
        seen.setdefault(window["channel_id"], set()).add(window["label"])
    for channel, categories in sorted(seen.items()):
        assert set(CATEGORIES) <= categories, f"{channel} missing {set(CATEGORIES) - categories}"


# --- retrieval returns one exemplar per category ------------------------------


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


def test_retrieval_returns_one_exemplar_of_each_category(provider):
    neighbours = provider.nearest(_detection(provider))
    labels = [n.label for n in neighbours]
    assert labels == list(CATEGORIES), labels
    assert len({n.window_start for n in neighbours}) == len(neighbours)


def test_every_retrieved_exemplar_carries_a_distance(provider):
    for neighbour in provider.nearest(_detection(provider)):
        assert neighbour.distance >= 0.0
        assert neighbour.features, "an exemplar with no features tells the model nothing"


def test_retrieval_stays_on_the_requested_channel(provider):
    for channel in ("channel_18", "channel_23", "channel_28"):
        neighbours = provider.nearest(_detection(provider, channel))
        assert {n.channel_id for n in neighbours} == {channel}
        assert [n.label for n in neighbours] == list(CATEGORIES)


def test_context_provider_returns_history_and_exemplars_together(provider):
    history, neighbours = provider(_detection(provider))
    assert history.channel_id == "channel_20"
    assert history.reference_windows > 0
    assert [n.label for n in neighbours] == list(CATEGORIES)


# --- the investigator is given the evidence its hypotheses require ------------


def test_decision_relevant_features_are_always_reported(provider):
    """command_induced, correlated_channel and gap_artifact can only be chosen
    on evidence that ranking by |z| alone could crowd out.
    """
    history, _ = provider(_detection(provider))
    reported = {stat.feature for stat in history.most_deviant}
    for feature in ALWAYS_REPORTED:
        assert feature in reported, f"{feature} withheld; a hypothesis becomes unchoosable"


# --- the verifier adjudicates rather than agrees ------------------------------


def test_verifier_instruction_asks_for_adjudication_not_hypothesis_support():
    """The framing must be the fault-vs-expected question.

    Note the instruction is allowed to *mention* the hypothesis-support framing,
    because it explicitly rejects it. What must not happen is that framing being
    asked for: a rare event genuinely IS deviant, so "does the evidence support
    the hypothesis" has the answer yes for faults and non-faults alike, which
    makes rejection structurally impossible however good the model is.
    """
    text = wf.VERIFIER_INSTRUCTION.lower()
    assert "genuine fault" in text
    assert "unusual-but-expected" in text or "unusual but expected" in text

    support_framing = "does the evidence support"
    if support_framing in text:
        prefix = text.split(support_framing)[0]
        assert prefix.rstrip().endswith("not") or "is not" in prefix[-40:], (
            "the hypothesis-support framing appears without being negated"
        )


def test_verifier_instruction_directs_reasoning_over_exemplar_distances():
    text = wf.VERIFIER_INSTRUCTION.lower()
    assert "distance" in text
    for category in CATEGORIES:
        assert category.replace("_", "_") in text.replace(" ", "_") or category in text


def test_verifier_instruction_states_rejection_is_expected():
    """Without this the model treats reject as a failure state and avoids it."""
    text = wf.VERIFIER_INSTRUCTION.lower()
    assert "reject" in text
    assert "not a failure" in text or "useful" in text


def test_verifier_status_meanings_are_documented_on_the_enum():
    assert VerifierStatus.__doc__ and "fault" in VerifierStatus.__doc__.lower()


def test_investigator_instruction_points_at_the_features_per_hypothesis():
    text = wf.INVESTIGATOR_INSTRUCTION
    assert "tc_count_in_window" in text and "seconds_since_last_tc" in text
    assert "mahalanobis" in text
    assert "gap_fraction" in text
    assert "rule the others out" in text or "ruled the others out" in text
