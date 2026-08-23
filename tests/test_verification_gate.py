"""Tests for the deterministic verification gate.

The property under test is that the LLM cannot change the outcome AT ALL — not
that its influence is bounded, but that it has none. That is provable rather
than samplable because `gate_verdict()` takes no LLM argument: there is no
parameter through which an opinion could enter.

The earlier design let the model escalate a gate-reject to `disputed`. It is
gone, and these tests exist partly to stop it coming back: a reintroduced
override would have to change `gate_verdict`'s signature, and every exhaustive
test below would fail.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from aksha_agent.graph import workflow as wf
from aksha_agent.graph.context import ReferenceContextProvider
from aksha_agent.graph.schemas import Verdict
from tests.test_graph_workflow import DETECTION, run_graph

ARTIFACTS = Path(__file__).resolve().parent.parent / "aksha_core" / "artifacts"
CALIBRATION_PATH = ARTIFACTS / "mission2_recognition_calibration.json"
GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_set.json"

ALL_STATUSES = [Verdict.CONFIRM, Verdict.REJECT, Verdict.DISPUTED, None]
BAND = (1.0, 3.0)


# --- the LLM cannot change the verdict -----------------------------------------


def test_gate_verdict_has_no_parameter_an_llm_verdict_could_enter_through():
    """The structural claim, checked structurally.

    Every behavioural test below could be satisfied by a function that happens
    to ignore its LLM argument today. This one says there is no such argument.
    """
    params = list(inspect.signature(wf.gate_verdict).parameters)
    assert params == ["distance", "band"], params


def test_the_override_helper_is_gone():
    """A reintroduced `apply_override` would silently restore model influence."""
    assert not hasattr(wf, "apply_override"), (
        "apply_override is back; the LLM can affect the verdict again"
    )


@pytest.mark.parametrize("llm", ALL_STATUSES)
@pytest.mark.parametrize("distance", [0.0, 0.99, 1.0, 1.5, 2.9, 3.0, 3.5, 99.0])
def test_llm_verdict_cannot_alter_the_final_verdict(distance, llm):
    """Exhaustive over gate outcome x every verdict the model can return.

    Runs the graph with the model pinned to `llm`, and asserts the final verdict
    is whatever the distance alone implies.
    """
    expected = wf.gate_verdict(distance, BAND)
    incident, _ = run_graph(
        verifier_status=llm.value if llm else "banana",
        gate_distance=distance,
        band=BAND,
    )
    assert incident["final_verdict"] == expected.value, (
        f"llm={llm} changed the verdict at distance {distance}"
    )
    assert incident["gate_verdict"] == expected.value


@pytest.mark.parametrize("llm", ALL_STATUSES)
def test_a_confirming_model_cannot_promote_a_reject(llm):
    incident, _ = run_graph(
        verifier_status=llm.value if llm else "banana", gate_distance=0.5, band=BAND
    )
    assert incident["final_verdict"] == Verdict.REJECT.value
    assert incident["severity"] == "Advisory"


@pytest.mark.parametrize("llm", ALL_STATUSES)
def test_a_rejecting_model_cannot_de_escalate_a_confirm(llm):
    incident, _ = run_graph(
        verifier_status=llm.value if llm else "banana", gate_distance=50.0, band=BAND
    )
    assert incident["final_verdict"] == Verdict.CONFIRM.value


# --- disputed comes from the band, and only from the band ----------------------


@pytest.mark.parametrize(
    "distance,expected",
    [
        (0.0, Verdict.REJECT),
        (1.0, Verdict.REJECT),      # inclusive lower bound
        (1.0001, Verdict.DISPUTED),
        (2.0, Verdict.DISPUTED),
        (2.9999, Verdict.DISPUTED),
        (3.0, Verdict.CONFIRM),     # inclusive upper bound
        (100.0, Verdict.CONFIRM),
    ],
)
def test_verdict_is_a_pure_function_of_distance_and_band(distance, expected):
    assert wf.gate_verdict(distance, BAND) is expected


@pytest.mark.parametrize("llm", ALL_STATUSES)
def test_disputed_arises_only_inside_the_band(llm):
    """Outside the band the verdict is never `disputed`, whatever the model says.

    This is what changed: `disputed` used to mean "the model disagreed", which
    made it a report on model behaviour. It now means one thing — the calibrated
    distance did not decide.
    """
    for distance in (0.0, 0.5, 1.0, 3.0, 4.0, 40.0):
        verdict = wf.gate_verdict(distance, BAND)
        assert verdict is not Verdict.DISPUTED, distance
        incident, _ = run_graph(
            verifier_status=llm.value if llm else "banana",
            gate_distance=distance,
            band=BAND,
        )
        assert incident["final_verdict"] != Verdict.DISPUTED.value


def test_disputed_is_reachable_only_between_the_bounds():
    low, high = BAND
    inside = [d for d in (0.0, 1.0, 2.0, 3.0, 5.0) if wf.gate_verdict(d, BAND) is Verdict.DISPUTED]
    assert inside == [2.0]
    assert all(low < d < high for d in inside)


# --- the band bounds are loaded, not hardcoded ---------------------------------


def test_band_bounds_come_from_the_calibration_artifact():
    calibration = json.loads(CALIBRATION_PATH.read_text())
    stored = calibration["ambiguous_band"]
    provider = ReferenceContextProvider(calibration_path=CALIBRATION_PATH)
    assert provider.ambiguous_band == (
        pytest.approx(stored["low"]),
        pytest.approx(stored["high"]),
    )


def test_no_band_literal_is_hardcoded_in_the_graph():
    """Recalibrating must move the band. A literal here would silently not."""
    source = Path(wf.__file__).read_text()
    for literal in ("1.3428", "2.7888", "1.34", "2.79", "0.35"):
        assert literal not in source, f"band literal {literal} hardcoded in workflow.py"


def test_missing_band_in_the_artifact_yields_none_rather_than_a_guess(tmp_path):
    provider = ReferenceContextProvider(calibration_path=tmp_path / "absent.json")
    assert provider.ambiguous_band is None


def test_a_reversed_band_is_rejected_rather_than_used(tmp_path):
    """low >= high would make `disputed` unreachable and silently disable a
    third of the gate's behaviour."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"ambiguous_band": {"low": 5.0, "high": 1.0}}))
    assert ReferenceContextProvider(calibration_path=path).ambiguous_band is None


# --- the gate is 0 LLM ---------------------------------------------------------


def test_gate_node_makes_no_llm_call():
    _, trace = run_graph(verifier_status="confirm")
    steps = dict(trace)
    assert "verification_gate" in steps, "the gate did not run"
    assert steps["verification_gate"]["llm_calls"] == 0


def test_graph_still_makes_exactly_two_llm_calls():
    """Adding the gate must not add a third reasoning call."""
    graph = wf.build_workflow(
        detection=DETECTION,
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        explain_model="gemini-3.5-flash-lite",
    )
    agents = {
        node.name
        for edge in graph.graph.edges
        for node in (edge.from_node, edge.to_node)
        if getattr(node, "model", None)
    }
    assert agents == {"investigate", "explain"}


def test_gate_sits_between_explain_and_the_router():
    graph = wf.build_workflow(
        detection=DETECTION,
        incident_id="i",
        investigate_model="gemini-3.5-flash",
        explain_model="gemini-3.5-flash-lite",
    )
    edges = {(e.from_node.name, e.to_node.name) for e in graph.graph.edges if e.route is None}
    assert ("explain", "verification_gate") in edges
    assert ("verification_gate", "route_by_status") in edges
    assert ("explain", "route_by_status") not in edges


# --- both verdicts are recorded, so disagreement is visible --------------------


def test_incident_records_gate_and_llm_verdicts_separately():
    incident, trace = run_graph(verifier_status="confirm")
    assert incident["gate_verdict"] in {s.value for s in Verdict}
    assert incident["llm_verdict"] == "confirm"
    assert incident["final_verdict"] in {s.value for s in Verdict}
    assert incident["gate_threshold"] is not None
    assert incident["band_low"] is not None and incident["band_high"] is not None

    step = dict(trace)["verification_gate"]
    assert {"gate_verdict", "llm_verdict", "final_verdict", "gate_llm_agree"} <= set(step)


def test_final_verdict_always_equals_the_gate_verdict():
    """They are equal by construction. Asserting it here means a future change
    that reintroduces a combining step fails loudly rather than quietly.
    """
    for status in ("confirm", "reject", "disputed", "banana"):
        for distance in (0.5, 2.0, 50.0):
            incident, _ = run_graph(
                verifier_status=status, gate_distance=distance, band=BAND
            )
            assert incident["final_verdict"] == incident["gate_verdict"], (
                f"{status} @ {distance}"
            )


def test_model_disagreement_is_recorded_but_not_applied():
    """The audit column: the model said reject, the gate confirmed, and the
    incident shows both without the disagreement changing anything.
    """
    incident, trace = run_graph(
        verifier_status="reject", gate_distance=50.0, band=BAND
    )
    assert incident["gate_verdict"] == "confirm"
    assert incident["llm_verdict"] == "reject"
    assert incident["final_verdict"] == "confirm"
    assert dict(trace)["verification_gate"]["gate_llm_agree"] is False


def test_the_explanation_text_still_comes_from_the_model():
    """The model lost the verdict, not the job. If `llm_reason` stopped being
    carried, the alert would go out with no reasoning in it.
    """
    incident, _ = run_graph(verifier_status="confirm")
    assert incident["llm_reason"], "the operator-facing explanation was dropped"


# --- threshold and calibration are loaded, not hardcoded -----------------------


def test_threshold_comes_from_the_calibration_artifact():
    calibration = json.loads(CALIBRATION_PATH.read_text())
    stored = calibration["operating_threshold"]["distance"]
    provider = ReferenceContextProvider(calibration_path=CALIBRATION_PATH)
    assert provider.operating_threshold == pytest.approx(stored)


def test_no_threshold_literal_is_hardcoded_in_the_graph():
    """Recalibrating must move the gate. A literal here would silently not."""
    source = Path(wf.__file__).read_text()
    for literal in ("2.29", "2.07", "2.0658"):
        assert literal not in source, f"threshold literal {literal} hardcoded in workflow.py"


def test_missing_calibration_defers_to_the_llm_rather_than_inventing_a_cut(tmp_path):
    provider = ReferenceContextProvider(calibration_path=tmp_path / "absent.json")
    assert provider.operating_threshold is None


# --- golden set ----------------------------------------------------------------


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden set not built")
def test_golden_set_ids_resolve_and_are_unique():
    golden = json.loads(GOLDEN_PATH.read_text())
    windows = golden["windows"]
    ids = [w["id"] for w in windows]
    assert len(ids) == len(set(ids)), "duplicate ids in the golden set"
    for window in windows:
        assert window["id"] == f"{window['channel_id']}@{window['window_start'].replace(' ', 'T')}"


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden set not built")
def test_golden_set_covers_every_group_including_faults():
    golden = json.loads(GOLDEN_PATH.read_text())
    groups = {w["group"] for w in golden["windows"]}
    assert groups == {"clear_fault", "clear_expected", "clear_nominal", "ambiguous"}
    faults = [w for w in golden["windows"] if w["group"] == "clear_fault"]
    assert faults, "a golden set with no faults cannot measure the expensive failure"
    assert all(w["true_label"] == "anomaly" for w in faults)


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden set not built")
def test_ambiguous_entries_carry_no_expected_verdict():
    golden = json.loads(GOLDEN_PATH.read_text())
    for window in golden["windows"]:
        if window["group"] == "ambiguous":
            assert window["expected_verdict"] is None
        else:
            assert window["expected_verdict"] in {"confirm", "reject"}


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden set not built")
def test_golden_set_marks_which_rows_are_not_test_data():
    """Clear faults come from the train holdout because the test split has none
    above threshold. That must be visible, or the set overstates its evidence.
    """
    golden = json.loads(GOLDEN_PATH.read_text())
    for window in golden["windows"]:
        assert window["source"] in {"test", "train_holdout"}
        if window["group"] != "clear_fault":
            assert window["source"] == "test"


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden set not built")
def test_golden_set_windows_carry_a_full_feature_vector():
    golden = json.loads(GOLDEN_PATH.read_text())
    widths = {len(w["features"]) for w in golden["windows"]}
    assert widths == {22}, widths


# --- the discarded framings must not return -----------------------------------


def test_no_recognition_framing_survives_in_prompts_or_schemas():
    """The calibration showed nominal windows sit CLOSER to rare-event exemplars
    than rare events do to each other, so resemblance language describes
    something false. Only explanations of why it is banned may mention it.
    """
    import aksha_agent.graph.schemas as schemas

    banned = ("recognised expected pattern", "recognized expected pattern")
    for text in (wf.EXPLAINER_INSTRUCTION, wf.INVESTIGATOR_INSTRUCTION):
        lowered = text.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{phrase!r} returned to a prompt"

    schema_source = Path(schemas.__file__).read_text().lower()
    assert "is a recognised expected pattern" not in schema_source


def test_verifier_instruction_states_what_the_distance_is_not():
    """Without the explicit correction the model reads small distance as
    resemblance, which the data contradicts.
    """
    text = wf.EXPLAINER_INSTRUCTION.lower()
    assert "outlierness" in text
    assert "closest of all" in text or "does not mean" in text
